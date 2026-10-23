"""月度调仓的多因子组合回测（研究用）。

- 因子按**文献方向**取号、截面稳健 z 分数，可选行业中性（行业内去均值）；
- 合成：等权，或滚动 ICIR 加权——只用调仓日之前**已经实现**的 IC，
  与文献方向相反的因子权重置 0；
- 月末收盘打分，次日开盘调仓，等权持有前 N 只；已持有且仍在前 ``N × buffer`` 名内的不换；
- 成本按换手比例扣：买入 0.13%（佣金+滑点），卖出 0.18%（佣金+印花税+滑点）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, cast

import numpy as np
import pandas as pd

from tradecompass.config.loader import Paths
from tradecompass.research import data as rdata
from tradecompass.research.evaluate import (
    PERIODS_PER_YEAR,
    period_returns,
    rank_ic,
    rebalance_dates,
    robust_z,
)
from tradecompass.research.factors import FactorSpec
from tradecompass.research.panel import Panel

BUY_COST = 0.0013
SELL_COST = 0.0018
MIN_IC_HISTORY = 12


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    top: int = 10
    buffer: float = 2.0
    max_per_industry: int | None = 3
    neutralize: bool = False
    weighting: Literal["equal", "icir"] = "equal"
    ic_lookback: int = 36
    max_price: float | None = None

    def label(self) -> str:
        cap = self.max_per_industry if self.max_per_industry is not None else "-"
        price = f" 价≤{self.max_price:.0f}" if self.max_price else ""
        return (
            f"N{self.top} 缓冲{self.buffer:g} 行业≤{cap} "
            f"{'中性' if self.neutralize else '原始'} {self.weighting}{price}"
        )


def _neutral(z: pd.Series, industry: dict[str, str]) -> pd.Series:
    groups = pd.Series([industry.get(code, "?") for code in z.index], index=z.index)
    sizes = groups.map(groups.value_counts())
    demeaned = z - z.groupby(groups).transform("mean")
    # 小行业和未知行业不去均值，避免一两只票被强行拉回 0
    out = demeaned.where((sizes >= 5) & (groups != "?"), z)
    std = out.std()
    return out / std if std > 0 else out


def exposures(
    panel: Panel,
    specs: tuple[FactorSpec, ...],
    days: list[pd.Timestamp],
    industry: dict[str, str],
    *,
    neutralize: bool,
    frames: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """每个因子在每个打分日的已取号 z 分数（日期 × 代码）。"""
    eligible = panel.eligible()
    raw = frames if frames is not None else {}
    out: dict[str, dict[pd.Timestamp, pd.Series]] = {spec.name: {} for spec in specs}
    for spec in specs:
        if spec.name not in raw:
            raw[spec.name] = spec.compute(panel)
    for day in days:
        mask = cast(pd.Series, eligible.loc[day])
        for spec in specs:
            values = cast(pd.Series, raw[spec.name].loc[day])[mask].dropna()
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < 30:
                continue
            sign = spec.expected if spec.expected != 0 else 1
            z = robust_z(values * sign)
            out[spec.name][day] = _neutral(z, industry) if neutralize else z
    return {name: pd.DataFrame(rows).T for name, rows in out.items()}


def factor_weights(
    expo: dict[str, pd.DataFrame],
    forward: pd.DataFrame,
    days: list[pd.Timestamp],
    *,
    weighting: str,
    lookback: int,
) -> pd.DataFrame:
    """每个打分日的因子权重（行和为 1）。"""
    names = list(expo)
    equal = pd.DataFrame(1.0 / len(names), index=days, columns=names)
    if weighting == "equal":
        return equal
    ic = pd.DataFrame(index=list(forward.index), columns=names, dtype="float64")
    for name, frame in expo.items():
        for day in forward.index:
            if day in frame.index:
                ic.loc[day, name] = rank_ic(frame.loc[day], forward.loc[day])
    weights = equal.copy()
    for i, day in enumerate(days):
        # 第 j 期收益在 days[j+1] 的次日开盘才实现，打分日 day 只能用 j+1 < i 的期
        known = ic[ic.index < day].iloc[:-1] if i > 0 else ic.iloc[:0]
        known = known.tail(lookback)
        if len(known) < MIN_IC_HISTORY:
            continue
        icir = known.mean() / known.std().replace(0.0, np.nan)
        raw = icir.clip(lower=0.0).fillna(0.0)
        if raw.sum() > 0:
            weights.loc[day] = raw / raw.sum()
    return weights


def combine(
    expo: dict[str, pd.DataFrame], weights: pd.DataFrame, days: list[pd.Timestamp]
) -> pd.DataFrame:
    rows: dict[pd.Timestamp, pd.Series] = {}
    for day in days:
        rows[day] = pd.Series(dtype="float64")
        parts = {name: frame.loc[day] for name, frame in expo.items() if day in frame.index}
        if not parts:
            continue
        joined = pd.DataFrame(parts)
        w = weights.loc[day].reindex(joined.columns).fillna(0.0)
        if w.sum() <= 0:
            continue
        # 至少一半因子有值才打分
        enough = joined.notna().sum(axis=1) >= max(1, len(expo) // 2)
        avail = joined.notna().mul(w, axis=1).sum(axis=1)
        score = joined.fillna(0.0).mul(w, axis=1).sum(axis=1) / avail.where(avail > 0)
        rows[day] = score[enough].dropna()
    return pd.DataFrame(rows).T


def composite_score(
    panel: Panel, specs: tuple[FactorSpec, ...], days: list[pd.Timestamp]
) -> pd.DataFrame:
    """等权、不中性的合成分（兼容旧接口）。"""
    expo = exposures(panel, specs, days, {}, neutralize=False)
    weights = pd.DataFrame(1.0 / len(specs), index=days, columns=[s.name for s in specs])
    return combine(expo, weights, days)


@dataclass(frozen=True, slots=True)
class PortfolioResult:
    returns: pd.Series
    benchmarks: dict[str, pd.Series]
    turnover: pd.Series
    holdings: dict[pd.Timestamp, list[str]]

    @staticmethod
    def stats(returns: pd.Series) -> dict[str, float]:
        equity = (1.0 + returns).cumprod()
        years = len(returns) / PERIODS_PER_YEAR
        total = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
        annual = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
        vol = float(returns.std() * np.sqrt(PERIODS_PER_YEAR))
        drawdown = float((equity / equity.cummax() - 1.0).min()) if not equity.empty else 0.0
        return {
            "total": total,
            "annual": annual,
            "vol": vol,
            "sharpe": annual / vol if vol > 0 else float("nan"),
            "max_dd": drawdown,
            "win_month": float((returns > 0).mean()),
        }

    def yearly(self, series: pd.Series) -> dict[int, float]:
        years = pd.DatetimeIndex(series.index).year
        values = series.to_numpy(dtype="float64")
        return {
            int(y): float(np.prod(1.0 + values[years == y]) - 1.0) for y in sorted(set(years))
        }


def _fill(
    kept: list[str],
    ranked: list[str],
    top: int,
    industry: dict[str, str],
    max_per_industry: int | None,
) -> list[str]:
    """在保留持仓基础上按排名补足 top 只；未知行业不受上限约束。"""
    target = list(kept)
    counts: dict[str, int] = {}
    for code in target:
        name = industry.get(code)
        if name:
            counts[name] = counts.get(name, 0) + 1
    for code in ranked:
        if len(target) >= top:
            break
        if code in target:
            continue
        name = industry.get(code)
        if max_per_industry is not None and name and counts.get(name, 0) >= max_per_industry:
            continue
        target.append(code)
        if name:
            counts[name] = counts.get(name, 0) + 1
    return target


def run_portfolio(
    panel: Panel,
    specs: tuple[FactorSpec, ...],
    start: date,
    end: date,
    *,
    top: int = 10,
    buffer: float = 2.0,
    benchmark_bars: dict[str, pd.DataFrame] | None = None,
    industry: dict[str, str] | None = None,
    max_per_industry: int | None = None,
) -> PortfolioResult:
    days = rebalance_dates(panel.dates, start, end)
    if len(days) < 3:
        raise ValueError(f"{start} → {end} 内调仓日不足 3 个")
    forward = period_returns(panel, days)
    scores = composite_score(panel, specs, list(forward.index))
    held: list[str] = []
    returns: dict[pd.Timestamp, float] = {}
    turnovers: dict[pd.Timestamp, float] = {}
    holdings: dict[pd.Timestamp, list[str]] = {}
    universe: dict[pd.Timestamp, float] = {}
    eligible = panel.eligible()
    for day in forward.index:
        ranked = scores.loc[day].dropna().sort_values(ascending=False)
        keep_zone = set(ranked.index[: int(top * buffer)])
        kept = [code for code in held if code in keep_zone]
        target = _fill(kept, list(ranked.index), top, industry or {}, max_per_industry)
        if not target:
            continue
        changed = len(set(target) - set(held)) / len(target)
        cost = changed * (BUY_COST + (SELL_COST if held else 0.0))
        period = forward.loc[day].reindex(target).fillna(0.0)
        returns[day] = float(period.mean()) - cost
        turnovers[day] = changed
        holdings[day] = target
        universe[day] = float(forward.loc[day][eligible.loc[day]].mean())
        held = target

    benchmarks = {"股票池等权": pd.Series(universe)}
    for name, bars in (benchmark_bars or {}).items():
        if bars.empty:
            continue
        price = bars["open"].reindex(panel.dates).ffill()
        series = {}
        for i, day in enumerate(days[:-1]):
            entry = panel.dates[int(panel.dates.searchsorted(day)) + 1]
            exit_ = panel.dates[int(panel.dates.searchsorted(days[i + 1])) + 1]
            series[day] = float(price.loc[exit_] / price.loc[entry] - 1.0)
        benchmarks[name] = pd.Series(series).reindex(list(returns))
    return PortfolioResult(
        returns=pd.Series(returns),
        benchmarks=benchmarks,
        turnover=pd.Series(turnovers),
        holdings=holdings,
    )


def load_benchmarks(paths: Paths) -> dict[str, pd.DataFrame]:
    names = {"hs300": "沪深300", "zz500": "中证500"}
    return {
        names[key]: rdata.read_index_bars(paths, code) for key, code in rdata.BENCHMARKS.items()
    }


def latest_picks(
    panel: Panel,
    specs: tuple[FactorSpec, ...],
    top: int,
    industry: dict[str, str] | None = None,
    max_per_industry: int | None = None,
) -> list[str]:
    """用最新一个交易日收盘打分的前 N 名（即下一次调仓的目标）。"""
    day = pd.Timestamp(panel.dates[-1])
    scores = composite_score(panel, specs, [day])
    row = pd.Series(scores.loc[day], dtype="float64").dropna()
    ranked = list(row.sort_values(ascending=False).index)
    return _fill([], ranked, top, industry or {}, max_per_industry)


def render_portfolio_markdown(
    result: PortfolioResult, specs: tuple[FactorSpec, ...], start: date, end: date, top: int
) -> str:
    series = {"组合（扣成本）": result.returns, **result.benchmarks}
    lines = [
        f"# 多因子组合回测  {start} → {end}",
        "",
        "因子（文献方向等权合成）："
        + "、".join(f"{s.label}({'+' if s.expected >= 0 else '−'})" for s in specs),
        f"持仓 {top} 只等权，月末打分、次日开盘调仓；平均换手 {result.turnover.mean():.0%}/月。",
        "",
        "| 序列 | 总收益 | 年化 | 波动 | 夏普 | 最大回撤 | 月胜率 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, values in series.items():
        s = PortfolioResult.stats(values.dropna())
        lines.append(
            f"| {name} | {s['total']:+.1%} | {s['annual']:+.1%} | {s['vol']:.1%} | "
            f"{s['sharpe']:.2f} | {s['max_dd']:.1%} | {s['win_month']:.0%} |"
        )
    years = sorted(result.yearly(result.returns))
    lines += ["", "## 分年度", "", "| 序列 | " + " | ".join(str(y) for y in years) + " |"]
    lines.append("|---|" + "---|" * len(years))
    for name, values in series.items():
        yearly = result.yearly(values.dropna())
        lines.append(
            f"| {name} | " + " | ".join(f"{yearly.get(y, float('nan')):+.1%}" for y in years) + " |"
        )
    last_day = max(result.holdings) if result.holdings else None
    if last_day is not None:
        names = "、".join(result.holdings[last_day])
        lines += ["", f"回测最后一期（{last_day.date()}）持仓：{names}"]
    return "\n".join(lines) + "\n"
