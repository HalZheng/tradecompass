"""横截面因子评估：RankIC、市值中性 IC、五分组超额、换手、分年度稳定性。

口径：每月最后一个交易日收盘计算因子，**次日开盘**进场，下一个月末的次日开盘出场；
股票池为当时的沪深300+中证500 成分（无幸存者偏差），剔除停牌、ST、上市不足 120 日、
次日开盘涨停买不进的标的。收益为相对股票池等权平均的超额，未扣交易成本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from tradecompass.research.factors import FactorSpec, log_frame
from tradecompass.research.panel import Panel

PERIODS_PER_YEAR = 12
QUANTILES = 5


def rebalance_dates(dates: pd.DatetimeIndex, start: date, end: date) -> list[pd.Timestamp]:
    """每月最后一个交易日，且之后还有交易日（保证次日开盘可成交）。"""
    frame = pd.Series(dates, index=dates)
    month_end = frame.groupby([dates.year, dates.month]).max()
    out = [pd.Timestamp(d) for d in month_end if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    return [d for d in out if d < dates[-1]]


def period_returns(panel: Panel, rebalance: list[pd.Timestamp]) -> pd.DataFrame:
    """第 i 行：rebalance[i] 次日开盘 → rebalance[i+1] 次日开盘的收益。"""
    dates = panel.dates
    price = panel.adj_open.ffill()
    entry_pos = [int(dates.searchsorted(d)) + 1 for d in rebalance]
    rows = {}
    for i in range(len(rebalance) - 1):
        start_px = price.iloc[entry_pos[i]]
        end_px = price.iloc[entry_pos[i + 1]]
        rows[rebalance[i]] = end_px / start_px - 1.0
    return pd.DataFrame(rows).T


def robust_z(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return values * 0.0
    clipped = values.clip(median - 3 * mad, median + 3 * mad)
    std = clipped.std()
    return (clipped - clipped.mean()) / std if std > 0 else clipped * 0.0


def neutralize(values: pd.Series, exposure: pd.Series) -> pd.Series:
    """对 exposure 做截面 OLS，返回残差。"""
    joined = pd.concat([values, exposure], axis=1, keys=["y", "x"]).dropna()
    if len(joined) < 30:
        return pd.Series(dtype="float64")
    x = np.column_stack([np.ones(len(joined)), joined["x"].to_numpy()])
    y = joined["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return pd.Series(y - x @ beta, index=joined.index)


def rank_ic(values: pd.Series, forward: pd.Series) -> float:
    joined = pd.concat([values, forward], axis=1).dropna()
    if len(joined) < 30:
        return float("nan")
    return float(joined.iloc[:, 0].rank().corr(joined.iloc[:, 1].rank()))


@dataclass(frozen=True, slots=True)
class FactorResult:
    spec: FactorSpec
    ic: pd.Series
    ic_neutral: pd.Series
    quantile_excess: pd.DataFrame
    coverage: float
    long_turnover: float
    yearly_ic: dict[int, float] = field(default_factory=dict)

    @property
    def ic_mean(self) -> float:
        return float(self.ic.mean())

    @property
    def icir(self) -> float:
        std = float(self.ic.std())
        return self.ic_mean / std if std > 0 else float("nan")

    @property
    def t_stat(self) -> float:
        return self.icir * float(np.sqrt(self.ic.count()))

    @property
    def positive_rate(self) -> float:
        return float((self.ic > 0).mean())

    @property
    def direction(self) -> int:
        return 1 if self.ic_mean >= 0 else -1

    def annual_excess(self) -> list[float]:
        return [float(v) * PERIODS_PER_YEAR for v in self.quantile_excess.mean()]

    @property
    def long_excess(self) -> float:
        """按样本 IC 方向做多那一端的年化超额。"""
        values = self.annual_excess()
        return values[-1] if self.direction > 0 else values[0]

    @property
    def long_short(self) -> float:
        values = self.annual_excess()
        return (values[-1] - values[0]) * self.direction


def evaluate_factor(
    spec: FactorSpec,
    values: pd.DataFrame,
    panel: Panel,
    rebalance: list[pd.Timestamp],
    forward: pd.DataFrame,
) -> FactorResult:
    eligible = panel.eligible()
    log_mcap = log_frame(panel.float_mcap)
    ics: dict[pd.Timestamp, float] = {}
    neutral_ics: dict[pd.Timestamp, float] = {}
    buckets: dict[pd.Timestamp, list[float]] = {}
    counts: list[int] = []
    top_sets: list[set[str]] = []
    bottom_sets: list[set[str]] = []
    for day in forward.index:
        mask = eligible.loc[day]
        fwd = forward.loc[day][mask]
        raw = values.loc[day][mask]
        joined = pd.concat([raw, fwd], axis=1, keys=["f", "r"]).dropna()
        if len(joined) < 50:
            continue
        counts.append(len(joined))
        ics[day] = rank_ic(joined["f"], joined["r"])
        z = robust_z(joined["f"])
        if spec.name != "size":
            resid = neutralize(z, log_mcap.loc[day].reindex(joined.index))
            neutral_ics[day] = rank_ic(resid, joined["r"])
        excess = joined["r"] - joined["r"].mean()
        group = pd.qcut(joined["f"].rank(method="first"), QUANTILES, labels=False)
        buckets[day] = [float(v) for v in excess.groupby(group).mean().reindex(range(QUANTILES))]
        top_sets.append(set(group[group == QUANTILES - 1].index))
        bottom_sets.append(set(group[group == 0].index))

    ic = pd.Series(ics, dtype="float64")
    quantile_excess = pd.DataFrame.from_dict(
        buckets, orient="index", columns=[f"Q{i + 1}" for i in range(QUANTILES)]
    )
    long_sets = top_sets if ic.empty or ic.mean() >= 0 else bottom_sets
    turnover = [
        1.0 - len(a & b) / len(b) for a, b in zip(long_sets, long_sets[1:], strict=False) if b
    ]
    years = pd.DatetimeIndex(ic.index).year
    yearly = {int(year): float(ic[years == year].mean()) for year in sorted(set(years))}
    return FactorResult(
        spec=spec,
        ic=ic,
        ic_neutral=pd.Series(neutral_ics, dtype="float64"),
        quantile_excess=quantile_excess,
        coverage=float(np.mean(counts)) if counts else 0.0,
        long_turnover=float(np.mean(turnover)) if turnover else float("nan"),
        yearly_ic=yearly,
    )


def evaluate_all(
    panel: Panel, specs: tuple[FactorSpec, ...], start: date, end: date
) -> list[FactorResult]:
    rebalance = rebalance_dates(panel.dates, start, end)
    if len(rebalance) < 3:
        raise ValueError(f"{start} → {end} 内调仓日不足 3 个")
    forward = period_returns(panel, rebalance)
    return [evaluate_factor(spec, spec.compute(panel), panel, rebalance, forward) for spec in specs]


def render_ic_markdown(results: list[FactorResult], start: date, end: date) -> str:
    years = sorted({y for r in results for y in r.yearly_ic})
    lines = [
        f"# 横截面因子体检  {start} → {end}",
        "",
        "股票池：当时的沪深300+中证500 成分；月末收盘算因子，次日开盘调仓；"
        "剔除停牌/ST/新股/次日涨停。收益为相对股票池等权的超额，**未扣成本**。",
        "",
        "判读：|IC| ≥ 0.03 且 |t| ≥ 2 才算有信号；Qlib Alpha158+LightGBM 在 CSI300 上 "
        "RankIC 约 0.047，可作为“成熟水平”的参照。",
        "",
        "| 因子 | 组 | 文献方向 | RankIC | ICIR | t | IC>0 | 市值中性IC | "
        "Q1 | Q2 | Q3 | Q4 | Q5 | 多头超额 | 多空 | 多头月换手 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: -abs(x.t_stat) if np.isfinite(x.t_stat) else 0):
        q = r.annual_excess()
        expected = {1: "+", -1: "−", 0: "?"}[r.spec.expected]
        neutral = r.ic_neutral.mean() if not r.ic_neutral.empty else float("nan")
        lines.append(
            f"| {r.spec.label} `{r.spec.name}` | {r.spec.group} | {expected} | "
            f"{r.ic_mean:+.3f} | {r.icir:+.2f} | {r.t_stat:+.1f} | {r.positive_rate:.0%} | "
            f"{neutral:+.3f} | "
            + " | ".join(f"{v:+.1%}" for v in q)
            + f" | {r.long_excess:+.1%} | {r.long_short:+.1%} | {r.long_turnover:.0%} |"
        )
    lines += ["", "## 分年度 RankIC", "", "| 因子 | " + " | ".join(str(y) for y in years) + " |"]
    lines.append("|---|" + "---|" * len(years))
    for r in results:
        cells = [f"{r.yearly_ic.get(y, float('nan')):+.3f}" for y in years]
        lines.append(f"| {r.spec.name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"平均每期样本数：{np.mean([r.coverage for r in results]):.0f}；"
        f"调仓期数：{max((r.ic.count() for r in results), default=0)}。",
    ]
    return "\n".join(lines) + "\n"
