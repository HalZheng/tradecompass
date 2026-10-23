"""事件研究：把现有 B1/B2 买点放到无幸存者偏差的大股票池里，看触发后的超额收益。

每个事件在信号日收盘确认、**次日开盘**进场，持有 h 个交易日后按开盘价计；
超额 = 事件收益 − 同日全部可选股票的等权平均收益。t 值按“事件日”聚类
（同一天多只同时触发只算一个观测），避免同日相关性夸大显著性。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from tradecompass.research.factors import ewm_rsi, macd_hist, position, vol_ratio20
from tradecompass.research.panel import Panel

HORIZONS = (5, 20, 60)


def tdx_sma_wide(values: np.ndarray, n: int, m: int, seed: float = 50.0) -> np.ndarray:
    """通达信 SMA 的逐行递推，对所有列同时计算；NaN 段沿用前值。"""
    out = np.full(values.shape, np.nan)
    prev = np.full(values.shape[1], seed)
    started = np.zeros(values.shape[1], dtype=bool)
    for i, row in enumerate(values):
        valid = ~np.isnan(row)
        prev = np.where(valid, (m * row + (n - m) * prev) / n, prev)
        started |= valid
        out[i] = np.where(started, prev, np.nan)
    return out


@dataclass(frozen=True, slots=True)
class Weekly:
    """已完结周线，按日对齐：某日看到的是“当日或之前最后一个完结周”。"""

    close: pd.DataFrame
    ma20: pd.DataFrame
    ma20_up: pd.DataFrame
    j: pd.DataFrame


def weekly_on_daily(panel: Panel) -> Weekly:
    week_close = panel.adj_close.resample("W-FRI").last()
    week_high = panel.adj_high.resample("W-FRI").max()
    week_low = panel.adj_low.resample("W-FRI").min()
    lowest = week_low.rolling(9, min_periods=9).min()
    highest = week_high.rolling(9, min_periods=9).max()
    span = highest - lowest
    rsv = ((week_close - lowest) / span.where(span != 0) * 100.0).where(span != 0, 50.0)
    rsv = rsv.where(week_close.notna() & lowest.notna())
    k = tdx_sma_wide(rsv.to_numpy(dtype="float64"), 3, 1)
    d = tdx_sma_wide(k, 3, 1)
    j = pd.DataFrame(3.0 * k - 2.0 * d, index=week_close.index, columns=week_close.columns)
    ma20 = week_close.rolling(20, min_periods=20).mean()

    def to_daily(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.reindex(panel.dates, method="ffill")

    return Weekly(
        close=to_daily(week_close),
        ma20=to_daily(ma20),
        ma20_up=to_daily((ma20 > ma20.shift(1)).astype("float64")) == 1.0,
        j=to_daily(j),
    )


def b2_strict(panel: Panel, weekly: Weekly) -> pd.DataFrame:
    """当前 B2 硬条件（HANDOFF §3）的横截面版本；不含大盘/行业准入。"""
    close = panel.adj_close
    breakout = close > close.shift(1).rolling(60, min_periods=60).max()
    return (
        breakout
        & (vol_ratio20(panel) >= 1.5)
        & (macd_hist(close) > 0)
        & (ewm_rsi(close, 6) <= 80)
        & (position(panel, 250) < 95)
        & (weekly.close > weekly.ma20)
    )


def b2_loose(panel: Panel, weekly: Weekly) -> pd.DataFrame:
    """PLAN v1 原始放量突破：创 60 日新高 + 量比 ≥ 2。"""
    close = panel.adj_close
    return (close > close.shift(1).rolling(60, min_periods=60).max()) & (vol_ratio20(panel) >= 2.0)


def b1_v1(panel: Panel, weekly: Weekly) -> pd.DataFrame:
    """PLAN v1 回调买点：周线多头 + 周 J < 20 + 日线重新站上 MA5。"""
    close = panel.adj_close
    ma5 = close.rolling(5, min_periods=5).mean()
    cross = (close > ma5) & (close.shift(1) <= ma5.shift(1))
    return (
        (weekly.close > weekly.ma20)
        & weekly.ma20_up
        & (weekly.j < 20)
        & cross
        & (position(panel, 250) < 95)
    )


EVENTS: dict[str, tuple[str, Callable[[Panel, Weekly], pd.DataFrame]]] = {
    "b2_strict": ("B2 当前硬条件", b2_strict),
    "b2_loose": ("B2 原始放量突破", b2_loose),
    "b1_v1": ("B1 周J低+站上MA5", b1_v1),
}


@dataclass(frozen=True, slots=True)
class EventResult:
    name: str
    label: str
    count: int
    days: int
    stats: dict[int, tuple[float, float, float, float]]
    yearly: dict[int, tuple[int, float]]


def forward_open_returns(panel: Panel, horizon: int) -> pd.DataFrame:
    price = panel.adj_open.ffill()
    return price.shift(-1 - horizon) / price.shift(-1) - 1.0


def event_study(
    panel: Panel,
    name: str,
    label: str,
    events: pd.DataFrame,
    start: date,
    end: date,
    horizons: tuple[int, ...] = HORIZONS,
) -> EventResult:
    eligible = panel.eligible()
    window = (panel.dates >= pd.Timestamp(start)) & (panel.dates <= pd.Timestamp(end))
    mask = (events.fillna(False).astype(bool) & eligible).loc[window]
    stats: dict[int, tuple[float, float, float, float]] = {}
    yearly: dict[int, tuple[int, float]] = {}
    count = int(mask.to_numpy().sum())
    days = int(mask.any(axis=1).to_numpy().sum())
    for horizon in horizons:
        fwd = forward_open_returns(panel, horizon).loc[window]
        base = fwd.where(eligible.loc[window]).mean(axis=1)
        excess = fwd.sub(base, axis=0).where(mask)
        stacked = excess.stack().dropna()
        if stacked.empty:
            stats[horizon] = (float("nan"),) * 4
            continue
        per_day = excess.mean(axis=1).dropna()
        t = 0.0
        if len(per_day) > 1:
            t = float(per_day.mean() / per_day.std() * np.sqrt(len(per_day)))
        stats[horizon] = (
            float(stacked.mean()),
            float(stacked.median()),
            float((stacked > 0).mean()),
            t,
        )
        if horizon == 20:
            years = pd.DatetimeIndex(stacked.index.get_level_values(0)).year
            for year in sorted(set(years)):
                chunk = stacked[years == year]
                yearly[int(year)] = (len(chunk), float(chunk.mean()))
    return EventResult(name, label, count, days, stats, yearly)


def run_events(panel: Panel, start: date, end: date) -> list[EventResult]:
    weekly = weekly_on_daily(panel)
    return [
        event_study(panel, name, label, builder(panel, weekly), start, end)
        for name, (label, builder) in EVENTS.items()
    ]


def render_events_markdown(results: list[EventResult], start: date, end: date) -> str:
    lines = [
        f"# 现有买点事件研究  {start} → {end}",
        "",
        "股票池同因子体检（历史沪深300+中证500）。次日开盘进场，超额相对同日可选股票等权；"
        "t 值按事件日聚类。**未扣成本**（一次往返约 0.2%–0.3%）；不含大盘/行业准入。",
        "",
        "| 买点 | 事件数 | 事件日 | " + " | ".join(
            f"{h}日 均值/中位/胜率/t" for h in HORIZONS
        ) + " |",
        "|---|---|---|" + "---|" * len(HORIZONS),
    ]
    for r in results:
        cells = []
        for h in HORIZONS:
            mean, median, hit, t = r.stats.get(h, (float("nan"),) * 4)
            cells.append(f"{mean:+.2%} / {median:+.2%} / {hit:.0%} / {t:+.1f}")
        lines.append(f"| {r.label} `{r.name}` | {r.count} | {r.days} | " + " | ".join(cells) + " |")
    years = sorted({y for r in results for y in r.yearly})
    lines += ["", "## 20 日超额分年度（事件数 / 均值）", ""]
    lines.append("| 买点 | " + " | ".join(str(y) for y in years) + " |")
    lines.append("|---|" + "---|" * len(years))
    for r in results:
        cells = []
        for y in years:
            n, mean = r.yearly.get(y, (0, float("nan")))
            cells.append(f"{n} / {mean:+.2%}")
        lines.append(f"| {r.name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
