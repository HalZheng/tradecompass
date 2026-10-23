"""市场层：决定今天最多能拿几成仓（STATE_MACHINE §2）。

输入只有指数：沪深300 的周线趋势、20 日动量、波动率分位。
输出四个状态与对应额度，并带两个记忆项避免抖动：
1. 状态切换需连续 2 日确认；
2. 从 PANIC 恢复后 10 个交易日内额度上限压到一半。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd

from tradecompass.indicators.frame import IndicatorParams, weekly_frame

CONFIRM_DAYS = 2
PANIC_COOLDOWN_DAYS = 10
PANIC_VALUE_QUOTA = 0.5
PANIC_DROP = -0.08
PANIC_VOL_PERCENTILE = 0.80


class Regime(StrEnum):
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"
    PANIC = "PANIC"


QUOTA: dict[Regime, float] = {
    Regime.RISK_ON: 1.00,
    Regime.NEUTRAL: 0.60,
    Regime.RISK_OFF: 0.25,
    Regime.PANIC: 0.00,
}


@dataclass(frozen=True, slots=True)
class RegimePoint:
    day: date
    raw: Regime
    state: Regime
    quota: float
    note: str


@dataclass(frozen=True, slots=True)
class IndexFeatures:
    """单日的指数特征，全部来自收盘后计算。"""

    close: float
    ma20w: float | None
    ma20w_up: bool | None
    ma10: float | None
    ret20: float
    vol_percentile: float | None


def classify(features: IndexFeatures) -> Regime:
    """把特征映射成"当日原始判定"，未经过确认。"""
    if features.ma20w is None or features.ma20w_up is None:
        return Regime.NEUTRAL
    below_weekly = features.close < features.ma20w
    weekly_down = features.ma20w_up is False

    if below_weekly and weekly_down:
        vol_hot = (
            features.vol_percentile is not None
            and features.vol_percentile >= PANIC_VOL_PERCENTILE
        )
        if features.ret20 <= PANIC_DROP or vol_hot:
            return Regime.PANIC
        return Regime.RISK_OFF

    trend_up = not below_weekly and features.ma20w_up is True
    if trend_up and features.ret20 > 0:
        return Regime.RISK_ON
    return Regime.NEUTRAL


def _features(index_bars: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
    """在指数日线上算出每日特征。索引与 ``index_bars`` 对齐。"""
    close = index_bars["close"].astype(float)
    weekly = weekly_frame(index_bars, index_bars.index[-1].date(), params)

    # 把周线指标映射回交易日：用当周之前**已完结**的那一周
    labels = weekly.index
    positions = labels.searchsorted(index_bars.index, side="right") - 1
    same_week = np.array(
        [
            positions[i] >= 0
            and labels[positions[i]].isocalendar()[:2]
            == index_bars.index[i].isocalendar()[:2]
            for i in range(len(positions))
        ]
    )
    positions = positions - same_week

    ma20w = np.full(len(index_bars), np.nan)
    up = np.full(len(index_bars), np.nan)
    valid = positions >= 0
    ma20w[valid] = weekly["ma20w"].to_numpy()[positions[valid]]
    up[valid] = weekly["ma20w_up"].to_numpy()[positions[valid]]

    returns = close.pct_change()
    vol = returns.rolling(20).std()
    vol_rank = vol.rolling(250, min_periods=60).rank(pct=True)

    frame = pd.DataFrame(
        {
            "close": close,
            "ma20w": ma20w,
            "ma20w_up": up,
            "ma10": close.rolling(10).mean(),
            "ret20": close.pct_change(20),
            "vol_percentile": vol_rank,
        },
        index=index_bars.index,
    )
    return frame


def build_series(
    index_bars: pd.DataFrame, params: IndicatorParams | None = None
) -> list[RegimePoint]:
    """逐日状态序列，含确认与冷却期记忆。"""
    if index_bars.empty:
        return []
    frame = _features(index_bars, params or IndicatorParams())

    points: list[RegimePoint] = []
    state = Regime.NEUTRAL
    pending: Regime | None = None
    pending_days = 0
    panic_exit_until = -1
    day_index = 0

    for i in range(len(frame)):
        row = frame.iloc[i]
        day_value = frame.index[i].date()
        features = IndexFeatures(
            close=float(row["close"]),
            ma20w=None if pd.isna(row["ma20w"]) else float(row["ma20w"]),
            ma20w_up=None if pd.isna(row["ma20w_up"]) else bool(row["ma20w_up"] > 0),
            ma10=None if pd.isna(row["ma10"]) else float(row["ma10"]),
            ret20=0.0 if pd.isna(row["ret20"]) else float(row["ret20"]),
            vol_percentile=(
                None if pd.isna(row["vol_percentile"]) else float(row["vol_percentile"])
            ),
        )
        raw = classify(features)
        note = ""

        # 连续 CONFIRM_DAYS 天才切换；PANIC 立即生效（来不及等）
        if raw == state:
            pending, pending_days = None, 0
        elif raw == Regime.PANIC:
            state = Regime.PANIC
            pending, pending_days = None, 0
            panic_exit_until = day_index + PANIC_COOLDOWN_DAYS
            note = "恐慌：立即降为空仓"
        else:
            if pending == raw:
                pending_days += 1
            else:
                pending, pending_days = raw, 1
            if pending_days >= CONFIRM_DAYS:
                # 从 PANIC 回升必须先站上 MA10
                if state == Regime.PANIC and features.ma10 is not None:
                    if features.close < features.ma10:
                        note = "原为恐慌，未站上 MA10，暂不恢复"
                        points.append(
                            RegimePoint(
                                day=day_value,
                                raw=raw,
                                state=state,
                                quota=QUOTA[state],
                                note=note,
                            )
                        )
                        day_index += 1
                        continue
                state = raw
                pending, pending_days = None, 0
                note = "确认切换"

        quota = QUOTA[state]
        if 0 <= day_index < panic_exit_until:
            capped = min(quota, PANIC_VALUE_QUOTA)
            if capped < quota:
                note = (note + "；" if note else "") + "恐慌恢复期，额度减半"
            quota = capped

        points.append(
            RegimePoint(
                day=day_value,
                raw=raw,
                state=state,
                quota=quota,
                note=note,
            )
        )
        day_index += 1

    return points


def quota_on(points: list[RegimePoint], day: date) -> float:
    """某日的仓位额度；没有数据时按 0 处理（宁可空仓）。"""
    for point in reversed(points):
        if point.day <= day:
            return point.quota
    return 0.0
