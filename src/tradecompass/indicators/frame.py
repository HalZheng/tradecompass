"""把一个标的的日线/周线整理成完整指标表（SPEC §9.2、§8.4）。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from tradecompass.indicators import core


@dataclass(frozen=True, slots=True)
class IndicatorParams:
    ma_periods: tuple[int, ...] = (5, 10, 20, 60)
    kdj_n: int = 9
    kdj_m1: int = 3
    kdj_m2: int = 3
    atr_n: int = 14
    vol_ma_periods: tuple[int, ...] = (5, 20)
    high_lookback: int = 60
    low_lookback: int = 10
    weekly_ma: int = 20
    weekly_short_ma: tuple[int, ...] = (5, 10)
    weekly_slope_lookback: int = 5
    rsi_periods: tuple[int, ...] = (6, 12, 24)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    position_lookbacks: tuple[int, ...] = (60, 250)
    return_periods: tuple[int, ...] = (5, 20, 60)
    ma_slope_lookback: int = 3


def daily_frame(bars_qfq: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
    """日线指标表。

    必需列：``ma5 ma10 ma20 ma60 k d j atr14 vol_ma5 vol_ma20
    hhv_close_60_prev llv_low_10``；此外还带 RSI、MACD、价格位置、
    量能趋势与收益率，供因子层使用。
    """
    close = bars_qfq["close"]
    volume = bars_qfq["volume"]
    frame = pd.DataFrame(index=bars_qfq.index)
    # 把前复权收盘价一并带上：策略判定"收盘 > MA5"这类条件时需要它，
    # 省得每个策略都自己再对齐一次行情。
    frame["close"] = close.astype(float)

    for period in params.ma_periods:
        frame[f"ma{period}"] = core.ma(close, period)
        frame[f"ma{period}_up"] = core.slope_up(
            frame[f"ma{period}"], params.ma_slope_lookback
        )

    kdj = core.kdj(
        bars_qfq["high"],
        bars_qfq["low"],
        close,
        n=params.kdj_n,
        m1=params.kdj_m1,
        m2=params.kdj_m2,
    )
    frame["k"] = kdj["k"]
    frame["d"] = kdj["d"]
    frame["j"] = kdj["j"]
    frame["j_up"] = (frame["j"] > frame["j"].shift(1)).fillna(False)
    frame["j_cross_k"] = (frame["k"] > frame["d"]).fillna(False)
    # 相对该票自身历史的 J 位置：不同波动率的票，绝对 J 值不可直接比较。
    frame["j_pct"] = frame["j"].rolling(250, min_periods=60).rank(pct=True) * 100.0

    frame[f"atr{params.atr_n}"] = core.atr(
        bars_qfq["high"], bars_qfq["low"], close, n=params.atr_n
    )

    for period in params.rsi_periods:
        frame[f"rsi{period}"] = core.rsi(close, period)

    macd = core.macd(
        close,
        fast=params.macd_fast,
        slow=params.macd_slow,
        signal=params.macd_signal,
    )
    frame["dif"] = macd["dif"]
    frame["dea"] = macd["dea"]
    frame["macd_hist"] = macd["macd_hist"]
    frame["macd_hist_up"] = (
        frame["macd_hist"] > frame["macd_hist"].shift(1)
    ).fillna(False)
    frame["macd_cross_up"] = (
        (frame["dif"] > frame["dea"]) & (frame["dif"].shift(1) <= frame["dea"].shift(1))
    ).fillna(False)
    frame["macd_cross_down"] = (
        (frame["dif"] < frame["dea"]) & (frame["dif"].shift(1) >= frame["dea"].shift(1))
    ).fillna(False)

    for period in params.vol_ma_periods:
        frame[f"vol_ma{period}"] = core.ma(volume, period)

    frame["vol_ratio5"] = volume.astype(float) / frame["vol_ma5"].replace(0.0, np.nan)
    frame["vol_ratio20"] = volume.astype(float) / frame["vol_ma20"].replace(0.0, np.nan)
    frame["vol_ma5_gt_ma20"] = (frame["vol_ma5"] > frame["vol_ma20"]).fillna(False)

    ret1 = close.pct_change()
    frame["ret1"] = ret1
    direction = pd.Series(0.0, index=close.index, dtype="float64")
    direction[(ret1 > 0) & (frame["vol_ratio20"] > 1.2)] = 1.0
    direction[(ret1 < 0) & (frame["vol_ratio20"] > 1.2)] = -1.0
    frame["vol_price_dir"] = direction

    for period in params.return_periods:
        frame[f"ret{period}"] = close.pct_change(period)

    for lookback in params.position_lookbacks:
        lowest = core.llv(bars_qfq["low"], lookback)
        highest = core.hhv(bars_qfq["high"], lookback)
        span = highest - lowest
        position = (close.astype(float) - lowest) / span.where(span != 0.0) * 100.0
        frame[f"pos_{lookback}"] = position.where(span != 0.0, 50.0).clip(0.0, 100.0)

    frame[f"hhv_close_{params.high_lookback}_prev"] = core.hhv(
        close.shift(1), params.high_lookback
    )
    frame["hhv_close_250_prev"] = core.hhv(close.shift(1), 250)
    frame[f"llv_low_{params.low_lookback}"] = core.llv(bars_qfq["low"], params.low_lookback)
    frame[f"llv_low_{params.low_lookback}_prev"] = core.llv(
        bars_qfq["low"].shift(1), params.low_lookback
    )

    frame.attrs["params"] = params
    return frame


def to_weekly(bars_daily: pd.DataFrame) -> pd.DataFrame:
    """日线聚合为周线；索引标记为**当周最后一个实际交易日**（SPEC §8.4）。"""
    if bars_daily.empty:
        return bars_daily.copy()

    idx = pd.DatetimeIndex(bars_daily.index)
    iso = idx.to_series().dt.isocalendar()
    keys = pd.Series(
        list(zip(iso["year"].tolist(), iso["week"].tolist(), strict=True)),
        index=bars_daily.index,
    )
    grouped = bars_daily.groupby(keys)
    weekly = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    )
    last_days = bars_daily.index.to_series().groupby(keys).max()
    weekly.index = pd.DatetimeIndex(last_days.to_numpy())
    weekly.index.name = "date"
    return weekly.sort_index()


def _week_is_complete(
    label: date,
    as_of: date,
    last_bar: date,
    trading_days: frozenset[date] | None,
) -> bool:
    if trading_days is not None:
        iso = label.isocalendar()
        week_end = max(
            (d for d in trading_days if d.isocalendar()[:2] == iso[:2]),
            default=label,
        )
        return week_end <= as_of
    # 没有日历时保守处理：只有明确存在更晚的交易日才算完成
    return label < last_bar


def weekly_frame(
    bars_qfq_daily: pd.DataFrame,
    as_of: date,
    params: IndicatorParams,
    *,
    trading_days: frozenset[date] | None = None,
) -> pd.DataFrame:
    """周线指标表：``ma20w ma20w_up kw dw jw is_complete``。

    ``is_complete`` 只有在该周所有交易日都已收盘后才为 True；
    策略只允许读取 ``is_complete=True`` 的行。
    """
    weekly = to_weekly(bars_qfq_daily)
    if weekly.empty:
        return weekly

    frame = weekly.copy()
    ma_period = params.weekly_ma
    for period in (*params.weekly_short_ma, ma_period):
        frame[f"ma{period}w"] = core.ma(frame["close"], period)
        frame[f"ma{period}w_up"] = core.slope_up(
            frame[f"ma{period}w"], params.weekly_slope_lookback
        )
    kdj = core.kdj(
        frame["high"],
        frame["low"],
        frame["close"],
        n=params.kdj_n,
        m1=params.kdj_m1,
        m2=params.kdj_m2,
    )
    frame["kw"] = kdj["k"]
    frame["dw"] = kdj["d"]
    frame["jw"] = kdj["j"]
    frame["jw_up"] = (frame["jw"] > frame["jw"].shift(1)).fillna(False)
    frame["kw_cross_dw"] = (frame["kw"] > frame["dw"]).fillna(False)
    # 周 J 在自身近两年历史中的百分位：低百分位=相对超卖，高百分位=相对超买。
    frame["jw_pct"] = frame["jw"].rolling(104, min_periods=52).rank(pct=True) * 100.0

    for period in params.rsi_periods:
        frame[f"rsi{period}w"] = core.rsi(frame["close"], period)

    macd = core.macd(
        frame["close"],
        fast=params.macd_fast,
        slow=params.macd_slow,
        signal=params.macd_signal,
    )
    frame["difw"] = macd["dif"]
    frame["deaw"] = macd["dea"]
    frame["macd_histw"] = macd["macd_hist"]
    frame["macd_histw_up"] = (
        frame["macd_histw"] > frame["macd_histw"].shift(1)
    ).fillna(False)
    frame["macd_cross_upw"] = (
        (frame["difw"] > frame["deaw"])
        & (frame["difw"].shift(1) <= frame["deaw"].shift(1))
    ).fillna(False)
    frame["macd_cross_downw"] = (
        (frame["difw"] < frame["deaw"])
        & (frame["difw"].shift(1) >= frame["deaw"].shift(1))
    ).fillna(False)

    frame["vol_ma5w"] = core.ma(frame["volume"], 5)
    frame["vol_ratio5w"] = frame["volume"].astype(float) / frame["vol_ma5w"].replace(
        0.0, np.nan
    )
    frame["ret4w"] = frame["close"].pct_change(4)

    labels = [pd.Timestamp(item).date() for item in frame.index]
    last_bar = labels[-1] if labels else as_of
    frame["is_complete"] = [
        _week_is_complete(label, as_of, last_bar, trading_days) for label in labels
    ]
    return frame


def weekly_until(weekly: pd.DataFrame, day: date) -> pd.DataFrame:
    """取到 ``day`` 为止**已经走完**的周线。

    若最后一根周线的标签与 ``day`` 落在同一个 ISO 周，说明当周还没走完，丢掉它。
    策略只允许读这里返回的数据，否则会用到当周未完成的信息（未来函数）。
    """
    if weekly.empty:
        return weekly
    window = weekly.loc[: pd.Timestamp(day)]
    if window.empty:
        return window
    last_label = pd.Timestamp(window.index[-1]).date()
    if last_label.isocalendar()[:2] == day.isocalendar()[:2]:
        return window.iloc[:-1]
    return window


def complete_weekly(frame: pd.DataFrame) -> pd.DataFrame:
    """只保留完结周线（策略的唯一合法输入）。"""
    if "is_complete" not in frame.columns:
        return frame.iloc[0:0]
    return frame[frame["is_complete"].astype(bool)]
