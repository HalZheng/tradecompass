"""因子引擎：一次算出一只标的在当前交易日的全部上下文。

输出是一个扁平的 ``FactorSnapshot``：键值既包含日线/周线的原始指标，
也包含派生出来的"是否多头排列""距 MA20 多远"这类判断量。
信号层只读快照，不再碰行情或指标表，避免不同信号各自重算一套口径。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from tradecompass.config.loader import AppConfig
from tradecompass.data import adjust, store
from tradecompass.domain.models import Instrument
from tradecompass.indicators.frame import (
    IndicatorParams,
    daily_frame,
    weekly_frame,
    weekly_until,
)
from tradecompass.market.regime import build_series
from tradecompass.market.sectors import SectorRank, industry_of

FactorValue = float | bool | None


@dataclass(frozen=True, slots=True)
class MarketContext:
    """大盘上下文。"""

    as_of: date | None
    state: str
    quota: float
    above_weekly: bool | None
    ma20w_up: bool | None
    ret20: float | None
    vol_ratio20: float | None
    note: str = ""

    @property
    def allows_new_buy(self) -> bool:
        # RISK_OFF 的 25% 额度是给已有持仓的，不是给新开仓的。
        # 防守档继续买突破，是回测里最明显的亏损来源之一。
        return self.quota > 0.0 and self.state in ("RISK_ON", "NEUTRAL")


@dataclass(frozen=True, slots=True)
class SectorContext:
    """行业上下文。"""

    code: str | None = None
    name: str | None = None
    rank: int | None = None
    count: int | None = None
    momentum: float | None = None
    above_ma20: bool | None = None
    vol_ratio20: float | None = None

    @property
    def strength(self) -> str:
        """``strong`` / ``neutral`` / ``weak`` / ``unknown``。"""
        if self.rank is None:
            return "unknown"
        if self.above_ma20 and self.rank <= 3:
            return "strong"
        half = (self.count or 10) // 2
        if self.above_ma20 and self.rank <= max(3, half):
            return "neutral"
        return "weak"


@dataclass(frozen=True, slots=True)
class FactorSnapshot:
    """单只标的、单个交易日的因子快照。"""

    symbol: str
    name: str
    as_of: date
    close_qfq: float
    close_raw: float
    price_scale: float
    factors: Mapping[str, FactorValue]
    market: MarketContext
    sector: SectorContext | None
    kind: str = "stock"
    weekly_as_of: date | None = None

    def num(self, key: str) -> float | None:
        value = self.factors.get(key)
        return float(value) if isinstance(value, float) else None

    def flag(self, key: str) -> bool | None:
        value = self.factors.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            return bool(value)
        return None


# --------------------------------------------------------------------------- 基础工具


def _num(row: pd.Series, column: str) -> float | None:
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return float(value)


def _flag(row: pd.Series, column: str) -> bool | None:
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return bool(value)


def _put_num(target: dict[str, FactorValue], key: str, row: pd.Series, column: str) -> None:
    target[key] = _num(row, column)


def _put_flag(target: dict[str, FactorValue], key: str, row: pd.Series, column: str) -> None:
    target[key] = _flag(row, column)


# --------------------------------------------------------------------------- 上下文


def build_market_context(index_bars: pd.DataFrame, as_of: date) -> MarketContext:
    """由沪深300 日线构造大盘上下文（regime + 周线趋势 + 量能）。"""
    if index_bars.empty:
        return MarketContext(None, "NO_DATA", 0.0, None, None, None, None, "没有指数行情")

    window = index_bars.loc[: pd.Timestamp(as_of)]
    if window.empty:
        return MarketContext(None, "NO_DATA", 0.0, None, None, None, None, "指数没有该日数据")

    last_day = pd.Timestamp(window.index[-1]).date()
    points = build_series(window)
    point = next((item for item in reversed(points) if item.day <= as_of), None)
    if point is None:
        return MarketContext(
            last_day, "NO_DATA", 0.0, None, None, None, None, "指数历史不足以判定 regime"
        )

    close = window["close"].astype(float)
    ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else None
    volume = window["volume"].astype(float)
    vol_ratio: float | None = None
    if len(volume) >= 20:
        average = float(volume.tail(20).mean())
        if average > 0:
            vol_ratio = float(volume.iloc[-1]) / average

    weekly = weekly_frame(window, last_day, IndicatorParams())
    week = weekly_until(weekly, last_day)
    above_weekly: bool | None = None
    ma20w_up: bool | None = None
    if len(week) >= 21:
        last = week.iloc[-1]
        ma20w = _num(last, "ma20w")
        if ma20w is not None:
            above_weekly = float(last["close"]) > ma20w
            ma20w_up = _flag(last, "ma20w_up")

    return MarketContext(
        as_of=last_day,
        state=str(point.state),
        quota=float(point.quota),
        above_weekly=above_weekly,
        ma20w_up=ma20w_up,
        ret20=ret20,
        vol_ratio20=vol_ratio,
        note=point.note,
    )


def build_sector_context(
    symbol: str, sector_sw: str | None, ranks: list[SectorRank]
) -> SectorContext | None:
    """把标的映射到行业指数，并附上该行业的排名与量能。"""
    code = industry_of(symbol, sector_sw)
    if code is None:
        return None
    match = next((item for item in ranks if item.code == code), None)
    if match is None:
        return SectorContext(code=code)
    return SectorContext(
        code=code,
        name=match.name,
        rank=match.rank,
        count=len(ranks),
        momentum=match.momentum,
        above_ma20=match.above_ma20,
        vol_ratio20=match.vol_ratio20,
    )


# --------------------------------------------------------------------------- 快照


def build_snapshot(
    *,
    symbol: str,
    name: str,
    raw: pd.DataFrame,
    factor: pd.Series,
    market: MarketContext,
    sector: SectorContext | None,
    as_of: date | None = None,
    params: IndicatorParams | None = None,
    kind: str = "stock",
) -> FactorSnapshot | None:
    """由不复权日线与复权因子构造因子快照。"""
    if raw.empty:
        return None
    window = raw.loc[: pd.Timestamp(as_of)] if as_of is not None else raw
    if window.empty:
        return None

    last_day = pd.Timestamp(window.index[-1]).date()
    qfq = adjust.to_qfq(window, factor) if not factor.empty else window
    resolved = params or IndicatorParams()
    daily = daily_frame(qfq, resolved)
    if daily.empty:
        return None
    weekly = weekly_frame(qfq, last_day, resolved)
    week = weekly_until(weekly, last_day)

    return snapshot_from_frames(
        symbol=symbol,
        name=name,
        daily=daily,
        week=week,
        close_raw=float(window["close"].iloc[-1]),
        market=market,
        sector=sector,
        kind=kind,
        as_of=last_day,
    )


def snapshot_from_frames(
    *,
    symbol: str,
    name: str,
    daily: pd.DataFrame,
    week: pd.DataFrame,
    close_raw: float,
    market: MarketContext,
    sector: SectorContext | None,
    kind: str = "stock",
    as_of: date | None = None,
) -> FactorSnapshot | None:
    """由**已算好的**日线/周线表构造快照，供回测逐日复用，避免重算指标。

    ``daily`` 必须截止到 ``as_of``（含），``week`` 只能包含已完结周线。
    """
    if daily.empty:
        return None
    last_daily = daily.iloc[-1]
    close_qfq = float(last_daily["close"])
    if pd.isna(close_qfq) or close_qfq == 0:
        return None

    factors: dict[str, FactorValue] = {}
    _add_daily_factors(factors, last_daily)
    _add_weekly_factors(factors, week)
    _add_derived_factors(factors, last_daily, week)

    row_day = pd.Timestamp(daily.index[-1]).date()
    weekly_as_of = pd.Timestamp(week.index[-1]).date() if not week.empty else None
    return FactorSnapshot(
        symbol=symbol,
        name=name,
        as_of=as_of or row_day,
        close_qfq=close_qfq,
        close_raw=close_raw,
        price_scale=close_raw / close_qfq,
        factors=factors,
        market=market,
        sector=sector,
        kind=kind,
        weekly_as_of=weekly_as_of,
    )


def _add_daily_factors(target: dict[str, FactorValue], row: pd.Series) -> None:
    for period in (5, 10, 20, 60):
        _put_num(target, f"ma{period}", row, f"ma{period}")
        _put_flag(target, f"ma{period}_up", row, f"ma{period}_up")
    for column in ("k", "d", "j", "j_pct"):
        _put_num(target, column, row, column)
    _put_flag(target, "j_up", row, "j_up")
    _put_flag(target, "j_cross_k", row, "j_cross_k")

    for period in (6, 12, 24):
        _put_num(target, f"rsi{period}", row, f"rsi{period}")
    for column in ("dif", "dea", "macd_hist"):
        _put_num(target, column, row, column)
    for column in ("macd_hist_up", "macd_cross_up", "macd_cross_down"):
        _put_flag(target, column, row, column)

    _put_num(target, "atr14", row, "atr14")
    _put_num(target, "vol_ratio5", row, "vol_ratio5")
    _put_num(target, "vol_ratio20", row, "vol_ratio20")
    _put_flag(target, "vol_ma5_gt_ma20", row, "vol_ma5_gt_ma20")
    _put_num(target, "vol_price_dir", row, "vol_price_dir")

    _put_num(target, "pos_60", row, "pos_60")
    _put_num(target, "pos_250", row, "pos_250")
    _put_num(target, "ret1", row, "ret1")
    _put_num(target, "ret5", row, "ret5")
    _put_num(target, "ret20", row, "ret20")
    _put_num(target, "ret60", row, "ret60")

    _put_num(target, "hhv_close_60_prev", row, "hhv_close_60_prev")
    _put_num(target, "hhv_close_250_prev", row, "hhv_close_250_prev")
    _put_num(target, "llv_low_10", row, "llv_low_10")
    _put_num(target, "llv_low_10_prev", row, "llv_low_10_prev")


def _add_weekly_factors(target: dict[str, FactorValue], week: pd.DataFrame) -> None:
    if week.empty:
        for key in (
            "weekly_close", "ma5w", "ma10w", "ma20w", "ma20w_up", "jw", "jw_pct",
            "kw", "dw", "jw_up", "kw_cross_dw", "rsi6w", "rsi12w", "difw", "deaw",
            "macd_histw", "macd_histw_up", "vol_ratio5w", "ret4w",
        ):
            target[key] = None
        return

    row = week.iloc[-1]
    for column in ("close", "ma5w", "ma10w", "ma20w", "kw", "dw", "jw", "jw_pct"):
        _put_num(target, column if column != "close" else "weekly_close", row, column)
    _put_flag(target, "ma20w_up", row, "ma20w_up")
    _put_flag(target, "jw_up", row, "jw_up")
    _put_flag(target, "kw_cross_dw", row, "kw_cross_dw")
    for column in ("rsi6w", "rsi12w", "difw", "deaw", "macd_histw"):
        _put_num(target, column, row, column)
    _put_flag(target, "macd_histw_up", row, "macd_histw_up")
    _put_num(target, "vol_ratio5w", row, "vol_ratio5w")
    _put_num(target, "ret4w", row, "ret4w")


def _add_derived_factors(
    target: dict[str, FactorValue],
    daily_row: pd.Series,
    week: pd.DataFrame,
) -> None:
    close = _num(daily_row, "close")
    if close is not None:
        for period in (5, 10, 20, 60):
            ma_value = _num(daily_row, f"ma{period}")
            if ma_value:
                target[f"above_ma{period}"] = close > ma_value
                target[f"dist_ma{period}_pct"] = close / ma_value - 1.0
            else:
                target[f"above_ma{period}"] = None
                target[f"dist_ma{period}_pct"] = None

        allowed = [_num(daily_row, f"ma{period}") for period in (5, 10, 20, 60)]
        if all(value is not None for value in allowed):
            ordered = [float(value) for value in allowed if value is not None]
            target["ma_bull_aligned"] = ordered[0] > ordered[1] > ordered[2] > ordered[3]
        else:
            target["ma_bull_aligned"] = None

    if week.empty:
        for key in (
            "weekly_bull", "weekly_bull_aligned", "weekly_above_ma5w",
            "weekly_above_ma10w", "dist_ma20w_pct",
        ):
            target[key] = None
        return

    row = week.iloc[-1]
    weekly_close = _num(row, "close")
    ma20w = _num(row, "ma20w")
    ma20w_up = _flag(row, "ma20w_up")
    if weekly_close is not None and ma20w is not None:
        target["weekly_bull"] = bool(weekly_close > ma20w and ma20w_up is True)
        target["dist_ma20w_pct"] = weekly_close / ma20w - 1.0
        target["weekly_above_ma20w"] = weekly_close > ma20w
    else:
        target["weekly_bull"] = None
        target["dist_ma20w_pct"] = None
        target["weekly_above_ma20w"] = None

    ma5w = _num(row, "ma5w")
    ma10w = _num(row, "ma10w")
    if weekly_close is not None and ma5w is not None:
        target["weekly_above_ma5w"] = weekly_close > ma5w
    else:
        target["weekly_above_ma5w"] = None
    if weekly_close is not None and ma10w is not None:
        target["weekly_above_ma10w"] = weekly_close > ma10w
    else:
        target["weekly_above_ma10w"] = None
    if ma5w is not None and ma10w is not None and ma20w is not None:
        target["weekly_bull_aligned"] = ma5w > ma10w > ma20w
    else:
        target["weekly_bull_aligned"] = None


# --------------------------------------------------------------------------- 取数入口


def load_snapshot(
    cfg: AppConfig,
    instrument: Instrument,
    *,
    market: MarketContext,
    sector: SectorContext | None,
    as_of: date | None = None,
    params: IndicatorParams | None = None,
) -> FactorSnapshot | None:
    """从 TC_HOME 读取行情并构建快照；没有数据时返回 ``None``。"""
    raw = store.read_bars_or_empty(
        store.raw_bars_path(cfg.paths, instrument.market, instrument.symbol)
    )
    if raw.empty:
        return None
    factor = store.read_factor_or_empty(
        store.factor_path(cfg.paths, instrument.market, instrument.symbol)
    )
    return build_snapshot(
        symbol=instrument.symbol,
        name=instrument.name,
        raw=raw,
        factor=factor,
        market=market,
        sector=sector,
        as_of=as_of,
        params=params,
        kind=instrument.kind,
    )
