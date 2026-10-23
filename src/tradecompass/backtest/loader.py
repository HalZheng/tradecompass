"""把本地行情装配成回测输入。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from tradecompass.backtest.engine import BacktestData
from tradecompass.config.loader import AppConfig
from tradecompass.data import adjust, store
from tradecompass.domain.models import Instrument
from tradecompass.indicators.frame import IndicatorParams, daily_frame, weekly_frame


def load_backtest_data(
    cfg: AppConfig,
    instrument: Instrument,
    *,
    start: date,
    end: date,
    params: IndicatorParams | None = None,
) -> BacktestData | None:
    """读缓存里的不复权日线与因子，装配成回测输入；没有数据时返回 None。

    指标在**全历史**上计算后再裁剪到区间，避免从区间起点重新预热把 MA60
    的前 59 根算成 NaN（那会让策略在区间开头少做很多笔交易）。
    """
    raw_path = store.raw_bars_path(cfg.paths, instrument.market, instrument.symbol)
    factor_path = store.factor_path(cfg.paths, instrument.market, instrument.symbol)
    raw = store.read_bars_or_empty(raw_path)
    if raw.empty:
        return None
    factor = store.read_factor_or_empty(factor_path)
    qfq = adjust.to_qfq(raw, factor) if not factor.empty else raw
    resolved = params or IndicatorParams()
    indicators = daily_frame(qfq, resolved)
    # 周线在**全历史**上合成：策略只读截至当日的已完结周，见 frame.weekly_until
    weekly = weekly_frame(qfq, end, resolved)

    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    mask = (raw.index >= lo) & (raw.index <= hi)
    if not mask.any():
        return None
    return BacktestData(
        instrument=instrument,
        bars_raw=raw.loc[mask],
        bars_qfq=qfq.loc[mask],
        factor=factor,
        indicators=indicators,
        weekly=weekly,
    )
