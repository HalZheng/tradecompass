"""因子层：快照完整性、无未来函数、行业/大盘上下文。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tradecompass.factors.engine import (
    MarketContext,
    SectorContext,
    build_market_context,
    build_sector_context,
    build_snapshot,
    snapshot_from_frames,
)
from tradecompass.indicators.frame import (
    IndicatorParams,
    daily_frame,
    weekly_frame,
    weekly_until,
)
from tradecompass.market.sectors import SectorRank


def bars(n: int = 400) -> pd.DataFrame:
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = pd.Series(
        np.linspace(10.0, 16.0, n) + np.sin(np.arange(n) / 6.0) * 0.6,
        index=idx,
    )
    volume = pd.Series(100_000.0 + np.arange(n) * 20.0, index=idx)
    volume.iloc[-1] = 500_000.0
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
            "amount": volume * close,
        },
        index=idx,
    )


def market() -> MarketContext:
    return MarketContext(
        as_of=date(2026, 10, 22),
        state="RISK_ON",
        quota=1.0,
        above_weekly=True,
        ma20w_up=True,
        ret20=0.03,
        vol_ratio20=1.1,
    )


def test_snapshot_contains_all_factor_groups() -> None:
    data = bars()
    snapshot = build_snapshot(
        symbol="600000",
        name="测试",
        raw=data,
        factor=pd.Series(dtype=float),
        market=market(),
        sector=None,
    )

    assert snapshot is not None
    for key in (
        "ma20",
        "above_ma20",
        "rsi6",
        "rsi12",
        "rsi24",
        "dif",
        "dea",
        "macd_hist",
        "pos_60",
        "pos_250",
        "vol_ratio20",
        "ret5",
        "ret20",
        "ret60",
        "llv_low_10_prev",
        "jw",
        "ma20w",
        "weekly_bull",
        "rsi6w",
        "macd_histw",
    ):
        assert key in snapshot.factors, key

    assert snapshot.num("rsi12") is not None
    assert snapshot.num("macd_hist") is not None
    assert snapshot.num("pos_250") is not None
    assert snapshot.flag("weekly_bull") is not None
    assert snapshot.weekly_as_of is not None


def test_snapshot_is_truncation_stable() -> None:
    """同一日期、只用截止该日的数据，因子必须完全相同（无未来函数）。"""
    data = bars()
    cut = 330
    cut_day = pd.Timestamp(data.index[cut]).date()

    full = build_snapshot(
        symbol="600000",
        name="测试",
        raw=data,
        factor=pd.Series(dtype=float),
        market=market(),
        sector=None,
        as_of=cut_day,
    )
    truncated = build_snapshot(
        symbol="600000",
        name="测试",
        raw=data.iloc[: cut + 1],
        factor=pd.Series(dtype=float),
        market=market(),
        sector=None,
        as_of=cut_day,
    )

    assert full is not None and truncated is not None
    assert full.as_of == truncated.as_of
    for key, expected in full.factors.items():
        actual = truncated.factors.get(key)
        if isinstance(expected, bool):
            assert actual is expected, key
        elif expected is None:
            assert actual is None, key
        elif isinstance(expected, float):
            if np.isnan(expected):
                assert actual is None or np.isnan(actual), key
            else:
                assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9), key


def test_snapshot_from_frames_matches_live_path() -> None:
    """回测用的快路径必须与实盘 build_snapshot 给出完全相同的因子。"""
    data = bars()
    as_of = pd.Timestamp(data.index[-1]).date()
    params = IndicatorParams()
    live = build_snapshot(
        symbol="600000",
        name="测试",
        raw=data,
        factor=pd.Series(dtype=float),
        market=market(),
        sector=None,
    )
    daily = daily_frame(data, params)
    weekly = weekly_until(weekly_frame(data, as_of, params), as_of)
    fast = snapshot_from_frames(
        symbol="600000",
        name="测试",
        daily=daily.loc[[pd.Timestamp(as_of)]],
        week=weekly,
        close_raw=float(data["close"].iloc[-1]),
        market=market(),
        sector=None,
        as_of=as_of,
    )

    assert live is not None and fast is not None
    assert fast.as_of == live.as_of
    for key, expected in live.factors.items():
        actual = fast.factors.get(key)
        if isinstance(expected, bool):
            assert actual is expected, key
        elif expected is None:
            assert actual is None, key
        elif isinstance(expected, float):
            if np.isnan(expected):
                assert actual is None or np.isnan(actual), key
            else:
                assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9), key


def test_market_context_reads_index_and_volume() -> None:
    data = bars(300)
    context = build_market_context(data, pd.Timestamp(data.index[-1]).date())

    assert context.state in {"RISK_ON", "NEUTRAL", "RISK_OFF", "PANIC"}
    assert context.quota >= 0.0
    assert context.ret20 is not None
    assert context.vol_ratio20 is not None
    assert context.above_weekly is not None


def test_empty_index_is_no_data() -> None:
    context = build_market_context(
        pd.DataFrame(columns=["open", "high", "low", "close", "volume", "amount"]),
        date(2026, 10, 22),
    )
    assert context.state == "NO_DATA"
    assert context.quota == 0.0
    assert context.allows_new_buy is False


def test_sector_context_maps_and_ranks() -> None:
    ranks = [
        SectorRank("000935", "信息", 0.03, True, 1, 1.2),
        SectorRank("000932", "主要消费", 0.01, True, 2, 0.9),
    ]
    context = build_sector_context("600000", "计算机", ranks)

    assert context is not None
    assert context.code == "000935"
    assert context.rank == 1
    assert context.strength == "strong"
    assert build_sector_context("600000", None, ranks) is None


def test_sector_strength_uses_ma20_and_rank() -> None:
    weak = SectorContext(code="000935", rank=1, count=10, above_ma20=False)
    neutral = SectorContext(code="000935", rank=4, count=10, above_ma20=True)
    strong = SectorContext(code="000935", rank=2, count=10, above_ma20=True)

    assert weak.strength == "weak"
    assert neutral.strength == "neutral"
    assert strong.strength == "strong"
