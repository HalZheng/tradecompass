"""市场层：状态判定、额度映射、防抖记忆（STATE_MACHINE §2）。"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from tradecompass.market.regime import (
    QUOTA,
    IndexFeatures,
    Regime,
    build_series,
    classify,
    quota_on,
)


def make_index(n_up: int = 300, n_down: int = 60) -> pd.DataFrame:
    """先涨后跌的合成指数：上涨段缓慢上行，下跌段急跌且波动放大。"""
    up = np.linspace(3000, 4000, n_up)
    down = np.linspace(4000, 3300, n_down) + np.sin(np.arange(n_down)) * 30
    close = np.concatenate([up, down])
    idx = pd.bdate_range("2024-01-01", periods=len(close))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1e8,
            "amount": close * 1e8,
        },
        index=idx,
    )


def features(**overrides: object) -> IndexFeatures:
    base = {
        "close": 4000.0,
        "ma20w": 3800.0,
        "ma20w_up": True,
        "ma10": 3950.0,
        "ret20": 0.03,
        "vol_percentile": 0.4,
    }
    base.update(overrides)
    return IndexFeatures(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ 判定


def test_risk_on_when_trend_up_and_momentum_positive() -> None:
    assert classify(features()) is Regime.RISK_ON


def test_neutral_when_trend_up_but_momentum_flat() -> None:
    assert classify(features(ret20=-0.01)) is Regime.NEUTRAL


def test_neutral_when_momentum_positive_but_below_weekly() -> None:
    assert classify(features(close=3700.0, ma20w=3800.0)) is Regime.NEUTRAL


def test_risk_off_when_below_weekly_and_weekly_falling() -> None:
    assert classify(features(close=3600.0, ma20w=3800.0, ma20w_up=False, ret20=-0.03)) is (
        Regime.RISK_OFF
    )


def test_panic_on_deep_drop() -> None:
    assert classify(
        features(close=3000.0, ma20w=3800.0, ma20w_up=False, ret20=-0.09)
    ) is Regime.PANIC


def test_panic_on_volatility_spike() -> None:
    assert classify(
        features(close=3600.0, ma20w=3800.0, ma20w_up=False, ret20=-0.02, vol_percentile=0.9)
    ) is Regime.PANIC


def test_missing_weekly_data_is_neutral() -> None:
    assert classify(features(ma20w=None, ma20w_up=None)) is Regime.NEUTRAL


# ------------------------------------------------------------------ 序列


def test_quota_mapping_is_monotonic() -> None:
    assert QUOTA[Regime.RISK_ON] > QUOTA[Regime.NEUTRAL] > QUOTA[Regime.RISK_OFF]
    assert QUOTA[Regime.PANIC] == 0.0


def test_series_covers_every_trading_day() -> None:
    bars = make_index()
    points = build_series(bars)
    assert len(points) == len(bars)
    assert all(point.quota <= QUOTA[point.state] + 1e-9 for point in points)


def test_series_detects_downturn() -> None:
    points = build_series(make_index())
    states = [point.state for point in points]
    assert Regime.RISK_ON in states
    assert states[-1] in (Regime.RISK_OFF, Regime.PANIC)


def test_series_does_not_jitter() -> None:
    """状态切换必须稀疏：360 天里不超过 20 次，否则说明防抖失效。"""
    points = build_series(make_index())
    transitions = sum(
        1 for a, b in zip(points, points[1:], strict=False) if a.state != b.state
    )
    assert transitions <= 20, f"状态切换 {transitions} 次，抖动过多"


def test_panic_is_immediate_not_confirmed() -> None:
    """PANIC 不等 2 日确认——等两天可能已经亏掉 10%。"""
    bars = make_index(n_up=300, n_down=30)
    points = build_series(bars)
    panic_points = [point for point in points if point.state is Regime.PANIC]
    if panic_points:
        first = panic_points[0]
        assert first.raw is Regime.PANIC


def test_quota_on_returns_latest_known() -> None:
    points = build_series(make_index())
    assert quota_on(points, points[-1].day) == pytest.approx(points[-1].quota)
    assert quota_on(points, points[0].day - timedelta(days=1)) == 0.0
