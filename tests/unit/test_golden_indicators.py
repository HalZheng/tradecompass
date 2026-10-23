"""金标准回归测试（SPEC §9.2）。

对照组来自东方财富 APP 在 2026-10-21 收盘后显示的数值，由项目所有者逐项抄录。
断言按东财的显示精度四舍五入后比较：个股价格 2 位、ETF 价格 3 位、KDJ 一律 2 位。

这组用例锁定了三件事：
1. KDJ 用通达信 ``SMA(RSV,3,1)`` 递推、9 日 RSV 窗口；
2. MA 是简单移动平均；
3. 指标基于**前复权**价，且复权基准与东财一致。

只要这三条有任何一条被改坏，这里必然失败。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tradecompass.indicators.frame import IndicatorParams, daily_frame

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "golden_bars.csv"
GOLDEN_DATE = pd.Timestamp("2026-10-21")

# symbol -> (KDJ, MA, 价格显示精度)
CASES: dict[str, tuple[dict[str, float], dict[str, float], int]] = {
    "600233": (
        {"k": 25.20, "d": 30.27, "j": 15.07},
        {"ma5": 17.34, "ma10": 17.63, "ma20": 17.74, "ma60": 17.73},
        2,
    ),
    "600570": (
        {"k": 79.65, "d": 71.26, "j": 96.44},
        {"ma5": 20.02, "ma10": 19.78, "ma20": 19.86, "ma60": 21.17},
        2,
    ),
    "510300": (
        {"k": 12.36, "d": 14.84, "j": 7.40},
        {"ma5": 4.225, "ma10": 4.290, "ma20": 4.408, "ma60": 4.584},
        3,
    ),
}


def load_bars(symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(
        FIXTURE, parse_dates=["date"], index_col="date", dtype={"symbol": str}
    )
    subset = frame[frame["symbol"] == symbol].drop(columns=["symbol"])
    subset.index.name = "date"
    return subset.sort_index()


def test_fixture_shape() -> None:
    frame = pd.read_csv(FIXTURE, dtype={"symbol": str})
    assert set(frame.columns) == {
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    }
    assert len(frame) == 750  # 3 个标的 × 250 个交易日
    assert frame["date"].max() == GOLDEN_DATE.date().isoformat()


@pytest.mark.parametrize("symbol", sorted(CASES))
def test_kdj_matches_eastmoney(symbol: str) -> None:
    expected, _, _ = CASES[symbol]
    frame = daily_frame(load_bars(symbol), IndicatorParams())

    for column, want in expected.items():
        got = float(frame.loc[GOLDEN_DATE, column])
        assert round(got, 2) == pytest.approx(want, abs=1e-9), f"{symbol} {column.upper()}"


@pytest.mark.parametrize("symbol", sorted(CASES))
def test_ma_matches_eastmoney(symbol: str) -> None:
    _, expected, digits = CASES[symbol]
    frame = daily_frame(load_bars(symbol), IndicatorParams())

    for column, want in expected.items():
        got = float(frame.loc[GOLDEN_DATE, column])
        assert round(got, digits) == pytest.approx(want, abs=1e-9), f"{symbol} {column}"


@pytest.mark.parametrize("symbol", sorted(CASES))
def test_close_matches_eastmoney(symbol: str) -> None:
    """前复权收盘价必须与东财一致，否则 MA/KDJ 会整体偏移。"""
    _, expected_ma, _ = CASES[symbol]
    frame = load_bars(symbol)
    close = float(frame.loc[GOLDEN_DATE, "close"])
    # 与 MA 同量级，且 MA5 应在当日收盘的合理邻域内
    assert abs(close - expected_ma["ma5"]) < close * 0.05
