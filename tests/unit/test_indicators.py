"""指标公式与周线合成（SPEC §9.2、§8.4）。

金标准（东财 APP 抄录值）尚未提供，这里先用可手算的小样本 + 因果性检验。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tradecompass.indicators import core
from tradecompass.indicators.frame import (
    IndicatorParams,
    complete_weekly,
    daily_frame,
    to_weekly,
    weekly_frame,
)


def bars(n: int = 80) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-05", periods=n)  # 从周一开始，便于周线断言
    close = pd.Series(np.linspace(10.0, 12.0, n) + np.sin(np.arange(n)) * 0.5, index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": pd.Series(1000.0 + np.arange(n), index=idx),
            "amount": pd.Series((1000.0 + np.arange(n)) * close.to_numpy(), index=idx),
        },
        index=idx,
    )


# ------------------------------------------------------------------ core 公式


def test_ma_matches_manual_mean() -> None:
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = core.ma(x, 3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_tdx_sma_seeds_with_50_then_recurses() -> None:
    x = pd.Series([10.0, 10.0, 10.0, 10.0])
    result = core.tdx_sma(x, n=3, m=1)

    # y0 = (1*10 + 2*50)/3
    assert result.iloc[0] == pytest.approx(110.0 / 3)
    # y1 = (10 + 2*y0)/3
    assert result.iloc[1] == pytest.approx((10.0 + 2.0 * 110.0 / 3) / 3.0)
    assert result.iloc[3] < result.iloc[2] < result.iloc[1]


def test_kdj_flat_prices_gives_50() -> None:
    flat = pd.Series([10.0] * 20)
    result = core.kdj(flat, flat, flat)

    # 高开低收全相等 → 分母为 0 → RSV 取 50，K/D 收敛到 50，J 也是 50
    assert result["j"].iloc[-1] == pytest.approx(50.0)
    assert result["k"].iloc[-1] == pytest.approx(50.0)
    assert result["d"].iloc[-1] == pytest.approx(50.0)


def test_kdj_follows_tdx_recurrence() -> None:
    idx = pd.RangeIndex(12)
    close = pd.Series(np.arange(1.0, 13.0), index=idx)
    high = close + 0.5
    low = close - 0.5
    result = core.kdj(high, low, close, n=9, m1=3, m2=3)

    # 第 9 根的 RSV = (9 − LLV(9)) / (HHV(9) − LLV(9)) × 100，窗口是 idx 0..8
    rsv = (9.0 - 0.5) / (9.5 - 0.5) * 100.0
    k = (rsv + 2 * 50.0) / 3.0
    d = (k + 2 * 50.0) / 3.0

    assert result["k"].iloc[8] == pytest.approx(k)
    assert result["d"].iloc[8] == pytest.approx(d)
    assert result["j"].iloc[8] == pytest.approx(3 * k - 2 * d)


def test_atr_seed_is_mean_of_first_trs() -> None:
    idx = pd.RangeIndex(15)
    close = pd.Series([10.0] * 15, index=idx)
    high = pd.Series([11.0] * 15, index=idx)
    low = pd.Series([9.0] * 15, index=idx)

    result = core.atr(high, low, close, n=14)

    # TR 恒为 max(2, 1, 1) = 2
    assert result.iloc[12] != result.iloc[12]  # NaN 在前 13 个
    assert result.iloc[13] == pytest.approx(2.0)
    assert result.iloc[14] == pytest.approx(2.0)


def test_rsi_extremes_and_flat() -> None:
    up = pd.Series(np.arange(1.0, 30.0))
    down = pd.Series(np.arange(30.0, 1.0, -1.0))
    flat = pd.Series([10.0] * 30)

    assert core.rsi(up, 6).iloc[-1] == pytest.approx(100.0)
    assert core.rsi(down, 6).iloc[-1] == pytest.approx(0.0)
    assert core.rsi(flat, 6).iloc[-1] == pytest.approx(50.0)


def test_macd_matches_ema_definition() -> None:
    close = pd.Series(np.linspace(10.0, 20.0, 120))
    result = core.macd(close)

    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()

    assert result["dif"].iloc[-1] == pytest.approx(float(dif.iloc[-1]))
    assert result["dea"].iloc[-1] == pytest.approx(float(dea.iloc[-1]))
    assert result["macd_hist"].iloc[-1] == pytest.approx(
        2.0 * (float(dif.iloc[-1]) - float(dea.iloc[-1]))
    )


def test_hhv_llv_ref_slope() -> None:
    x = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
    assert core.hhv(x, 3).tolist()[2:] == [3.0, 5.0, 5.0]
    assert core.llv(x, 3).tolist()[2:] == [1.0, 2.0, 2.0]
    assert core.ref(x, 1).tolist()[1:] == [1.0, 3.0, 2.0, 5.0]
    assert core.slope_up(x, 2).tolist() == [False, False, True, True, True]


# ------------------------------------------------------------------ 指标表


def test_daily_frame_contains_contracted_columns() -> None:
    frame = daily_frame(bars(), IndicatorParams())
    for column in (
        "ma5",
        "ma10",
        "ma20",
        "ma60",
        "k",
        "d",
        "j",
        "atr14",
        "vol_ma5",
        "vol_ma20",
        "hhv_close_60_prev",
        "llv_low_10",
    ):
        assert column in frame.columns


def test_hhv_close_60_prev_excludes_current_bar() -> None:
    data = bars()
    frame = daily_frame(data, IndicatorParams())
    row = data.index[70]
    expected = data["close"].iloc[10:70].max()  # 前 60 根，不含当日
    assert frame.loc[row, "hhv_close_60_prev"] == pytest.approx(expected)


def test_no_lookahead_truncated_input_gives_same_row() -> None:
    data = bars()
    params = IndicatorParams()
    full = daily_frame(data, params)

    for length in (40, 55, 70):
        truncated = daily_frame(data.iloc[:length], params)
        last = data.index[length - 1]
        for column in full.columns:
            a = full.loc[last, column]
            b = truncated.loc[last, column]
            assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), f"{column}@{last}"


# ------------------------------------------------------------------ 周线


def test_to_weekly_uses_actual_last_trading_day_as_label() -> None:
    data = bars(10)  # 2026-01-01 起 10 个工作日
    weekly = to_weekly(data)

    assert len(weekly) == 2
    first_week_days = data[data.index <= weekly.index[0]]
    assert weekly.index[0] == first_week_days.index[-1]
    assert weekly["open"].iloc[0] == pytest.approx(first_week_days["open"].iloc[0])
    assert weekly["close"].iloc[0] == pytest.approx(first_week_days["close"].iloc[-1])
    assert weekly["high"].iloc[0] == pytest.approx(first_week_days["high"].max())
    assert weekly["volume"].iloc[0] == pytest.approx(first_week_days["volume"].sum())


def test_weekly_frame_marks_only_finished_weeks_complete() -> None:
    data = bars(60)
    params = IndicatorParams()
    trading_days = frozenset(pd.Timestamp(item).date() for item in data.index)

    # 取某周中间的周三，当周应未完成
    mid_week = data.index[32]
    weekly = weekly_frame(data[data.index <= mid_week], mid_week.date(), params,
                          trading_days=trading_days)
    finished = complete_weekly(weekly)
    assert not weekly.empty
    assert weekly["is_complete"].iloc[-1] in (False, np.bool_(False))
    assert finished.index[-1] < weekly.index[-1]


def test_weekly_frame_marks_week_complete_after_its_last_day() -> None:
    data = bars(60)
    params = IndicatorParams()
    trading_days = frozenset(pd.Timestamp(item).date() for item in data.index)
    last = data.index[-1].date()

    weekly = weekly_frame(data, last, params, trading_days=trading_days)

    assert bool(weekly["is_complete"].iloc[-1]) is True


def test_complete_weekly_requires_is_complete_column() -> None:
    assert complete_weekly(pd.DataFrame({"close": [1.0]})).empty


def test_weekly_indicators_present() -> None:
    data = bars(90)
    params = IndicatorParams()
    weekly = weekly_frame(data, data.index[-1].date(), params)

    for column in ("ma20w", "ma20w_up", "kw", "dw", "jw", "is_complete"):
        assert column in weekly.columns


def test_weekly_label_is_a_date_object() -> None:
    data = bars(30)
    weekly = to_weekly(data)
    assert isinstance(pd.Timestamp(weekly.index[0]).date(), date)
