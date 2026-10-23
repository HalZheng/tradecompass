"""复权因子与口径转换（SPEC §8.2）。"""

from __future__ import annotations

import pandas as pd
import pytest

from tradecompass.data.adjust import (
    AdjustError,
    compute_factor,
    factor_changed,
    factor_jump_dates,
    latest_factor,
    qfq_to_raw_price,
    rescale_price_after_adjust,
    to_hfq,
    to_qfq,
)

DATES = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"], name="date")


def _frame(
    closes: list[float],
    volumes: list[float] | None = None,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    volumes = volumes or [1000.0] * len(closes)
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.98 for c in closes],
            "close": closes,
            "volume": volumes,
            "amount": [c * v for c, v in zip(closes, volumes, strict=True)],
        },
        index=DATES if index is None else index,
    )


def test_compute_factor_divides_hfq_by_raw() -> None:
    raw = _frame([10.0, 10.5, 11.0, 11.5])
    hfq = _frame([10.0, 10.5, 12.1, 12.65])  # 第 3 天起 10% 除权

    factor = compute_factor(hfq, raw)

    assert list(factor.round(6)) == [1.0, 1.0, 1.1, 1.1]
    assert factor.name == "factor_hfq"


def test_compute_factor_rejects_disjoint_frames() -> None:
    raw = _frame([10.0, 10.5, 11.0, 11.5])
    other = _frame(
        [10.0, 11.0], index=pd.DatetimeIndex(["2026-02-01", "2026-02-02"], name="date")
    )

    with pytest.raises(AdjustError, match="没有重叠"):
        compute_factor(other, raw)


def test_to_hfq_multiplies_ohlc_only() -> None:
    raw = _frame([10.0, 10.5, 11.0, 11.5], volumes=[100.0, 200.0, 300.0, 400.0])
    factor = pd.Series([1.0, 1.0, 1.1, 1.1], index=DATES, name="factor_hfq")

    hfq = to_hfq(raw, factor)

    assert hfq["close"].tolist() == pytest.approx([10.0, 10.5, 12.1, 12.65])
    assert hfq["volume"].tolist() == pytest.approx([100.0, 200.0, 300.0, 400.0])
    assert hfq["amount"].tolist() == pytest.approx(raw["amount"].tolist())


def test_to_qfq_keeps_latest_close_equal_to_raw() -> None:
    raw = _frame([10.0, 10.5, 11.0, 11.5])
    # 因子恒定（没有除权）时，前复权价与不复权价完全一致
    factor = pd.Series([1.1, 1.1, 1.1, 1.1], index=DATES, name="factor_hfq")

    qfq = to_qfq(raw, factor)

    assert qfq["close"].iloc[-1] == pytest.approx(raw["close"].iloc[-1])
    assert qfq["close"].tolist() == pytest.approx(raw["close"].tolist())
    assert qfq["volume"].tolist() == pytest.approx(raw["volume"].tolist())


def test_to_qfq_adjusts_history_downward_after_dividend() -> None:
    raw = _frame([10.0, 10.5, 11.0, 11.5])
    factor = pd.Series([1.0, 1.0, 1.0, 1.1], index=DATES, name="factor_hfq")

    qfq = to_qfq(raw, factor)

    assert qfq["close"].iloc[-1] == pytest.approx(11.5)
    assert qfq["close"].iloc[0] == pytest.approx(10.0 * 1.0 / 1.1)


def test_qfq_to_raw_price_round_trip() -> None:
    raw_price = 15.32
    factor_at = 1.0
    factor_latest = 1.25
    qfq = raw_price * factor_at / factor_latest

    assert qfq_to_raw_price(qfq, factor_at, factor_latest) == pytest.approx(raw_price)


def test_factors_must_be_positive() -> None:
    with pytest.raises(AdjustError, match="因子必须为正"):
        qfq_to_raw_price(10.0, 0.0, 1.0)
    with pytest.raises(AdjustError, match="因子必须为正"):
        rescale_price_after_adjust(10.0, 1.0, 0.0)


def test_factor_jump_dates_detects_ex_date() -> None:
    factor = pd.Series([1.0, 1.0, 1.1, 1.1, 1.1], index=pd.date_range("2026-01-05", periods=5))

    jumps = factor_jump_dates(factor)

    assert [pd.Timestamp(d).date().isoformat() for d in jumps] == ["2026-01-07"]


def test_factor_changed_when_history_revised() -> None:
    dates = pd.date_range("2026-01-05", periods=4)
    old = pd.Series([1.0, 1.0, 1.0, 1.0], index=dates)
    revised = pd.Series([1.0, 1.0, 1.0001, 1.0], index=dates)

    assert factor_changed(old, revised) is True
    assert factor_changed(old, old.copy()) is False


def test_factor_changed_when_new_jump_appears() -> None:
    dates = pd.date_range("2026-01-05", periods=4)
    old = pd.Series([1.0, 1.0, 1.0, 1.0], index=dates)
    extended = pd.Series([1.0, 1.0, 1.0, 1.0, 1.1], index=pd.date_range("2026-01-05", periods=5))

    assert factor_changed(old, extended) is True


def test_factor_changed_when_only_new_flat_days_appended() -> None:
    old = pd.Series([1.0, 1.0], index=pd.date_range("2026-01-05", periods=2))
    extended = pd.Series([1.0, 1.0, 1.0], index=pd.date_range("2026-01-05", periods=3))

    assert factor_changed(old, extended) is False


def test_latest_factor_skips_nan() -> None:
    factor = pd.Series([1.0, 1.2, float("nan")], index=pd.date_range("2026-01-05", periods=3))
    assert latest_factor(factor) == pytest.approx(1.2)


def test_rescale_price_after_adjust() -> None:
    # 除权前一日因子 1.0，除权日 1.1 → 止损价按比例下调
    assert rescale_price_after_adjust(11.0, 1.0, 1.1) == pytest.approx(10.0)
