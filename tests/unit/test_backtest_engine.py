"""回测引擎：手算校验与 §12.8 防偏差测试。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tradecompass.backtest.engine import BacktestData, run_backtest
from tradecompass.backtest.fills import Bar, simulate
from tradecompass.backtest.types import Order
from tradecompass.config.models import CostsConfig
from tradecompass.data.universe import board_of
from tradecompass.domain.models import Instrument
from tradecompass.indicators.frame import IndicatorParams, daily_frame
from tradecompass.strategies.ma20_cross import Ma20CrossStrategy

ZERO_COST = CostsConfig(
    commission=0.0, commission_fund=0.0, commission_min=0.0, stamp_sell=0.0,
    transfer=0.0, slippage=0.0,
)


def make_data(
    closes: list[float],
    *,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    symbol: str = "600000",
    kind: str = "stock",
) -> BacktestData:
    idx = pd.bdate_range("2026-01-05", periods=len(closes))
    o = opens or closes
    base_high = [max(open_, close) for open_, close in zip(o, closes, strict=True)]
    base_low = [min(open_, close) for open_, close in zip(o, closes, strict=True)]
    h = highs or [value * 1.01 for value in base_high]
    lw = lows or [value * 0.99 for value in base_low]
    bars = pd.DataFrame(
        {
            "open": o,
            "high": h,
            "low": lw,
            "close": closes,
            "volume": [1_000_000.0] * len(closes),
            "amount": [c * 1_000_000.0 for c in closes],
        },
        index=idx,
    )
    market = "SH"
    instrument = Instrument(
        symbol=symbol,
        market=market,
        name=symbol,
        kind=kind,  # type: ignore[arg-type]
        board=board_of(symbol, market=market),
    )
    return BacktestData(
        instrument=instrument,
        bars_raw=bars,
        bars_qfq=bars,
        factor=pd.Series(dtype=float),
        indicators=daily_frame(bars, IndicatorParams()),
    )


def hand_calc_scenario() -> BacktestData:
    """20 根 10 元横盘 → 11 → 12 → 9，MA20 交叉由此产生一次完整买卖。"""
    closes = [10.0] * 20 + [11.0, 12.0, 9.0, 9.0, 9.0]
    opens = [10.0] * 20 + [10.0, 11.0, 12.0, 9.0, 9.0]
    return make_data(closes, opens=opens)


def run(data: BacktestData, *, costs: CostsConfig = ZERO_COST, cash: float = 100_000.0,
        **kwargs: object):
    return run_backtest(
        Ma20CrossStrategy(),
        {"600000": data},
        start=date(2026, 1, 5),
        end=date(2026, 12, 31),
        costs=costs,
        initial_cash=cash,
        size_mode=kwargs.pop("size_mode", "all_in"),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


# ------------------------------------------------------------------ 手算一致


def test_hand_calculated_trade() -> None:
    result = run(hand_calc_scenario())

    assert len(result.trades) == 1
    trade = result.trades[0]
    # 10 万 / 11 元 → 90 手 = 9000 股；次日以 9 元卖出
    assert trade.quantity == 9000
    assert trade.entry_price_raw == pytest.approx(11.0)
    assert trade.exit_price_raw == pytest.approx(9.0)
    assert trade.gross_pnl == pytest.approx((9.0 - 11.0) * 9000)
    assert trade.net_pnl == pytest.approx(-18_000.0)
    assert result.equity[-1].equity == pytest.approx(100_000.0 - 18_000.0)


def test_equity_curve_matches_hand_computation() -> None:
    result = run(hand_calc_scenario())
    values = [point.equity for point in result.equity]
    # 买入前一直是 10 万；买入后按收盘价计
    assert values[20] == pytest.approx(100_000.0)
    assert values[21] == pytest.approx(1000.0 + 9000 * 12.0)
    assert values[22] == pytest.approx(1000.0 + 9000 * 9.0)


def test_ma20_gate_blocks_entry_below_average() -> None:
    data = make_data([10.0] * 30)
    result = run(data)
    assert result.trades == ()


# ------------------------------------------------------------- §12.8 防偏差


def test_same_input_is_deterministic() -> None:
    first = run(hand_calc_scenario())
    second = run(hand_calc_scenario())
    assert first.trades == second.trades
    assert [p.equity for p in first.equity] == [p.equity for p in second.equity]


def test_truncated_data_gives_same_earlier_trades() -> None:
    full = hand_calc_scenario()
    truncated = make_data(
        [10.0] * 20 + [11.0, 12.0], opens=[10.0] * 20 + [10.0, 11.0]
    )

    full_result = run(full)
    cut = run_backtest(
        Ma20CrossStrategy(),
        {"600000": truncated},
        start=date(2026, 1, 5),
        end=date(2026, 12, 31),
        costs=ZERO_COST,
        initial_cash=100_000.0,
    )
    assert cut.trades == ()
    assert len(full_result.trades) == 1  # 后面的卖出只可能出现在完整数据里


def test_every_fill_lies_inside_its_bar() -> None:
    costs = CostsConfig(slippage=0.002)
    result = run(hand_calc_scenario(), costs=costs)
    data = hand_calc_scenario()

    for fill in result.fills:
        row = data.raw_on(fill.date)
        assert row is not None
        # 去掉滑点后必须落在当日 [low, high] 内
        raw_price = (
            fill.price_raw / (1 + costs.slippage)
            if fill.side == "buy"
            else fill.price_raw / (1 - costs.slippage)
        )
        assert float(row["low"]) <= raw_price <= float(row["high"]), fill


def test_one_word_limit_up_blocks_buy() -> None:
    closes = [10.0] * 20 + [11.0, 12.0, 12.0]
    opens = [10.0] * 20 + [10.0, 11.0, 12.0]
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    # 索引 21 造一字涨停：昨收 11.0 → 涨停价 12.1，四价全部贴住
    highs[21] = lows[21] = opens[21] = closes[21] = 12.1
    data = make_data(closes, opens=opens, highs=highs, lows=lows)

    result = run(data)
    limit_day = pd.Timestamp(data.bars_raw.index[21]).date()
    assert not [f for f in result.fills if f.side == "buy" and f.date == limit_day], (
        "一字涨停当天不应有买入成交"
    )


def test_stop_order_defers_on_one_word_limit_down() -> None:
    order = Order("600000", "sell", 1000, "cond_le", 9.0, "止损")
    bar = Bar(open=9.0, high=9.0, low=9.0, close=9.0, prev_close=10.0)
    outcome = simulate(order, bar, limit_pct=0.10, slippage=0.0, max_gap_open=0.05)
    assert outcome.filled is False
    assert outcome.deferred is True


def test_min_commission_applies() -> None:
    costs = CostsConfig(commission=0.0001654, commission_fund=0.00018, commission_min=5.0,
                        stamp_sell=0.0, transfer=0.0, slippage=0.0)
    data = make_data([10.0] * 20 + [11.0, 12.0, 9.0, 9.0])
    result = run(data, costs=costs)

    buy = next(fill for fill in result.fills if fill.side == "buy")
    assert buy.fee > 5.0  # 成交额够大，按下单费率算已经超过最低佣金
    assert buy.fee == pytest.approx(buy.price_raw * buy.quantity * 0.0001654)


def test_commission_floor_is_hit_for_tiny_trade() -> None:
    costs = CostsConfig(commission=0.0001654, commission_fund=0.00018, commission_min=5.0,
                        stamp_sell=0.0, transfer=0.0, slippage=0.0)
    data = make_data([10.0] * 20 + [11.0, 12.0, 9.0, 9.0])
    result = run(data, costs=costs, cash=2000.0)

    buy = next(fill for fill in result.fills if fill.side == "buy")
    assert buy.quantity == 100  # 2000 元只能买 1 手
    assert buy.fee == pytest.approx(5.0)


def test_etf_uses_fund_commission() -> None:
    costs = CostsConfig(commission=0.0001654, commission_fund=0.00018, commission_min=0.0,
                        stamp_sell=0.0005, transfer=0.00001, slippage=0.0)
    data = make_data(
        [4.0] * 20 + [4.4, 4.8, 3.6, 3.6], symbol="510300", kind="base_etf"
    )

    result = run(data, costs=costs)
    buy = next(fill for fill in result.fills if fill.side == "buy")
    amount = buy.price_raw * buy.quantity
    # ETF 不收印花税与过户费，只有佣金
    assert buy.fee == pytest.approx(amount * 0.00018)


def test_t_plus_one_blocks_same_day_sell() -> None:
    """买入当日即便已跌破 MA20，也只能次日开盘卖。"""
    closes = [10.0] * 20 + [11.0, 11.0, 9.0, 9.0]
    opens = [10.0] * 20 + [10.0, 11.0, 11.0, 9.0]
    data = make_data(closes, opens=opens)

    result = run(data)
    buy = next(f for f in result.fills if f.side == "buy")
    sells = [f for f in result.fills if f.side == "sell"]
    assert sells, "跌破 MA20 后应当卖出"
    assert sells[0].date > buy.date


def test_cost_reconciliation_with_fixed_quantity() -> None:
    """§12.8：无成本与有成本的权益差额必须等于已付费用总额。"""
    costs = CostsConfig(commission=0.0001654, commission_fund=0.00018, commission_min=5.0,
                        stamp_sell=0.0005, transfer=0.00001, slippage=0.0)
    data = hand_calc_scenario()

    costed = run(data, costs=costs, size_mode="fixed", fixed_quantity=9000)
    free = run(data, costs=ZERO_COST, size_mode="fixed", fixed_quantity=9000)

    assert len(costed.trades) == len(free.trades) == 1
    diff = free.equity[-1].equity - costed.equity[-1].equity
    assert diff == pytest.approx(costed.fees_total, abs=0.01)


def test_fixed_size_quantity_is_respected() -> None:
    result = run(hand_calc_scenario(), size_mode="fixed", fixed_quantity=500)
    assert result.trades[0].quantity == 500
