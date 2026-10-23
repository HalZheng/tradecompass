"""回测引擎（SPEC §12.1）。

逐交易日循环：开盘执行昨日订单 → 盘中止损 → 收盘更新状态并生成次日订单 → 记账。

价格口径：成交用**不复权**实际价（与券商一致），权益按复权比例折算，
使除权除息不会凭空吃掉收益（SPEC §12.2 最后一条）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, cast

import pandas as pd

from tradecompass.backtest import costs as cost_mod
from tradecompass.backtest.fills import Bar, simulate
from tradecompass.backtest.types import (
    BacktestResult,
    EquityPoint,
    Fill,
    Order,
    PositionView,
    Strategy,
    StrategyContext,
    Trade,
)
from tradecompass.config.models import CostsConfig
from tradecompass.data.universe import limit_pct_of
from tradecompass.domain.models import Instrument, OrderType
from tradecompass.indicators.frame import weekly_until

LOT_SIZE = 100
SizeMode = Literal["all_in", "equal_slots", "fixed"]


@dataclass(frozen=True, slots=True)
class BacktestData:
    """一个标的在回测区间内的全部输入。"""

    instrument: Instrument
    bars_raw: pd.DataFrame
    bars_qfq: pd.DataFrame
    factor: pd.Series
    indicators: pd.DataFrame
    weekly: pd.DataFrame = field(default_factory=pd.DataFrame)

    def raw_on(self, day: date) -> pd.Series | None:
        ts = pd.Timestamp(day)
        if ts not in self.bars_raw.index:
            return None
        return cast(pd.Series, self.bars_raw.loc[ts])

    def factor_on(self, day: date) -> float:
        if self.factor.empty:
            return 1.0
        ts = pd.Timestamp(day)
        window = self.factor.loc[:ts]
        if window.empty:
            return 1.0
        value = window.iloc[-1]
        return float(value) if pd.notna(value) else 1.0


@dataclass(slots=True)
class _OpenPosition:
    symbol: str
    shares: int
    entry_price_raw: float
    entry_date: date
    entry_factor: float
    entry_fees: float
    stop_raw: float | None
    highest_close_raw: float

    def view(self, close_raw: float) -> PositionView:
        change = (close_raw / self.entry_price_raw - 1.0) if self.entry_price_raw else 0.0
        return PositionView(
            symbol=self.symbol,
            shares=self.shares,
            entry_price_raw=self.entry_price_raw,
            entry_date=self.entry_date,
            current_stop_raw=self.stop_raw,
            highest_close_raw=self.highest_close_raw,
            unrealized_pct=change,
        )


@dataclass(slots=True)
class _Book:
    cash: float
    positions: dict[str, _OpenPosition] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    fees_total: float = 0.0
    equity: list[EquityPoint] = field(default_factory=list)


def _affordable(cash: float, price: float, costs: CostsConfig, kind: str, board: str) -> int:
    if price <= 0:
        return 0
    rate = costs.commission_for(kind)
    budget = cash / (1.0 + rate + costs.slippage) if cash > 0 else 0.0
    lots = int(budget // (price * LOT_SIZE))
    return max(lots, 0) * LOT_SIZE


def _target_quantity(
    size_mode: SizeMode,
    equity: float,
    price: float,
    costs: CostsConfig,
    data: BacktestData,
    max_positions: int,
    fixed_quantity: int,
) -> int:
    if size_mode == "fixed":
        return fixed_quantity
    budget = equity if size_mode == "all_in" else equity / max(max_positions, 1)
    return _affordable(budget, price, costs, data.instrument.kind, data.instrument.board)


def _equity(book: _Book, data_map: dict[str, BacktestData], day: date) -> float:
    total = book.cash
    for symbol, pos in book.positions.items():
        data = data_map[symbol]
        row = data.raw_on(day)
        if row is None:
            total += pos.shares * pos.entry_price_raw
            continue
        ratio = data.factor_on(day) / pos.entry_factor if pos.entry_factor else 1.0
        total += pos.shares * float(row["close"]) * ratio
    return total


def _close_position(
    book: _Book,
    data: BacktestData,
    pos: _OpenPosition,
    *,
    day: date,
    price_raw: float,
    reason: str,
    costs: CostsConfig,
) -> None:
    instrument = data.instrument
    quantity = pos.shares
    fee = cost_mod.trade_fee(
        costs,
        kind=instrument.kind,
        board=instrument.board,
        side="sell",
        price_raw=price_raw,
        quantity=quantity,
    )
    ratio = data.factor_on(day) / pos.entry_factor if pos.entry_factor else 1.0
    gross = quantity * (price_raw * ratio - pos.entry_price_raw)
    net = gross - fee - pos.entry_fees
    risk = pos.shares * (pos.entry_price_raw - pos.stop_raw) if pos.stop_raw else 0.0
    r_multiple = net / risk if risk > 0 else None

    book.cash += quantity * price_raw * ratio - fee
    book.fees_total += fee
    book.fills.append(
        Fill(
            date=day,
            symbol=pos.symbol,
            side="sell",
            quantity=quantity,
            price_raw=price_raw,
            fee=fee,
            reason=reason,
        )
    )
    book.trades.append(
        Trade(
            symbol=pos.symbol,
            entry_date=pos.entry_date,
            entry_price_raw=pos.entry_price_raw,
            exit_date=day,
            exit_price_raw=price_raw,
            quantity=quantity,
            gross_pnl=gross,
            fees=fee + pos.entry_fees,
            net_pnl=net,
            r_multiple=r_multiple,
            holding_days=(day - pos.entry_date).days,
            exit_reason=reason,
        )
    )
    del book.positions[pos.symbol]


def _try_buy(
    book: _Book,
    data: BacktestData,
    order: Order,
    *,
    day: date,
    price_raw: float,
    costs: CostsConfig,
    size_mode: SizeMode,
    max_positions: int,
    fixed_quantity: int,
) -> bool:
    instrument = data.instrument
    quantity = order.quantity
    if size_mode == "fixed":
        if quantity * price_raw > book.cash:
            return False
    else:
        affordable = _affordable(book.cash, price_raw, costs, instrument.kind, instrument.board)
        quantity = min(quantity, affordable) if quantity > 0 else affordable
    quantity -= quantity % LOT_SIZE
    if quantity <= 0:
        return False

    fee = cost_mod.trade_fee(
        costs,
        kind=instrument.kind,
        board=instrument.board,
        side="buy",
        price_raw=price_raw,
        quantity=quantity,
    )
    if quantity * price_raw + fee > book.cash + 1e-9:
        return False

    book.cash -= quantity * price_raw + fee
    book.fees_total += fee
    book.fills.append(
        Fill(
            date=day,
            symbol=order.symbol,
            side="buy",
            quantity=quantity,
            price_raw=price_raw,
            fee=fee,
            reason=order.reason,
        )
    )
    book.positions[order.symbol] = _OpenPosition(
        symbol=order.symbol,
        shares=quantity,
        entry_price_raw=price_raw,
        entry_date=day,
        entry_factor=data.factor_on(day),
        entry_fees=fee,
        stop_raw=None,
        highest_close_raw=price_raw,
    )
    return True


def run_backtest(
    strategy: Strategy,
    data_map: dict[str, BacktestData],
    *,
    start: date,
    end: date,
    costs: CostsConfig,
    initial_cash: float,
    max_positions: int = 5,
    size_mode: SizeMode = "equal_slots",
    fixed_quantity: int = LOT_SIZE,
    max_gap_open: float = 0.05,
) -> BacktestResult:
    """在给定数据上跑一遍回测。``data_map`` 为 代码 → BacktestData。

    ``size_mode`` 默认 ``equal_slots``：按 ``max_positions`` 均分资金。
    用 ``all_in`` 配合 ``max_positions > 1`` 等于"全仓押第一只"，会让
    「最多同时持 5 只」这条风控形同虚设——历史上这里踩过坑，别改回去。
    """
    book = _Book(cash=initial_cash)
    pending: list[Order] = []
    all_days = sorted(
        {
            pd.Timestamp(ts).date()
            for data in data_map.values()
            for ts in data.bars_raw.index
            if start <= pd.Timestamp(ts).date() <= end
        }
    )

    for day in all_days:
        # 1. 开盘：执行昨日订单
        still_pending: list[Order] = []
        for order in pending:
            data = data_map[order.symbol]
            row = data.raw_on(day)
            if row is None:  # 停牌
                if order.side == "sell":
                    still_pending.append(order)
                continue
            prev_close = _previous_close(data, day)
            bar = Bar(
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                prev_close=prev_close,
            )
            outcome = simulate(
                order,
                bar,
                limit_pct=limit_pct_of(data.instrument),
                slippage=costs.slippage,
                max_gap_open=max_gap_open,
            )
            if outcome.deferred:
                still_pending.append(order)
                continue
            if not outcome.filled or outcome.price_raw is None:
                continue
            if order.side == "buy":
                if order.symbol in book.positions:
                    continue
                if len(book.positions) >= max_positions:
                    continue
                if _try_buy(
                    book,
                    data,
                    order,
                    day=day,
                    price_raw=outcome.price_raw,
                    costs=costs,
                    size_mode=size_mode,
                    max_positions=max_positions,
                    fixed_quantity=fixed_quantity,
                ):
                    opened = book.positions[order.symbol]
                    opened.stop_raw = _initial_stop(strategy, data, day, outcome.price_raw)
            else:
                held = book.positions.get(order.symbol)
                if held is None or held.entry_date >= day:
                    continue
                _close_position(
                    book,
                    data,
                    held,
                    day=day,
                    price_raw=outcome.price_raw,
                    reason=order.reason,
                    costs=costs,
                )
        pending = still_pending

        # 2. 盘中：止损（T+1 约束下只能卖昨日之前买入的）
        for symbol, pos in list(book.positions.items()):
            data = data_map[symbol]
            row = data.raw_on(day)
            if row is None or pos.entry_date >= day or pos.stop_raw is None:
                continue
            if float(row["low"]) > pos.stop_raw:
                continue
            bar = Bar(
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                prev_close=_previous_close(data, day),
            )
            outcome = simulate(
                Order(symbol, "sell", pos.shares, "cond_le", pos.stop_raw, "止损"),
                bar,
                limit_pct=limit_pct_of(data.instrument),
                slippage=costs.slippage,
                max_gap_open=max_gap_open,
            )
            if outcome.filled and outcome.price_raw is not None:
                _close_position(
                    book,
                    data,
                    pos,
                    day=day,
                    price_raw=outcome.price_raw,
                    reason="止损",
                    costs=costs,
                )

        # 3. 收盘后：更新最高收盘、生成次日订单
        equity_now = _equity(book, data_map, day)
        next_orders: list[Order] = []
        for symbol, data in data_map.items():
            row = data.raw_on(day)
            if row is None:
                continue
            close_raw = float(row["close"])
            current = book.positions.get(symbol)
            if current is not None:
                current.highest_close_raw = max(current.highest_close_raw, close_raw)

            ctx = StrategyContext(
                date=day,
                instrument=data.instrument,
                bars=data.bars_qfq.loc[: pd.Timestamp(day)],
                daily=data.indicators.loc[: pd.Timestamp(day)],
                weekly=weekly_until(data.weekly, day),
                position=current.view(close_raw) if current else None,
            )
            if current is None:
                if len(book.positions) >= max_positions:
                    continue
                signal = strategy.entry(ctx)
                if signal is None:
                    continue
                trigger, reason = signal
                estimate = trigger if trigger is not None else close_raw
                quantity = _target_quantity(
                    size_mode,
                    equity_now,
                    estimate,
                    costs,
                    data,
                    max_positions,
                    fixed_quantity,
                )
                if quantity <= 0:
                    continue
                order_type: OrderType = "open" if trigger is None else "cond_ge"
                next_orders.append(Order(symbol, "buy", quantity, order_type, trigger, reason))
            else:
                if current.entry_date >= day:
                    continue
                signal = strategy.exit(ctx)
                if signal is None:
                    continue
                trigger, reason = signal
                if trigger is None:
                    next_orders.append(Order(symbol, "sell", current.shares, "open", None, reason))
                else:
                    next_orders.append(
                        Order(symbol, "sell", current.shares, "cond_ge", trigger, reason)
                    )
        pending = pending + next_orders

        book.equity.append(
            EquityPoint(
                date=day,
                equity=equity_now,
                cash=book.cash,
                position_value=equity_now - book.cash,
            )
        )

    return BacktestResult(
        strategy_id=strategy.id,
        start=start,
        end=end,
        initial_cash=initial_cash,
        trades=tuple(book.trades),
        equity=tuple(book.equity),
        fills=tuple(book.fills),
        fees_total=book.fees_total,
    )


def _previous_close(data: BacktestData, day: date) -> float:
    ts = pd.Timestamp(day)
    window = data.bars_raw.loc[:ts]
    if len(window) < 2:
        return float(window["close"].iloc[-1]) if len(window) else 0.0
    return float(window["close"].iloc[-2])


def _initial_stop(
    strategy: Strategy, data: BacktestData, day: date, entry_price_raw: float
) -> float | None:
    ctx = StrategyContext(
        date=day,
        instrument=data.instrument,
        bars=data.bars_qfq.loc[: pd.Timestamp(day)],
        daily=data.indicators.loc[: pd.Timestamp(day)],
        weekly=weekly_until(data.weekly, day),
        position=None,
    )
    return strategy.initial_stop(ctx, entry_price_raw)
