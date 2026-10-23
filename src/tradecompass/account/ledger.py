"""持仓推导（SPEC §11.2）。

``trades`` 表是唯一事实来源；持仓不落库，每次由流水算出，避免"两份事实"打架。

摊薄成本口径（与东财一致）：
``(累计买入金额 + 买入费用 − 累计卖出金额 + 卖出费用 − 红利) / 当前股数``
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date


@dataclass(slots=True)
class Holding:
    symbol: str
    name: str | None = None
    quantity: int = 0
    buy_amount: float = 0.0
    sell_amount: float = 0.0
    buy_fees: float = 0.0
    sell_fees: float = 0.0
    dividends: float = 0.0
    taxes: float = 0.0
    first_buy: date | None = None
    last_trade: date | None = None
    realized_pnl: float = 0.0
    _cost_pool: float = field(default=0.0, repr=False)

    @property
    def avg_cost_raw(self) -> float:
        """摊薄成本（不复权）。已清仓时返回 0。"""
        if self.quantity <= 0:
            return 0.0
        return (
            self.buy_amount + self.buy_fees - self.sell_amount + self.sell_fees - self.dividends
        ) / self.quantity

    @property
    def total_fees(self) -> float:
        return self.buy_fees + self.sell_fees + self.taxes

    def market_value(self, close_raw: float) -> float:
        return self.quantity * close_raw

    def unrealized(self, close_raw: float) -> float:
        return self.quantity * (close_raw - self.avg_cost_raw)

    def unrealized_pct(self, close_raw: float) -> float:
        cost = self.avg_cost_raw
        return (close_raw - cost) / cost if cost > 0 else 0.0


@dataclass(slots=True)
class SymbolHistory:
    """单票交易历史摘要（已清仓的票也保留）。"""

    symbol: str
    name: str | None = None
    last_buy_date: date | None = None
    last_sell_date: date | None = None
    last_sell_pnl: float | None = None
    realized_pnl: float = 0.0
    consecutive_losses: int = 0


def _row_date(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def derive_ledger(
    trades: list[sqlite3.Row],
) -> tuple[dict[str, Holding], dict[str, SymbolHistory]]:
    """按时间顺序处理流水，得到当前持仓、已实现盈亏与每票历史摘要。"""
    holdings: dict[str, Holding] = {}
    histories: dict[str, SymbolHistory] = {}

    def get(symbol: str, name: str | None) -> Holding:
        if symbol not in holdings:
            holdings[symbol] = Holding(symbol=symbol, name=name)
        elif name and not holdings[symbol].name:
            holdings[symbol].name = name
        return holdings[symbol]

    def get_history(symbol: str, name: str | None) -> SymbolHistory:
        if symbol not in histories:
            histories[symbol] = SymbolHistory(symbol=symbol, name=name)
        elif name and not histories[symbol].name:
            histories[symbol].name = name
        return histories[symbol]

    for row in sorted(trades, key=lambda r: (str(r["trade_date"]), str(r["trade_time"] or ""))):
        symbol = str(row["symbol"])
        holding = get(symbol, row["name"])
        history = get_history(symbol, row["name"])
        side = str(row["side"])
        quantity = int(row["quantity"] or 0)
        price = float(row["price_raw"] or 0.0)
        amount = float(row["amount"] or price * quantity)
        fees = (
            float(row["fee_commission"] or 0)
            + float(row["fee_stamp"] or 0)
            + float(row["fee_transfer"] or 0)
            + float(row["fee_other"] or 0)
        )
        day = _row_date(row["trade_date"])
        holding.last_trade = max(holding.last_trade or day, day)

        if side == "buy":
            holding.quantity += quantity
            holding.buy_amount += amount
            holding.buy_fees += fees
            holding._cost_pool += amount + fees
            holding.first_buy = holding.first_buy or day
            history.last_buy_date = day
        elif side == "sell":
            sold = min(quantity, holding.quantity)
            pnl: float | None = None
            if sold > 0:
                # FIFO：按当前摊薄成本结转已实现盈亏
                unit_cost = holding._cost_pool / holding.quantity if holding.quantity else 0.0
                pnl = sold * (price - unit_cost) - fees
                holding.realized_pnl += pnl
                holding._cost_pool -= unit_cost * sold
            holding.quantity -= quantity
            holding.sell_amount += amount
            holding.sell_fees += fees
            history.last_sell_date = day
            history.last_sell_pnl = pnl
            if pnl is not None:
                history.realized_pnl += pnl
                history.consecutive_losses = (
                    history.consecutive_losses + 1 if pnl < 0 else 0
                )
        elif side == "bonus_share":
            holding.quantity += quantity
        elif side == "dividend":
            holding.dividends += amount or price * quantity
            holding._cost_pool -= amount or price * quantity
        elif side == "tax":
            holding.taxes += amount or price * quantity
            holding._cost_pool += amount or price * quantity

    active = {
        symbol: holding for symbol, holding in holdings.items() if holding.quantity > 0
    }
    return active, histories


def derive(trades: list[sqlite3.Row]) -> dict[str, Holding]:
    return derive_ledger(trades)[0]


def load_trades(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    cursor = conn.execute("SELECT * FROM trades ORDER BY trade_date, trade_time, id")
    return list(cursor.fetchall())


def positions_from_db(conn: sqlite3.Connection) -> dict[str, Holding]:
    return derive(load_trades(conn))


def history_from_db(conn: sqlite3.Connection) -> dict[str, SymbolHistory]:
    return derive_ledger(load_trades(conn))[1]
