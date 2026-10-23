"""交易行为体检：从对账单成交流水算已实现盈亏（移动平均成本）、持有期、换手与习惯。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass(slots=True)
class _Lot:
    quantity: int = 0
    cost: float = 0.0
    opened: date | None = None


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    code: str
    name: str
    day: date
    quantity: int
    pnl: float
    pnl_pct: float
    hold_days: int


@dataclass(slots=True)
class BehaviorReport:
    trades: int = 0
    buys: int = 0
    sells: int = 0
    bought: float = 0.0
    sold: float = 0.0
    fees: float = 0.0
    closed: list[ClosedTrade] = field(default_factory=list)
    same_day_round_trips: int = 0
    symbols_traded: int = 0
    by_symbol: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def realized(self) -> float:
        return sum(c.pnl for c in self.closed)


def analyze(
    trades: pd.DataFrame, opening: dict[str, tuple[int, float]] | None = None
) -> BehaviorReport:
    """``trades`` 为 portfolio/eastmoney/trades.csv；``opening`` 为区间前已有持仓（数量, 成本）。"""
    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame.sort_values("date", kind="stable")
    lots: dict[str, _Lot] = {}
    for code, (qty, cost) in (opening or {}).items():
        lots[code] = _Lot(qty, qty * cost, None)
    report = BehaviorReport()
    seen_days: dict[tuple[str, date], set[str]] = {}
    for row in frame.to_dict("records"):
        day: date = row["date"]
        code = str(row["code"]).zfill(6)
        fee = float(row["fee"]) + float(row["stamp"]) + float(row["transfer_fee"])
        report.fees += fee
        report.trades += 1
        seen_days.setdefault((code, day), set()).add(str(row["side"]))
        lot = lots.setdefault(code, _Lot())
        qty = int(row["quantity"])
        price = float(row["price"])
        amount = float(row["amount"])
        if row["side"] == "buy":
            report.buys += 1
            report.bought += qty * price
            if lot.quantity == 0:
                lot.opened = day
            lot.cost += -amount
            lot.quantity += qty
        else:
            report.sells += 1
            report.sold += qty * price
            if lot.quantity <= 0:
                continue
            take = min(qty, lot.quantity)
            unit = lot.cost / lot.quantity
            proceeds = amount * take / qty
            pnl = proceeds - unit * take
            hold = (day - lot.opened).days if lot.opened else -1
            report.closed.append(
                ClosedTrade(code, str(row["name"]), day, take, pnl, pnl / (unit * take), hold)
            )
            lot.cost -= unit * take
            lot.quantity -= take
    report.same_day_round_trips = sum(1 for sides in seen_days.values() if len(sides) == 2)
    report.symbols_traded = len({str(c).zfill(6) for c in frame["code"]})
    if report.closed:
        closed = pd.DataFrame(
            [
                {"code": c.code, "name": c.name, "pnl": c.pnl, "hold_days": c.hold_days}
                for c in report.closed
            ]
        )
        report.by_symbol = (
            closed.groupby(["code", "name"])
            .agg(
                笔数=("pnl", "size"), 已实现盈亏=("pnl", "sum"), 平均持有天数=("hold_days", "mean")
            )
            .sort_values("已实现盈亏")
            .reset_index()
        )
    return report
