"""基准（SPEC §12.5）。"""

from __future__ import annotations

import random
from datetime import date

import pandas as pd

from tradecompass.backtest.engine import BacktestData
from tradecompass.backtest.metrics import max_drawdown
from tradecompass.backtest.types import EquityPoint

LOT_SIZE = 100


def equal_weight_buy_hold(
    data_map: dict[str, BacktestData], *, initial_cash: float
) -> tuple[EquityPoint, ...]:
    """样本等权买入持有：期初等额买入，之后不再调仓，不计成本。"""
    if not data_map:
        return ()
    per_symbol = initial_cash / len(data_map)

    shares: dict[str, int] = {}
    cash = initial_cash
    entry_day: dict[str, date] = {}
    for symbol, data in data_map.items():
        if data.bars_raw.empty:
            continue
        first_day = pd.Timestamp(data.bars_raw.index[0]).date()
        price = float(data.bars_raw["open"].iloc[0])
        lots = int(per_symbol // (price * LOT_SIZE)) if price > 0 else 0
        quantity = lots * LOT_SIZE
        if quantity <= 0:
            continue
        shares[symbol] = quantity
        cash -= quantity * price
        entry_day[symbol] = first_day

    days = sorted(
        {
            pd.Timestamp(ts).date()
            for data in data_map.values()
            for ts in data.bars_raw.index
        }
    )
    points: list[EquityPoint] = []
    for day in days:
        position_value = 0.0
        for symbol, quantity in shares.items():
            data = data_map[symbol]
            row = data.raw_on(day)
            if row is None:
                continue
            entry_factor = data.factor_on(entry_day[symbol])
            ratio = data.factor_on(day) / entry_factor if entry_factor else 1.0
            position_value += quantity * float(row["close"]) * ratio
        points.append(
            EquityPoint(
                date=day,
                equity=cash + position_value,
                cash=cash,
                position_value=position_value,
            )
        )
    return tuple(points)


def benchmark_drawdown(points: tuple[EquityPoint, ...]) -> float:
    return max_drawdown(points)


def random_entry_days(
    qualified: list[date], count: int, *, seed: int = 42
) -> list[date]:
    """从"满足趋势条件"的交易日里随机抽 count 个入场日（SPEC §12.5 第 3 项）。"""
    if not qualified or count <= 0:
        return []
    rng = random.Random(seed)
    picks = min(count, len(qualified))
    return sorted(rng.sample(qualified, picks))
