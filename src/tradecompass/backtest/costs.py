"""交易成本（SPEC §12.3）。

- 佣金按标的类型取费率（股票与场内基金常不同），并有单笔最低。
- 印花税只在股票卖出时收，ETF 免。
- 过户费只在沪深股票上收，基金与指数免。
"""

from __future__ import annotations

from tradecompass.config.models import CostsConfig
from tradecompass.domain.models import Board, Kind

_STOCK_BOARDS: frozenset[Board] = frozenset({"main", "gem", "star", "bj"})


def commission_rate(costs: CostsConfig, kind: Kind) -> float:
    return costs.commission_for(kind)


def trade_fee(
    costs: CostsConfig,
    *,
    kind: Kind,
    board: Board,
    side: str,
    price_raw: float,
    quantity: int,
) -> float:
    """单笔成交的总费用（元）。"""
    amount = price_raw * quantity
    if amount <= 0:
        return 0.0

    commission = max(
        costs.min_commission_for(kind), amount * commission_rate(costs, kind)
    )
    stamp = amount * costs.stamp_sell if side == "sell" and kind == "stock" else 0.0
    transfer = amount * costs.transfer if board in _STOCK_BOARDS else 0.0
    return commission + stamp + transfer


def slippage_price(price_raw: float, side: str, slippage: float) -> float:
    """买入向上、卖出向下让价（SPEC §12.2）。"""
    if side == "buy":
        return price_raw * (1.0 + slippage)
    return price_raw * (1.0 - slippage)
