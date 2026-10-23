"""费用计算：用真实交割单做金标准回归（SPEC §12.3）。

对照组来自项目所有者 2026-10 的真实成交，7 笔覆盖了
「股票买入 / 股票卖出 / ETF 买入 / ETF 卖出」四种情形。

它锁定的核心事实是：**股票有 5 元最低佣金，ETF 免五**。
这一条对小单成本的影响远大于费率本身——1 万元的股票买单，实际费率是名义费率的 3 倍。
"""

from __future__ import annotations

import pytest

from tradecompass.backtest.costs import slippage_price, trade_fee
from tradecompass.config.models import CostsConfig

# 用户账户的真实费率（2026-10 交割单核实）
REAL = CostsConfig(
    commission=0.0001654,
    commission_fund=0.00018,
    commission_min=5.0,
    commission_min_fund=0.0,
    stamp_sell=0.0005,
    transfer=0.00001,
    slippage=0.001,
)

# (标的, kind, board, 方向, 成交金额, 交割单上的费用合计)
GOLDEN = [
    ("山西汾酒", "stock", "main", "buy", 11931.0, 5.12),
    ("大金重工", "stock", "main", "sell", 3459.0, 6.73),
    ("大秦铁路", "stock", "main", "buy", 10045.0, 5.10),
    ("北方稀土", "stock", "main", "sell", 3513.0, 6.80),
    ("黄金ETF", "base_etf", "etf", "buy", 10590.0, 1.91),
    ("黄金ETF", "base_etf", "etf", "sell", 16101.0, 2.90),
    ("纳指基金", "sector_etf", "etf", "buy", 16899.0, 3.04),
]


@pytest.mark.parametrize(
    ("name", "kind", "board", "side", "amount", "expected"), GOLDEN
)
def test_fee_matches_broker_statement(
    name: str, kind: str, board: str, side: str, amount: float, expected: float
) -> None:
    # 用 1 股、价格 = 成交金额 的方式构造等额成交，避免手数干扰
    fee = trade_fee(
        REAL,
        kind=kind,  # type: ignore[arg-type]
        board=board,  # type: ignore[arg-type]
        side=side,
        price_raw=amount,
        quantity=1,
    )
    # 券商按分位取整；另有大金重工（深市）差 0.03 元，恰好等于过户费，
    # 推测该账户对深市未计过户费。按交易所规则仍应计收，故这里只放宽容差。
    assert fee == pytest.approx(expected, abs=0.05), name


def test_stock_commission_floor_is_five_yuan() -> None:
    fee = trade_fee(
        REAL, kind="stock", board="main", side="buy", price_raw=11931.0, quantity=1
    )
    # 5 元佣金 + 0.12 元过户费
    assert fee == pytest.approx(5.12, abs=0.01)


def test_etf_has_no_commission_floor() -> None:
    fee = trade_fee(
        REAL, kind="base_etf", board="etf", side="buy", price_raw=10590.0, quantity=1
    )
    assert fee == pytest.approx(10590.0 * 0.00018, abs=1e-6)
    assert fee < 5.0  # 关键：没有 5 元托底


def test_etf_is_exempt_from_stamp_duty() -> None:
    sell = trade_fee(
        REAL, kind="base_etf", board="etf", side="sell", price_raw=10000.0, quantity=1
    )
    assert sell == pytest.approx(10000.0 * 0.00018, abs=1e-6)


def test_stock_sell_includes_stamp_duty() -> None:
    sell = trade_fee(
        REAL, kind="stock", board="main", side="sell", price_raw=10000.0, quantity=1
    )
    expected = max(5.0, 10000.0 * 0.0001654) + 10000.0 * 0.0005 + 10000.0 * 0.00001
    assert sell == pytest.approx(expected, abs=1e-6)


def test_effective_rate_is_much_worse_for_small_stock_trades() -> None:
    """小单的真实成本：名义万 1.65，1 万元的单子实际要付万 5。"""
    amount = 10045.0
    fee = trade_fee(
        REAL, kind="stock", board="main", side="buy", price_raw=amount, quantity=1
    )
    effective = fee / amount * 10000
    assert effective > 5.0  # 万分之一为单位
    assert effective / 1.654 > 3.0  # 超过名义费率的 3 倍


def test_large_stock_trade_uses_rate_not_floor() -> None:
    amount = 100_000.0
    fee = trade_fee(
        REAL, kind="stock", board="main", side="buy", price_raw=amount, quantity=1
    )
    expected = amount * 0.0001654 + amount * 0.00001
    assert fee == pytest.approx(expected, abs=1e-6)


def test_shenzhen_transfer_fee_explains_small_gap() -> None:
    """深市卖出的一笔差额恰好等于过户费，记录下来避免日后被当成"算错了"。"""
    fee = trade_fee(
        REAL, kind="stock", board="main", side="sell", price_raw=3459.0, quantity=1
    )
    transfer = 3459.0 * 0.00001
    assert fee - transfer == pytest.approx(6.73, abs=0.01)


def test_slippage_direction() -> None:
    assert slippage_price(10.0, "buy", 0.001) == pytest.approx(10.01)
    assert slippage_price(10.0, "sell", 0.001) == pytest.approx(9.99)
