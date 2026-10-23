"""交易规则判定：板块、涨跌停幅度与涨跌停价（SPEC §8.5）。"""

from __future__ import annotations

import pytest

from tradecompass.data.universe import (
    board_of,
    is_at_limit_down,
    is_at_limit_up,
    is_fund_code,
    is_one_word_limit_up,
    limit_down_price,
    limit_pct,
    limit_up_price,
    market_of,
)


@pytest.mark.parametrize(
    ("symbol", "market", "expected"),
    [
        ("600000", "SH", "main"),
        ("601318", "SH", "main"),
        ("688981", "SH", "star"),
        ("000001", "SZ", "main"),
        ("300750", "SZ", "gem"),
        ("830799", "BJ", "bj"),
        ("430047", "BJ", "bj"),
        ("920001", "BJ", "bj"),
    ],
)
def test_board_of_stocks(symbol: str, market: str, expected: str) -> None:
    assert board_of(symbol, market=market) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "symbol",
    [
        "510300",  # 沪市宽基 ETF
        "518880",  # 沪市黄金 ETF
        "511090",  # 沪市国债 ETF
        "512880",  # 沪市行业 ETF
        "159501",  # 深市 QDII ETF —— SPEC §8.5 原文会漏判
        "159201",  # 深市 ETF
        "159636",  # 深市 ETF
        "159755",
        "159992",
        "160632",  # 深市 LOF
        "180101",  # 深市封闭式基金
    ],
)
def test_deep_market_and_shanghai_funds_are_etf(symbol: str) -> None:
    market = "SH" if symbol.startswith("5") else "SZ"
    assert is_fund_code(symbol) is True
    assert board_of(symbol, market=market) == "etf"  # type: ignore[arg-type]


@pytest.mark.parametrize("symbol", ["110043", "113050", "123456", "127789"])
def test_convertible_bonds_are_not_funds(symbol: str) -> None:
    assert is_fund_code(symbol) is False


def test_index_must_be_declared_explicitly() -> None:
    assert board_of("000001", market="SH", is_index=True) == "index"
    assert board_of("000001", market="SZ") == "main"


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("600000", "SH"),
        ("688981", "SH"),
        ("510300", "SH"),
        ("000001", "SZ"),
        ("300750", "SZ"),
        ("159501", "SZ"),
        ("399001", "SZ"),
        ("830799", "BJ"),
        ("920001", "BJ"),
    ],
)
def test_market_of(symbol: str, expected: str) -> None:
    assert market_of(symbol) == expected


def test_market_of_rejects_unknown_prefix() -> None:
    with pytest.raises(ValueError, match="无法从代码推断交易所"):
        market_of("990001")


@pytest.mark.parametrize(
    ("board", "is_st", "expected"),
    [
        ("main", False, 0.10),
        ("main", True, 0.05),
        ("gem", False, 0.20),
        ("gem", True, 0.20),  # 创业板 ST 仍是 20%
        ("star", False, 0.20),
        ("star", True, 0.20),
        ("bj", False, 0.30),
        ("bj", True, 0.30),
        ("etf", False, 0.10),
        ("index", False, 0.0),
    ],
)
def test_limit_pct(board: str, is_st: bool, expected: float) -> None:
    assert limit_pct(board, is_st=is_st) == expected  # type: ignore[arg-type]


def test_limit_prices_round_half_up_to_cent() -> None:
    assert limit_up_price(10.0, 0.10) == pytest.approx(11.00)
    assert limit_up_price(3.05, 0.10) == pytest.approx(3.36)  # 3.355 → 四舍五入
    assert limit_down_price(3.05, 0.10) == pytest.approx(2.75)  # 2.745 → 四舍五入
    assert limit_up_price(9.99, 0.20) == pytest.approx(11.99)  # 11.988 → 11.99


def test_limit_hit_helpers() -> None:
    assert is_at_limit_up(11.0, 10.0, 0.10)
    assert not is_at_limit_up(10.99, 10.0, 0.10)
    assert is_at_limit_down(9.0, 10.0, 0.10)
    assert not is_at_limit_up(5.0, 10.0, 0.0)  # 指数无涨跌停


def test_one_word_limit_up() -> None:
    limit = limit_up_price(10.0, 0.10)
    assert is_one_word_limit_up(limit, limit, limit, limit, limit)
    assert not is_one_word_limit_up(10.90, limit, limit, limit, limit)
