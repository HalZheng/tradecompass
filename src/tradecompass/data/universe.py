"""交易规则判定：板块、涨跌停幅度、涨跌停价（SPEC §8.5）。

纯函数，不读配置、不看时间。所有价格参数都带 ``_raw`` 语义（不复权实际价）。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from tradecompass.config.models import InstrumentConfig
from tradecompass.domain.models import Board, Instrument, Kind, Market

# 深市基金：15/16 开头为 ETF/LOF，18 开头为封闭式基金。
_SZ_FUND_PREFIXES = frozenset({"15", "16", "18"})

# 各板块涨跌停幅度（小数）
_LIMIT_PCT: dict[Board, float] = {
    "main": 0.10,
    "gem": 0.20,
    "star": 0.20,
    "bj": 0.30,
    "etf": 0.10,
    "index": 0.0,
}
_ST_LIMIT_PCT = 0.05


def market_of(symbol: str) -> Market:
    """按代码前缀推断交易所（SPEC §8.5）。"""
    if symbol[0] == "6":
        return "SH"
    if symbol[:2] in {"50", "51", "52", "56", "58", "90"}:
        return "SH"
    if symbol[:2] in {"00", "30", "20", "15", "16", "18", "12", "39"}:
        return "SZ"
    if symbol[:2] in {"83", "87", "88", "43"} or symbol[:3] == "920":
        return "BJ"
    raise ValueError(f"无法从代码推断交易所: {symbol!r}")


def is_fund_code(symbol: str) -> bool:
    """是否为场内基金（ETF/LOF/封闭式）。

    SPEC §8.5 原文只写了「5/1 开头」，会漏掉深市 15/16/18 段的 ETF
    （如 159501、159201、159636），也会把 110/113/123 开头的可转债误判为基金。
    这里按实际代码段判定，见 docs/DECISIONS.md D006。
    """
    return symbol[0] == "5" or symbol[:2] in _SZ_FUND_PREFIXES


def board_of(
    symbol: str,
    *,
    market: Market,
    is_index: bool = False,
    is_st: bool = False,
) -> Board:
    """推导板块。``is_index`` 必须由调用方显式给出：000001 既是平安银行也是上证指数。"""
    if is_index:
        return "index"
    if is_fund_code(symbol):
        return "etf"
    if market == "BJ":
        return "bj"
    if market == "SH":
        return "star" if symbol.startswith("688") else "main"
    if market == "SZ":
        if symbol.startswith("30"):
            return "gem"
        return "main"
    raise ValueError(f"未知交易所: {market!r}")


def limit_pct(board: Board, *, is_st: bool = False) -> float:
    """涨跌停幅度（小数）。

    只有主板 ST 收窄到 5%；创业板/科创板的 ST 股仍是 20%，北交所仍是 30%。
    """
    if board == "main" and is_st:
        return _ST_LIMIT_PCT
    return _LIMIT_PCT[board]


def limit_pct_of(instrument: Instrument) -> float:
    return limit_pct(instrument.board, is_st=instrument.is_st)


def _round_cent(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def limit_up_price(prev_close_raw: float, pct: float) -> float:
    """涨停价 = 四舍五入到分(昨收 × (1 + 幅度))（SPEC §8.5）。"""
    return _round_cent(Decimal(str(prev_close_raw)) * (Decimal("1") + Decimal(str(pct))))


def limit_down_price(prev_close_raw: float, pct: float) -> float:
    """跌停价，同样四舍五入到分。"""
    return _round_cent(Decimal(str(prev_close_raw)) * (Decimal("1") - Decimal(str(pct))))


def is_at_limit_up(close_raw: float, prev_close_raw: float, pct: float) -> bool:
    return pct > 0 and close_raw >= limit_up_price(prev_close_raw, pct)


def is_at_limit_down(close_raw: float, prev_close_raw: float, pct: float) -> bool:
    return pct > 0 and close_raw <= limit_down_price(prev_close_raw, pct)


def is_one_word_limit_up(
    open_raw: float, high_raw: float, low_raw: float, close_raw: float, limit_price: float
) -> bool:
    """一字涨停：四价全等于涨停价（SPEC §12.2）。"""
    return min(open_raw, high_raw, low_raw, close_raw) >= limit_price


def is_one_word_limit_down(
    open_raw: float, high_raw: float, low_raw: float, close_raw: float, limit_price: float
) -> bool:
    return max(open_raw, high_raw, low_raw, close_raw) <= limit_price


def board_of_kind(symbol: str, market: Market, kind: Kind, *, is_st: bool = False) -> Board:
    """按 universe 的 ``kind`` 推板块，指数走 index 分支。"""
    return board_of(symbol, market=market, is_index=kind == "index", is_st=is_st)


def index_market(symbol: str) -> Market:
    """指数所属交易所：399 开头为深证系列，其余 000 开头为上证/中证系列。"""
    if symbol.startswith("399"):
        return "SZ"
    if symbol.startswith("000"):
        return "SH"
    raise ValueError(f"无法推断指数所属交易所: {symbol!r}")


def to_instrument(item: InstrumentConfig) -> Instrument:
    """把配置里的标的口径转成领域对象（board 由代码前缀推导）。"""
    return Instrument(
        symbol=item.symbol,
        market=item.market,
        name=item.name,
        kind=item.kind,
        board=board_of_kind(item.symbol, item.market, item.kind, is_st=item.is_st),
        sector_sw=item.sector_sw,
        t0=item.t0,
        is_st=item.is_st,
    )
