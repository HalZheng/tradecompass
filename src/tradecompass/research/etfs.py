"""ETF 判定池：按资产类别（收益驱动）分信号族，按操作角色分底仓/稳健/波段。

资产类别在 ``asset_class``（equity_broad/equity_value/sector/overseas/
overseas_sector/commodity/bond），策略族由 ``strategy_of`` 推导：

- ``core_band``：A股宽基/红利、海外宽基——核心持有 + 250 日位置仓位带；
- ``core_band_high_floor``：黄金/商品——核心仓位更高（floor 0.7），只在极端高位减；
- ``hold``：债券——战略持有/再平衡，不做股票式仓位带；
- ``swing_cleanup``：行业/主题、海外行业——不新增，按 MA20 清理，趋势强度只做展示。

``t0/qdii/aum/avg_amount/tracking_error`` 是元数据占位：目前只用于分类/展示，
执行仍按 T+1、真实费率与滑点。名单为常见品种（2026-10 人工整理），有幸存者偏差。
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class EtfInfo:
    code: str
    name: str
    category: str
    asset_class: str = ""
    t0: bool | None = None
    """是否 T+0；None = 未接入/未确认。当前执行仍按 T+1。"""
    qdii: bool = False
    aum: float | None = None
    """规模（亿元），暂未接入数据。"""
    avg_amount: float | None = None
    """日均成交额（亿元），暂未接入数据。"""
    tracking_error: float | None = None
    """跟踪误差，暂未接入数据。"""


# 资产类别：按“收益驱动”分，不按表面名称分。
ASSET_CLASS: dict[str, str] = {
    "sh.510300": "equity_broad",
    "sh.510050": "equity_broad",
    "sh.510500": "equity_broad",
    "sh.512100": "equity_broad",
    "sz.159915": "equity_broad",
    "sh.588000": "equity_broad",
    "sh.515080": "equity_value",
    "sz.159201": "equity_value",
    "sh.513910": "equity_value",
    "sh.513100": "overseas",
    "sz.159501": "overseas",
    "sh.513400": "overseas",
    "sz.159636": "overseas_sector",
    "sh.520600": "overseas_sector",
    "sh.518880": "commodity",
    "sh.511090": "bond",
}

ASSET_CLASS_LABELS = {
    "equity_broad": "A股宽基",
    "equity_value": "红利/价值（A股/港股）",
    "sector": "A股行业/主题",
    "overseas": "海外/QDII",
    "overseas_sector": "海外行业",
    "commodity": "黄金/商品",
    "bond": "债券",
}

STRATEGY_LABELS = {
    "core_band": "核心+仓位带",
    "hold": "持有/再平衡",
    "swing_cleanup": "MA20清理/不新增",
}


@dataclass(frozen=True, slots=True)
class BandParams:
    high_cut: float
    mid_cut: float
    low_cut: float
    high: float
    mid: float
    neutral: float
    low_wait: float


ETF_BAND: dict[str, BandParams] = {
    "core_band": BandParams(85.0, 70.0, 30.0, 0.5, 0.75, 1.0, 0.6),
}


_RAW_ETFS: tuple[EtfInfo, ...] = (
    EtfInfo("sh.510300", "沪深300ETF", "底仓"),
    EtfInfo("sh.518880", "黄金ETF", "底仓"),
    EtfInfo("sh.511090", "30年国债ETF", "底仓"),
    EtfInfo("sh.510050", "上证50ETF", "宽基"),
    EtfInfo("sh.510500", "中证500ETF", "宽基"),
    EtfInfo("sh.512100", "中证1000ETF", "宽基"),
    EtfInfo("sz.159915", "创业板ETF", "宽基"),
    EtfInfo("sh.588000", "科创50ETF", "宽基"),
    EtfInfo("sh.515080", "中证红利ETF", "宽基"),
    EtfInfo("sz.159201", "自由现金流ETF", "宽基"),
    EtfInfo("sh.512880", "证券ETF", "行业"),
    EtfInfo("sh.512800", "银行ETF", "行业"),
    EtfInfo("sh.512010", "医药ETF", "行业"),
    EtfInfo("sh.512170", "医疗ETF", "行业"),
    EtfInfo("sz.159992", "创新药ETF", "行业"),
    EtfInfo("sh.512690", "酒ETF", "行业"),
    EtfInfo("sz.159928", "消费ETF", "行业"),
    EtfInfo("sh.512480", "半导体ETF", "行业"),
    EtfInfo("sz.159995", "芯片ETF", "行业"),
    EtfInfo("sh.515050", "5G通信ETF", "行业"),
    EtfInfo("sh.515980", "人工智能ETF", "行业"),
    EtfInfo("sz.159732", "消费电子ETF", "行业"),
    EtfInfo("sh.512980", "传媒ETF", "行业"),
    EtfInfo("sh.512660", "军工ETF", "行业"),
    EtfInfo("sz.159227", "航空航天ETF", "行业"),
    EtfInfo("sh.562500", "机器人ETF", "行业"),
    EtfInfo("sh.512400", "有色金属ETF", "行业"),
    EtfInfo("sh.515220", "煤炭ETF", "行业"),
    EtfInfo("sh.515210", "钢铁ETF", "行业"),
    EtfInfo("sh.516950", "基建ETF", "行业"),
    EtfInfo("sh.512200", "房地产ETF", "行业"),
    EtfInfo("sh.516160", "新能源ETF", "行业"),
    EtfInfo("sh.515790", "光伏ETF", "行业"),
    EtfInfo("sh.515030", "新能源车ETF", "行业"),
    EtfInfo("sz.159755", "电池ETF", "行业"),
    EtfInfo("sz.159566", "储能电池ETF", "行业"),
    EtfInfo("sz.159825", "农业ETF", "行业"),
    EtfInfo("sz.159865", "养殖ETF", "行业"),
    EtfInfo("sz.159611", "电力ETF", "行业"),
    EtfInfo("sh.513100", "纳指ETF", "跨境"),
    EtfInfo("sz.159501", "纳指基金", "跨境"),
    EtfInfo("sh.513400", "道琼斯ETF", "跨境"),
    EtfInfo("sh.513910", "港股通央企红利ETF", "跨境"),
    EtfInfo("sz.159636", "港科技30ETF", "跨境"),
    EtfInfo("sh.520600", "汽车港股ETF", "跨境"),
)


def _derive_asset_class(info: EtfInfo) -> str:
    if info.code in ASSET_CLASS:
        return ASSET_CLASS[info.code]
    if info.category == "行业":
        return "sector"
    if info.category == "跨境":
        return "overseas_sector"
    if info.category == "底仓":
        return "commodity"
    return "equity_broad"


ETFS: tuple[EtfInfo, ...] = tuple(
    replace(
        info,
        asset_class=_derive_asset_class(info),
        qdii=info.category == "跨境",
    )
    for info in _RAW_ETFS
)

ROTATION_CATEGORIES = ("宽基", "行业")


def asset_class_of(info: EtfInfo) -> str:
    return info.asset_class or _derive_asset_class(info)


def strategy_of(info: EtfInfo) -> str:
    """由资产类别推导策略族；债券/黄金只做再平衡，仓位带只用于宽基/红利/海外。"""
    asset = asset_class_of(info)
    if asset in ("bond", "commodity"):
        return "hold"
    if asset in ("equity_broad", "equity_value", "overseas"):
        return "core_band"
    return "swing_cleanup"


def role_of(info: EtfInfo) -> str:
    if info.category == "底仓":
        return "底仓"
    if strategy_of(info) in ("core_band", "hold"):
        return "稳健"
    return "波段"


def etf_roles() -> dict[str, str]:
    return {info.code: role_of(info) for info in ETFS}


def etf_by_code() -> dict[str, EtfInfo]:
    return {info.code: info for info in ETFS}
