"""ETF 判定池：宽基、行业、跨境三类；底仓 ETF 只做再平衡。

名单为常见的宽基/行业/跨境 ETF（2026-10 人工整理）。
名单本身是“现在”选出来的，有幸存者偏差：回测里每只 ETF 上市满 250 日才可选，
且报告需注明这一点。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EtfInfo:
    code: str
    name: str
    category: str


ETFS: tuple[EtfInfo, ...] = (
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

ROTATION_CATEGORIES = ("宽基", "行业")


def etf_by_code() -> dict[str, EtfInfo]:
    return {info.code: info for info in ETFS}
