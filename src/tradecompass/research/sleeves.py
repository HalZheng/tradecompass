"""把证监会行业归并成几个“板块”（分域），用于分域因子模型与节奏差异研究。

归并依据国内券商“分域选股”的常见做法：大金融 / 周期资源 / 地产基建 / 公用交运 /
消费 / 医药 / 科技成长 / 先进制造。证监会分类是当前快照，不是历史时点。
"""

from __future__ import annotations

SLEEVES: tuple[str, ...] = (
    "大金融", "周期资源", "地产基建", "公用交运", "消费", "医药", "科技成长", "先进制造", "其他",
)

_BY_LETTER = {
    "J": "大金融",
    "K": "地产基建",
    "E": "地产基建",
    "B": "周期资源",
    "D": "公用交运",
    "G": "公用交运",
    "N": "公用交运",
    "I": "科技成长",
    "M": "科技成长",
    "Q": "医药",
    "A": "消费",
    "F": "消费",
    "H": "消费",
    "L": "消费",
    "R": "消费",
}
_BY_MANUFACTURING = {
    **dict.fromkeys(range(13, 25), "消费"),
    **dict.fromkeys((25, 26, 28, 29, 30, 31, 32, 33), "周期资源"),
    27: "医药",
    **dict.fromkeys((34, 35, 36, 37, 38, 41, 42, 43), "先进制造"),
    **dict.fromkeys((39, 40), "科技成长"),
}


def sleeve_of(industry: str | None) -> str:
    """``industry`` 形如 ``C39计算机、通信和其他电子设备制造业``。"""
    if not industry:
        return "其他"
    letter = industry[0]
    if letter == "C" and industry[1:3].isdigit():
        return _BY_MANUFACTURING.get(int(industry[1:3]), "其他")
    return _BY_LETTER.get(letter, "其他")


def sleeve_map(industry: dict[str, str]) -> dict[str, str]:
    return {code: sleeve_of(name) for code, name in industry.items()}
