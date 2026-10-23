"""行业层：用中证一级行业指数决定"钱能投哪些行业"（STATE_MACHINE §3）。

只买动量排名前 ``top_k`` 且站上 MA20 的行业；个股所属行业不在集合内就不准入场。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from tradecompass.config.loader import AppConfig
from tradecompass.data import store

#: 中证一级行业指数（baostock 可取，见 DECISIONS D024）
INDUSTRIES: tuple[tuple[str, str], ...] = (
    ("000928", "能源"),
    ("000929", "材料"),
    ("000930", "工业"),
    ("000931", "可选消费"),
    ("000932", "主要消费"),
    ("000933", "医药"),
    ("000934", "金融"),
    ("000935", "信息"),
    ("000936", "电信"),
    ("000937", "公用事业"),
)

DEFAULT_TOP_K = 3
MOMENTUM_WINDOW = 20


@dataclass(frozen=True, slots=True)
class SectorRank:
    code: str
    name: str
    momentum: float
    above_ma20: bool
    rank: int
    vol_ratio20: float | None = None


def industry_of(symbol: str, sector_sw: str | None = None) -> str | None:
    """把个股映射到行业指数代码。

    优先用 ``universe.yaml`` 里的 ``sector_sw`` 名称匹配（支持"计算机""银行"等），
    匹配不到返回 None——此时该标的**不参与**行业过滤（按放行处理，避免过度限制）。
    """
    if not sector_sw:
        return None
    mapping = {
        "能源": ("石油石化", "煤炭"),
        "材料": ("有色金属", "基础化工", "钢铁", "建筑材料"),
        "工业": ("机械设备", "国防军工", "建筑装饰", "电力设备", "汽车", "交通运输"),
        "可选消费": ("家用电器", "纺织服饰", "轻工制造", "商贸零售", "社会服务", "传媒"),
        "主要消费": ("食品饮料", "农林牧渔", "美容护理"),
        "医药": ("医药生物",),
        "金融": ("银行", "非银金融", "房地产"),
        "信息": ("电子", "计算机", "通信"),
        "电信": ("通信服务",),
        "公用事业": ("公用事业", "环保"),
    }
    for code, name in INDUSTRIES:
        if sector_sw in mapping.get(name, ()):
            return code
    return None


def _momentum(close: pd.Series, window: int) -> float | None:
    if len(close) <= window:
        return None
    value = float(close.iloc[-1] / close.iloc[-1 - window] - 1.0)
    return value


def ranking(cfg: AppConfig, day: date, *, window: int = MOMENTUM_WINDOW) -> list[SectorRank]:
    """某日各行业动量排名（只含有数据的行业）。"""
    timestamp = pd.Timestamp(day)
    rows: list[tuple[str, str, float, bool, float | None]] = []
    for code, name in INDUSTRIES:
        bars = store.read_bars_or_empty(store.raw_bars_path(cfg.paths, "SH", code))
        if bars.empty:
            continue
        window_bars = bars.loc[:timestamp]
        if window_bars.empty:
            continue
        close = window_bars["close"].astype(float)
        momentum = _momentum(close, window)
        if momentum is None:
            continue
        ma20 = float(close.tail(20).mean())
        volume = window_bars["volume"].astype(float)
        vol_ratio: float | None = None
        if len(volume) >= 20:
            avg = float(volume.tail(20).mean())
            if avg > 0:
                vol_ratio = float(volume.iloc[-1]) / avg
        rows.append((code, name, momentum, float(close.iloc[-1]) > ma20, vol_ratio))

    rows.sort(key=lambda item: item[2], reverse=True)
    return [
        SectorRank(
            code=code,
            name=name,
            momentum=momentum,
            above_ma20=above,
            rank=index + 1,
            vol_ratio20=vol_ratio,
        )
        for index, (code, name, momentum, above, vol_ratio) in enumerate(rows)
    ]


def buyable(ranks: list[SectorRank], top_k: int = DEFAULT_TOP_K) -> set[str]:
    """可买行业代码集合：排名前 top_k 且站上 MA20。"""
    return {item.code for item in ranks[:top_k] if item.above_ma20}
