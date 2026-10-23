"""行情源统一接口。

一个"源"只需要做一件事：给定标的、区间和复权口径，返回符合
:mod:`tradecompass.data.store` 契约的日线 DataFrame。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

import pandas as pd

from tradecompass.data.store import normalize_bars
from tradecompass.domain.models import Kind, Market

Adjust = Literal["none", "hfq"]


class SourceError(RuntimeError):
    """数据源不可用、返回异常或接口变更。"""


@dataclass(frozen=True, slots=True)
class BarQuery:
    symbol: str
    market: Market
    kind: Kind
    start: date
    end: date


class BarSource(Protocol):
    name: str

    def fetch(self, query: BarQuery, adjust: Adjust) -> pd.DataFrame: ...


def finalize(frame: pd.DataFrame) -> pd.DataFrame:
    """把各源的原始列名统一成契约格式；空表原样返回空表。"""
    if frame.empty:
        return frame
    return normalize_bars(frame)


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume", "amount"],
        index=pd.DatetimeIndex([], name="date"),
    )
