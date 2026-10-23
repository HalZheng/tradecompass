"""数据源注册表：按优先级降级取数。

``system.yaml`` 里的源名可以包含未安装的源（如 mootdx 属于可选 extra），
这里在运行时会跳过不可导入的源，而不是让整个任务崩溃。
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from tradecompass.data.sources import akshare_source, baostock_source
from tradecompass.data.sources.base import Adjust, BarQuery, SourceError, empty_bars

Fetcher = Callable[[BarQuery, str], pd.DataFrame]

_REGISTRY: dict[str, Fetcher] = {
    baostock_source.NAME: baostock_source.fetch,
    akshare_source.NAME: akshare_source.fetch,
}


def known_sources() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def resolve(names: list[str] | tuple[str, ...]) -> list[str]:
    """过滤出已实现且可用的源，保持配置里的顺序。"""
    return [name for name in names if name in _REGISTRY]


def fetch_with_fallback(
    query: BarQuery,
    adjust: Adjust,
    names: list[str] | tuple[str, ...],
    *,
    allow_empty: bool = False,
) -> tuple[pd.DataFrame, str]:
    """按顺序尝试各源，返回第一个非空结果与其来源名。

    ``allow_empty=True`` 时，"源正常响应但没有数据"（如当日行情还没发布）
    不算失败，返回空表；只有全部源都报错才抛 :class:`SourceError`。
    """
    usable = resolve(names)
    if not usable:
        raise SourceError(f"没有可用的数据源（配置为 {list(names)}）")
    errors: list[str] = []
    responded_empty: str | None = None
    for name in usable:
        try:
            frame = _REGISTRY[name](query, adjust)
        except SourceError as exc:
            errors.append(f"{name}: {exc}")
            continue
        if not frame.empty:
            return frame, name
        responded_empty = responded_empty or name
        errors.append(f"{name}: 返回空数据")
    if allow_empty and responded_empty is not None:
        return empty_bars(), responded_empty
    raise SourceError(f"{query.symbol} 全部数据源失败 -> {' | '.join(errors)}")


def fetch_named(query: BarQuery, adjust: Adjust, name: str) -> pd.DataFrame:
    """指定源取数，用于交叉校验。源不可用时抛 SourceError。"""
    if name not in _REGISTRY:
        raise SourceError(f"未知或未安装的数据源: {name}")
    return _REGISTRY[name](query, adjust)
