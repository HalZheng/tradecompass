"""Parquet 读写（SPEC §8.2）。

路径约定：
- 不复权日线：``data/bars/raw/{market}{symbol}_d.parquet``
- 复权因子：  ``data/bars/factor/{market}{symbol}.parquet``

行情 DataFrame 契约：索引为无时区的 ``DatetimeIndex``（日期），
列 ``open, high, low, close, volume, amount``，volume 单位为股/份，amount 单位为元。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tradecompass.config.loader import Paths

BAR_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")
INDEX_NAME = "date"


class StoreError(RuntimeError):
    """行情文件缺失或格式不符。"""


def raw_bars_path(paths: Paths, market: str, symbol: str) -> Path:
    return paths.bars_raw_dir / f"{market}{symbol}_d.parquet"


def factor_path(paths: Paths, market: str, symbol: str) -> Path:
    return paths.bars_factor_dir / f"{market}{symbol}.parquet"


def normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """统一索引与列顺序，并做最基本的清洗。"""
    out = frame.copy()
    if "datetime" in out.columns:
        out = out.set_index("datetime")
    index = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()
    out.index = index
    out.index.name = INDEX_NAME
    out = out.drop_duplicates(keep="last").sort_index()

    missing = [c for c in BAR_COLUMNS if c not in out.columns]
    if missing:
        raise StoreError(f"行情缺少列: {', '.join(missing)}")
    return out.loc[:, list(BAR_COLUMNS)].astype(float)


def write_bars(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_bars(frame).to_parquet(path)


def read_bars(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise StoreError(f"行情文件不存在: {path}")
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
    frame.index.name = INDEX_NAME
    return frame.sort_index()


def read_bars_or_empty(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=list(BAR_COLUMNS), index=pd.DatetimeIndex([], name=INDEX_NAME))
    return read_bars(path)


def write_factor(path: Path, factor: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series = factor.copy()
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).normalize()
    series.index.name = INDEX_NAME
    series.sort_index().to_frame().to_parquet(path)


def read_factor(path: Path) -> pd.Series:
    if not path.is_file():
        raise StoreError(f"因子文件不存在: {path}")
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
    frame.index.name = INDEX_NAME
    return frame.iloc[:, 0].sort_index()


def read_factor_or_empty(path: Path) -> pd.Series:
    if not path.is_file():
        return pd.Series(dtype=float, name="factor_hfq")
    return read_factor(path)


def merge_bars(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """增量合并：新数据覆盖同一天，随后按日期排序。"""
    if old.empty:
        return normalize_bars(new)
    if new.empty:
        return normalize_bars(old)
    combined = pd.concat([old, normalize_bars(new)])
    return normalize_bars(combined)


def source_path(paths: Paths, market: str, symbol: str) -> Path:
    """记录该标的历史上是用哪个源拉的，避免增量时换了源导致口径混用。"""
    return paths.bars_raw_dir / f"{market}{symbol}.source"


def write_source(paths: Paths, market: str, symbol: str, name: str) -> None:
    path = source_path(paths, market, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name, encoding="utf-8")


def read_source(paths: Paths, market: str, symbol: str, default: str = "baostock") -> str:
    path = source_path(paths, market, symbol)
    if not path.is_file():
        return default
    return path.read_text(encoding="utf-8").strip() or default
