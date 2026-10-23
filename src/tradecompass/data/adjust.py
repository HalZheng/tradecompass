"""复权因子与口径转换（SPEC §8.2）。

因子定义：``factor_hfq[t] = close_hfq[t] / close_raw[t]``。

- 只缓存不复权日线与因子，前/后复权价一律按需计算。
- 所有函数都是纯函数：不读文件、不看当前时间。
"""

from __future__ import annotations

import pandas as pd

FACTOR_COLUMN = "factor_hfq"
_PRICE_COLUMNS = ("open", "high", "low", "close")


class AdjustError(ValueError):
    """因子与行情无法对齐，或因子非法。"""


def compute_factor(bars_hfq: pd.DataFrame, bars_raw: pd.DataFrame) -> pd.Series:
    """由同一标的的后复权与不复权日线相除得到因子序列。"""
    for name, frame in (("hfq", bars_hfq), ("raw", bars_raw)):
        if "close" not in frame.columns:
            raise AdjustError(f"{name} 日线缺少 close 列")
    joined = pd.concat(
        {"hfq": bars_hfq["close"], "raw": bars_raw["close"]}, axis=1, join="inner"
    ).dropna()
    joined = joined[joined["raw"] > 0]
    if joined.empty:
        raise AdjustError("后复权与不复权日线没有重叠的有效交易日")
    factor = (joined["hfq"] / joined["raw"]).sort_index()
    factor.name = FACTOR_COLUMN
    return factor


def _align(factor: pd.Series, index: pd.Index) -> pd.Series:
    aligned = factor.sort_index().reindex(index).ffill()
    if aligned.isna().all():
        raise AdjustError("因子序列与行情索引完全不重叠")
    return aligned


def latest_factor(factor: pd.Series) -> float:
    """最新有效因子，用于 qfq 归一化基准。"""
    valid = factor.dropna()
    if valid.empty:
        raise AdjustError("因子序列为空")
    return float(valid.iloc[-1])


def normalize_factor(factor: pd.Series) -> pd.Series:
    """把因子缩放到「最新日 = 1」。

    必须做这一步：baostock 的后复权基期会随查询窗口变化（实测同一天两次查询可差 36%），
    只有归一化后的因子才是稳定的、可跨次比较的。代价是 ``to_hfq`` 的结果带一个常数倍，
    但后复权价只用来看收益率，常数倍会约掉（SPEC §12.2）。
    """
    base = latest_factor(factor)
    out = factor / base
    out.name = FACTOR_COLUMN
    return out


def to_hfq(bars_raw: pd.DataFrame, factor: pd.Series) -> pd.DataFrame:
    """不复权 → 后复权：OHLC × factor；volume 与 amount 不变。"""
    f = _align(factor, bars_raw.index)
    out = bars_raw.copy()
    for column in _PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float) * f
    return out


def to_qfq(bars_raw: pd.DataFrame, factor: pd.Series) -> pd.DataFrame:
    """不复权 → 前复权：``raw × factor / factor_latest``（SPEC §8.2）。"""
    base = latest_factor(factor)
    f = _align(factor, bars_raw.index) / base
    out = bars_raw.copy()
    for column in _PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float) * f
    return out


def qfq_to_raw_price(price_qfq: float, factor_at_date: float, factor_latest: float) -> float:
    """把前复权价换算回不复权实际价，用于下单价与止损价。"""
    if factor_at_date <= 0 or factor_latest <= 0:
        raise AdjustError(f"因子必须为正: at={factor_at_date}, latest={factor_latest}")
    return price_qfq * factor_latest / factor_at_date


def raw_to_qfq_price(price_raw: float, factor_at_date: float, factor_latest: float) -> float:
    if factor_at_date <= 0 or factor_latest <= 0:
        raise AdjustError(f"因子必须为正: at={factor_at_date}, latest={factor_latest}")
    return price_raw * factor_at_date / factor_latest


def factor_jump_dates(factor: pd.Series, tol: float = 1e-6) -> list[pd.Timestamp]:
    """因子相对前一交易日发生跳变的日期（即除权除息日候选）。"""
    series = factor.dropna().sort_index()
    if len(series) < 2:
        return []
    ratio = series / series.shift(1)
    mask = (ratio - 1.0).abs() > tol
    return [pd.Timestamp(idx) for idx in series.index[mask.fillna(False)]]


def factor_changed(
    old: pd.Series, new: pd.Series, tol: float = 1e-6, *, detect_new_jumps: bool = True
) -> bool:
    """缓存因子与新拉取因子是否不一致（SPEC §8.2 的除权/修订检测）。

    判定为变化的情形：
    1. 共同日期上因子相对变化超过 ``tol``；
    2. ``detect_new_jumps=True`` 时，新增日期里出现新的跳变（说明期间发生了除权）。

    日常增量同步应传 ``detect_new_jumps=False``：最新一根 K 线的因子点本来就是新的，
    把它当成"历史被修订"会天天误报。真正的除权处理在下游按日历判定。
    """
    if old.empty or new.empty:
        return not old.empty or not new.empty

    old_s = old.dropna().sort_index()
    new_s = new.dropna().sort_index()
    common = old_s.index.intersection(new_s.index)
    if len(common) == 0:
        return True
    old_c = old_s.reindex(common)
    new_c = new_s.reindex(common)
    ratio = (new_c - old_c).abs() / old_c.abs()
    if bool((ratio > tol).any()):
        return True

    if not detect_new_jumps:
        return False
    added = new_s.index.difference(old_s.index)
    if len(added) == 0:
        return False
    merged = pd.concat([old_s, new_s.loc[added]]).sort_index()
    return len(factor_jump_dates(merged, tol)) > len(factor_jump_dates(old_s, tol))


def rescale_price_after_adjust(
    price_raw: float, factor_before_ex: float, factor_on_ex: float
) -> float:
    """除权后换算已存的价格（止损价、未执行行动价）。

    ``新价 = 旧价 × factor[除权前一日] / factor[除权日]``（SPEC §8.2）。
    """
    if factor_before_ex <= 0 or factor_on_ex <= 0:
        raise AdjustError(f"因子必须为正: before={factor_before_ex}, on={factor_on_ex}")
    return price_raw * factor_before_ex / factor_on_ex
