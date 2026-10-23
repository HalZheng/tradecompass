"""数据校验（SPEC §8.3）。

每个标的每日检查，任一项失败即标记 ``data_suspect``，该标的当日不出信号（风控 R1）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

CROSS_SOURCE_TOL = 0.003  # 多源收盘价相对偏差上限
LIMIT_MOVE_BUFFER = 0.01  # 涨跌停判定放宽 1 个百分点


@dataclass(frozen=True, slots=True)
class Issue:
    code: str
    detail: str


def check_ohlc(frame: pd.DataFrame) -> list[Issue]:
    """high < max(open, close) 或 low > min(open, close)。"""
    if frame.empty:
        return []
    upper = frame[["open", "close"]].max(axis=1)
    lower = frame[["open", "close"]].min(axis=1)
    bad = frame[(frame["high"] < upper) | (frame["low"] > lower)]
    return [
        Issue("ohlc_invalid", f"{idx.date()} high/low 与 open/close 矛盾")
        for idx in bad.index
    ]


def check_limit_moves(
    frame: pd.DataFrame,
    limit_pct: float,
    *,
    exclude_dates: frozenset[date] | None = None,
) -> list[Issue]:
    """相邻两日收盘涨跌幅超过「涨跌停幅度 + 1%」，除权日除外。"""
    if limit_pct <= 0 or len(frame) < 2:
        return []
    close = frame["close"].astype(float)
    change = close.pct_change().abs()
    threshold = limit_pct + LIMIT_MOVE_BUFFER
    excluded = exclude_dates or frozenset()
    issues: list[Issue] = []
    for ts, raw_value in zip(change.index.tolist(), change.to_numpy(), strict=True):
        magnitude = float(raw_value)
        if not magnitude > threshold:  # NaN 也会走到这里，等价于跳过
            continue
        moment = pd.Timestamp(ts).date()
        if moment in excluded:
            continue
        issues.append(
            Issue("limit_move", f"{moment} 收盘变动 {magnitude:.2%} 超过 {threshold:.0%}")
        )
    return issues


def check_latest_bar(
    frame: pd.DataFrame, expected_last: date | None, *, suspended: bool = False
) -> Issue | None:
    """最新交易日 K 线是否缺失；停牌日不算缺失。"""
    if suspended or expected_last is None:
        return None
    if frame.empty:
        return Issue("missing_last_bar", f"缺少 {expected_last} 的日线（且未标记停牌）")
    last = pd.Timestamp(frame.index[-1]).date()
    if last < expected_last:
        return Issue("missing_last_bar", f"最新日线为 {last}，缺少 {expected_last}")
    return None


def check_cross_source(
    primary: pd.Series, secondary: pd.Series, *, tol: float = CROSS_SOURCE_TOL
) -> Issue | None:
    """多源不复权收盘价偏差（同日相对差的最大值）。"""
    common = primary.index.intersection(secondary.index)
    if len(common) == 0:
        return None
    a = primary.reindex(common).astype(float)
    b = secondary.reindex(common).astype(float)
    denom = a.where(a != 0)
    deviation = ((a - b).abs() / denom).dropna()
    if deviation.empty:
        return None
    worst_date = deviation.idxmax()
    worst = float(deviation.max())
    if worst > tol:
        return Issue(
            "cross_source_mismatch",
            f"{pd.Timestamp(worst_date).date()} 两源收盘价偏差 {worst:.2%} 超过 {tol:.2%}",
        )
    return None


def validate_all(
    frame: pd.DataFrame,
    *,
    limit_pct: float,
    expected_last: date | None = None,
    suspended: bool = False,
    ex_dates: frozenset[date] | None = None,
    secondary_close: pd.Series | None = None,
) -> list[Issue]:
    """把当日全部校验跑一遍，返回问题列表（空表示通过）。"""
    issues: list[Issue] = []
    issues.extend(check_ohlc(frame))
    issues.extend(check_limit_moves(frame, limit_pct, exclude_dates=ex_dates))
    missing = check_latest_bar(frame, expected_last, suspended=suspended)
    if missing is not None:
        issues.append(missing)
    if secondary_close is not None and not frame.empty:
        mismatch = check_cross_source(frame["close"], secondary_close)
        if mismatch is not None:
            issues.append(mismatch)
    return issues
