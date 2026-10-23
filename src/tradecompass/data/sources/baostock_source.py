"""baostock 行情源：唯一同时提供「不复权」与「后复权」日线的主力源。

实测（2026-10-21）：
- ``adjustflag="3"`` 不复权，``"1"`` 后复权，``"2"`` 前复权。
- 个股、ETF（含深市 159501）、指数均可取。
- ``volume`` 单位为股/份，``amount`` 单位为元，与契约一致，无需换算。
"""

from __future__ import annotations

import atexit

import baostock as bs
import pandas as pd

from tradecompass.data.sources.base import BarQuery, SourceError, empty_bars, finalize

NAME = "baostock"
_FIELDS = "date,open,high,low,close,volume,amount"
_ADJUST_FLAG = {"none": "3", "hfq": "1"}

_logged_in = False


def _login() -> None:
    global _logged_in
    if _logged_in:
        return
    result = bs.login()
    if result.error_code != "0":
        raise SourceError(f"baostock 登录失败: {result.error_msg}")
    _logged_in = True
    atexit.register(_logout)


def _logout() -> None:
    global _logged_in
    if _logged_in:
        bs.logout()
        _logged_in = False


def bs_code(query: BarQuery) -> str:
    return f"{query.market.lower()}.{query.symbol}"


def fetch(query: BarQuery, adjust: str = "none") -> pd.DataFrame:
    flag = _ADJUST_FLAG.get(adjust)
    if flag is None:
        raise SourceError(f"baostock 不支持的口径: {adjust}")
    _login()
    result = bs.query_history_k_data_plus(
        bs_code(query),
        _FIELDS,
        start_date=query.start.isoformat(),
        end_date=query.end.isoformat(),
        frequency="d",
        adjustflag=flag,
    )
    if result.error_code != "0":
        raise SourceError(f"baostock 查询失败 {bs_code(query)}: {result.error_msg}")

    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    if not rows:
        return empty_bars()

    frame = pd.DataFrame(rows, columns=result.fields)
    frame = frame.replace("", pd.NA).dropna(subset=["close"])
    if frame.empty:
        return empty_bars()
    for column in ("open", "high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["datetime"] = pd.to_datetime(frame["date"])
    return finalize(frame)
