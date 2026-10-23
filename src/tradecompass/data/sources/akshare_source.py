"""akshare 行情源：交叉校验与备用源。

实测（2026-10-21）的两个坑：
1. **``成交量`` 单位是「手」**（baostock 是「股」），必须 ×100 才与契约一致。
2. **后复权基准与 baostock 不同**：600000 在同一日 baostock hfq=124.98、akshare hfq=100.97，
   差一个常数倍。因子只能做**比例**比较（除以自身首值），不能比绝对值。
"""

from __future__ import annotations

from datetime import date

import akshare as ak
import pandas as pd

from tradecompass.data.sources.base import BarQuery, SourceError, empty_bars, finalize
from tradecompass.data.universe import is_fund_code

NAME = "akshare"

_COLUMN_MAP = {
    "日期": "datetime",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}
_HANDS_TO_SHARES = 100


def _yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")


def _frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return empty_bars()
    missing = [c for c in _COLUMN_MAP if c not in raw.columns]
    if missing:
        raise SourceError(f"akshare 返回列缺失: {', '.join(missing)}")
    frame = raw.rename(columns=_COLUMN_MAP).loc[:, list(_COLUMN_MAP.values())]
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") * _HANDS_TO_SHARES
    for column in ("open", "high", "low", "close", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return finalize(frame)


def fetch(query: BarQuery, adjust: str = "none") -> pd.DataFrame:
    start = _yyyymmdd(query.start)
    end = _yyyymmdd(query.end)
    aj = "hfq" if adjust == "hfq" else ""
    try:
        if query.kind == "index":
            # 指数不复权，两种口径返回同一份数据
            raw = ak.index_zh_a_hist(
                symbol=query.symbol, period="daily", start_date=start, end_date=end
            )
        elif query.kind in ("base_etf", "sector_etf") or is_fund_code(query.symbol):
            raw = ak.fund_etf_hist_em(
                symbol=query.symbol, period="daily", start_date=start, end_date=end, adjust=aj
            )
        else:
            raw = ak.stock_zh_a_hist(
                symbol=query.symbol, period="daily", start_date=start, end_date=end, adjust=aj
            )
    except Exception as exc:  # noqa: BLE001 - 第三方库抛出的异常类型不稳定
        raise SourceError(f"akshare 查询失败 {query.symbol}: {exc}") from exc
    return _frame(raw)
