"""研究日线的快速通道：新浪日线 + 东财历史估值，HTTP 可并发。

2026-10-23 与 baostock 交叉校验（8 只，2018 年起）：收盘/成交量/换手/复权收益完全一致；
PE(TTM) 基本一致，PB 个别日差 5%–40%（净资产更新时点不同），市现率口径完全不同。
因此估值字段对**全部**标的统一用东财，量价可来自 baostock 或新浪。

差异：新浪没有停牌日的行、没有已退市标的、没有 ST 标记（记 0）；``preclose`` 由东财涨跌幅反推，
东财无数据的早期日子退化为前一日收盘。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date

import pandas as pd

from tradecompass.data.sources.base import SourceError

RETRIES = 3
VALUATION_COLUMNS = ("pct_chg", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm")


def _call(label: str, fn: Callable[..., object], **kwargs: object) -> pd.DataFrame:
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            result = fn(**kwargs)
            if not isinstance(result, pd.DataFrame):
                raise SourceError(f"{label} 返回类型异常: {type(result).__name__}")
            return result
        except Exception as exc:  # noqa: BLE001 - 第三方抓取异常种类不可控
            last = exc
            # 新浪对已退市标的返回空体，重试无意义
            if "No value to decode" in str(exc):
                break
            time.sleep(1.5 * (attempt + 1))
    raise SourceError(f"{label} 失败: {last}")


def fetch_valuation(code: str) -> pd.DataFrame:
    """东财历史估值；``code`` 为 baostock 风格，例如 ``sz.000001``。"""
    import akshare as ak

    value = _call(f"东财估值 {code}", ak.stock_value_em, symbol=code.split(".")[1])
    if value.empty:
        return pd.DataFrame(columns=list(VALUATION_COLUMNS), dtype="float64")
    value.index = pd.DatetimeIndex(pd.to_datetime(value.pop("数据日期")), name="date")
    value = value.apply(pd.to_numeric, errors="coerce").rename(
        columns={
            "当日涨跌幅": "pct_chg",
            "PE(TTM)": "pe_ttm",
            "市净率": "pb_mrq",
            "市销率": "ps_ttm",
            "市现率": "pcf_ttm",
        }
    )
    out: pd.DataFrame = value[list(VALUATION_COLUMNS)].sort_index().astype("float64")
    return out


def fetch_fast(code: str, start: date, valuation: pd.DataFrame) -> pd.DataFrame:
    """新浪不复权日线，用东财涨跌幅反推交易所前收。"""
    import akshare as ak

    market, symbol = code.split(".")
    daily = _call(f"新浪 {code}", ak.stock_zh_a_daily, symbol=f"{market}{symbol}", adjust="")
    if daily.empty:
        return pd.DataFrame()
    daily.index = pd.DatetimeIndex(pd.to_datetime(daily.pop("date")), name="date")
    daily = daily.astype("float64")
    frame = pd.DataFrame(
        {
            "open": daily["open"],
            "high": daily["high"],
            "low": daily["low"],
            "close": daily["close"],
            "volume": daily["volume"],
            "amount": daily["amount"],
            "turn": daily["turnover"] * 100.0,
        }
    ).join(valuation[["pct_chg"]]).sort_index()
    implied = frame["close"] / (1.0 + frame["pct_chg"] / 100.0)
    frame["preclose"] = implied.fillna(frame["close"].shift(1))
    frame["tradestatus"] = (frame["volume"] > 0).astype("float64")
    frame["is_st"] = 0.0
    frame = frame[frame.index >= pd.Timestamp(start)].dropna(subset=["close"])
    return frame.astype("float64")
