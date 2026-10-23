"""研究数据：按历史时点的指数成分（无幸存者偏差）+ 带换手/估值/停牌/ST 的日线。

全部来自 baostock（2026-10-23 实测可用）：
- ``query_hs300_stocks(date)`` / ``query_zz500_stocks(date)``：任一历史日期的成分；
- ``query_history_k_data_plus`` 不复权日线带 ``preclose``（交易所除权后的前收），
  因此 ``close / preclose − 1`` 就是复权日收益，不必另存复权因子。

布局：``TC_HOME/data/research/{constituents.parquet, bars/<code>.parquet, index/<code>.parquet}``。
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Callable, Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import baostock as bs
import pandas as pd

from tradecompass.config.loader import Paths
from tradecompass.data.calendar import default_calendar
from tradecompass.data.sources import baostock_source
from tradecompass.data.sources.base import SourceError

log = logging.getLogger(__name__)

STOCK_FIELDS = (
    "date,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST,"
    "peTTM,pbMRQ,psTTM,pcfNcfTTM"
)
INDEX_FIELDS = "date,open,high,low,close,preclose,volume,amount,pctChg"
RENAME = {
    "pctChg": "pct_chg",
    "isST": "is_st",
    "peTTM": "pe_ttm",
    "pbMRQ": "pb_mrq",
    "psTTM": "ps_ttm",
    "pcfNcfTTM": "pcf_ttm",
}
INDEX_QUERIES: dict[str, Callable[..., Any]] = {
    "hs300": bs.query_hs300_stocks,
    "zz500": bs.query_zz500_stocks,
}
BENCHMARKS: dict[str, str] = {"hs300": "sh.000300", "zz500": "sh.000905"}
BAOSTOCK_TIMEOUT = 60.0


def research_dir(paths: Paths) -> Path:
    return paths.data / "research"


def constituents_path(paths: Paths) -> Path:
    return research_dir(paths) / "constituents.parquet"


def stock_bars_path(paths: Paths, code: str) -> Path:
    return research_dir(paths) / "bars" / f"{code}.parquet"


def index_bars_path(paths: Paths, code: str) -> Path:
    return research_dir(paths) / "index" / f"{code}.parquet"


def valuation_path(paths: Paths, code: str) -> Path:
    return research_dir(paths) / "valuation" / f"{code}.parquet"


def etf_bars_path(paths: Paths, code: str) -> Path:
    return research_dir(paths) / "etf" / f"{code}.parquet"


def sync_etf_bars(paths: Paths, code: str, start: date, end: date, *, full: bool = False) -> int:
    return _sync_bars(etf_bars_path(paths, code), code, INDEX_FIELDS, start, end, full)


def read_etf_bars(paths: Paths, code: str) -> pd.DataFrame:
    path = etf_bars_path(paths, code)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def append_from_tencent(path: Path, code: str, end: date, *, index: bool = False) -> int:
    """baostock 不可用时，用腾讯日线把 ETF/指数补到 ``end``（前收取上一根收盘，不处理当日拆分）。"""
    import akshare as ak

    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if old.empty:
        return 0
    last = pd.Timestamp(old.index.max()).date()
    if last >= end:
        return 0
    symbol = code.replace(".", "")
    if index:
        raw = ak.stock_zh_index_daily_tx(symbol=symbol)
    else:
        raw = ak.stock_zh_a_hist_tx(
            symbol=symbol, start_date=last.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
            adjust="",
        )
    if raw is None or raw.empty:
        return 0
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.pop("date")), name="date")
    raw = raw[(raw.index > pd.Timestamp(last)) & (raw.index <= pd.Timestamp(end))]
    if raw.empty:
        return 0
    new = pd.DataFrame(index=raw.index)
    for column in ("open", "high", "low", "close", "amount"):
        new[column] = pd.to_numeric(raw[column], errors="coerce")
    volume = raw["volume"] if "volume" in raw else pd.Series(float("nan"), index=raw.index)
    new["volume"] = pd.to_numeric(volume, errors="coerce")
    closes = pd.concat([old["close"].iloc[-1:], new["close"]])
    new["preclose"] = closes.shift(1).loc[new.index]
    new["pct_chg"] = (new["close"] / new["preclose"] - 1.0) * 100.0
    merged = pd.concat([old, new.reindex(columns=old.columns)]).sort_index()
    merged.to_parquet(path)
    return len(new)


def industry_path(paths: Paths) -> Path:
    return research_dir(paths) / "industry.parquet"


def sync_industry(paths: Paths) -> pd.DataFrame:
    """证监会行业分类（当前快照，非历史时点）；已退市标的可能缺失。"""
    baostock_source.login()
    fields, rows = _rows(bs.query_stock_industry(), "行业分类")
    frame = pd.DataFrame(rows, columns=fields)[["code", "industry"]]
    frame = frame[frame["industry"] != ""]
    path = industry_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def load_industry(paths: Paths) -> dict[str, str]:
    path = industry_path(paths)
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    return dict(zip(frame["code"], frame["industry"], strict=True))


def month_end_days(start: date, end: date) -> list[date]:
    """区间内每个月最后一个交易日。"""
    days = default_calendar().trading_days(start, end)
    out: list[date] = []
    for current, following in zip(days, [*days[1:], None], strict=True):
        if following is None or following.month != current.month:
            out.append(current)
    return out


def _rows(result: Any, what: str) -> tuple[list[str], list[list[str]]]:
    if result.error_code != "0":
        raise SourceError(f"baostock {what} 失败: {result.error_msg}")
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    return list(result.fields), rows


def fetch_constituents(index: str, day: date) -> list[str]:
    baostock_source.login()
    fields, rows = _rows(INDEX_QUERIES[index](date=day.isoformat()), f"{index} 成分 {day}")
    position = fields.index("code")
    return sorted({row[position] for row in rows})


def load_constituents(paths: Paths) -> pd.DataFrame:
    """列 ``date``（Timestamp）、``code``、``index``。"""
    path = constituents_path(paths)
    if not path.exists():
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "code": [], "index": []})
    return pd.read_parquet(path)


def sync_constituents(
    paths: Paths, start: date, end: date, indexes: Iterable[str] = ("hs300", "zz500")
) -> pd.DataFrame:
    """补齐每个月末的成分快照；已存在的月份不重复请求。"""
    existing = load_constituents(paths)
    have = {
        (str(i), pd.Timestamp(d).date())
        for d, i in zip(existing["date"], existing["index"], strict=True)
    }
    frames = [existing]
    for index in indexes:
        for day in month_end_days(start, end):
            if (index, day) in have:
                continue
            codes = fetch_constituents(index, day)
            if not codes:
                log.warning("%s %s 成分为空，跳过", index, day)
                continue
            frames.append(
                pd.DataFrame({"date": pd.Timestamp(day), "code": codes, "index": index})
            )
    merged = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    merged = merged.drop_duplicates(["date", "code", "index"])
    merged = merged.sort_values(["date", "index", "code"])
    path = constituents_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return merged.reset_index(drop=True)


def _query_k(code: str, fields: str, start: date, end: date) -> Any:
    socket.setdefaulttimeout(BAOSTOCK_TIMEOUT)
    return bs.query_history_k_data_plus(
        code, fields, start_date=start.isoformat(), end_date=end.isoformat(),
        frequency="d", adjustflag="3",
    )


def _fetch_k(code: str, fields: str, start: date, end: date) -> pd.DataFrame:
    baostock_source.login()
    try:
        result = _query_k(code, fields, start, end)
    except OSError:
        # baostock 无超时参数，靠全局 socket 超时把挂起的请求打断
        baostock_source.relogin()
        result = _query_k(code, fields, start, end)
    columns, rows = _rows(result, f"{code} 日线")
    frame = pd.DataFrame(rows, columns=columns).rename(columns=RENAME)
    if frame.empty:
        return frame
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("date")), name="date")
    frame = frame.replace("", pd.NA)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame.dropna(subset=["close"])


def _sync_bars(path: Path, code: str, fields: str, start: date, end: date, full: bool) -> int:
    old = pd.read_parquet(path) if path.exists() and not full else pd.DataFrame()
    fetch_from = start
    if not old.empty:
        last = pd.Timestamp(old.index.max()).date()
        calendar = default_calendar()
        # 研究按月调仓，晚一天无所谓；避免盘中/收盘前对每只票空跑一次请求
        if last >= calendar.previous_trading_day(calendar.last_trading_day_on_or_before(end)):
            return 0
        fetch_from = last + timedelta(days=1)
    new = _fetch_k(code, fields, fetch_from, end)
    if new.empty:
        if old.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["close"], dtype="float64").to_parquet(path)
        return 0
    merged = pd.concat([old, new]) if not old.empty else new
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path)
    return len(new)


def sync_stock_bars(paths: Paths, code: str, start: date, end: date, *, full: bool = False) -> int:
    return _sync_bars(stock_bars_path(paths, code), code, STOCK_FIELDS, start, end, full)


def sync_index_bars(paths: Paths, code: str, start: date, end: date, *, full: bool = False) -> int:
    return _sync_bars(index_bars_path(paths, code), code, INDEX_FIELDS, start, end, full)


def read_stock_bars(paths: Paths, code: str) -> pd.DataFrame:
    path = stock_bars_path(paths, code)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def read_index_bars(paths: Paths, code: str) -> pd.DataFrame:
    path = index_bars_path(paths, code)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def read_valuation(paths: Paths, code: str) -> pd.DataFrame:
    path = valuation_path(paths, code)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def sync_fast(paths: Paths, code: str, start: date) -> str:
    """刷新东财估值，并用新浪补齐/追加日线。返回 "bars" / "append" / "valuation"。"""
    from tradecompass.research.fast_source import fetch_fast, fetch_valuation

    valuation = fetch_valuation(code)
    path = valuation_path(paths, code)
    path.parent.mkdir(parents=True, exist_ok=True)
    valuation.to_parquet(path)
    bars_path = stock_bars_path(paths, code)
    old = pd.read_parquet(bars_path) if bars_path.exists() else pd.DataFrame()
    if not old.empty and not valuation.empty and old.index.max() >= valuation.index.max():
        return "valuation"
    bars = fetch_fast(code, start, valuation)
    if bars.empty:
        raise SourceError(f"新浪 {code} 无日线")
    bars_path.parent.mkdir(parents=True, exist_ok=True)
    if old.empty:
        bars.to_parquet(bars_path)
        return "bars"
    new = bars[bars.index > old.index.max()]
    if new.empty:
        return "valuation"
    merged = pd.concat([old, new]).sort_index()
    merged.to_parquet(bars_path)
    return "append"
