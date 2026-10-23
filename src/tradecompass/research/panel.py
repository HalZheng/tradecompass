"""把研究日线拼成 日期 × 代码 的宽表面板。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from tradecompass.config.loader import Paths
from tradecompass.research import data as rdata

FIELDS = (
    "open", "high", "low", "close", "preclose", "volume", "amount", "turn",
    "tradestatus", "is_st", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm",
)
# 创业板 2020-08-24 起涨跌幅 20%
CHINEXT_20PCT_FROM = pd.Timestamp("2020-08-24")


@dataclass(frozen=True, slots=True)
class Panel:
    """全部为 日期 × 代码 宽表；价格为不复权，``adj_*`` 为复权价格指数。"""

    raw: dict[str, pd.DataFrame]
    ret: pd.DataFrame
    adj_close: pd.DataFrame
    adj_open: pd.DataFrame
    adj_high: pd.DataFrame
    adj_low: pd.DataFrame
    float_mcap: pd.DataFrame
    trading: pd.DataFrame
    st: pd.DataFrame
    member: pd.DataFrame
    listed_days: pd.DataFrame
    buy_blocked: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.ret.index)

    @property
    def codes(self) -> pd.Index:
        return self.ret.columns

    def field(self, name: str) -> pd.DataFrame:
        return self.raw[name]

    def eligible(self) -> pd.DataFrame:
        """当日收盘可作为候选：在指数内、正常交易、非 ST、上市满 120 日、次日开盘买得进。"""
        return (
            self.member
            & self.trading
            & ~self.st
            & (self.listed_days >= 120)
            & ~self.buy_blocked.shift(-1, fill_value=False)
        )


def limit_pct(codes: pd.Index, dates: pd.DatetimeIndex, st: pd.DataFrame) -> pd.DataFrame:
    star = np.array([c.startswith("sh.688") for c in codes])
    chinext = np.array([c.startswith(("sz.300", "sz.301")) for c in codes])
    after = np.asarray(dates >= CHINEXT_20PCT_FROM)[:, None]
    pct = np.where(star[None, :] | (chinext[None, :] & after), 0.2, 0.1)
    frame = pd.DataFrame(pct, index=dates, columns=codes)
    return frame.mask(st & (frame < 0.2), 0.05)


def build_panel(bars: dict[str, pd.DataFrame], members: pd.DataFrame) -> Panel:
    """``bars``：代码 → 研究日线；``members``：列 date/code（月末成分快照）。"""
    usable = {code: frame for code, frame in bars.items() if not frame.empty}
    if not usable:
        raise ValueError("没有研究日线，先运行 research sync")
    long = pd.concat(usable, names=["code", "date"])
    raw = {
        name: long[name].unstack("code").sort_index().astype("float64")
        for name in FIELDS
        if name in long.columns
    }
    close = raw["close"]
    dates = pd.DatetimeIndex(close.index)
    codes = close.columns

    ret = (close / raw["preclose"] - 1.0).where(close.notna())
    adj_close = (1.0 + ret.fillna(0.0)).cumprod().where(close.notna())
    scale = adj_close / close
    adj_open = raw["open"] * scale
    adj_high = raw["high"] * scale
    adj_low = raw["low"] * scale

    turn = raw["turn"]
    float_shares = (raw["volume"] / (turn / 100.0)).where(turn > 0).ffill()
    float_mcap = close * float_shares

    trading = (raw["tradestatus"] == 1) & (raw["volume"] > 0)
    st = raw["is_st"].fillna(0.0) == 1
    listed_days = close.notna().cumsum()

    pct = limit_pct(codes, dates, st)
    gap = raw["open"] / raw["preclose"] - 1.0
    buy_blocked = ~trading | (gap >= pct - 0.003)

    member = _membership(members, dates, codes)
    return Panel(
        raw=raw,
        ret=ret,
        adj_close=adj_close,
        adj_open=adj_open,
        adj_high=adj_high,
        adj_low=adj_low,
        float_mcap=float_mcap,
        trading=trading,
        st=st,
        member=member,
        listed_days=listed_days,
        buy_blocked=buy_blocked,
    )


def _membership(members: pd.DataFrame, dates: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    snap = (
        members.assign(flag=1.0)
        .pivot_table(index="date", columns="code", values="flag", aggfunc="max")
        .reindex(columns=codes)
        .fillna(0.0)
    )
    # 月末快照对当日及之后、下一次快照之前生效
    daily = snap.reindex(dates.union(pd.DatetimeIndex(snap.index))).ffill().reindex(dates)
    return daily.fillna(0.0) == 1.0


VALUATION_FIELDS = ("pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm")


def _with_valuation(bars: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    """有东财估值时覆盖估值列，保证全样本同一口径。"""
    if bars.empty or valuation.empty:
        return bars
    aligned = valuation.reindex(bars.index)
    return bars.assign(**{name: aligned[name] for name in VALUATION_FIELDS})


def load_panel(
    paths: Paths, *, start: date | None = None, indexes: tuple[str, ...] = ("hs300", "zz500")
) -> Panel:
    members = rdata.load_constituents(paths)
    members = members[members["index"].isin(indexes)]
    if members.empty:
        raise ValueError("没有成分股快照，先运行 research sync")
    bars = {
        code: _with_valuation(rdata.read_stock_bars(paths, code), rdata.read_valuation(paths, code))
        for code in sorted(set(members["code"]))
    }
    if start is not None:
        cut = pd.Timestamp(start)
        bars = {code: frame[frame.index >= cut] for code, frame in bars.items()}
    return build_panel(bars, members)
