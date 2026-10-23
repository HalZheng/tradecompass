"""ETF 轮动（A 股宽基 + 行业）：与个股引擎相同的逐日“开盘执行 → 收盘估值 → 收盘决策”。

规则（社区经典的“动量轮动”，参数取常见值，未在本样本上寻优）：
- 打分：近 25 日对数价格线性回归的年化斜率 × R²（趋势强且平滑才高分）；
- 准入：上市满 250 日、收盘在 MA20 之上、得分 > 0；
- 每 5 个交易日重排，持有前 K 只；仍在前 2K 且趋势未破的不换；
- 每日风控：持仓收盘跌破 MA20 次日卖出（趋势策略的天然止损）；空仓时持现金；
- 费用按场内基金：佣金按配置（通常万 1.8、免五），无印花税、无过户费。

2026-10-23 实测（2019–2026、4 个起点）：快参数年化 +0.2%、慢参数（120 日/MA120/月度）−1.2%，
回撤均约 −52%，样本内外都不赚钱 → **不作为交易策略启用**，只保留为研究对照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from tradecompass.backtest.costs import trade_fee
from tradecompass.config.models import CostsConfig
from tradecompass.research.etfs import ROTATION_CATEGORIES, EtfInfo

LOT = 100
ETF_LIMIT = 0.10


@dataclass(frozen=True, slots=True)
class EtfParams:
    top: int = 2
    period: int = 5
    lookback: int = 25
    ma: int = 20
    min_listed: int = 250


@dataclass(frozen=True, slots=True)
class EtfTrade:
    day: pd.Timestamp
    code: str
    side: str
    reason: str
    price: float
    shares: int
    fee: float
    pnl_pct: float | None = None


@dataclass(slots=True)
class EtfState:
    cash: float
    # 复权单位：份额拆分/合并时不变，实际份数 = units × (复权价 / 原始价)
    units: dict[str, float] = field(default_factory=dict)
    entry: dict[str, float] = field(default_factory=dict)
    pending: list[tuple[str, str, str, float]] = field(default_factory=list)
    last_rebalance: int | None = None


def slope_r2_score(adj: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """滚动对数价格回归：年化斜率 × R²；全部只用窗口内（含当日）数据。"""
    logp = np.log(adj.where(adj > 0).to_numpy(dtype="float64"))
    x = np.arange(lookback, dtype="float64")
    x_c = x - x.mean()
    sxx = float((x_c**2).sum())
    out = np.full(adj.shape, np.nan)
    values = logp
    for i in range(lookback - 1, len(values)):
        window = values[i - lookback + 1 : i + 1]
        valid = ~np.isnan(window).any(axis=0)
        if not valid.any():
            continue
        y = window[:, valid]
        y_c = y - y.mean(axis=0)
        slope = (x_c[:, None] * y_c).sum(axis=0) / sxx
        ss_tot = (y_c**2).sum(axis=0)
        ss_res = ((y_c - slope * x_c[:, None]) ** 2).sum(axis=0)
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), 0.0)
        annual = np.exp(slope * 250.0) - 1.0
        out[i, valid] = annual * r2
    return pd.DataFrame(out, index=adj.index, columns=adj.columns)


class EtfRotation:
    def __init__(
        self,
        bars: dict[str, pd.DataFrame],
        infos: dict[str, EtfInfo],
        dates: pd.DatetimeIndex,
        costs: CostsConfig,
        params: EtfParams | None = None,
    ) -> None:
        self.params = params or EtfParams()
        self.costs = costs
        self.dates = dates
        codes = [
            c for c, b in bars.items()
            if not b.empty and infos[c].category in ROTATION_CATEGORIES
        ]
        self.codes = codes
        self.names = {c: infos[c].name for c in codes}

        def wide(column: str) -> pd.DataFrame:
            return pd.DataFrame({c: bars[c][column] for c in codes}).reindex(dates)

        close = wide("close")
        preclose = wide("preclose")
        self.open = wide("open").to_numpy(dtype="float64")
        self.close_raw = close.ffill().to_numpy(dtype="float64")
        self.preclose = preclose.to_numpy(dtype="float64")
        ret = (close / preclose - 1.0).fillna(0.0).where(close.notna().cummax(), np.nan)
        adj = (1.0 + ret.fillna(0.0)).cumprod().where(close.notna().cummax())
        self.adj = adj.ffill().to_numpy(dtype="float64")
        self.scale = (adj / close).ffill().to_numpy(dtype="float64")
        self.trading = (wide("volume") > 0).to_numpy(dtype=bool)
        listed = close.notna().cumsum()
        ma = adj.rolling(self.params.ma, min_periods=self.params.ma).mean()
        self.above_ma = (adj > ma).to_numpy(dtype=bool)
        self.score = slope_r2_score(adj, self.params.lookback).to_numpy(dtype="float64")
        self.listed_ok = (listed >= self.params.min_listed).to_numpy(dtype=bool)

    def equity(self, state: EtfState, i: int) -> float:
        value = state.cash
        for code, units in state.units.items():
            value += units * self.adj[i, self.codes.index(code)]
        return value

    def _fee(self, side: str, price: float, shares: int) -> float:
        return trade_fee(self.costs, kind="sector_etf", board="etf", side=side,
                         price_raw=price, quantity=shares)

    def execute_open(self, state: EtfState, i: int, trades: list[EtfTrade]) -> None:
        day = self.dates[i]
        slip = self.costs.slippage
        carry: list[tuple[str, str, str, float]] = []
        for code, side, reason, amount in sorted(state.pending, key=lambda o: o[1] != "sell"):
            j = self.codes.index(code)
            gap = self.open[i, j] / self.preclose[i, j] - 1.0
            if not self.trading[i, j] or not np.isfinite(gap):
                if side == "sell":
                    carry.append((code, side, reason, amount))
                continue
            if side == "sell":
                units = state.units.get(code, 0.0)
                if units <= 0:
                    continue
                if gap <= -ETF_LIMIT + 0.003:
                    carry.append((code, side, reason, amount))
                    continue
                shares = int(round(units * self.scale[i, j]))
                price = self.open[i, j] * (1.0 - slip)
                fee = self._fee("sell", price, shares)
                state.cash += shares * price - fee
                entry_adj = state.entry.pop(code, price * self.scale[i, j])
                trades.append(EtfTrade(day, code, "sell", reason, price, shares, fee,
                                       price * self.scale[i, j] / entry_adj - 1.0))
                del state.units[code]
            else:
                if code in state.units or gap >= ETF_LIMIT - 0.003:
                    continue
                price = self.open[i, j] * (1.0 + slip)
                shares = int(min(amount, state.cash) // (price * LOT)) * LOT
                while shares > 0 and shares * price + self._fee("buy", price, shares) > state.cash:
                    shares -= LOT
                if shares <= 0:
                    continue
                fee = self._fee("buy", price, shares)
                state.cash -= shares * price + fee
                state.units[code] = shares / self.scale[i, j]
                state.entry[code] = price * self.scale[i, j]
                trades.append(EtfTrade(day, code, "buy", reason, price, shares, fee))
        state.pending = carry

    def decide(self, state: EtfState, i: int) -> None:
        p = self.params
        selling = {o[0] for o in state.pending if o[1] == "sell"}
        for code in list(state.units):
            j = self.codes.index(code)
            if code not in selling and not self.above_ma[i, j]:
                state.pending.append((code, "sell", f"跌破 MA{p.ma}", 0.0))
                selling.add(code)
        if state.last_rebalance is not None and i - state.last_rebalance < p.period:
            return
        state.last_rebalance = i
        ok = self.listed_ok[i] & self.above_ma[i] & (self.score[i] > 0)
        order = [j for j in np.argsort(-np.nan_to_num(self.score[i], nan=-np.inf)) if ok[j]]
        ranked = [self.codes[j] for j in order]
        held = [c for c in state.units if c not in selling]
        keep = [c for c in held if c in ranked[: 2 * p.top]]
        for code in held:
            if code not in keep:
                state.pending.append((code, "sell", "排名跌出", 0.0))
        slot = self.equity(state, i) / p.top
        for code in [c for c in ranked if c not in keep][: max(0, p.top - len(keep))]:
            state.pending.append((code, "buy", "动量轮动入选", slot))

    def run(self, start: date, end: date, capital: float) -> tuple[pd.Series, list[EtfTrade]]:
        lo = int(self.dates.searchsorted(pd.Timestamp(start)))
        hi = int(self.dates.searchsorted(pd.Timestamp(end), side="right"))
        state = EtfState(cash=capital)
        trades: list[EtfTrade] = []
        values = {}
        for i in range(lo, hi):
            self.execute_open(state, i, trades)
            values[self.dates[i]] = self.equity(state, i)
            self.decide(state, i)
        return pd.Series(values), trades
