"""持仓体检（C 方案的核心产出）。

不做选股，只回答三个问题：
1. 这只票的周线趋势还在不在？
2. 有没有止损？止损该放哪？
3. 该继续拿、还是找机会减？
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from tradecompass.account.ledger import Holding
from tradecompass.config.loader import AppConfig
from tradecompass.data import adjust, store
from tradecompass.data.universe import limit_pct_of, to_instrument
from tradecompass.domain.models import Instrument
from tradecompass.indicators.frame import (
    IndicatorParams,
    daily_frame,
    weekly_frame,
    weekly_until,
)

LOSS_ALERT = -0.15  # 浮亏超过 15% 视为失控
STOP_ATR_MULT = 2.0


@dataclass(frozen=True, slots=True)
class HoldingRow:
    symbol: str
    name: str
    quantity: int
    avg_cost: float
    last_close: float
    market_value: float
    pnl: float
    pnl_pct: float
    weekly_ok: bool | None
    weekly_jw: float | None
    daily_j: float | None
    suggested_stop: float | None
    verdict: str
    action: str
    has_data: bool = True


@dataclass(frozen=True, slots=True)
class CheckupReport:
    as_of: date
    rows: tuple[HoldingRow, ...]
    total_value: float
    total_cost: float
    total_pnl: float
    rebalance: tuple[str, ...]

    @property
    def total_pnl_pct(self) -> float:
        return self.total_pnl / self.total_cost if self.total_cost else 0.0

    @property
    def alerts(self) -> tuple[HoldingRow, ...]:
        return tuple(row for row in self.rows if row.action.startswith("⚠"))


def classify(weekly_ok: bool | None, pnl_pct: float) -> tuple[str, str]:
    """把"趋势 + 盈亏"映射成结论与动作。纯函数，便于测试。"""
    if weekly_ok is None:
        return "缺数据", "⚠ 先跑 data sync"
    if weekly_ok and pnl_pct >= 0:
        return "趋势健康", "持有，止损上移"
    if weekly_ok and pnl_pct < 0:
        return "趋势健康但浮亏", "持有观察，跌破止损再走"
    if not weekly_ok and pnl_pct >= 0:
        return "周线走坏", "⚠ 止盈离场，别等回本"
    return "周线走坏且浮亏", "⚠ 反弹减仓（clean）"


def _instrument_of(cfg: AppConfig, symbol: str) -> Instrument | None:
    for item in cfg.universe.instruments:
        if item.symbol == symbol:
            return to_instrument(item)
    return None


def _load_series(
    cfg: AppConfig, instrument: Instrument
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    raw = store.read_bars_or_empty(
        store.raw_bars_path(cfg.paths, instrument.market, instrument.symbol)
    )
    if raw.empty:
        return None
    factor = store.read_factor_or_empty(
        store.factor_path(cfg.paths, instrument.market, instrument.symbol)
    )
    qfq = adjust.to_qfq(raw, factor) if not factor.empty else raw
    params = IndicatorParams()
    return raw, daily_frame(qfq, params), weekly_frame(qfq, raw.index[-1].date(), params)


def build_report(cfg: AppConfig, holdings: dict[str, Holding]) -> CheckupReport:
    rows: list[HoldingRow] = []
    rebalance: list[str] = []
    total_value = 0.0
    total_cost = 0.0

    for symbol, holding in sorted(holdings.items()):
        instrument = _instrument_of(cfg, symbol)
        name = holding.name or (instrument.name if instrument else symbol)
        cost = holding.avg_cost_raw
        series = _load_series(cfg, instrument) if instrument else None
        if instrument is None or series is None:
            rows.append(
                HoldingRow(
                    symbol=symbol, name=name, quantity=holding.quantity, avg_cost=cost,
                    last_close=0.0, market_value=0.0, pnl=0.0, pnl_pct=0.0,
                    weekly_ok=None, weekly_jw=None, daily_j=None, suggested_stop=None,
                    verdict="缺数据", action="⚠ 先 sync 该标的（不在 universe.yaml 里则先加入）",
                    has_data=False,
                )
            )
            total_cost += cost * holding.quantity
            continue

        raw, daily, weekly = series
        close = float(raw["close"].iloc[-1])
        week = weekly_until(weekly, raw.index[-1].date())
        weekly_ok: bool | None = None
        jw: float | None = None
        if len(week) >= 21:
            last = week.iloc[-1]
            weekly_ok = bool(last["close"] > last["ma20w"] and bool(last["ma20w_up"]))
            if not pd.isna(last["jw"]):
                jw = float(last["jw"])
        daily_j = None
        atr = None
        if not daily.empty:
            if not pd.isna(daily["j"].iloc[-1]):
                daily_j = float(daily["j"].iloc[-1])
            if not pd.isna(daily["atr14"].iloc[-1]):
                atr = float(daily["atr14"].iloc[-1])
        prev_low = None
        if not daily.empty and not pd.isna(daily["llv_low_10"].iloc[-1]):
            prev_low = float(daily["llv_low_10"].iloc[-1])

        stop = None
        if atr:
            candidates = [close - STOP_ATR_MULT * atr]
            if prev_low and prev_low < close:
                candidates.append(prev_low)
            stop = max(candidates)

        pnl = holding.unrealized(close)
        pnl_pct = holding.unrealized_pct(close)
        verdict, action = classify(weekly_ok, pnl_pct)
        if pnl_pct <= LOSS_ALERT and not action.startswith("⚠"):
            action = f"⚠ 浮亏 {pnl_pct:.0%} 已失控，必须先定止损"

        if instrument.kind == "base_etf":
            target = next(
                (item.target for item in cfg.universe.instruments if item.symbol == symbol), None
            )
            if target:
                rebalance.append(f"{name} 目标 {target:.0%}")

        rows.append(
            HoldingRow(
                symbol=symbol, name=name, quantity=holding.quantity, avg_cost=cost,
                last_close=close, market_value=holding.market_value(close), pnl=pnl,
                pnl_pct=pnl_pct, weekly_ok=weekly_ok, weekly_jw=jw, daily_j=daily_j,
                suggested_stop=stop, verdict=verdict, action=action,
            )
        )
        total_value += holding.market_value(close)
        total_cost += cost * holding.quantity

    return CheckupReport(
        as_of=date.today(),
        rows=tuple(rows),
        total_value=total_value,
        total_cost=total_cost,
        total_pnl=total_value - total_cost,
        rebalance=tuple(rebalance),
    )


def stop_for(instrument: Instrument, close: float, atr: float, prev_low: float | None) -> float:
    """止损建议价（不复权），与回测 X1 同口径。"""
    candidates = [close - STOP_ATR_MULT * atr]
    if prev_low and prev_low < close:
        candidates.append(prev_low)
    stop = max(candidates)
    # 至少留出 1 个最小价位
    return max(stop, close * (1 - limit_pct_of(instrument)))
