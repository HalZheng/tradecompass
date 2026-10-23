"""信号系统专用回测：逐日重放 B1/B2/S1/S2。

为什么不复用 `backtest/engine.py`：旧引擎的 `Strategy` 协议只看单标的
`StrategyContext`，拿不到大盘 regime、行业排名、账户现金、全部持仓和交易历史；
而新信号层的 `evaluate()` 恰恰依赖这些上下文。硬套适配器会把仓位/冷却/连亏
这些承重状态丢掉，回测结果没有意义。

本模块的口径与实盘 `daily` 完全一致：
- 因子用同一套 `snapshot_from_frames`；
- 信号用同一个 `evaluate()`；
- 成交在**次日开盘**，真实佣金/印花税/过户费/滑点，T+1，涨跌停约束；
- 组合层执行最多持仓数、单票权重、禁止补仓、止损冷却、S1 减仓冷却。
"""

from __future__ import annotations

import json
import random
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from tradecompass.backtest.baselines import equal_weight_buy_hold
from tradecompass.backtest.costs import trade_fee
from tradecompass.backtest.engine import BacktestData
from tradecompass.backtest.fills import Bar, simulate
from tradecompass.backtest.metrics import Metrics, compute_metrics
from tradecompass.backtest.types import BacktestResult, EquityPoint, Fill, Order, Trade
from tradecompass.config.loader import AppConfig
from tradecompass.config.models import CostsConfig, InstrumentConfig
from tradecompass.data import adjust, store
from tradecompass.data.universe import limit_pct_of
from tradecompass.domain.models import Instrument
from tradecompass.factors.engine import (
    FactorSnapshot,
    MarketContext,
    SectorContext,
    build_sector_context,
    snapshot_from_frames,
)
from tradecompass.indicators.frame import (
    IndicatorParams,
    daily_frame,
    weekly_frame,
    weekly_until,
)
from tradecompass.market.regime import build_series
from tradecompass.market.sectors import INDUSTRIES, SectorRank
from tradecompass.signals.engine import evaluate, stop_levels
from tradecompass.signals.types import (
    PortfolioContext,
    PositionContext,
    Signal,
    SignalKind,
    SignalParams,
    SymbolHistory,
)

LOT_SIZE = 100
REDUCE_COOLDOWN_DAYS = 5
EntryMode = Literal["signals", "random"]


# --------------------------------------------------------------------------- 数据


@dataclass(frozen=True, slots=True)
class SignalBacktestData:
    instrument: Instrument
    raw: pd.DataFrame
    qfq: pd.DataFrame
    factor: pd.Series
    factor_by_day: pd.Series
    daily: pd.DataFrame
    weekly: pd.DataFrame

    def raw_on(self, day: date) -> pd.Series | None:
        ts = pd.Timestamp(day)
        if ts not in self.raw.index:
            return None
        return cast(pd.Series, self.raw.loc[ts])

    def factor_on(self, day: date) -> float:
        ts = pd.Timestamp(day)
        value = self.factor_by_day.get(ts)
        return float(value) if value is not None and pd.notna(value) else 1.0

    def atr_adj(self, day: date, ratio: float) -> float | None:
        """当日 ATR 换算到“入场成本币值”，供移动止损使用。"""
        ts = pd.Timestamp(day)
        if ts not in self.daily.index or ts not in self.raw.index or ts not in self.qfq.index:
            return None
        daily_row = cast(pd.Series, self.daily.loc[ts])
        raw_row = cast(pd.Series, self.raw.loc[ts])
        qfq_row = cast(pd.Series, self.qfq.loc[ts])
        atr = float(daily_row["atr14"])
        raw_close = float(raw_row["close"])
        qfq_close = float(qfq_row["close"])
        if pd.isna(atr) or qfq_close <= 0:
            return None
        return atr * (raw_close / qfq_close) * ratio


def load_signal_data(
    cfg: AppConfig, instrument: Instrument, *, end: date
) -> SignalBacktestData | None:
    raw = store.read_bars_or_empty(
        store.raw_bars_path(cfg.paths, instrument.market, instrument.symbol)
    )
    if raw.empty:
        return None
    raw = raw.loc[: pd.Timestamp(end)]
    if raw.empty:
        return None
    factor = store.read_factor_or_empty(
        store.factor_path(cfg.paths, instrument.market, instrument.symbol)
    )
    if not factor.empty:
        factor = factor.loc[: pd.Timestamp(end)]
    qfq = adjust.to_qfq(raw, factor) if not factor.empty else raw
    params = IndicatorParams()
    daily = daily_frame(qfq, params)
    weekly = weekly_frame(qfq, pd.Timestamp(end).date(), params)
    factor_by_day = (
        factor.reindex(raw.index, method="ffill")
        if not factor.empty
        else pd.Series(1.0, index=raw.index)
    )
    return SignalBacktestData(
        instrument=instrument,
        raw=raw,
        qfq=qfq,
        factor=factor,
        factor_by_day=factor_by_day,
        daily=daily,
        weekly=weekly,
    )


class _MarketProvider:
    """预计算大盘日线/周线/regime，逐日给出 ``MarketContext``。"""

    def __init__(self, bars: pd.DataFrame, end: date) -> None:
        self.bars = bars.loc[: pd.Timestamp(end)]
        params = IndicatorParams()
        self.weekly = weekly_frame(self.bars, end, params)
        self.points = build_series(self.bars)
        self.point_days = [point.day for point in self.points]
        close = self.bars["close"].astype(float)
        volume = self.bars["volume"].astype(float)
        self.ret20 = close.pct_change(20)
        self.vol_ratio20 = volume / volume.rolling(20).mean()

    def on(self, day: date) -> MarketContext:
        index = bisect_right(self.point_days, day) - 1
        point = self.points[index] if index >= 0 else None
        week = weekly_until(self.weekly, day)
        above: bool | None = None
        ma20w_up: bool | None = None
        if len(week) >= 21:
            last = week.iloc[-1]
            ma20w = last["ma20w"]
            if not pd.isna(ma20w):
                above = float(last["close"]) > float(ma20w)
                ma20w_up = bool(last["ma20w_up"]) if not pd.isna(last["ma20w_up"]) else None

        ts = pd.Timestamp(day)
        ret20 = _series_at(self.ret20, ts)
        vol_ratio = _series_at(self.vol_ratio20, ts)
        return MarketContext(
            as_of=day,
            state=str(point.state) if point is not None else "NO_DATA",
            quota=float(point.quota) if point is not None else 0.0,
            above_weekly=above,
            ma20w_up=ma20w_up,
            ret20=ret20,
            vol_ratio20=vol_ratio,
            note=point.note if point is not None else "",
        )


class _SectorProvider:
    """预计算行业指数的动量/MA20/量比，逐日给出排名与单标的上下文。"""

    def __init__(self, cfg: AppConfig, end: date) -> None:
        self.frames: dict[str, tuple[str, pd.DataFrame]] = {}
        for code, name in INDUSTRIES:
            bars = store.read_bars_or_empty(
                store.raw_bars_path(cfg.paths, "SH", code)
            )
            if bars.empty:
                continue
            bars = bars.loc[: pd.Timestamp(end)]
            close = bars["close"].astype(float)
            volume = bars["volume"].astype(float)
            frame = pd.DataFrame(
                {
                    "close": close,
                    "momentum": close / close.shift(20) - 1.0,
                    "ma20": close.rolling(20).mean(),
                    "vol_ratio20": volume / volume.rolling(20).mean(),
                },
                index=bars.index,
            )
            self.frames[code] = (name, frame)
        self.cache: dict[date, list[SectorRank]] = {}

    def ranks(self, day: date) -> list[SectorRank]:
        if day in self.cache:
            return self.cache[day]
        ts = pd.Timestamp(day)
        rows: list[tuple[str, str, float, bool, float | None]] = []
        for code, (name, frame) in self.frames.items():
            if ts not in frame.index:
                continue
            row = frame.loc[ts]
            momentum = float(row["momentum"])
            ma20 = float(row["ma20"])
            if not np.isfinite(momentum) or not np.isfinite(ma20):
                continue
            raw_vol_ratio = float(row["vol_ratio20"])
            vol_ratio = raw_vol_ratio if np.isfinite(raw_vol_ratio) else None
            rows.append(
                (
                    code,
                    name,
                    momentum,
                    bool(float(row["close"]) > ma20),
                    vol_ratio,
                )
            )
        rows.sort(key=lambda item: item[2], reverse=True)
        result = [
            SectorRank(
                code=code,
                name=name,
                momentum=momentum,
                above_ma20=above,
                rank=index + 1,
                vol_ratio20=vol_ratio,
            )
            for index, (code, name, momentum, above, vol_ratio) in enumerate(rows)
        ]
        self.cache[day] = result
        return result

    def context(
        self, day: date, symbol: str, sector_sw: str | None
    ) -> SectorContext | None:
        return build_sector_context(symbol, sector_sw, self.ranks(day))


# --------------------------------------------------------------------------- 组合状态


@dataclass(slots=True)
class _SimPosition:
    symbol: str
    instrument: Instrument
    shares: int
    entry_price_raw: float
    entry_date: date
    entry_factor: float
    entry_fees: float
    stop_raw: float | None
    initial_stop_raw: float | None
    highest_close_raw: float
    entry_kind: str
    signal_date: date
    entry_features: dict[str, object] = field(default_factory=dict)
    highest_adj: float = 0.0
    lowest_adj: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    mfe_date: date | None = None
    last_reduce_date: date | None = None


@dataclass(frozen=True, slots=True)
class _Pending:
    symbol: str
    side: Literal["buy", "sell"]
    kind: str
    action: str
    reason: str
    stop_raw: float | None
    fraction: float = 1.0
    confidence: int = 0
    signal_date: date | None = None
    features: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SignalTrade:
    symbol: str
    entry_kind: str
    entry_date: date
    exit_date: date
    net_pnl: float
    holding_days: int
    exit_reason: str
    signal_date: date
    entry_price_raw: float
    exit_price_raw: float
    initial_risk_pct: float
    mfe_pct: float
    mae_pct: float
    days_to_mfe: int
    entry_features: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class _SimBook:
    cash: float
    positions: dict[str, _SimPosition] = field(default_factory=dict)
    histories: dict[str, SymbolHistory] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    signal_trades: list[SignalTrade] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    fees_total: float = 0.0
    equity: list[EquityPoint] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SignalBacktestOutcome:
    start: date
    end: date
    initial_cash: float
    signal_result: BacktestResult
    signal_metrics: Metrics
    benchmark_points: tuple[EquityPoint, ...]
    benchmark_metrics: Metrics
    random_results: tuple[tuple[int, BacktestResult, Metrics], ...]
    trade_breakdown: dict[str, tuple[SignalTrade, ...]]
    random_rate: float
    active_signals: int

    @property
    def random_median_return(self) -> float:
        if not self.random_results:
            return 0.0
        values = sorted(metrics.total_return for _, _, metrics in self.random_results)
        return values[len(values) // 2]


# --------------------------------------------------------------------------- 主循环


def run_signal_backtest(
    cfg: AppConfig,
    *,
    start: date,
    end: date,
    initial_cash: float = 100_000.0,
    params: SignalParams | None = None,
    random_seeds: int = 3,
) -> SignalBacktestOutcome:
    resolved = params or SignalParams.from_config(cfg)
    instruments = [
        _instrument_of(item)
        for item in cfg.universe.instruments
        if item.symbol != cfg.market.indices.hs300
    ]
    data_map: dict[str, SignalBacktestData] = {}
    for instrument in instruments:
        data = load_signal_data(cfg, instrument, end=end)
        if data is not None and not data.raw.loc[pd.Timestamp(start) :].empty:
            data_map[instrument.symbol] = data
    if not data_map:
        raise ValueError("没有可用于回测的行情，先运行 data sync")

    index_bars = store.read_bars_or_empty(
        store.raw_bars_path(cfg.paths, "SH", cfg.market.indices.hs300)
    )
    if index_bars.empty:
        raise ValueError("没有沪深300行情")
    market_provider = _MarketProvider(index_bars, end)
    sector_provider = _SectorProvider(cfg, end)
    days = _trading_days(data_map, start, end)

    signal_book = _replay(
        cfg,
        data_map,
        days,
        start=start,
        end=end,
        initial_cash=initial_cash,
        params=resolved,
        market_provider=market_provider,
        sector_provider=sector_provider,
        entry_mode="signals",
        rng=None,
        random_rate=0.0,
    )
    signal_result = _result(signal_book, start, end, initial_cash, "signal_system")
    signal_metrics = compute_metrics(signal_result)

    benchmark_data = {
        symbol: BacktestData(
            instrument=data.instrument,
            bars_raw=data.raw.loc[pd.Timestamp(start) : pd.Timestamp(end)],
            bars_qfq=data.qfq.loc[pd.Timestamp(start) : pd.Timestamp(end)],
            factor=data.factor,
            indicators=data.daily,
            weekly=data.weekly,
        )
        for symbol, data in data_map.items()
    }
    benchmark_points = equal_weight_buy_hold(benchmark_data, initial_cash=initial_cash)
    benchmark_metrics = _equity_metrics(
        benchmark_points, start, end, initial_cash, "equal_weight"
    )

    n_buys = sum(
        1 for fill in signal_book.fills if fill.side == "buy"
    )
    eligible = _eligible_symbol_days(data_map, days, market_provider, sector_provider, resolved)
    random_rate = min(max(n_buys / eligible, 0.001), 0.5) if eligible > 0 else 0.0

    random_results: list[tuple[int, BacktestResult, Metrics]] = []
    for seed in range(random_seeds):
        rng = random.Random(42 + seed)
        random_book = _replay(
            cfg,
            data_map,
            days,
            start=start,
            end=end,
            initial_cash=initial_cash,
            params=resolved,
            market_provider=market_provider,
            sector_provider=sector_provider,
            entry_mode="random",
            rng=rng,
            random_rate=random_rate,
        )
        random_result = _result(
            random_book, start, end, initial_cash, f"random_entry_{seed + 1}"
        )
        random_results.append(
            (seed + 1, random_result, compute_metrics(random_result))
        )

    breakdown: dict[str, tuple[SignalTrade, ...]] = {}
    for kind in ("B1", "B2", "RANDOM"):
        selected = tuple(
            trade for trade in signal_book.signal_trades if trade.entry_kind == kind
        )
        if selected:
            breakdown[kind] = selected

    return SignalBacktestOutcome(
        start=start,
        end=end,
        initial_cash=initial_cash,
        signal_result=signal_result,
        signal_metrics=signal_metrics,
        benchmark_points=benchmark_points,
        benchmark_metrics=benchmark_metrics,
        random_results=tuple(random_results),
        trade_breakdown=breakdown,
        random_rate=random_rate,
        active_signals=len(signal_book.positions),
    )


def _replay(
    cfg: AppConfig,
    data_map: dict[str, SignalBacktestData],
    days: list[date],
    *,
    start: date,
    end: date,
    initial_cash: float,
    params: SignalParams,
    market_provider: _MarketProvider,
    sector_provider: _SectorProvider,
    entry_mode: EntryMode,
    rng: random.Random | None,
    random_rate: float,
) -> _SimBook:
    book = _SimBook(cash=initial_cash)
    pending: list[_Pending] = []
    equity_prev = initial_cash
    costs = cfg.backtest.costs
    max_positions = params.max_positions
    max_gap = cfg.backtest.max_gap_open

    for day in days:
        # 1. 开盘执行昨日订单
        still_pending: list[_Pending] = []
        for order in pending:
            data = data_map[order.symbol]
            row = data.raw_on(day)
            if row is None:
                if order.side == "sell":
                    still_pending.append(order)
                continue
            bar = _bar(data, day)
            sim_order = Order(order.symbol, order.side, 0, "open", None, order.reason)
            outcome = simulate(
                sim_order,
                bar,
                limit_pct=limit_pct_of(data.instrument),
                slippage=costs.slippage,
                max_gap_open=max_gap,
            )
            if outcome.deferred:
                still_pending.append(order)
                continue
            if not outcome.filled or outcome.price_raw is None:
                continue
            if order.side == "buy":
                if order.symbol in book.positions:
                    continue
                if len(book.positions) >= max_positions:
                    continue
                quantity = _target_quantity(
                    equity_prev,
                    book.cash,
                    outcome.price_raw,
                    order.stop_raw,
                    order.kind,
                    params,
                    data,
                    costs,
                )
                if quantity <= 0:
                    continue
                fee = trade_fee(
                    costs,
                    kind=data.instrument.kind,
                    board=data.instrument.board,
                    side="buy",
                    price_raw=outcome.price_raw,
                    quantity=quantity,
                )
                if quantity * outcome.price_raw + fee > book.cash + 1e-9:
                    continue
                book.cash -= quantity * outcome.price_raw + fee
                book.fees_total += fee
                position = _SimPosition(
                    symbol=order.symbol,
                    instrument=data.instrument,
                    shares=quantity,
                    entry_price_raw=outcome.price_raw,
                    entry_date=day,
                    entry_factor=data.factor_on(day),
                    entry_fees=fee,
                    stop_raw=order.stop_raw,
                    initial_stop_raw=order.stop_raw,
                    highest_close_raw=outcome.price_raw,
                    entry_kind=order.kind,
                    signal_date=order.signal_date or day,
                    entry_features=dict(order.features),
                    highest_adj=outcome.price_raw,
                    lowest_adj=outcome.price_raw,
                    mfe_date=day,
                )
                book.positions[order.symbol] = position
                history = book.histories.setdefault(
                    order.symbol, SymbolHistory(symbol=order.symbol)
                )
                history.last_buy_date = day
                book.fills.append(
                    Fill(
                        date=day,
                        symbol=order.symbol,
                        side="buy",
                        quantity=quantity,
                        price_raw=outcome.price_raw,
                        fee=fee,
                        reason=order.reason,
                    )
                )
            else:
                held = book.positions.get(order.symbol)
                if held is None or held.entry_date >= day:
                    continue
                _sell(
                    book,
                    data,
                    held,
                    day=day,
                    price_raw=outcome.price_raw,
                    fraction=order.fraction,
                    reason=order.reason,
                    costs=costs,
                )
        pending = still_pending

        # 2. 盘中止损（T+1：只卖昨日之前买入的）
        for symbol, position in list(book.positions.items()):
            data = data_map[symbol]
            row = data.raw_on(day)
            if row is None or position.entry_date >= day or position.stop_raw is None:
                continue
            if float(row["low"]) > position.stop_raw:
                continue
            trailing = (
                position.initial_stop_raw is not None
                and position.stop_raw > position.initial_stop_raw
            )
            reason = "移动止损" if trailing else "止损"
            bar = _bar(data, day)
            outcome = simulate(
                Order(symbol, "sell", position.shares, "cond_le", position.stop_raw, reason),
                bar,
                limit_pct=limit_pct_of(data.instrument),
                slippage=costs.slippage,
                max_gap_open=max_gap,
            )
            if outcome.filled and outcome.price_raw is not None:
                _sell(
                    book,
                    data,
                    position,
                    day=day,
                    price_raw=outcome.price_raw,
                    fraction=1.0,
                    reason=reason,
                    costs=costs,
                )

        # 2b. 收盘后更新 MFE/MAE 与分级移动止损（次日生效，避免日内未来函数）
        for symbol, position in book.positions.items():
            _update_excursion(position, data_map[symbol], day, params)

        # 3. 收盘：更新净值、生成次日订单
        portfolio = _portfolio_context(book, data_map, day)
        equity_prev = portfolio.equity
        book.equity.append(
            EquityPoint(
                date=day,
                equity=portfolio.equity,
                cash=book.cash,
                position_value=portfolio.equity - book.cash,
            )
        )
        pending.extend(
            _orders_for_day(
                data_map,
                day,
                book,
                portfolio,
                market_provider,
                sector_provider,
                params,
                entry_mode=entry_mode,
                rng=rng,
                random_rate=random_rate,
            )
        )

    return book


# --------------------------------------------------------------------------- 订单


def _orders_for_day(
    data_map: dict[str, SignalBacktestData],
    day: date,
    book: _SimBook,
    portfolio: PortfolioContext,
    market_provider: _MarketProvider,
    sector_provider: _SectorProvider,
    params: SignalParams,
    *,
    entry_mode: EntryMode,
    rng: random.Random | None,
    random_rate: float,
) -> list[_Pending]:
    market = market_provider.on(day)
    orders: list[_Pending] = []

    if entry_mode == "random":
        if rng is None or not market.allows_new_buy:
            return orders
        if len(book.positions) >= params.max_positions:
            return orders
        for symbol in sorted(data_map):
            if symbol in book.positions:
                continue
            data = data_map[symbol]
            if data.instrument.kind == "base_etf":
                continue
            if data.raw_on(day) is None:
                continue
            sector = sector_provider.context(day, symbol, data.instrument.sector_sw)
            if sector is not None and sector.above_ma20 is False:
                continue
            snapshot = _snapshot(data, day, market, sector)
            if snapshot is None:
                continue
            stop, _ = stop_levels(snapshot, params)
            if stop is None or stop >= snapshot.close_raw:
                continue
            if (snapshot.close_raw - stop) / snapshot.close_raw > params.max_stop_pct:
                continue
            if rng.random() >= random_rate:
                continue
            orders.append(
                _Pending(
                    symbol=symbol,
                    side="buy",
                    kind="RANDOM",
                    action="随机对照买入",
                    reason="随机入场对照",
                    stop_raw=stop,
                    fraction=1.0,
                    confidence=0,
                    signal_date=day,
                    features=_entry_features(snapshot, sector),
                )
            )
            if len(book.positions) + len(orders) >= params.max_positions:
                break
        return orders

    signals: list[tuple[Signal, FactorSnapshot, SectorContext | None]] = []
    for symbol, data in data_map.items():
        if data.raw_on(day) is None:
            continue
        sector = sector_provider.context(day, symbol, data.instrument.sector_sw)
        snapshot = _snapshot(data, day, market, sector)
        if snapshot is None:
            continue
        signal = evaluate(snapshot, portfolio, params)
        if signal is None or not signal.actionable:
            continue
        if signal.kind in (SignalKind.B1, SignalKind.B2) and symbol in book.positions:
            continue
        position = book.positions.get(symbol)
        if signal.kind is SignalKind.S1 and _recent_reduce(position, day):
            continue
        signals.append((signal, snapshot, sector))

    # 卖单优先执行；买单按“动态置信度”排序，不再让 B2 永远挤掉 B1。
    signals.sort(
        key=lambda item: (
            0 if item[0].kind in (SignalKind.S1, SignalKind.S2) else 1,
            -item[0].confidence,
            -item[0].priority,
            item[0].symbol,
        )
    )
    buy_count = sum(1 for order in orders if order.side == "buy")
    for signal, snapshot, sector in signals:
        if signal.kind in (SignalKind.B1, SignalKind.B2):
            if len(book.positions) + buy_count >= params.max_positions:
                continue
            if book.cash <= 0 or signal.stop_price_raw is None:
                continue
            orders.append(
                _Pending(
                    symbol=signal.symbol,
                    side="buy",
                    kind=str(signal.kind),
                    action=signal.action,
                    reason="；".join(signal.reasons[:3]),
                    stop_raw=signal.stop_price_raw,
                    fraction=1.0 if signal.kind is SignalKind.B2 else 0.5,
                    confidence=signal.confidence,
                    signal_date=day,
                    features=_entry_features(snapshot, sector),
                )
            )
            buy_count += 1
        else:
            if signal.symbol not in book.positions:
                continue
            fraction = 0.5 if "减仓" in signal.action else 1.0
            orders.append(
                _Pending(
                    symbol=signal.symbol,
                    side="sell",
                    kind=str(signal.kind),
                    action=signal.action,
                    reason=f"{signal.kind} {signal.action}",
                    stop_raw=None,
                    fraction=fraction,
                    confidence=signal.confidence,
                )
            )
    return orders


def _entry_features(
    snapshot: FactorSnapshot, sector: SectorContext | None
) -> dict[str, object]:
    return {
        "market_state": snapshot.market.state,
        "market_ret20": snapshot.market.ret20,
        "market_vol_ratio20": snapshot.market.vol_ratio20,
        "sector_rank": float(sector.rank) if sector and sector.rank is not None else None,
        "sector_strength": sector.strength if sector is not None else "unknown",
        "sector_code": sector.code if sector is not None else None,
        "sector_name": sector.name if sector is not None else None,
        "vol_ratio20": snapshot.num("vol_ratio20"),
        "pos_250": snapshot.num("pos_250"),
        "dist_ma20_pct": snapshot.num("dist_ma20_pct"),
        "dist_ma60_pct": snapshot.num("dist_ma60_pct"),
        "rsi6": snapshot.num("rsi6"),
        "macd_hist": snapshot.num("macd_hist"),
        "jw": snapshot.num("jw"),
        "jw_pct": snapshot.num("jw_pct"),
        "ret20": snapshot.num("ret20"),
        "ret60": snapshot.num("ret60"),
        "weekly_bull": snapshot.flag("weekly_bull"),
    }


def _recent_reduce(position: _SimPosition | None, day: date) -> bool:
    if position is None or position.last_reduce_date is None:
        return False
    return (day - position.last_reduce_date).days <= REDUCE_COOLDOWN_DAYS


# --------------------------------------------------------------------------- 成交/记账


def _update_excursion(
    position: _SimPosition,
    data: SignalBacktestData,
    day: date,
    params: SignalParams,
) -> None:
    row = data.raw_on(day)
    if row is None:
        return
    ratio = data.factor_on(day) / position.entry_factor if position.entry_factor else 1.0
    high_adj = float(row["high"]) * ratio
    low_adj = float(row["low"]) * ratio
    if high_adj > position.highest_adj:
        position.highest_adj = high_adj
        position.mfe_date = day
    if low_adj < position.lowest_adj:
        position.lowest_adj = low_adj
    if position.entry_price_raw > 0:
        position.mfe_pct = position.highest_adj / position.entry_price_raw - 1.0
        position.mae_pct = position.lowest_adj / position.entry_price_raw - 1.0

    # 分级移动止损：+0.5R 保本 → +1R 锁 +0.5R → +2R 用 ATR 跟踪。
    initial_stop = position.initial_stop_raw
    if initial_stop is None or initial_stop >= position.entry_price_raw:
        return
    risk = position.entry_price_raw - initial_stop
    if risk <= 0:
        return
    mfe_r = (position.highest_adj - position.entry_price_raw) / risk
    target_adj: float | None = None
    if mfe_r >= params.trail_at_r:
        atr_adj = data.atr_adj(day, ratio)
        if atr_adj is not None and atr_adj > 0:
            target_adj = position.highest_adj - params.trail_atr_mult * atr_adj
    if mfe_r >= params.lock_half_r_at_r:
        lock = position.entry_price_raw + 0.5 * risk
        target_adj = max(target_adj, lock) if target_adj is not None else lock
    if mfe_r >= params.breakeven_at_r:
        target_adj = (
            max(target_adj, position.entry_price_raw)
            if target_adj is not None
            else position.entry_price_raw
        )
    if target_adj is None:
        return
    target_raw = target_adj / ratio if ratio > 0 else target_adj
    if position.stop_raw is None or target_raw > position.stop_raw:
        position.stop_raw = target_raw


def _bar(data: SignalBacktestData, day: date) -> Bar:
    row = data.raw.loc[pd.Timestamp(day)]
    return Bar(
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        prev_close=_previous_close(data, day),
    )


def _previous_close(data: SignalBacktestData, day: date) -> float:
    ts = pd.Timestamp(day)
    window = data.raw.loc[:ts]
    if len(window) < 2:
        return float(window["close"].iloc[-1]) if len(window) else 0.0
    return float(window["close"].iloc[-2])


def _target_quantity(
    equity: float,
    cash: float,
    price: float,
    stop: float | None,
    kind: str,
    params: SignalParams,
    data: SignalBacktestData,
    costs: CostsConfig,
) -> int:
    if price <= 0 or stop is None or stop >= price or equity <= 0:
        return 0
    risk_per_share = price - stop
    risk_budget = equity * params.risk_per_trade
    qty_risk = int(risk_budget / risk_per_share // LOT_SIZE) * LOT_SIZE
    slot = equity * params.swing_ratio / max(params.max_positions, 1)
    qty_slot = int(slot / price // LOT_SIZE) * LOT_SIZE
    qty = min(qty_risk, qty_slot)
    if kind == "B1":
        qty = int(qty // 2 // LOT_SIZE) * LOT_SIZE
    affordable = _affordable(cash, price, data, costs)
    qty = min(qty, affordable)
    if qty < LOT_SIZE or qty * price < params.min_trade_amount:
        return 0
    return qty


def _affordable(
    cash: float, price: float, data: SignalBacktestData, costs: CostsConfig
) -> int:
    rate = costs.commission_for(data.instrument.kind)
    budget = cash / (1.0 + rate + costs.slippage) if cash > 0 else 0.0
    return max(int(budget // (price * LOT_SIZE)), 0) * LOT_SIZE


def _sell(
    book: _SimBook,
    data: SignalBacktestData,
    position: _SimPosition,
    *,
    day: date,
    price_raw: float,
    fraction: float,
    reason: str,
    costs: CostsConfig,
) -> None:
    shares_before = position.shares
    quantity = int(shares_before * fraction // LOT_SIZE) * LOT_SIZE
    if quantity <= 0 or quantity > shares_before:
        quantity = shares_before
    if quantity <= 0:
        return

    fee = trade_fee(
        costs,
        kind=data.instrument.kind,
        board=data.instrument.board,
        side="sell",
        price_raw=price_raw,
        quantity=quantity,
    )
    ratio = data.factor_on(day) / position.entry_factor if position.entry_factor else 1.0
    entry_fee_part = position.entry_fees * quantity / shares_before
    gross = quantity * (price_raw * ratio - position.entry_price_raw)
    net = gross - fee - entry_fee_part
    risk = (
        quantity * (position.entry_price_raw - position.stop_raw)
        if position.stop_raw is not None
        else 0.0
    )
    r_multiple = net / risk if risk > 0 else None

    book.cash += quantity * price_raw - fee
    book.fees_total += fee
    position.shares -= quantity
    position.entry_fees -= entry_fee_part
    if fraction < 1.0:
        position.last_reduce_date = day
    if position.shares <= 0:
        del book.positions[position.symbol]

    book.fills.append(
        Fill(
            date=day,
            symbol=position.symbol,
            side="sell",
            quantity=quantity,
            price_raw=price_raw,
            fee=fee,
            reason=reason,
        )
    )
    book.trades.append(
        Trade(
            symbol=position.symbol,
            entry_date=position.entry_date,
            entry_price_raw=position.entry_price_raw,
            exit_date=day,
            exit_price_raw=price_raw,
            quantity=quantity,
            gross_pnl=gross,
            fees=fee + entry_fee_part,
            net_pnl=net,
            r_multiple=r_multiple,
            holding_days=(day - position.entry_date).days,
            exit_reason=reason,
        )
    )
    initial_risk_pct = (
        (position.entry_price_raw - position.initial_stop_raw) / position.entry_price_raw
        if position.initial_stop_raw is not None and position.entry_price_raw > 0
        else 0.0
    )
    mfe_date = position.mfe_date or position.entry_date
    book.signal_trades.append(
        SignalTrade(
            symbol=position.symbol,
            entry_kind=position.entry_kind,
            entry_date=position.entry_date,
            exit_date=day,
            net_pnl=net,
            holding_days=(day - position.entry_date).days,
            exit_reason=reason,
            signal_date=position.signal_date,
            entry_price_raw=position.entry_price_raw,
            exit_price_raw=price_raw,
            initial_risk_pct=initial_risk_pct,
            mfe_pct=position.mfe_pct,
            mae_pct=position.mae_pct,
            days_to_mfe=(mfe_date - position.entry_date).days,
            entry_features=dict(position.entry_features),
        )
    )
    history = book.histories.setdefault(
        position.symbol, SymbolHistory(symbol=position.symbol)
    )
    history.last_sell_date = day
    history.last_sell_pnl = net
    history.realized_pnl += net
    history.consecutive_losses = (
        history.consecutive_losses + 1 if net < 0 else 0
    )


# --------------------------------------------------------------------------- 上下文


def _portfolio_context(
    book: _SimBook, data_map: dict[str, SignalBacktestData], day: date
) -> PortfolioContext:
    values: dict[str, float] = {}
    adjusted_prices: dict[str, float] = {}
    for symbol, position in book.positions.items():
        data = data_map[symbol]
        row = data.raw_on(day)
        ratio = data.factor_on(day) / position.entry_factor if position.entry_factor else 1.0
        if row is None:
            adjusted = position.entry_price_raw
        else:
            adjusted = float(row["close"]) * ratio
        adjusted_prices[symbol] = adjusted
        values[symbol] = position.shares * adjusted

    equity = book.cash + sum(values.values())
    positions: dict[str, PositionContext] = {}
    for symbol, position in book.positions.items():
        adjusted = adjusted_prices[symbol]
        pnl_pct = (
            adjusted / position.entry_price_raw - 1.0
            if position.entry_price_raw > 0
            else 0.0
        )
        positions[symbol] = PositionContext(
            symbol=symbol,
            name=position.instrument.name,
            quantity=position.shares,
            avg_cost_raw=position.entry_price_raw,
            close_raw=adjusted,
            weight=values[symbol] / equity if equity > 0 else 0.0,
            pnl_pct=pnl_pct,
            holding_days=(day - position.entry_date).days,
        )

    return PortfolioContext(
        equity=equity,
        cash=book.cash,
        holdings_count=len(positions),
        positions=positions,
        history=dict(book.histories),
        source="backtest",
    )


def _snapshot(
    data: SignalBacktestData,
    day: date,
    market: MarketContext,
    sector: SectorContext | None,
) -> FactorSnapshot | None:
    ts = pd.Timestamp(day)
    if ts not in data.daily.index:
        return None
    daily_row = data.daily.loc[[ts]]
    week = weekly_until(data.weekly, day)
    row = data.raw.loc[ts]
    return snapshot_from_frames(
        symbol=data.instrument.symbol,
        name=data.instrument.name,
        daily=daily_row,
        week=week,
        close_raw=float(row["close"]),
        market=market,
        sector=sector,
        kind=data.instrument.kind,
        as_of=day,
    )


# --------------------------------------------------------------------------- 绩效/输出


def _result(
    book: _SimBook, start: date, end: date, initial_cash: float, strategy_id: str
) -> BacktestResult:
    return BacktestResult(
        strategy_id=strategy_id,
        start=start,
        end=end,
        initial_cash=initial_cash,
        trades=tuple(book.trades),
        equity=tuple(book.equity),
        fills=tuple(book.fills),
        fees_total=book.fees_total,
    )


def _equity_metrics(
    points: tuple[EquityPoint, ...],
    start: date,
    end: date,
    initial_cash: float,
    strategy_id: str,
) -> Metrics:
    result = BacktestResult(
        strategy_id=strategy_id,
        start=start,
        end=end,
        initial_cash=initial_cash,
        trades=(),
        equity=points,
        fills=(),
        fees_total=0.0,
    )
    return compute_metrics(result)


def _trading_days(
    data_map: dict[str, SignalBacktestData], start: date, end: date
) -> list[date]:
    days = {
        pd.Timestamp(ts).date()
        for data in data_map.values()
        for ts in data.raw.index
        if start <= pd.Timestamp(ts).date() <= end
    }
    return sorted(days)


def _eligible_symbol_days(
    data_map: dict[str, SignalBacktestData],
    days: list[date],
    market_provider: _MarketProvider,
    sector_provider: _SectorProvider,
    params: SignalParams,
) -> int:
    count = 0
    for day in days:
        market = market_provider.on(day)
        if not market.allows_new_buy:
            continue
        for symbol, data in data_map.items():
            if data.raw_on(day) is None:
                continue
            sector = sector_provider.context(day, symbol, data.instrument.sector_sw)
            if sector is not None and sector.above_ma20 is False:
                continue
            count += 1
    return count


def _instrument_of(item: InstrumentConfig) -> Instrument:
    from tradecompass.data.universe import to_instrument

    return to_instrument(item)


def _series_at(series: pd.Series, ts: pd.Timestamp) -> float | None:
    if ts not in series.index:
        return None
    value = series.loc[ts]
    return None if pd.isna(value) else float(value)


# --------------------------------------------------------------------------- 报告


def render_backtest_markdown(outcome: SignalBacktestOutcome) -> str:
    signal = outcome.signal_metrics
    benchmark = outcome.benchmark_metrics
    lines = [
        f"# 信号系统回测（动态阈值）  {outcome.start} → {outcome.end}",
        "",
        f"初始资金：{outcome.initial_cash:,.0f} 元；标的数：见配置；口径：次日开盘、"
        "真实费率、T+1、涨跌停限制、滑点 0.1%。",
        "",
        "## 总览",
        "| 方案 | 笔数 | 胜率 | 利润因子 | 总收益 | 年化 | 最大回撤 | 资金占用 | 平均持有 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines.append(_metrics_row("B1/B2/S1/S2", signal))
    lines.append(_metrics_row("等权买入持有", benchmark))
    for seed, _, metrics in outcome.random_results:
        lines.append(_metrics_row(f"随机入场 seed {seed}", metrics))
    lines.extend(
        [
            "",
            f"随机入场率：{outcome.random_rate:.4f}（按信号买入次数 / 可入场标的日校准）",
            f"回测结束时仍有 {outcome.active_signals} 只持仓（未强制平仓）。",
            "",
            "## 按入场信号",
            "| 信号 | 笔数 | 胜率 | 平均净盈亏 | 平均持有天数 |",
            "|---|---|---|---|---|",
        ]
    )
    for kind, trades in sorted(outcome.trade_breakdown.items()):
        wins = [item.net_pnl for item in trades if item.net_pnl > 0]
        lines.append(
            f"| {kind} | {len(trades)} | {len(wins) / len(trades):.1%} | "
            f"{np.mean([item.net_pnl for item in trades]):+.2f} | "
            f"{np.mean([item.holding_days for item in trades]):.1f} |"
        )

    closed_trades = list(outcome.signal_result.trades)
    if closed_trades:
        wins = [item.net_pnl for item in closed_trades if item.net_pnl > 0]
        losses = [item.net_pnl for item in closed_trades if item.net_pnl <= 0]
        lines.extend(
            [
                "",
                "## 盈亏结构",
                f"- 平均盈利：{np.mean(wins):+.2f} 元；平均亏损：{np.mean(losses):+.2f} 元"
                if wins and losses
                else "- 盈利或亏损样本缺失",
                f"- 最大单笔盈利：{max(wins):+.2f} 元；最大单笔亏损：{min(losses):+.2f} 元"
                if wins and losses
                else "",
                "",
                "### 按退出原因",
                "| 退出原因 | 笔数 | 合计净盈亏 | 平均净盈亏 |",
                "|---|---|---|---|",
            ]
        )
        groups: dict[str, list[Trade]] = {}
        for closed in closed_trades:
            groups.setdefault(closed.exit_reason, []).append(closed)
        for reason, items in sorted(
            groups.items(), key=lambda item: -sum(t.net_pnl for t in item[1])
        ):
            total = sum(t.net_pnl for t in items)
            lines.append(
                f"| {reason} | {len(items)} | {total:+.0f} | {total / len(items):+.2f} |"
            )
        lines.extend(
            [
                "",
                "### 最赚的 5 笔",
                "| 代码 | 入场 | 出场 | 净盈亏 | 持有 | 退出 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for trade in sorted(closed_trades, key=lambda item: -item.net_pnl)[:5]:
            lines.append(
                f"| {trade.symbol} | {trade.entry_date} | {trade.exit_date} | "
                f"{trade.net_pnl:+.0f} | {trade.holding_days} 天 | {trade.exit_reason} |"
            )
        lines.extend(
            [
                "",
                "### 最亏的 5 笔",
                "| 代码 | 入场 | 出场 | 净盈亏 | 持有 | 退出 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for trade in sorted(closed_trades, key=lambda item: item.net_pnl)[:5]:
            lines.append(
                f"| {trade.symbol} | {trade.entry_date} | {trade.exit_date} | "
                f"{trade.net_pnl:+.0f} | {trade.holding_days} 天 | {trade.exit_reason} |"
            )

    lines.extend(
        [
            "",
            "## 年度收益",
            "| 年份 | 信号系统 | 等权买入持有 |",
            "|---|---|---|",
        ]
    )
    signal_years = _yearly_returns(outcome.signal_result.equity)
    benchmark_years = _yearly_returns(outcome.benchmark_points)
    for year in sorted(signal_years):
        lines.append(
            f"| {year} | {signal_years[year]:+.1%} | "
            f"{benchmark_years.get(year, 0.0):+.1%} |"
        )
    lines.extend(
        [
            "",
            "## 必须打的折扣",
            "1. 自选池是「现在挑出来的」，存在幸存者/后视偏差；结果不能直接外推。",
            "2. 回测从空仓开始，未使用你当前的持仓，因此不是「实盘复现」，"
            "只是同一套信号的规则检验。",
            "3. 随机入场对照只校准了入场次数，不能完全消除路径差异；看利润因子和回撤，"
            "不要只看总收益。",
            "4. 结束时未强制平仓，未实现盈亏只体现在净值里，不计入利润因子。",
        ]
    )
    return "\n".join(lines) + "\n"


def _yearly_returns(points: tuple[EquityPoint, ...]) -> dict[int, float]:
    if len(points) < 2:
        return {}
    by_year: dict[int, float] = {}
    for point in points:
        by_year[point.date.year] = point.equity
    result: dict[int, float] = {}
    previous = points[0].equity
    for year in sorted(by_year):
        last = by_year[year]
        result[year] = last / previous - 1.0 if previous > 0 else 0.0
        previous = last
    return result


def write_backtest_report(
    cfg: AppConfig, outcome: SignalBacktestOutcome, *, tag: str = "signal_system"
) -> Path:
    directory = cfg.paths.reports / "backtest"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{tag}-{outcome.end}.md"
    path.write_text(render_backtest_markdown(outcome), encoding="utf-8")
    json_path = directory / f"{tag}-{outcome.end}.json"
    json_path.write_text(
        json.dumps(
            {
                "start": str(outcome.start),
                "end": str(outcome.end),
                "signal": _metrics_dict(outcome.signal_metrics),
                "benchmark": _metrics_dict(outcome.benchmark_metrics),
                "random": [
                    {"seed": seed, **_metrics_dict(metrics)}
                    for seed, _, metrics in outcome.random_results
                ],
                "random_rate": outcome.random_rate,
                "active_signals": outcome.active_signals,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _metrics_row(label: str, metrics: Metrics) -> str:
    return (
        f"| {label} | {metrics.trades} | {metrics.win_rate:.1%} | "
        f"{metrics.profit_factor:.2f} | {metrics.total_return:+.1%} | "
        f"{metrics.annual_return:+.1%} | {metrics.max_drawdown:.1%} | "
        f"{metrics.capital_utilization:.1%} | {metrics.avg_holding_days:.1f} 天 |"
    )


def _metrics_dict(metrics: Metrics) -> dict[str, float | int | None]:
    return {
        "trades": metrics.trades,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "total_return": metrics.total_return,
        "annual_return": metrics.annual_return,
        "max_drawdown": metrics.max_drawdown,
        "capital_utilization": metrics.capital_utilization,
        "avg_holding_days": metrics.avg_holding_days,
        "fees": None,
    }
