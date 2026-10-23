"""每日编排：取数 → 因子 → 信号 → 报告。

这一层是唯一做 I/O 的地方；信号层与因子层都保持纯函数，
这样回测、盘中提醒、事后统计可以复用同一套判断。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from tradecompass.account.ledger import (
    Holding,
    history_from_db,
    positions_from_db,
)
from tradecompass.account.ledger import (
    SymbolHistory as LedgerHistory,
)
from tradecompass.config.loader import AppConfig
from tradecompass.config.models import OpeningPositionsConfig
from tradecompass.data import store
from tradecompass.data.universe import board_of, is_fund_code, market_of
from tradecompass.domain.models import Instrument, Kind
from tradecompass.factors.engine import (
    FactorSnapshot,
    MarketContext,
    build_market_context,
    build_sector_context,
    load_snapshot,
)
from tradecompass.market.sectors import SectorRank, ranking
from tradecompass.market.sectors import buyable as buyable_sectors
from tradecompass.signals.engine import evaluate
from tradecompass.signals.types import (
    PortfolioContext,
    PositionContext,
    Signal,
    SignalParams,
    SymbolHistory,
)
from tradecompass.storage.db import connect as connect_db


class DailyError(RuntimeError):
    """无法生成每日报告（通常是缺指数或行情）。"""


@dataclass(frozen=True, slots=True)
class DailyReport:
    as_of: date
    market: MarketContext
    sectors: tuple[SectorRank, ...]
    buyable: frozenset[str]
    portfolio: PortfolioContext
    actions: tuple[Signal, ...]
    watch: tuple[Signal, ...]
    errors: tuple[str, ...]

    @property
    def has_actions(self) -> bool:
        return bool(self.actions)


def build_daily(
    cfg: AppConfig,
    *,
    as_of: date | None = None,
    params: SignalParams | None = None,
) -> DailyReport:
    """生成某个交易日的完整报告。默认取沪深300 的最后一根日线。"""
    index_symbol = cfg.market.indices.hs300
    index_bars = store.read_bars_or_empty(
        store.raw_bars_path(cfg.paths, "SH", index_symbol)
    )
    if index_bars.empty:
        raise DailyError(f"没有沪深300（{index_symbol}）行情，先运行 data sync")

    target = as_of or pd.Timestamp(index_bars.index[-1]).date()
    market = build_market_context(index_bars, target)
    ranks = ranking(cfg, target)
    buyable = frozenset(buyable_sectors(ranks))

    holdings, ledger_history, source, cash = _load_portfolio(cfg)
    instruments = _scan_instruments(cfg, holdings)
    snapshots, errors = _load_snapshots(cfg, instruments, market, ranks, target)

    portfolio = _portfolio_context(
        holdings, ledger_history, snapshots, cash=cash, source=source, as_of=target
    )

    resolved = params or SignalParams.from_config(cfg)
    actions: list[Signal] = []
    watch: list[Signal] = []
    for symbol in sorted(snapshots):
        signal = evaluate(snapshots[symbol], portfolio, resolved)
        if signal is None:
            continue
        (actions if signal.actionable else watch).append(signal)

    actions.sort(key=lambda item: (-item.priority, -item.confidence, item.symbol))
    watch.sort(key=lambda item: (-item.priority, -item.confidence, item.symbol))
    errors.extend(
        f"{symbol} 有持仓但没有行情，已按成本计入权益"
        for symbol in holdings
        if symbol not in snapshots
    )

    return DailyReport(
        as_of=target,
        market=market,
        sectors=tuple(ranks[:5]),
        buyable=buyable,
        portfolio=portfolio,
        actions=tuple(actions),
        watch=tuple(watch),
        errors=tuple(errors),
    )


# --------------------------------------------------------------------------- 账户


def _load_portfolio(
    cfg: AppConfig,
) -> tuple[dict[str, Holding], dict[str, LedgerHistory], str, float]:
    """优先用交割单流水；没有流水时用期初持仓快照。"""
    holdings: dict[str, Holding] = {}
    histories: dict[str, LedgerHistory] = {}
    try:
        conn = connect_db(cfg.paths.db_path)
        try:
            holdings = positions_from_db(conn)
            histories = history_from_db(conn)
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        holdings, histories = {}, {}

    if holdings or histories:
        return holdings, histories, "trades", cfg.risk.cash

    opening = cfg.opening_positions
    if opening is not None:
        return _opening_holdings(opening), {}, "opening", opening.cash or cfg.risk.cash
    return {}, {}, "none", cfg.risk.cash


def _opening_holdings(opening: OpeningPositionsConfig) -> dict[str, Holding]:
    result: dict[str, Holding] = {}
    for item in opening.positions:
        holding = Holding(
            symbol=item.symbol,
            name=item.name,
            quantity=item.quantity,
            buy_amount=item.avg_cost * item.quantity,
            first_buy=None,
        )
        holding._cost_pool = holding.buy_amount
        result[item.symbol] = holding
    return result


def _scan_instruments(
    cfg: AppConfig, holdings: dict[str, Holding]
) -> list[Instrument]:
    """universe 全部标的 + 不在 universe 里的持仓。"""
    result: dict[str, Instrument] = {}
    for item in cfg.universe.instruments:
        result[item.symbol] = Instrument(
            symbol=item.symbol,
            market=item.market,
            name=item.name,
            kind=item.kind,
            board=board_of(
                item.symbol, market=item.market, is_index=item.kind == "index",
                is_st=item.is_st,
            ),
            sector_sw=item.sector_sw,
            t0=item.t0,
            is_st=item.is_st,
        )
    for symbol, holding in holdings.items():
        if symbol in result:
            continue
        market = market_of(symbol)
        kind: Kind = "sector_etf" if is_fund_code(symbol) else "stock"
        result[symbol] = Instrument(
            symbol=symbol,
            market=market,
            name=holding.name or symbol,
            kind=kind,
            board=board_of(symbol, market=market, is_index=False),
        )
    return list(result.values())


def _load_snapshots(
    cfg: AppConfig,
    instruments: list[Instrument],
    market: MarketContext,
    ranks: list[SectorRank],
    as_of: date,
) -> tuple[dict[str, FactorSnapshot], list[str]]:
    snapshots: dict[str, FactorSnapshot] = {}
    errors: list[str] = []
    for instrument in instruments:
        sector = build_sector_context(instrument.symbol, instrument.sector_sw, ranks)
        snapshot = load_snapshot(
            cfg, instrument, market=market, sector=sector, as_of=as_of
        )
        if snapshot is None:
            errors.append(f"{instrument.symbol} {instrument.name}：没有行情或历史不足")
            continue
        snapshots[instrument.symbol] = snapshot
    return snapshots, errors


def _portfolio_context(
    holdings: dict[str, Holding],
    histories: dict[str, LedgerHistory],
    snapshots: dict[str, FactorSnapshot],
    *,
    cash: float,
    source: str,
    as_of: date,
) -> PortfolioContext:
    values: dict[str, float] = {}
    for symbol, holding in holdings.items():
        snapshot = snapshots.get(symbol)
        price = snapshot.close_raw if snapshot is not None else holding.avg_cost_raw
        values[symbol] = holding.quantity * price
    equity = sum(values.values()) + cash

    positions: dict[str, PositionContext] = {}
    for symbol, holding in holdings.items():
        price = values[symbol] / holding.quantity if holding.quantity else 0.0
        cost = holding.avg_cost_raw
        pnl_pct = (price - cost) / cost if cost > 0 else 0.0
        weight = values[symbol] / equity if equity > 0 else 0.0
        holding_days = (
            (as_of - holding.first_buy).days if holding.first_buy is not None else None
        )
        positions[symbol] = PositionContext(
            symbol=symbol,
            name=holding.name,
            quantity=holding.quantity,
            avg_cost_raw=cost,
            close_raw=price,
            weight=weight,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
        )

    signal_history = {
        symbol: SymbolHistory(
            symbol=symbol,
            last_buy_date=item.last_buy_date,
            last_sell_date=item.last_sell_date,
            last_sell_pnl=item.last_sell_pnl,
            realized_pnl=item.realized_pnl,
            consecutive_losses=item.consecutive_losses,
        )
        for symbol, item in histories.items()
    }
    return PortfolioContext(
        equity=equity,
        cash=cash,
        holdings_count=len(positions),
        positions=positions,
        history=signal_history,
        source=source,
    )


# --------------------------------------------------------------------------- 报告


def render_markdown(report: DailyReport) -> str:
    """生成可直接推送/落盘的 Markdown。"""
    market = report.market
    lines: list[str] = [
        f"# TradeCompass 每日信号  {report.as_of}",
        "",
        "## 市场",
        f"- 沪深300：**{market.state}**，仓位额度 **{market.quota:.0%}**"
        + (f"（{market.note}）" if market.note else ""),
    ]
    if market.ret20 is not None:
        lines.append(f"- 20 日动量：{market.ret20:+.1%}")
    if market.vol_ratio20 is not None:
        lines.append(f"- 量比 20：{market.vol_ratio20:.2f}")
    lines.append(f"- 持仓来源：{_source_text(report.portfolio)}")
    lines.append(
        f"- 账户权益：{report.portfolio.equity:,.0f} 元，"
        f"持仓 {report.portfolio.holdings_count} 只"
    )

    if report.sectors:
        lines.extend(["", "## 行业（前 5）", "| 排名 | 行业 | 20日动量 | 量比20 | MA20 | 可买 |"])
        lines.append("|---|---|---|---|---|---|")
        for item in report.sectors:
            mark = "是" if item.code in report.buyable else "—"
            vol = f"{item.vol_ratio20:.2f}" if item.vol_ratio20 is not None else "-"
            lines.append(
                f"| {item.rank} | {item.name} | {item.momentum:+.1%} | {vol} | "
                f"{'上' if item.above_ma20 else '下'} | {mark} |"
            )

    lines.extend(["", "## 今日操作（高置信度）"])
    if report.actions:
        lines.extend([_signal_table_header()])
        lines.extend(_signal_row(item) for item in report.actions)
    else:
        lines.append("无。今天没有达到置信度阈值的操作信号。")

    if report.watch:
        lines.extend(["", "## 观察区（不行动）"])
        lines.extend([_signal_table_header()])
        lines.extend(_signal_row(item) for item in report.watch)

    if report.errors:
        lines.extend(["", "## 数据提示"])
        lines.extend(f"- {item}" for item in report.errors[:10])
        if len(report.errors) > 10:
            lines.append(f"- ……另有 {len(report.errors) - 10} 条")

    lines.extend(
        [
            "",
            "---",
            "> S2 是风控提醒：深度亏损的存量票可在反弹到 MA10/MA20 时分批减仓，"
            "不必在低点一次性割；新买入的票按信号立即执行。",
            "",
            "本报告是纪律提醒，不构成投资建议。信号只用于决定“今天该不该动手”，"
            "具体下单请在券商 APP 用条件单执行。",
        ]
    )
    return "\n".join(lines) + "\n"


def _signal_table_header() -> str:
    return (
        "| 代码 | 名称 | 信号 | 置信度/门槛 | 动作 | 参考价 | 止损 | 数量 | 理由 |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )


def _signal_row(signal: Signal) -> str:
    trigger = f"{signal.trigger_price_raw:.2f}" if signal.trigger_price_raw else "-"
    stop = f"{signal.stop_price_raw:.2f}" if signal.stop_price_raw else "-"
    quantity = str(signal.suggested_quantity) if signal.suggested_quantity else "-"
    reasons = "；".join(signal.reasons[:4])
    if signal.warnings:
        reasons += "（注意：" + "、".join(signal.warnings[:2]) + "）"
    confidence = (
        f"{signal.confidence}/{signal.threshold}"
        if signal.threshold
        else str(signal.confidence)
    )
    return (
        f"| {signal.symbol} | {signal.name} | {signal.kind} | {confidence} | "
        f"{signal.action} | {trigger} | {stop} | {quantity} | {reasons} |"
    )


def _source_text(portfolio: PortfolioContext) -> str:
    return {
        "trades": "交割单流水",
        "opening": "期初持仓快照",
        "none": "未配置持仓（按空仓处理）",
    }.get(portfolio.source, portfolio.source)


def write_report(cfg: AppConfig, report: DailyReport) -> tuple[Path, Path]:
    """把 Markdown 与 JSON 写入 ``TC_HOME/reports/daily``。"""
    directory = cfg.paths.reports / "daily"
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / f"{report.as_of}.md"
    json_path = directory / f"{report.as_of}.json"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(_report_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return markdown_path, json_path


def _report_dict(report: DailyReport) -> dict[str, object]:
    return {
        "as_of": str(report.as_of),
        "market": {
            "state": report.market.state,
            "quota": report.market.quota,
            "ret20": report.market.ret20,
            "vol_ratio20": report.market.vol_ratio20,
            "note": report.market.note,
        },
        "sectors": [
            {
                "rank": item.rank,
                "code": item.code,
                "name": item.name,
                "momentum": item.momentum,
                "vol_ratio20": item.vol_ratio20,
                "above_ma20": item.above_ma20,
                "buyable": item.code in report.buyable,
            }
            for item in report.sectors
        ],
        "portfolio": {
            "source": report.portfolio.source,
            "equity": report.portfolio.equity,
            "cash": report.portfolio.cash,
            "holdings_count": report.portfolio.holdings_count,
            "positions": [
                {
                    "symbol": item.symbol,
                    "name": item.name,
                    "quantity": item.quantity,
                    "avg_cost_raw": item.avg_cost_raw,
                    "close_raw": item.close_raw,
                    "weight": item.weight,
                    "pnl_pct": item.pnl_pct,
                    "holding_days": item.holding_days,
                }
                for item in report.portfolio.positions.values()
            ],
        },
        "actions": [_signal_dict(item) for item in report.actions],
        "watch": [_signal_dict(item) for item in report.watch],
        "errors": list(report.errors),
    }


def _signal_dict(signal: Signal) -> dict[str, object]:
    return {
        "kind": str(signal.kind),
        "symbol": signal.symbol,
        "name": signal.name,
        "as_of": str(signal.as_of),
        "confidence": signal.confidence,
        "threshold": signal.threshold,
        "actionable": signal.actionable,
        "action": signal.action,
        "reasons": list(signal.reasons),
        "warnings": list(signal.warnings),
        "trigger_price_raw": signal.trigger_price_raw,
        "stop_price_raw": signal.stop_price_raw,
        "suggested_quantity": signal.suggested_quantity,
        "position_note": signal.position_note,
        "metrics": dict(signal.metrics),
    }
