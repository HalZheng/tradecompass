"""B2 突破买点的失败归因。

只分析**实际成交并已平仓**的 B2 交易，回答四个问题：

1. 亏损主要来自入场、退出还是仓位？
2. 亏损单在失败前有没有出现过盈利（MFE/MAE）？
3. 哪类入场条件（大盘/行业/量比/追高度/位置/RSI/周J）最差？
4. 亏损是否集中在少数票或行业？
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from tradecompass.config.loader import AppConfig
from tradecompass.signals.backtest import SignalBacktestOutcome, SignalTrade


@dataclass(frozen=True, slots=True)
class B2Bucket:
    label: str
    count: int
    win_rate: float
    avg_pnl: float
    profit_factor: float
    stop_rate: float
    avg_mfe_pct: float
    avg_mae_pct: float
    avg_mfe_r: float
    avg_hold: float


@dataclass(frozen=True, slots=True)
class B2Analysis:
    total: B2Bucket
    exits: tuple[B2Bucket, ...]
    markets: tuple[B2Bucket, ...]
    sectors: tuple[B2Bucket, ...]
    volumes: tuple[B2Bucket, ...]
    extensions: tuple[B2Bucket, ...]
    positions: tuple[B2Bucket, ...]
    rsis: tuple[B2Bucket, ...]
    jws: tuple[B2Bucket, ...]
    weekly: tuple[B2Bucket, ...]
    losers: int
    never_profit: float
    half_r_fail: float
    one_r_fail: float
    avg_loss_hold: float
    avg_win_hold: float
    worst_symbols: tuple[tuple[str, float, int], ...]
    worst_sectors: tuple[tuple[str, float, int], ...]


def analyze_b2(outcome: SignalBacktestOutcome) -> B2Analysis:
    trades = list(outcome.trade_breakdown.get("B2", ()))
    if not trades:
        raise ValueError("回测期内没有已平仓的 B2 交易，无法归因")

    total = _bucket("全部 B2", trades)
    losers = [trade for trade in trades if trade.net_pnl <= 0]
    winners = [trade for trade in trades if trade.net_pnl > 0]

    mfe_rs: list[float] = []
    for trade in losers:
        value = _mfe_r(trade)
        if value is not None:
            mfe_rs.append(value)
    never_profit = (
        sum(1 for trade in losers if trade.mfe_pct <= 0.0) / len(losers)
        if losers
        else 0.0
    )
    half_r_fail = (
        sum(1 for value in mfe_rs if value >= 0.5) / len(mfe_rs) if mfe_rs else 0.0
    )
    one_r_fail = (
        sum(1 for value in mfe_rs if value >= 1.0) / len(mfe_rs) if mfe_rs else 0.0
    )

    return B2Analysis(
        total=total,
        exits=_group(trades, lambda trade: trade.exit_reason),
        markets=_group(
            trades,
            lambda trade: _feature_str(trade, "market_state") or "未知",
            order=("RISK_ON", "NEUTRAL", "未知"),
        ),
        sectors=_group(
            trades,
            lambda trade: _feature_str(trade, "sector_strength") or "unknown",
            order=("strong", "neutral", "weak", "unknown"),
        ),
        volumes=_group(trades, lambda trade: _vol_bucket(_feature_num(trade, "vol_ratio20"))),
        extensions=_group(
            trades, lambda trade: _extension_bucket(_feature_num(trade, "dist_ma20_pct"))
        ),
        positions=_group(
            trades, lambda trade: _position_bucket(_feature_num(trade, "pos_250"))
        ),
        rsis=_group(trades, lambda trade: _rsi_bucket(_feature_num(trade, "rsi6"))),
        jws=_group(
            trades, lambda trade: _jw_bucket(_feature_num(trade, "jw_pct"))
        ),
        weekly=_group(trades, _weekly_label),
        losers=len(losers),
        never_profit=never_profit,
        half_r_fail=half_r_fail,
        one_r_fail=one_r_fail,
        avg_loss_hold=(
            float(np.mean([trade.holding_days for trade in losers])) if losers else 0.0
        ),
        avg_win_hold=(
            float(np.mean([trade.holding_days for trade in winners])) if winners else 0.0
        ),
        worst_symbols=_worst_by(trades, "symbol"),
        worst_sectors=_worst_by(trades, "sector_name"),
    )


def _bucket(label: str, trades: Sequence[SignalTrade]) -> B2Bucket:
    count = len(trades)
    wins = [trade.net_pnl for trade in trades if trade.net_pnl > 0]
    losses = [trade.net_pnl for trade in trades if trade.net_pnl <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = float("inf") if gross_win > 0 else 0.0
    mfe_rs = [value for value in (_mfe_r(trade) for trade in trades) if value is not None]
    return B2Bucket(
        label=label,
        count=count,
        win_rate=len(wins) / count if count else 0.0,
        avg_pnl=float(np.mean([trade.net_pnl for trade in trades])) if count else 0.0,
        profit_factor=profit_factor,
        stop_rate=(
            sum(1 for trade in trades if "止损" in trade.exit_reason) / count
            if count
            else 0.0
        ),
        avg_mfe_pct=float(np.mean([trade.mfe_pct for trade in trades])) if count else 0.0,
        avg_mae_pct=float(np.mean([trade.mae_pct for trade in trades])) if count else 0.0,
        avg_mfe_r=float(np.mean(mfe_rs)) if mfe_rs else 0.0,
        avg_hold=float(np.mean([trade.holding_days for trade in trades])) if count else 0.0,
    )


def _group(
    trades: Sequence[SignalTrade],
    labeler: Callable[[SignalTrade], str],
    *,
    order: Sequence[str] | None = None,
) -> tuple[B2Bucket, ...]:
    groups: dict[str, list[SignalTrade]] = {}
    for trade in trades:
        groups.setdefault(labeler(trade), []).append(trade)
    if order is None:
        labels = sorted(groups)
    else:
        labels = list(order) + [label for label in sorted(groups) if label not in order]
    return tuple(_bucket(label, groups[label]) for label in labels if label in groups)


def _weekly_label(trade: SignalTrade) -> str:
    return "周线多头" if trade.entry_features.get("weekly_bull") is True else "周线非多头"


def _mfe_r(trade: SignalTrade) -> float | None:
    if trade.initial_risk_pct <= 0:
        return None
    return trade.mfe_pct / trade.initial_risk_pct


def _feature_num(trade: SignalTrade, key: str) -> float | None:
    value = trade.entry_features.get(key)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _feature_str(trade: SignalTrade, key: str) -> str | None:
    value = trade.entry_features.get(key)
    return str(value) if value is not None else None


def _vol_bucket(value: float | None) -> str:
    if value is None:
        return "未知"
    if value < 2.0:
        return "1.5–2.0"
    if value < 3.0:
        return "2.0–3.0"
    return ">3.0"


def _extension_bucket(value: float | None) -> str:
    if value is None:
        return "未知"
    if value < 0.0:
        return "低于MA20"
    if value < 0.03:
        return "0–3%"
    if value < 0.06:
        return "3–6%"
    return ">6%"


def _position_bucket(value: float | None) -> str:
    if value is None:
        return "未知"
    if value < 50.0:
        return "<50%"
    if value < 70.0:
        return "50–70%"
    if value < 85.0:
        return "70–85%"
    return ">85%"


def _rsi_bucket(value: float | None) -> str:
    if value is None:
        return "未知"
    if value < 60.0:
        return "<60"
    if value < 70.0:
        return "60–70"
    if value < 80.0:
        return "70–80"
    return ">80"


def _jw_bucket(value: float | None) -> str:
    if value is None:
        return "未知"
    if value < 50.0:
        return "<50%"
    if value < 70.0:
        return "50–70%"
    if value < 85.0:
        return "70–85%"
    return ">85%"


def _worst_by(
    trades: Sequence[SignalTrade], key: str
) -> tuple[tuple[str, float, int], ...]:
    groups: dict[str, list[float]] = {}
    for trade in trades:
        label = _feature_str(trade, key) if key != "symbol" else trade.symbol
        if label is None:
            continue
        groups.setdefault(label, []).append(trade.net_pnl)
    rows = [(label, sum(values), len(values)) for label, values in groups.items()]
    rows.sort(key=lambda item: item[1])
    return tuple(rows[:5])


# --------------------------------------------------------------------------- 报告


def render_b2_markdown(analysis: B2Analysis) -> str:
    total = analysis.total
    lines = [
        "# B2 突破买点失败归因",
        "",
        "口径：实际成交并已平仓的 B2 交易；次日开盘成交、真实费率、T+1、涨跌停、滑点 0.1%。",
        "",
        "## 总览",
        "| 笔数 | 胜率 | 利润因子 | 平均净盈亏 | 初始止损率 | 平均 MFE | "
        "平均 MAE | 平均 MFE(R) | 平均持有 |",
        "|---|---|---|---|---|---|---|---|---|",
        _bucket_row(total),
        "",
        "## 失败单有没有先给过机会",
        f"- 亏损单数量：{analysis.losers}",
        f"- 从未出现过浮盈（MFE≤0）的比例：{analysis.never_profit:.1%}",
        f"- 曾达到 +0.5R 后才失败的比例：{analysis.half_r_fail:.1%}",
        f"- 曾达到 +1R 后才失败的比例：{analysis.one_r_fail:.1%}",
        f"- 亏损单平均持有：{analysis.avg_loss_hold:.1f} 天；"
        f"盈利单平均持有：{analysis.avg_win_hold:.1f} 天",
        "",
    ]
    sections: tuple[tuple[str, tuple[B2Bucket, ...]], ...] = (
        ("按退出原因", analysis.exits),
        ("按大盘状态", analysis.markets),
        ("按行业强度", analysis.sectors),
        ("按量比 20", analysis.volumes),
        ("按距 MA20（追高度）", analysis.extensions),
        ("按 250 日位置", analysis.positions),
        ("按 RSI6", analysis.rsis),
        ("按周 J 历史分位", analysis.jws),
        ("按周线趋势", analysis.weekly),
    )
    for title, buckets in sections:
        lines.extend([f"## {title}", _bucket_header()])
        lines.extend(_bucket_row(bucket) for bucket in buckets)
        lines.append("")

    lines.extend(["## 最亏的票", "| 代码 | 合计净盈亏 | 笔数 |", "|---|---|---|"])
    lines.extend(
        f"| {symbol} | {pnl:+.0f} | {count} |"
        for symbol, pnl, count in analysis.worst_symbols
    )
    lines.extend(["", "## 最亏的行业", "| 行业 | 合计净盈亏 | 笔数 |", "|---|---|---|"])
    lines.extend(
        f"| {name or '未映射'} | {pnl:+.0f} | {count} |"
        for name, pnl, count in analysis.worst_sectors
    )
    lines.extend(
        [
            "",
            "## 读法提醒",
            "1. 分组样本少时（<10 笔）只看方向，不要当统计结论。",
            "2. MFE 是持仓期间最大浮盈，MAE 是最大浮亏；用 R 看，避免不同价格量级干扰。",
            "3. 如果大量亏损单在失败前到过 +0.5R/+1R，问题在退出；"
            "如果多数从未盈利，问题在入场。",
        ]
    )
    return "\n".join(lines) + "\n"


def _bucket_header() -> str:
    return (
        "| 分组 | 笔数 | 胜率 | 利润因子 | 平均净盈亏 | 止损率 | "
        "平均 MFE | 平均 MAE | 平均 MFE(R) | 平均持有 |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )


def _bucket_row(bucket: B2Bucket) -> str:
    pf = "inf" if bucket.profit_factor == float("inf") else f"{bucket.profit_factor:.2f}"
    return (
        f"| {bucket.label} | {bucket.count} | {bucket.win_rate:.1%} | {pf} | "
        f"{bucket.avg_pnl:+.2f} | {bucket.stop_rate:.1%} | {bucket.avg_mfe_pct:+.1%} | "
        f"{bucket.avg_mae_pct:+.1%} | {bucket.avg_mfe_r:.2f} | {bucket.avg_hold:.1f} 天 |"
    )


def write_b2_report(
    cfg: AppConfig, analysis: B2Analysis, *, end: date
) -> Path:
    directory = cfg.paths.reports / "backtest"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"b2_analysis_{end}.md"
    path.write_text(render_b2_markdown(analysis), encoding="utf-8")
    (directory / f"b2_analysis_{end}.json").write_text(
        json.dumps(
            {
                "total": _bucket_dict(analysis.total),
                "losers": analysis.losers,
                "never_profit": analysis.never_profit,
                "half_r_fail": analysis.half_r_fail,
                "one_r_fail": analysis.one_r_fail,
                "worst_symbols": analysis.worst_symbols,
                "worst_sectors": analysis.worst_sectors,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _bucket_dict(bucket: B2Bucket) -> dict[str, object]:
    return {
        "label": bucket.label,
        "count": bucket.count,
        "win_rate": bucket.win_rate,
        "avg_pnl": bucket.avg_pnl,
        "profit_factor": (
            None if bucket.profit_factor == float("inf") else bucket.profit_factor
        ),
        "stop_rate": bucket.stop_rate,
        "avg_mfe_pct": bucket.avg_mfe_pct,
        "avg_mae_pct": bucket.avg_mae_pct,
        "avg_mfe_r": bucket.avg_mfe_r,
        "avg_hold": bucket.avg_hold,
    }
