"""每日收盘后的滚动决策报告：真实持仓 + 交易记录 + 截至当日收盘的策略状态 → 下一交易日的操作。

不推送，只写 ``reports/plan/``。每天重算一次，条件单价位随最新均线更新。规则：
- 资金结构沿用 PLAN：底仓 50%（按 universe.yaml 目标比例），个股 50% 交给分板块滚动引擎；
- 个股：在所属板块组“名额 × 缓冲”内的保留；不在的进入存量清理——反弹到 MA10 卖一半、MA20 卖完，
  不在低点一次性割（PLAN §4 存量清理）；
- 新买入：每个交易日最多 ``max_new_buys`` 只，按引擎排名，限价 ≤ 最新收盘 × 1.02，一手可买得起；
- 近 20 个交易日内卖出过的不回补（交易记录冷却）；
- 行业/跨境 ETF：轮动策略实测无效（HANDOFF §0.2），按存量清理处理，资金并入底仓。
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from tradecompass.research.data import read_etf_bars
from tradecompass.research.rolling import LOT, RollingEngine, group_indices

BUY_LIMIT_MARKUP = 0.02
COOLDOWN_TRADING_DAYS = 20
CLEANUP_PATIENCE = 10
DEEP_LOSS = -0.20


@dataclass(frozen=True, slots=True)
class HoldingView:
    code: str
    name: str
    kind: str
    quantity: int
    price: float
    cost: float
    value: float
    pnl_pct: float
    group: str
    rank: int | None
    zone: int | None
    ma10: float | None
    ma20: float | None
    ma60: float | None
    action: str
    detail: str


@dataclass(frozen=True, slots=True)
class BuyIdea:
    code: str
    name: str
    group: str
    rank: int
    close: float
    limit: float
    shares: int
    amount: float


def _bs_code(code: str) -> str:
    return ("sh." if code[0] in "569" else "sz.") + code


def _ma_raw(adj: np.ndarray, scale: float, n: int) -> float | None:
    window = adj[-n:]
    if len(window) < n or np.isnan(window).any():
        return None
    return float(window.mean() * scale)


def _etf_mas(bars: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    if bars.empty:
        return None, None, None
    ret = (bars["close"] / bars["preclose"] - 1.0).fillna(0.0)
    adj = (1.0 + ret).cumprod().to_numpy()
    scale = float(bars["close"].iloc[-1] / adj[-1])
    return _ma_raw(adj, scale, 10), _ma_raw(adj, scale, 20), _ma_raw(adj, scale, 60)


def _cleanup(price: float, ma10: float | None, ma20: float | None) -> tuple[str, str]:
    targets = sorted({round(v, 3) for v in (ma10, ma20) if v is not None and v > price})
    if not targets:
        return "清理：已在均线上方", f"现价已高于 MA10/MA20，可在 {price:.3f} 附近分两次卖出"
    if len(targets) == 1:
        return "清理：反弹卖出", f"挂条件单 {targets[0]:.3f} 卖一半，突破后再卖另一半"
    return "清理：反弹分批", f"{targets[0]:.3f} 卖一半，{targets[1]:.3f} 卖剩余"


def _half_lot(quantity: int) -> int:
    if quantity < 2 * LOT:
        return quantity
    return (quantity // 2) // LOT * LOT


def apply_patience(
    views: list[HoldingView],
    dates: pd.DatetimeIndex,
    since: dict[str, str],
    patience: int = CLEANUP_PATIENCE,
) -> tuple[list[HoldingView], dict[str, str]]:
    """进入清理超过 ``patience`` 个交易日仍未反弹到位的，改为收盘前卖一半；返回新的清理起始日。"""
    today = dates[-1]
    state: dict[str, str] = {}
    out: list[HoldingView] = []
    for v in views:
        if not v.action.startswith("清理"):
            out.append(v)
            continue
        start = since.get(v.code, today.date().isoformat())
        state[v.code] = start
        waited = len(dates) - 1 - int(dates.searchsorted(pd.Timestamp(start)))
        if waited >= patience and v.action != "清理：已在均线上方":
            v = dataclasses.replace(
                v,
                action="清理：到期卖一半",
                detail=f"自 {start} 进入清理已 {waited} 个交易日未反弹到位；"
                f"收盘前卖 {_half_lot(v.quantity)} 股，剩余继续挂 {v.detail.split('；')[-1]}",
            )
        out.append(v)
    return out, state


def build_plan(
    engine: RollingEngine,
    holdings: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    cash: float,
    total_assets: float,
    base_targets: dict[str, float],
    etf_bars: dict[str, pd.DataFrame],
    etf_names: dict[str, str],
    names: dict[str, str],
    stock_share: float = 0.5,
    max_new_buys: int = 3,
    cleanup_since: dict[str, str] | None = None,
) -> dict[str, object]:
    i = len(engine.dates) - 1
    as_of = engine.dates[i]
    stock_capital = total_assets * stock_share
    slot_value = stock_capital / engine.slots
    slots = engine.rule_slots(i)

    ranks: dict[str, tuple[str, int, int]] = {}
    ranked_by_rule: dict[str, pd.Series] = {}
    for name, rule in engine.rules.items():
        ranked = engine.scores(rule, i)
        ranked_by_rule[name] = ranked
        keep_line = int(math.ceil(slots[name] * rule.buffer))
        for pos, code in enumerate(ranked.index, start=1):
            ranks[code] = (name, pos, keep_line)

    sold_recently = set()
    if not trades.empty:
        cutoff = engine.dates[max(0, i - COOLDOWN_TRADING_DAYS)].date()
        dates = pd.to_datetime(trades["date"]).dt.date
        recent = trades[(trades["side"] == "sell") & (dates > cutoff)]
        sold_recently = {_bs_code(str(c).zfill(6)) for c in recent["code"]}

    views: list[HoldingView] = []
    kept_by_rule: dict[str, int] = {}
    for row in holdings.to_dict("records"):
        code = str(row["code"]).zfill(6)
        bs = _bs_code(code)
        name_ = str(row["name"])
        price, cost, qty = float(row["price"]), float(row["cost"]), int(row["quantity"])
        value = float(row["value"])
        pnl = price / cost - 1.0 if cost else 0.0
        if bs in etf_names:
            ma10, ma20, ma60 = _etf_mas(etf_bars.get(bs, pd.DataFrame()))
            if code in base_targets:
                action, detail = "底仓：保留", f"目标占底仓 {base_targets[code]:.0%}，见底仓再平衡"
            else:
                action, detail = _cleanup(price, ma10, ma20)
            views.append(HoldingView(code, name_, "ETF", qty, price, cost, value,
                                     pnl, "ETF", None, None, ma10, ma20, ma60, action, detail))
            continue
        j = engine.col.get(bs)
        ma10 = ma20 = ma60 = None
        if j is not None:
            adj = engine.adj[: i + 1, j]
            scale = float(engine.close[i, j] / engine.adj[i, j])
            ma10, ma20, ma60 = (_ma_raw(adj, scale, n) for n in (10, 20, 60))
        info = ranks.get(bs)
        rank: int | None = None
        zone: int | None = None
        if info is not None and info[1] <= info[2]:
            group, rank, zone = info
            kept_by_rule[group] = kept_by_rule.get(group, 0) + 1
            action, detail = "保留", f"{group} 第 {rank} 名（保留线 {zone}）"
        else:
            fallback = str(engine.rule_of_code[j]) if j is not None else ""
            group = info[0] if info else (fallback or "池外")
            if info:
                rank, zone = info[1], info[2]
            action, detail = _cleanup(price, ma10, ma20)
            if info is None:
                detail = "不在当前沪深300/中证500 股票池；" + detail
            else:
                detail = f"{group} 第 {rank} 名，跌出保留线 {zone}；" + detail
        views.append(HoldingView(code, name_, "股票", qty, price, cost, value,
                                 pnl, group, rank, zone, ma10, ma20, ma60, action, detail))

    held = {_bs_code(str(c).zfill(6)) for c in holdings["code"]}
    views, cleanup_state = apply_patience(
        views, pd.DatetimeIndex(engine.dates[: i + 1]), cleanup_since or {}
    )
    etf_stale = sorted(
        v.code for v in views
        if v.kind == "ETF"
        and not etf_bars.get(_bs_code(v.code), pd.DataFrame()).empty
        and etf_bars[_bs_code(v.code)].index.max() < as_of
    )
    ideas: list[BuyIdea] = []
    for name, ranked in ranked_by_rule.items():
        room = max(0, slots[name] - kept_by_rule.get(name, 0))
        for pos, code in enumerate(ranked.index, start=1):
            if room <= 0:
                break
            if code in held or code in sold_recently:
                continue
            j = engine.col[code]
            close = float(engine.close[i, j])
            limit = round(close * (1.0 + BUY_LIMIT_MARKUP), 2)
            shares = int(slot_value // (limit * LOT)) * LOT
            if shares <= 0:
                continue
            ideas.append(BuyIdea(code[3:], names.get(code[3:], code[3:]), name, pos, close, limit,
                                 shares, shares * limit))
            room -= 1
    ideas.sort(key=lambda b: (b.rank, b.group))

    idx = group_indices(engine)
    trend = []
    for name in engine.rules:
        series = idx[name].iloc[: i + 1]
        ma120 = float(series.rolling(120).mean().iloc[-1])
        last = float(series.iloc[-1])
        trend.append((name, last / ma120 - 1.0, slots[name]))

    return {
        "as_of": as_of.date(),
        "views": views,
        "ideas": ideas,
        "trend": trend,
        "slots": slots,
        "slot_value": slot_value,
        "stock_capital": stock_capital,
        "cash": cash,
        "total_assets": total_assets,
        "sold_recently": sorted(c[3:] for c in sold_recently),
        "cleanup_state": cleanup_state,
        "etf_stale": etf_stale,
    }


def render_plan(
    plan: dict[str, object],
    *,
    base_targets: dict[str, float],
    behavior_lines: list[str],
    market_line: str,
    notes: list[str],
    max_new_buys: int = 3,
    next_day: date | None = None,
) -> str:
    views: list[HoldingView] = plan["views"]  # type: ignore[assignment]
    ideas: list[BuyIdea] = plan["ideas"]  # type: ignore[assignment]
    trend: list[tuple[str, float, int]] = plan["trend"]  # type: ignore[assignment]
    total = float(plan["total_assets"])  # type: ignore[arg-type]
    cash = float(plan["cash"])  # type: ignore[arg-type]
    as_of: date = plan["as_of"]  # type: ignore[assignment]
    stock_value = sum(v.value for v in views if v.kind == "股票")
    etf_value = sum(v.value for v in views if v.kind == "ETF")
    base_value = sum(v.value for v in views if v.code in base_targets)
    keep = [v for v in views if v.action.startswith(("保留", "底仓"))]
    clean = [v for v in views if v.action.startswith("清理")]
    clean_value = sum(v.value for v in clean)
    deep = "、".join(
        v.name for v in sorted(views, key=lambda x: x.pnl_pct) if v.pnl_pct <= DEEP_LOSS
    )

    target_day = f"{next_day}（下一交易日）" if next_day else "下一交易日"
    lines = [
        f"# 操作提醒：{target_day}（数据截至 {as_of} 收盘）",
        "",
        "> 每个交易日收盘后滚动重算；纪律提醒，不构成投资建议；不推送。"
        "价格为不复权实际价，请在券商 APP 用条件单执行。",
        "",
        "## 一、结论",
        "",
        f"- 现有 {len(views)} 个持仓过于分散；按策略只需 **{len(keep)} 个保留**，"
        f"**{len(clean)} 个进入存量清理**（市值约 {clean_value:,.0f} 元），**不在低点一次性割**，"
        "挂反弹条件单分批卖。",
        f"- {target_day}最多新买 **{max_new_buys} 只**策略标的，先用现金 {cash:,.0f} 元；"
        "清理回笼的钱由之后每天的报告继续安排。",
        "- 底仓（沪深300ETF/黄金ETF/30年国债ETF）偏低，清理回笼资金优先补底仓与策略名额，"
        "不再买行业/跨境 ETF。",
        "",
        "## 二、账户现状（对账单 10-23）",
        "",
        "| 项目 | 金额 | 占比 |",
        "|---|---|---|",
        f"| 总资产 | {total:,.0f} | 100% |",
        f"| 个股市值 | {stock_value:,.0f} | {stock_value / total:.0%} |",
        f"| ETF 市值 | {etf_value:,.0f} | {etf_value / total:.0%} |",
        f"| 其中底仓 ETF | {base_value:,.0f} | {base_value / total:.0%}（目标 50%） |",
        f"| 可用现金（含天天宝） | {cash:,.0f} | {cash / total:.0%} |",
        "",
        "## 三、市场与板块状态（只用截至 10-23 的数据）",
        "",
        f"- {market_line}",
        "",
        "| 板块组 | 板块指数相对 MA120 | 当前名额 |",
        "|---|---|---|",
    ]
    for name, gap, n in trend:
        state = "趋势向上（加配 ×1.3）" if gap > 0 else "趋势向下（减配 ×0.7）"
        lines.append(f"| {name} | {gap:+.1%} {state} | {n} |")
    lines += [
        "",
        f"个股部分资金按总资产 50% 计约 {float(plan['stock_capital']):,.0f} 元，"  # type: ignore[arg-type]
        f"每个名额约 {float(plan['slot_value']):,.0f} 元。",  # type: ignore[arg-type]
        "",
        "## 四、持仓逐只处理",
        "",
        "| 代码 | 名称 | 数量 | 现价 | 成本 | 浮盈亏 | 板块组/排名 | MA10 | MA20 | 结论 | 怎么做 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "-"

    order = {"保留": 0, "底仓：保留": 1}
    for v in sorted(views, key=lambda x: (order.get(x.action, 2), x.pnl_pct)):
        where = f"{v.group} #{v.rank}" if v.rank else v.group
        lines.append(
            f"| {v.code} | {v.name} | {v.quantity} | {v.price:.3f} | {v.cost:.3f} | "
            f"{v.pnl_pct:+.1%} | {where} | {fmt(v.ma10)} | {fmt(v.ma20)} | "
            f"**{v.action}** | {v.detail} |"
        )
    lines += [
        "",
        f"清理规则：条件单价位每天按最新均线更新；"
        f"进入清理满 {CLEANUP_PATIENCE} 个交易日仍未反弹到位的，"
        "报告会改为“收盘前卖一半”（起始日记在 portfolio/plan_state.json）。"
        + (
            f"深套票（浮亏超过 {-DEEP_LOSS:.0%}：{deep}）同样执行，不再补仓摊低。"
            if deep
            else ""
        ),
        "",
        f"## 五、新买入（{target_day}最多 {max_new_buys} 只，按排名先后，限价不追高）",
        "",
        "| 优先级 | 代码 | 名称 | 板块组/排名 | 10-23 收盘 | 限价（≤） | 股数 | 约金额 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    budget = cash
    shown = 0
    for k, b in enumerate(ideas, start=1):
        tag = "买入" if shown < max_new_buys and b.amount <= budget else "候补"
        if tag == "买入":
            budget -= b.amount
            shown += 1
        lines.append(
            f"| {tag} {k} | {b.code} | {b.name} | {b.group} #{b.rank} | {b.close:.2f} | "
            f"{b.limit:.2f} | {b.shares} | {b.amount:,.0f} |"
        )
    lines += [
        "",
        "- “买入”= 现金够用且在当日上限内，开盘后挂限价单；"
        "当天没成交就撤单，收盘后按新报告重排。",
        "- “候补”= 等清理回笼资金后按顺序补；每天最多新买 3 只，避免一次性大换仓。",
        f"- 近 {COOLDOWN_TRADING_DAYS} 个交易日卖出过的不回补："
        + ("、".join(plan["sold_recently"]) or "无"),  # type: ignore[arg-type]
        "- 新买入不设紧止损（回测证明对反弹型选股有害）；"
        "退出靠各板块组到期重排：防御 60 日、其余 20 日。",
        "",
        "## 六、底仓再平衡（目标：总资产 50%）",
        "",
        "| 代码 | 目标占底仓 | 目标金额 | 现有市值 | 差额 |",
        "|---|---|---|---|---|",
    ]
    for code, weight in base_targets.items():
        target = total * 0.5 * weight
        have = sum(v.value for v in views if v.code == code)
        lines.append(
            f"| {code} | {weight:.0%} | {target:,.0f} | {have:,.0f} | {target - have:+,.0f} |"
        )
    lines += [
        "",
        "底仓缺口较大，不用一次补齐：清理回笼资金的一半按差额比例分配，约 4–6 周补到位。",
        "",
        "## 七、你的交易记录体检（对账单两年，剔除天天宝）",
        "",
        *[f"- {line}" for line in behavior_lines],
        "",
        "## 八、注意事项",
        "",
        *[f"- {line}" for line in notes],
    ]
    return "\n".join(lines) + "\n"


def base_targets_from_universe(path: Path) -> dict[str, float]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    for item in data.get("instruments", []):
        if item.get("kind") == "base_etf" and item.get("target"):
            out[str(item["symbol"])] = float(item["target"])
    return out


def load_etf_bars(paths: object, codes: list[str]) -> dict[str, pd.DataFrame]:
    return {code: read_etf_bars(paths, code) for code in codes}  # type: ignore[arg-type]
