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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from tradecompass.research import data as rdata
from tradecompass.research.data import read_etf_bars
from tradecompass.research.factors import factor_by_name
from tradecompass.research.rolling import LOT, RollingEngine, group_indices

BUY_LIMIT_MARKUP = 0.02
COOLDOWN_TRADING_DAYS = 20
CLEANUP_PATIENCE = 10
DEEP_LOSS = -0.20
BASE_SHARE_OF_ASSETS = 0.5
BASE_BUDGET_UNIT = 10_000.0
BASE_REBALANCE_SHARE = 0.5
"""清理回笼资金里用于补底仓的比例；HANDOFF §4：回笼资金一半按差额补。"""


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
    score: float | None = None
    """所属板块组的综合分（组内 z 分数加权合成），只能同组比较，跨组不可比。"""
    add_shares: int = 0
    """仍在该组保留线内、但市值低于目标仓位时，建议补足的股数（0 表示不补）。"""
    drivers: str = ""
    """分数的主要来源：贡献最大的两个因子与拖累最大的一个。"""
    strength: float | None = None
    """0–100 的相对强度：组内分相对当日门槛的位置，50 = 刚好在线上。不是上涨概率。"""


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
    tag: str = "候补"
    score: float | None = None
    """所属板块组的综合分（组内 z 分数加权合成），只能同组比较，跨组不可比。"""
    origin: str = "池内"
    """候选来源：``池内`` 指数成分 / ``自选`` 个人自选池。"""
    drivers: str = ""
    """分数的主要来源：贡献最大的两个因子与拖累最大的一个。"""
    strength: float | None = None
    """0–100 的相对强度：相对“买入线”（第一个没买到的同组标的）的位置。"""


@dataclass(frozen=True, slots=True)
class WatchView:
    """自选池里一只标的的体检结果（不含持仓，持仓在持仓表里）。"""

    code: str
    name: str
    group: str
    rank: int | None
    zone: int | None
    score: float | None
    drivers: str
    status: str
    strength: float | None = None
    """0–100 的相对强度：组内分相对保留门槛的位置。"""


def _bs_code(code: str) -> str:
    return rdata.to_bs_code(code)


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


def _etf_trend_strength(bars: pd.DataFrame) -> float | None:
    """波段 ETF 的趋势健康度（0–100），只作展示与清理参考。

    ETF 不该套股票的横截面 z 分数：样本太少、因子集也对不上；而动量轮动已经实测无效。
    这里用时间序列趋势：以“现价相对 MA20 的偏离”为主（50 = 刚好在 MA20），
    再用 MA60/MA120 位置和 MA20 斜率加减分，高=趋势强、低=趋势弱。
    它不改变 ``etf_action`` 的暂行 MA20 规则，也不用于新买；要用于交易必须先回测。
    """
    if len(bars) < 120:
        return None
    ret = (bars["close"] / bars["preclose"] - 1.0).fillna(0.0)
    adj = (1.0 + ret).cumprod().to_numpy()
    scale = float(bars["close"].iloc[-1] / adj[-1])
    ma20 = _ma_raw(adj, scale, 20)
    ma60 = _ma_raw(adj, scale, 60)
    ma120 = _ma_raw(adj, scale, 120)
    if ma20 is None or ma60 is None:
        return None

    price = float(bars["close"].iloc[-1])
    # 以 MA20 为中性线：偏离 ±4% 大约对应 88/12 分，再叠加中长期位置与斜率
    score = 50.0 + 50.0 * math.tanh((price / ma20 - 1.0) / 0.04)
    score += 8.0 if price >= ma60 else -8.0
    if ma120 is not None:
        score += 8.0 if price >= ma120 else -8.0
    score += 8.0 if adj[-20:].mean() > adj[-40:-20].mean() else -8.0
    return max(0.0, min(100.0, score))


def _drivers(row: pd.Series, weights: pd.Series) -> str:
    """把一只票的因子 z 分数压成一行：贡献最大的两个 + 拖累最大的一个。

    z 已经乘过因子方向，所以正数就是“有利”。用因子原始 z 而不是加权贡献，
    是为了让读者直接看到它在这只票上有多突出，而不是被权重稀释后的数字。
    """
    usable = row.dropna()
    if usable.empty:
        return ""
    contrib = (usable * weights.reindex(usable.index).fillna(0.0)).sort_values(ascending=False)
    positive = [c for c in contrib.index if contrib[c] > 0][:2]
    negative = [c for c in contrib.index if contrib[c] < 0][-1:]
    text = "、".join(f"{factor_by_name(c).label} {usable[c]:+.1f}" for c in positive)
    if negative:
        lead = "；拖累 " if text else "拖累 "
        text += f"{lead}{factor_by_name(negative[0]).label} {usable[negative[0]]:+.1f}"
    return text


def _top_up_shares(value: float, target: float, price: float) -> int:
    """补到目标仓位还差几股（整手）；已经够了返回 0。"""
    deficit = target - value
    if deficit <= 0 or price <= 0:
        return 0
    return int(deficit // (price * LOT)) * LOT


def _strength(score: float | None, line: float | None) -> float | None:
    """把组内分相对门槛的位置映射成 0–100，方便人读。

    ``line`` 是该组当日的保留线（持仓）或买入线（新买）分数；分数刚好在线上给 50，
    高一个标准差约 84，低一个标准差约 16。它是相对强度，不是上涨概率，也不做买卖阈值。
    """
    if score is None or line is None:
        return None
    edge = score - line
    return 100.0 * 0.5 * (1.0 + math.erf(edge / math.sqrt(2.0)))


def _cleanup(price: float, ma10: float | None, ma20: float | None) -> tuple[str, str]:
    """清理建议：反弹到 MA10 卖一半、MA20 卖完。

    两条均线都没有时**不能**说“已在均线上方”——那只是没数据，不是价格位置。
    """
    if ma10 is None and ma20 is None:
        return (
            "清理：缺数据",
            f"缺 MA10/MA20（没有 {price:.3f} 当日 K 线），给不出反弹条件单价位；先补齐数据再执行",
        )
    targets = sorted({round(v, 3) for v in (ma10, ma20) if v is not None and v > price})
    if not targets:
        return "清理：已在均线上方", f"现价已高于 MA10/MA20，可在 {price:.3f} 附近分两次卖出"
    if len(targets) == 1:
        return "清理：反弹卖出", f"挂条件单 {targets[0]:.3f} 卖一半，突破后再卖另一半"
    return "清理：反弹分批", f"{targets[0]:.3f} 卖一半，{targets[1]:.3f} 卖剩余"


def etf_action(
    role: str,
    code: str,
    price: float,
    ma10: float | None,
    ma20: float | None,
    base_targets: dict[str, float],
) -> tuple[str, str]:
    """底仓再平衡、稳健持有、波段（暂行：站上 MA20 持有观察，跌破则反弹减仓）。"""
    if code in base_targets:
        return "底仓：保留", f"目标占底仓 {base_targets[code]:.0%}，见底仓再平衡"
    if role == "稳健":
        trend = ""
        if ma20 is not None:
            trend = "；在 MA20 上方" if price >= ma20 else "；在 MA20 下方，不低位卖出"
        return "稳健：持有", "稳健类 ETF 长期持有，不做波段" + trend
    if ma20 is not None and price >= ma20:
        return "波段：持有观察", (
            "站上 MA20；波段规则待验证（暂行），不新增，跌破 MA20 后改为反弹减仓"
        )
    if ma20 is None:
        return "清理：缺数据", "缺 MA20（没有当日 K 线），判断不了波段位置；先补齐数据再执行"
    action, detail = _cleanup(price, ma10, ma20)
    return action, "波段 ETF 跌破 MA20（暂行规则）；" + detail


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
        # 已经能卖（在均线上方）或压根没数据（给不出价位）的，不该按“等太久”升级
        if waited >= patience and v.action not in ("清理：已在均线上方", "清理：缺数据"):
            v = dataclasses.replace(
                v,
                action="清理：到期卖一半",
                detail=f"自 {start} 进入清理已 {waited} 个交易日未反弹到位；"
                f"收盘前卖 {_half_lot(v.quantity)} 股，剩余继续挂 {v.detail.split('；')[-1]}",
            )
        out.append(v)
    return out, state


@dataclass(frozen=True, slots=True)
class BaseOrder:
    """底仓再平衡的一行：目标、缺口，以及“每回笼一万元”能买多少。"""

    code: str
    weight: float
    target: float
    have: float
    gap: float
    price: float | None
    limit: float | None
    shares_now: int
    amount_now: float
    shares_per_unit: int
    amount_per_unit: float
    lot_cost: float
    share_amount: float
    units_to_lot: int | None
    shares_to_target: int


def _etf_last_close(bars: pd.DataFrame) -> float | None:
    if bars.empty:
        return None
    value = float(bars["close"].iloc[-1])
    return value if np.isfinite(value) and value > 0 else None


def _allocate(
    budget: float,
    limits: dict[str, float],
    gaps: dict[str, float],
) -> dict[str, int]:
    """按缺口比例分预算，取整到整手；剩下的零钱按“剩余缺口”再按比例分一轮。

    纯按比例取整会每轮剩几千元闲置（三只 ETF 价位差很大）。反复按剩余缺口分，
    既不破坏比例，又能把预算用掉。
    """
    lots = {code: 0 for code in limits}
    remaining = dict(gaps)
    left = budget
    while left > 0:
        total = sum(value for value in remaining.values() if value > 0)
        if total <= 0:
            break
        pass_budget = left
        spent = 0.0
        for code, limit in limits.items():
            if limit <= 0 or remaining.get(code, 0.0) <= 0:
                continue
            share = remaining[code] / total
            count = int(pass_budget * share // (limit * LOT)) * LOT
            affordable = int(remaining[code] // (limit * LOT)) * LOT
            count = min(count, affordable)
            if count <= 0:
                continue
            lots[code] += count
            remaining[code] -= count * limit
            spent += count * limit
        if spent <= 0:
            break
        left -= spent
    return lots


def base_orders(
    base_targets: dict[str, float],
    *,
    total_assets: float,
    views: list[HoldingView],
    etf_bars: dict[str, pd.DataFrame],
    cash_now: float,
) -> list[BaseOrder]:
    """按缺口比例把“现在的闲钱”和“每回笼一万元”分到三只底仓 ETF。

    分配只按缺口比例，不猜哪只更好；买入数量取整到 100 份，限价 = 收盘 × 1.02。
    现金不够一手时给 0，报告里会说明要攒几次。
    """
    have = {
        code: sum(v.value for v in views if v.code == code) for code in base_targets
    }
    target = {
        code: total_assets * BASE_SHARE_OF_ASSETS * weight
        for code, weight in base_targets.items()
    }
    gap = {code: max(0.0, target[code] - have[code]) for code in base_targets}
    unit_budget = BASE_BUDGET_UNIT * BASE_REBALANCE_SHARE
    limits: dict[str, float] = {}
    prices: dict[str, float | None] = {}
    for code in base_targets:
        price = _etf_last_close(etf_bars.get(rdata.to_bs_code(code), pd.DataFrame()))
        prices[code] = price
        limits[code] = 0.0 if price is None else round(price * (1.0 + BUY_LIMIT_MARKUP), 3)

    # 没有价格的 ETF 买不了，它的缺口不能参与本轮分配——否则会把另外两只的钱也稀释掉
    priced = [code for code in base_targets if limits[code] > 0]
    total_gap = sum(gap[code] for code in priced)
    share = {
        code: (gap[code] / total_gap if code in priced and total_gap > 0 else 0.0)
        for code in base_targets
    }
    alloc_gaps = {code: gap[code] for code in priced}

    now_lots = _allocate(max(0.0, cash_now), limits, alloc_gaps)
    unit_lots = _allocate(unit_budget, limits, alloc_gaps)

    rows: list[BaseOrder] = []
    for code, weight in base_targets.items():
        limit = limits[code]
        price = prices[code]
        lot_cost = limit * LOT
        share_amount = unit_budget * share[code]
        rows.append(
            BaseOrder(
                code=code,
                weight=weight,
                target=target[code],
                have=have[code],
                gap=gap[code],
                price=price,
                limit=None if price is None else limit,
                shares_now=now_lots[code],
                amount_now=now_lots[code] * limit,
                shares_per_unit=unit_lots[code],
                amount_per_unit=unit_lots[code] * limit,
                lot_cost=lot_cost,
                share_amount=share_amount,
                units_to_lot=(
                    None
                    if lot_cost <= 0 or share_amount <= 0
                    else math.ceil(lot_cost / share_amount)
                ),
                shares_to_target=0 if lot_cost <= 0 else math.ceil(gap[code] / lot_cost) * LOT,
            )
        )
    return rows


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
    etf_roles: dict[str, str] | None = None,
    stock_share: float = 0.5,
    max_new_buys: int = 3,
    cleanup_since: dict[str, str] | None = None,
    watchlist: Sequence[str] = (),
    top_up: bool = True,
) -> dict[str, object]:
    i = len(engine.dates) - 1
    as_of = engine.dates[i]
    stock_capital = total_assets * stock_share
    slot_value = stock_capital / engine.slots
    slots = engine.rule_slots(i)
    st_flags = engine.panel.st.to_numpy(dtype=bool)
    listed_days = engine.panel.listed_days.to_numpy(dtype="float64")

    held = {_bs_code(str(c).zfill(6)) for c in holdings["code"]}
    # 自选与持仓都参与打分：池内标的本来就在截面里，池外的用成分股截面参数映射进来
    tracked = sorted({*held, *(_bs_code(code) for code in watchlist)})

    ranks: dict[str, tuple[str, int, int, float]] = {}
    pool_ranks: dict[str, tuple[str, int, int, float]] = {}
    cut_scores: dict[str, float | None] = {}
    ranked_by_rule: dict[str, pd.Series] = {}
    expo_by_rule: dict[str, pd.DataFrame] = {}
    weight_by_rule: dict[str, pd.Series] = {}
    for name, rule in engine.rules.items():
        extra = [
            code for code in tracked
            if code in engine.col and engine.rule_of_code[engine.col[code]] == name
        ]
        expo = engine.exposures(rule, i, extra)
        ranked = engine.scores_from(rule, i, expo)
        expo_by_rule[name] = expo
        weight_by_rule[name] = engine.weights(rule, i).reindex(expo.columns).fillna(0.0)
        ranked_by_rule[name] = ranked
        keep_line = int(math.ceil(slots[name] * rule.buffer))
        for pos, code in enumerate(ranked.index, start=1):
            ranks[code] = (name, pos, keep_line, float(ranked.loc[code]))

        # 持仓的保留/清理按**池内**排名（回测口径），自选/池外票不挤占名次；
        # 池外标的另用“分数 ≥ 池内第 line 名的分数”来判断，等价于引擎的分数规则。
        pool_ranked = engine.scores(rule, i)
        pool_line = min(keep_line, len(pool_ranked))
        cut_scores[name] = float(pool_ranked.iloc[pool_line - 1]) if pool_line else None
        for pos, code in enumerate(pool_ranked.index, start=1):
            pool_ranks[code] = (name, pos, pool_line, float(pool_ranked.loc[code]))

    def drivers_of(code: str) -> str:
        group = ranks.get(code, ("", 0, 0, 0.0))[0]
        expo = expo_by_rule.get(group)
        if expo is None or code not in expo.index:
            return ""
        row = expo.loc[[code]].iloc[0]
        return _drivers(row, weight_by_rule[group])

    sold_recently = set()
    entry_days: dict[str, pd.Timestamp] = {}
    if not trades.empty:
        cutoff = engine.dates[max(0, i - COOLDOWN_TRADING_DAYS)].date()
        trade_dates = pd.to_datetime(trades["date"])
        recent = trades[(trades["side"] == "sell") & (trade_dates.dt.date > cutoff)]
        sold_recently = {_bs_code(str(c).zfill(6)) for c in recent["code"]}
        # 用“累计持仓由 0 变正”的时点当作建仓日：部分卖出不重置最短持有，
        # 与引擎的 entry_day 语义一致（只有清仓后再买才算新仓）。
        size = (
            pd.to_numeric(trades["quantity"], errors="coerce").fillna(0.0)
            if "quantity" in trades.columns
            else pd.Series(1.0, index=trades.index)
        )
        work = trades.assign(
            moment=trade_dates,
            code6=trades["code"].astype(str).str.zfill(6),
            signed=np.where(trades["side"] == "buy", size, -size),
        ).sort_values(["code6", "moment"])
        for code6, block in work.groupby("code6", sort=False):
            cumulative = block["signed"].cumsum()
            starts = block[(cumulative > 0) & (cumulative.shift(1, fill_value=0.0) <= 0)]
            if not starts.empty:
                entry_days[_bs_code(str(code6))] = pd.Timestamp(starts["moment"].iloc[-1])

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
            bars = etf_bars.get(bs, pd.DataFrame())
            ma10, ma20, ma60 = _etf_mas(bars)
            role = (etf_roles or {}).get(bs, "波段")
            action, detail = etf_action(role, code, price, ma10, ma20, base_targets)
            strength = _etf_trend_strength(bars) if role == "波段" else None
            if strength is not None and ma20 is not None:
                detail = (
                    f"趋势强度 {strength:.0f}/100（{price / ma20 - 1.0:+.1%} vs MA20）；"
                    + detail
                )
            group = "底仓" if code in base_targets else f"ETF·{role}"
            views.append(HoldingView(code, name_, "ETF", qty, price, cost, value,
                                     pnl, group, None, None, ma10, ma20, ma60, action, detail,
                                     strength=strength))
            continue
        j = engine.col.get(bs)
        ma10 = ma20 = ma60 = None
        if j is not None:
            adj = engine.adj[: i + 1, j]
            scale = float(engine.close[i, j] / engine.adj[i, j])
            ma10, ma20, ma60 = (_ma_raw(adj, scale, n) for n in (10, 20, 60))
        merged = ranks.get(bs)
        pool = pool_ranks.get(bs)
        info = pool or merged
        group = info[0] if info else (
            str(engine.rule_of_code[j]) if j is not None else "池外"
        )
        rank = pool[1] if pool else None
        zone = pool[2] if pool else None
        score = merged[3] if merged else None
        cut = cut_scores.get(group)
        held_days: int | None = None
        if bs in entry_days:
            entry = int(engine.dates.searchsorted(entry_days[bs]))
            if entry <= i:
                held_days = i - entry
        min_hold = engine.rules[group].min_hold if group in engine.rules else 0
        within_min_hold = held_days is not None and held_days < min_hold
        keep_by_score = score is not None and cut is not None and score >= cut

        add_shares = 0
        if within_min_hold or keep_by_score:
            kept_by_rule[group] = kept_by_rule.get(group, 0) + 1
            if pool is not None:
                action, detail = "保留", f"{group} 第 {rank} 名（保留线 {zone}）"
            elif score is not None and cut is not None:
                action, detail = (
                    "保留",
                    f"池外标的（不占池内名次），组内分 {score:+.2f} ≥ 保留门槛 {cut:+.2f}",
                )
            else:
                action, detail = "保留", f"{group}；建仓未满最短持有，先持有"
            if within_min_hold:
                detail += (
                    f"；建仓才 {held_days} 个交易日（最短持有 {min_hold} 日），按规则先持有"
                )
            # 加仓只给“组名额以内”的持仓；深套票不补，避免回到“补仓摊低”的老路
            if (
                top_up
                and pnl > DEEP_LOSS
                and rank is not None
                and rank <= slots.get(group, 0)
            ):
                add_shares = _top_up_shares(value, slot_value, price)
        else:
            action, detail = _cleanup(price, ma10, ma20)
            if pool is not None:
                detail = f"{group} 第 {rank} 名，跌出保留线 {zone}；" + detail
            elif merged is not None and score is not None and cut is not None:
                detail = (
                    f"池外标的（不占池内名次），组内分 {score:+.2f} < 保留门槛 "
                    f"{cut:+.2f}；" + detail
                )
            elif j is None:
                detail = "不在当前沪深300/中证500 股票池（无行情）；" + detail
            else:
                detail = "今日无法评分（停牌/ST/因子不足）；" + detail
        views.append(HoldingView(code, name_, "股票", qty, price, cost, value,
                                 pnl, group, rank, zone, ma10, ma20, ma60, action, detail,
                                 score, add_shares, drivers_of(bs), _strength(score, cut)))

    watch_set = {_bs_code(code) for code in watchlist}
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
    buy_cut: dict[str, float] = {}
    for name, ranked in ranked_by_rule.items():
        room = max(0, slots[name] - kept_by_rule.get(name, 0))
        last_pos = 0
        for pos, code in enumerate(ranked.index, start=1):
            if room <= 0:
                break
            if code in held or code in sold_recently:
                continue
            j = engine.col[code]
            # 池外自选可以买，但必须至少有“非 ST、上市满 120 日、今天在交易”的资格；
            # 池内标的本来就过了 eligible，这一层对所有候选都成立。
            if not engine.trading[i, j] or st_flags[i, j] or listed_days[i, j] < 120:
                continue
            close = float(engine.close[i, j])
            limit = round(close * (1.0 + BUY_LIMIT_MARKUP), 2)
            shares = int(slot_value // (limit * LOT)) * LOT
            if shares <= 0:
                continue
            ideas.append(
                BuyIdea(code[3:], names.get(code[3:], code[3:]), name, pos, close, limit,
                        shares, shares * limit, score=float(ranked.loc[code]),
                        origin="自选" if code in watch_set else "池内",
                        drivers=drivers_of(code))
            )
            last_pos = pos
            room -= 1
        if last_pos:
            # 买入线 = 同组第一个没买到的标的；没有下一只时退回最后一只
            line = last_pos if last_pos < len(ranked) else last_pos - 1
            buy_cut[name] = float(ranked.iloc[line])
    ideas.sort(key=lambda b: (b.rank, b.group))
    # 现金能覆盖的前 max_new_buys 只标“买入”，其余是“候补”；标记在这里做，
    # 报告与结构化落盘共用同一份结果，避免两处各算一遍。
    budget = cash
    shown = 0
    tagged: list[BuyIdea] = []
    for idea in ideas:
        tag = "买入" if shown < max_new_buys and idea.amount <= budget else "候补"
        if tag == "买入":
            budget -= idea.amount
            shown += 1
        tagged.append(
            dataclasses.replace(
                idea, tag=tag, strength=_strength(idea.score, buy_cut.get(idea.group))
            )
        )
    ideas = tagged

    idx = group_indices(engine)
    trend = []
    for name in engine.rules:
        series = idx[name].iloc[: i + 1]
        ma120 = float(series.rolling(120).mean().iloc[-1])
        last = float(series.iloc[-1])
        trend.append((name, last / ma120 - 1.0, slots[name]))

    spent_on_stocks = sum(b.amount for b in ideas if b.tag == "买入")

    # 加仓排在最后：回测（12 起点）显示现金先给新买入优于先给加仓——
    # 加仓优先年化 8.3% vs 8.5%、超额 +2.3% vs +2.5%，回撤没换来好处。
    # 所以只用“新买之后剩下的钱”补仓；排不上就退回“保留”，不硬凑。
    funded: dict[str, int] = {}
    for candidate in sorted((v for v in views if v.add_shares > 0),
                            key=lambda v: (v.rank or 0, v.code)):
        need = candidate.add_shares * candidate.price
        if need <= budget:
            funded[candidate.code] = candidate.add_shares
            budget -= need
    spent_on_topup = cash - spent_on_stocks - budget
    views = [
        dataclasses.replace(
            v,
            action="加仓" if v.code in funded else v.action,
            add_shares=funded.get(v.code, 0),
            detail=(
                f"{v.detail}；目标仓位 {slot_value:,.0f} 元、现 {v.value:,.0f} 元，"
                f"补 {funded[v.code]} 股（约 {funded[v.code] * v.price:,.0f} 元）"
                if v.code in funded
                else v.detail
            ),
        )
        for v in views
    ]

    watch_views: list[WatchView] = []
    for code in watchlist:
        bs = _bs_code(code)
        if bs in held:
            continue
        j = engine.col.get(bs)
        info = ranks.get(bs)
        if j is None:
            watch_views.append(WatchView(code, names.get(code, code), "—", None, None, None,
                                         "", "无行情数据"))
            continue
        if info is None:
            watch_views.append(WatchView(code, names.get(code, code), "—", None, None, None,
                                         "", "今日不可评分（停牌/ST/因子不足）"))
            continue
        group, pos, keep, score = info
        room = max(0, slots[group] - kept_by_rule.get(group, 0))
        if bs in sold_recently:
            status = "冷却期不回补"
        elif not engine.trading[i, j] or st_flags[i, j] or listed_days[i, j] < 120:
            status = "不可买（ST/新股/停牌）"
        elif pos <= room:
            status = "可买"
        elif pos <= keep:
            status = "持有才留"
        else:
            status = "不买"
        watch_views.append(WatchView(code, names.get(code, code), group, pos, keep, score,
                                     drivers_of(bs), status,
                                     _strength(score, cut_scores.get(group))))

    return {
        "as_of": as_of.date(),
        "views": views,
        "ideas": ideas,
        "watch": watch_views,
        "trend": trend,
        "slots": slots,
        "factor_weights": {
            name: [
                (factor_by_name(factor).label, float(weights[factor]))
                for factor in engine.rules[name].factors
                if factor in weights.index
            ]
            for name, weights in weight_by_rule.items()
        },
        "universe_size": {name: int(len(ranked)) for name, ranked in ranked_by_rule.items()},
        "slot_value": slot_value,
        "stock_capital": stock_capital,
        "cash": cash,
        "spent_on_topup": spent_on_topup,
        "total_assets": total_assets,
        "sold_recently": sorted(c[3:] for c in sold_recently),
        "cleanup_state": cleanup_state,
        "base_orders": base_orders(
            base_targets,
            total_assets=total_assets,
            views=views,
            etf_bars=etf_bars,
            cash_now=cash - spent_on_topup - spent_on_stocks,
        ),
        "etf_stale": etf_stale,
    }


def replay_paths(directory: Path, target_day: date, as_of: date) -> tuple[Path, Path]:
    """历史复现的输出路径：单独目录 + 带 as_of 后缀。

    复现会覆盖同名报告、并让 ``track`` 把“旧规则下的建议”当成实盘记录；放到
    ``plan/replay/`` 下（track 只扫 ``plan/*.json``，不递归）可以两者都避免。
    """
    base = directory / "replay"
    stem = f"{target_day}-asof-{as_of}"
    return base / f"{stem}.md", base / f"{stem}.json"


def plan_payload(
    plan: dict[str, object],
    *,
    target_day: date,
    rules_fingerprint: str,
    max_new_buys: int,
    statement_period: tuple[date, date] | None = None,
    generated_at: str | None = None,
) -> dict[str, object]:
    """把当次提醒落成结构化数据，供事后追踪使用（HANDOFF §9.1）。

    ``rules_fingerprint`` 是板块规则的指纹：复盘时用它区分“规则改了”还是“市场变了”。
    """
    views: list[HoldingView] = plan["views"]  # type: ignore[assignment]
    ideas: list[BuyIdea] = plan["ideas"]  # type: ignore[assignment]
    as_of: date = plan["as_of"]  # type: ignore[assignment]
    holdings: list[dict[str, object]] = [
        {
            "code": v.code,
            "name": v.name,
            "kind": v.kind,
            "quantity": v.quantity,
            "price": round(v.price, 3),
            "cost": round(v.cost, 3),
            "value": round(v.value, 2),
            "pnl_pct": round(v.pnl_pct, 4),
            "group": v.group,
            "rank": v.rank,
            "keep_line": v.zone,
            "ma10": None if v.ma10 is None else round(v.ma10, 3),
            "ma20": None if v.ma20 is None else round(v.ma20, 3),
            "ma60": None if v.ma60 is None else round(v.ma60, 3),
            "action": v.action,
            "detail": v.detail,
            "score": None if v.score is None else round(v.score, 4),
            "strength": None if v.strength is None else round(v.strength, 1),
            "add_shares": v.add_shares,
            "drivers": v.drivers,
        }
        for v in views
    ]
    buys: list[dict[str, object]] = [
        {
            "code": b.code,
            "name": b.name,
            "action": b.tag,
            "group": b.group,
            "rank": b.rank,
            "score": None if b.score is None else round(b.score, 4),
            "strength": None if b.strength is None else round(b.strength, 1),
            "close": round(b.close, 3),
            "limit": round(b.limit, 3),
            "shares": b.shares,
            "amount": round(b.amount, 2),
            "origin": b.origin,
            "drivers": b.drivers,
        }
        for b in ideas
    ]
    watch: list[dict[str, object]] = [
        {
            "code": w.code,
            "name": w.name,
            "group": w.group,
            "rank": w.rank,
            "keep_line": w.zone,
            "score": None if w.score is None else round(w.score, 4),
            "strength": None if w.strength is None else round(w.strength, 1),
            "drivers": w.drivers,
            "status": w.status,
        }
        for w in cast("list[WatchView]", plan.get("watch") or [])
    ]
    return {
        "schema": 1,
        "as_of": as_of.isoformat(),
        "target_day": target_day.isoformat(),
        "generated_at": generated_at or pd.Timestamp.now().isoformat(timespec="seconds"),
        "rules": {
            "fingerprint": rules_fingerprint,
            "max_new_buys": max_new_buys,
        },
        "slots": {
            name: int(count)
            for name, count in cast("dict[str, int]", plan["slots"]).items()
        },
        "statement_period": (
            [statement_period[0].isoformat(), statement_period[1].isoformat()]
            if statement_period
            else None
        ),
        "account": {
            "total_assets": round(float(plan["total_assets"]), 2),  # type: ignore[arg-type]
            "cash": round(float(plan["cash"]), 2),  # type: ignore[arg-type]
            "stock_capital": round(float(plan["stock_capital"]), 2),  # type: ignore[arg-type]
            "slot_value": round(float(plan["slot_value"]), 2),  # type: ignore[arg-type]
        },
        "holdings": holdings,
        "buys": buys,
        "watch": watch,
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
    statement_period: tuple[date, date] | None = None,
) -> str:
    """渲染报告。日期一律来自 ``plan["as_of"]`` 与对账单区间，不写死。"""
    views: list[HoldingView] = plan["views"]  # type: ignore[assignment]
    ideas: list[BuyIdea] = plan["ideas"]  # type: ignore[assignment]
    watch_rows: list[WatchView] = cast("list[WatchView]", plan.get("watch") or [])
    trend: list[tuple[str, float, int]] = plan["trend"]  # type: ignore[assignment]
    total = float(plan["total_assets"])  # type: ignore[arg-type]
    cash = float(plan["cash"])  # type: ignore[arg-type]
    as_of: date = plan["as_of"]  # type: ignore[assignment]
    period_start, period_end = statement_period or (as_of, as_of)
    stock_value = sum(v.value for v in views if v.kind == "股票")
    etf_value = sum(v.value for v in views if v.kind == "ETF")
    base_value = sum(v.value for v in views if v.code in base_targets)
    steady_value = sum(v.value for v in views if v.group == "ETF·稳健")
    swing_value = sum(v.value for v in views if v.group == "ETF·波段")
    keep = [
        v for v in views
        if v.action.startswith(("加仓", "保留", "底仓", "稳健", "波段"))
    ]
    topups = [v for v in views if v.action == "加仓"]
    topup_value = sum(v.add_shares * v.price for v in topups)
    clean = [v for v in views if v.action.startswith("清理")]
    actual_buys = sum(1 for b in ideas if b.tag == "买入")
    stock_now = [v for v in views if v.kind == "股票"]
    stock_after = [v for v in stock_now if not v.action.startswith("清理")]
    clean_value = sum(v.value for v in clean)
    deep = "、".join(
        v.name for v in sorted(views, key=lambda x: x.pnl_pct) if v.pnl_pct <= DEEP_LOSS
    )

    target_day = f"{next_day}（下一交易日）" if next_day else "下一交易日"
    lines = [
        f"# 操作提醒：{target_day}，数据截至 {as_of} 收盘",
        "",
        "> 每个交易日收盘后滚动重算；纪律提醒，不构成投资建议；不推送。"
        "价格为不复权实际价，请在券商 APP 用条件单执行。",
        "",
        "## 一、结论",
        "",
        f"- 现有 {len(views)} 个持仓：**{len(keep)} 个保留/持有**，"
        f"**{len(clean)} 个进入存量清理**（市值约 {clean_value:,.0f} 元），**不在低点一次性割**，"
        "挂反弹条件单分批卖。",
        (
            f"- 其中 **{len(topups)} 个仍在保留线内但低于目标仓位**，补到目标需 "
            f"{topup_value:,.0f} 元；这笔钱排在新买之后（回测显示先买新票更划算），"
            "剩下多少补多少。"
            if topups
            else "- 没有“仍在保留线内、但低于目标仓位”的持仓，本次不需要加仓"
            "（加仓只给还在保留线内的票，不给跌出线的票补仓摊低）。"
        ),
        f"- {target_day}最多新买 **{max_new_buys} 只**策略标的；按当前可用现金 "
        f"{cash:,.0f} 元，实际只够 **{actual_buys} 只**"
        + (
            ""
            if actual_buys >= max_new_buys
            else f"（还差 {max_new_buys - actual_buys} 只的钱）"
        )
        + "；清理回笼的钱由之后每天的报告继续安排。",
        f"- 底仓 {base_value / total:.0%}（目标 {BASE_SHARE_OF_ASSETS:.0%}）"
        + (
            "，清理回笼资金按差额优先补底仓"
            if base_value < total * BASE_SHARE_OF_ASSETS
            else "，已达标，回笼资金留给策略名额"
        )
        + "；稳健类 ETF 继续持有；波段类 ETF 规则待验证，暂不新增。",
        "",
        f"## 二、账户现状（对账单 {period_end:%m-%d}）",
        "",
        "| 项目 | 金额 | 占比 |",
        "|---|---|---|",
        f"| 总资产 | {total:,.0f} | 100% |",
        f"| 个股市值 | {stock_value:,.0f} | {stock_value / total:.0%} |",
        f"| ETF 市值 | {etf_value:,.0f} | {etf_value / total:.0%} |",
        f"| 其中底仓 ETF | {base_value:,.0f} | {base_value / total:.0%}（目标 50%） |",
        f"| 其中稳健 ETF | {steady_value:,.0f} | {steady_value / total:.0%} |",
        f"| 其中波段 ETF | {swing_value:,.0f} | {swing_value / total:.0%} |",
        f"| 可用现金（含天天宝） | {cash:,.0f} | {cash / total:.0%} |",
        "",
        f"## 三、市场与板块状态（只用截至 {as_of:%m-%d} 的数据）",
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
        "每个篮子的因子和权重都不一样——防御看低波/便宜/高 ROE，科技成长只看反转和价量，"
        "所以“排第几”只在同一个篮子里有意义：",
        "",
        "| 板块组 | 当日权重（先验 50% + 滚动 ICIR 50%） | 参与打分 |",
        "|---|---|---|",
    ]
    weights_table = cast("dict[str, list[tuple[str, float]]]", plan.get("factor_weights") or {})
    sizes = cast("dict[str, int]", plan.get("universe_size") or {})
    for name, items in weights_table.items():
        detail = " / ".join(f"{label} {weight:.2f}" for label, weight in items)
        lines.append(f"| {name} | {detail} | {sizes.get(name, 0)} 只 |")
    lines += [
        "",
        f"个股部分资金按总资产 50% 计约 {float(plan['stock_capital']):,.0f} 元，"  # type: ignore[arg-type]
        f"每个名额约 {float(plan['slot_value']):,.0f} 元。",  # type: ignore[arg-type]
        "",
        "## 四、持仓逐只处理",
        "",
        "| 代码 | 名称 | 数量 | 现价 | 成本 | 浮盈亏 | 板块组/排名 | 组内分 | 强度 | 驱动 | "
        "MA10 | MA20 | 结论 | 怎么做 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "-"

    def fmt_score(v: float | None) -> str:
        return f"{v:+.2f}" if v is not None else "-"

    def fmt_strength(v: float | None) -> str:
        return f"{v:.0f}" if v is not None else "-"

    order = {"加仓": 0, "保留": 0, "底仓：保留": 1, "稳健：持有": 2, "波段：持有观察": 3}
    for v in sorted(views, key=lambda x: (order.get(x.action, 4), x.pnl_pct)):
        where = f"{v.group} #{v.rank}" if v.rank else v.group
        lines.append(
            f"| {v.code} | {v.name} | {v.quantity} | {v.price:.3f} | {v.cost:.3f} | "
            f"{v.pnl_pct:+.1%} | {where} | {fmt_score(v.score)} | {fmt_strength(v.strength)} | "
            f"{v.drivers or '-'} | {fmt(v.ma10)} | {fmt(v.ma20)} | **{v.action}** | {v.detail} |"
        )
    lines += [
        "",
        "组内分：该标的在所属板块组里的因子综合分——各因子先做组内 z 分数，再按"
        "“先验等权 50% + 滚动 ICIR 50%”加权合成。排名就是这个分数的降序，"
        "所以**只能同组横向比较，跨组不可比**；负分表示在该组里偏弱。",
        "强度（0–100）：股票=组内分相对当日门槛（50 = 刚好在线上，>80 明显强，"
        "<20 明显弱；持仓/自选相对保留线，新买相对买入线）；"
        "波段 ETF=趋势健康度（站上 MA20/60/120、均线向上得分，50 为中性）；"
        "底仓/稳健 ETF 不打分。它不是上涨概率，也不能当绝对买卖阈值；"
        "波段 ETF 的强度只作清理参考，买卖仍按暂行 MA20 规则，规则回测通过前不用于新买。",
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
        f"清理完成后：如果 {len(clean)} 只清理仓全部按条件单成交，回笼约 {clean_value:,.0f} 元——"
        f"其中 {clean_value * BASE_REBALANCE_SHARE:,.0f} 元按差额补底仓，"
        f"剩下 {clean_value * (1 - BASE_REBALANCE_SHARE):,.0f} 元按排名重建个股；"
        f"个股持仓会从 {len(stock_now)} 只降到 {len(stock_after)} 只"
        "（这个数字是“全部成交”的上限，实际会分很多天）。",
        "",
        f"## 五、新买入（{target_day}，最多 {max_new_buys} 只，按排名先后，限价不追高）",
        "",
        f"| 优先级 | 代码 | 名称 | 来源 | 板块组/排名 | 组内分 | 强度 | 驱动 | "
        f"{as_of:%m-%d} 收盘 | 限价（≤） | 股数 | 约金额 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, b in enumerate(ideas, start=1):
        lines.append(
            f"| {b.tag} {k} | {b.code} | {b.name} | {b.origin} | {b.group} #{b.rank} | "
            f"{fmt_score(b.score)} | {fmt_strength(b.strength)} | {b.drivers or '-'} | "
            f"{b.close:.2f} | {b.limit:.2f} | {b.shares} | {b.amount:,.0f} |"
        )
    lines += [
        "",
        "- “买入”= 现金够用且在当日上限内，开盘后挂限价单；"
        "当天没成交就撤单，收盘后按新报告重排。",
        "- “候补”= 等清理回笼资金后按顺序补；每天最多新买 3 只，避免一次性大换仓。",
        "- “来源”= 池内（沪深300/中证500 成分）或 自选（universe.yaml 的个人自选池）。"
        "自选票与成分股用同一套因子、同一个组内分布打分，没有单独放宽。",
        f"- 近 {COOLDOWN_TRADING_DAYS} 个交易日卖出过的不回补："
        + ("、".join(plan["sold_recently"]) or "无"),  # type: ignore[arg-type]
        "- 新买入不设紧止损（回测证明对反弹型选股有害）；"
        "退出靠各板块组到期重排：防御 60 日、其余 20 日。",
        "",
        f"## 六、自选池体检（{len(watch_rows)} 只未持仓的，按组内排名）",
        "",
        "| 代码 | 名称 | 板块组 | 排名/保留线 | 组内分 | 强度 | 驱动 | 状态 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for w in sorted(watch_rows, key=lambda x: (x.rank is None, x.rank or 0, x.code)):
        where = f"{w.group} #{w.rank}" if w.rank else w.group
        keep_text = f"/{w.zone}" if w.zone else ""
        lines.append(
            f"| {w.code} | {w.name} | {where}{keep_text} | "
            f"{fmt_score(w.score)} | {fmt_strength(w.strength)} | {w.drivers or '-'} | "
            f"{w.status} |"
        )
    lines += [
        "",
        "**可买**= 排名进入该组当日空余名额，会出现在上一节的新买入里；"
        "**持有才留**= 若是持仓就继续拿着，但现在不新增；**不买**= 连持有标准都够不上。"
        "**冷却期不回补**= 近 20 个交易日卖出过；"
        "**不可买（ST/新股/停牌）**= 有资格问题，分数再高也不进新买。"
        "状态只看分数与可交易性，不看持有成本，也不看个人偏好。已持仓的自选票不在这里重复，"
        "见第四节持仓表。",
        "",
        f"## 七、底仓再平衡（目标：总资产 {BASE_SHARE_OF_ASSETS:.0%}）",
        "",
        "| 代码 | 目标占底仓 | 目标金额 | 现有市值 | 差额 | 限价（≤） | "
        "本次可买 | 每回笼 1 万元 | 补到目标还需 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    order_index = {code: position for position, code in enumerate(base_targets)}
    raw_orders = cast("list[BaseOrder]", plan["base_orders"])
    orders = sorted(raw_orders, key=lambda item: order_index.get(item.code, len(order_index)))
    for entry in orders:
        limit_text = "—" if entry.limit is None else f"{entry.limit:.3f}"
        now_text = (
            "—" if entry.shares_now == 0 else f"{entry.shares_now} 份（{entry.amount_now:,.0f} 元）"
        )
        unit_text = "—" if entry.shares_per_unit == 0 else f"{entry.shares_per_unit} 份"
        lines.append(
            f"| {entry.code} | {entry.weight:.0%} | {entry.target:,.0f} | {entry.have:,.0f} | "
            f"{entry.gap:+,.0f} | {limit_text} | {now_text} | {unit_text} | "
            f"{entry.shares_to_target} 份 |"
        )
    gap_total = sum(entry.gap for entry in orders)
    unit_amount = sum(entry.amount_per_unit for entry in orders)
    if gap_total <= 0:
        rebalance_notes = ["底仓已达目标，无需再平衡；回笼资金留给个股名额。"]
    elif unit_amount <= 0:
        rebalance_notes = [
            f"缺口合计 {gap_total:,.0f} 元，但每回笼 1 万元分到每只都不够一手，先攒够再买。"
        ]
    else:
        rounds = math.ceil(gap_total / unit_amount)
        rebalance_notes = [
            f"缺口合计 {gap_total:,.0f} 元。每回笼 1 万元，其中 "
            f"{BASE_REBALANCE_SHARE:.0%}（{BASE_BUDGET_UNIT * BASE_REBALANCE_SHARE:,.0f} 元）"
            f"按差额比例补底仓，约买 {unit_amount:,.0f} 元；照此约需 {rounds} 次回笼补齐。"
        ]
        pool = sum(v.value for v in clean)
        if pool > 0:
            available = pool * BASE_REBALANCE_SHARE
            rebalance_notes.append(
                f"当前 {len(clean)} 只清理仓市值 {pool:,.0f} 元，即便全部按条件单成交，"
                f"一半补底仓也只有 {available:,.0f} 元，"
                + (
                    f"仍差 {gap_total - available:,.0f} 元，缺口要靠后续结余或减少个股新买。"
                    if available < gap_total
                    else "够补上缺口。"
                )
            )
        for entry in orders:
            if (
                entry.gap > 0
                and entry.amount_per_unit <= 0
                and entry.units_to_lot is not None
                and entry.units_to_lot > 1
            ):
                rebalance_notes.append(
                    f"{entry.code} 一手 {entry.lot_cost:,.0f} 元，按比例每万元只分到 "
                    f"{entry.share_amount:,.0f} 元，约 {entry.units_to_lot} 次回笼才够一手——"
                    "这只先攒着，够一手再买。"
                )
    lines += ["", *rebalance_notes]
    lines += [
        "- “本次可买”用的是个股新买之后剩下的现金；为 0 说明现金已被当日新买占满。",
        "- 限价 = 最新收盘 × 1.02，100 份为一手；当天没成交就撤单，第二天按新报告重算。",
        "",
        "## 八、你的交易记录体检（对账单 "
        f"{period_start:%Y-%m-%d}~{period_end:%Y-%m-%d}，剔除天天宝）",
        "",
        *[f"- {line}" for line in behavior_lines],
        "",
        "## 九、注意事项",
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


def watchlist_from_universe(path: Path) -> list[str]:
    """个人自选池里的个股代码（6 位）。自选票与指数成分一起打分，只是来源不同。"""
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [
        str(item["symbol"]).zfill(6)
        for item in data.get("instruments", [])
        if item.get("kind") == "stock" and item.get("symbol")
    ]


def load_etf_bars(paths: object, codes: list[str]) -> dict[str, pd.DataFrame]:
    return {code: read_etf_bars(paths, code) for code in codes}  # type: ignore[arg-type]
