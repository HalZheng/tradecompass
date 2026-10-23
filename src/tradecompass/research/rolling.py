"""分板块滚动决策引擎：回测与实盘共用同一个逐日 ``step``。

每个交易日：
1. **开盘**执行前一日收盘做出的订单（先卖后买；跌停/停牌的卖单顺延，涨停/停牌的买单作废）；
2. **收盘**估值；
3. **收盘决策**：只用截至当日的数据 + 交易记录 + 当前持仓——
   - 每只持仓按所属板块规则做风控（止损后进入冷却期，冷却期内不回补）；
   - 到了该板块的调仓日才重排该板块：持有不足最短期限的不动，仍在“名额 × 缓冲”内的不换；
   - 因子权重 = 领域先验 与 滚动 ICIR（只用已实现的月度 IC）各半。

板块规则是领域先验（防御类慢、科技成长快），见 :data:`DEFAULT_RULES`；
它们参考过 2019–2026 全样本的分板块 IC，因此报告单独给出 2023 年以后的样本外段。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from tradecompass.backtest.costs import trade_fee
from tradecompass.config.models import CostsConfig
from tradecompass.domain.models import Board
from tradecompass.research.evaluate import period_returns, rank_ic, rebalance_dates, robust_z
from tradecompass.research.factors import factor_by_name
from tradecompass.research.panel import Panel, limit_pct
from tradecompass.research.sleeves import sleeve_of

LOT = 100
COOLDOWN_DAYS = 20
IC_LOOKBACK = 36
MIN_IC_HISTORY = 12
PRIOR_WEIGHT = 0.5
STALE_DAYS = 60


@dataclass(frozen=True, slots=True)
class SleeveRule:
    name: str
    sleeves: tuple[str, ...]
    factors: tuple[str, ...]
    period: int
    buffer: float
    min_hold: int
    stop_loss: float | None = None


DEFAULT_RULES: tuple[SleeveRule, ...] = (
    SleeveRule(
        "防御", ("大金融", "公用交运", "地产基建"),
        ("vol60", "turn60", "ep", "bp", "roe"), period=60, buffer=3.0, min_hold=40,
    ),
    SleeveRule(
        "消费医药", ("消费", "医药"),
        ("vol60", "turn60", "ep", "roe", "pv_corr20", "rev20"), period=20, buffer=2.0, min_hold=10,
    ),
    SleeveRule(
        "周期制造", ("周期资源", "先进制造"),
        ("vol60", "ivol20", "ep", "bp", "pv_corr20"), period=20, buffer=2.0, min_hold=10,
    ),
    # 反转为主的板块再配止损会自我抵消；2019–2022 样本内：10 日+止损 12% 每笔 +0.2%，改后 +1.6%
    SleeveRule(
        "科技成长", ("科技成长",),
        ("rev20", "pv_corr20", "ivol20", "turn20"), period=20, buffer=2.0, min_hold=10,
    ),
)


@dataclass(slots=True)
class Position:
    code: str
    rule: str
    units: float
    entry_adj: float
    entry_day: int


@dataclass(frozen=True, slots=True)
class Order:
    code: str
    side: str
    rule: str
    reason: str
    amount: float = 0.0


@dataclass(frozen=True, slots=True)
class Trade:
    day: pd.Timestamp
    code: str
    side: str
    rule: str
    reason: str
    price: float
    shares: int
    fee: float
    pnl_pct: float | None = None
    hold_days: int | None = None


@dataclass(slots=True)
class EngineState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    pending: list[Order] = field(default_factory=list)
    last_rebalance: dict[str, int] = field(default_factory=dict)
    cooldown_until: dict[str, int] = field(default_factory=dict)
    last_trading: dict[str, int] = field(default_factory=dict)


def board_of_code(code: str) -> Board:
    if code.startswith("sh.688"):
        return "star"
    if code.startswith(("sz.300", "sz.301")):
        return "gem"
    return "main"


def regime_quota(index_bars: pd.DataFrame, mapping: dict[str, float] | None = None) -> pd.Series:
    """沪深300 状态机的逐日额度；``mapping`` 为空时用原状态机额度（含恐慌恢复期减半）。"""
    from tradecompass.market.regime import build_series

    points = build_series(index_bars)
    values = {
        pd.Timestamp(p.day): (mapping[p.state.value] if mapping else p.quota) for p in points
    }
    return pd.Series(values, dtype="float64")


class RollingEngine:
    def __init__(
        self,
        panel: Panel,
        industry: dict[str, str],
        costs: CostsConfig,
        *,
        rules: tuple[SleeveRule, ...] = DEFAULT_RULES,
        slots: int = 12,
        prior_weight: float = PRIOR_WEIGHT,
        quota: pd.Series | None = None,
        tilt: pd.DataFrame | None = None,
    ) -> None:
        self.panel = panel
        self.rules = {rule.name: rule for rule in rules}
        self.costs = costs
        self.slots = slots
        self.prior_weight = prior_weight
        self.quota = (
            quota.reindex(panel.dates).ffill().fillna(1.0).to_numpy(dtype="float64")
            if quota is not None
            else None
        )
        self.dates = panel.dates
        self.codes = list(panel.codes)
        self.col = {code: i for i, code in enumerate(self.codes)}
        self.sleeve = np.array([sleeve_of(industry.get(code)) for code in self.codes])
        self.rule_of_code = np.array(
            [next((r.name for r in rules if s in r.sleeves), "") for s in self.sleeve]
        )
        self.adj = panel.adj_close.ffill().to_numpy(dtype="float64")
        self.close = panel.field("close").ffill().to_numpy(dtype="float64")
        self.open = panel.field("open").to_numpy(dtype="float64")
        self.preclose = panel.field("preclose").to_numpy(dtype="float64")
        self.trading = panel.trading.to_numpy(dtype=bool)
        self.buy_blocked = panel.buy_blocked.to_numpy(dtype=bool)
        # 决策时不能知道次日是否一字涨停（panel.eligible 含这一信息），成交时再判断
        self.eligible = (
            panel.member & panel.trading & ~panel.st & (panel.listed_days >= 120)
        ).to_numpy(dtype=bool)
        self.limit = limit_pct(panel.codes, self.dates, panel.st).to_numpy(dtype="float64")
        names = sorted({name for rule in rules for name in rule.factors})
        self.factors = {
            name: factor_by_name(name).compute(panel).to_numpy(dtype="float64") for name in names
        }
        self.expected = {name: factor_by_name(name).expected for name in names}
        self._ic_history = {rule.name: self._monthly_ic(rule) for rule in rules}
        self._tilt: pd.DataFrame | None = None
        self.set_tilt(tilt)

    def set_tilt(self, tilt: pd.DataFrame | None) -> None:
        self._tilt = (
            None
            if tilt is None
            else tilt.reindex(index=self.dates, columns=list(self.rules)).ffill().fillna(1.0)
        )

    # ---------- 打分 ----------

    def _rule_mask(self, rule: SleeveRule, i: int) -> np.ndarray:
        mask: np.ndarray = self.eligible[i] & (self.rule_of_code == rule.name)
        return mask

    def _exposures(self, rule: SleeveRule, i: int) -> pd.DataFrame:
        mask = self._rule_mask(rule, i)
        idx = np.flatnonzero(mask)
        codes = [self.codes[j] for j in idx]
        sleeves = pd.Series(self.sleeve[idx], index=codes)
        out: dict[str, pd.Series] = {}
        for name in rule.factors:
            values = pd.Series(self.factors[name][i, idx], index=codes)
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < 10:
                continue
            z = robust_z(values * self.expected[name])
            if sleeves.nunique() > 1:
                # 板块组内再按细分板块去均值，避免某一类整体便宜/低波而被集中选中
                z = z - z.groupby(sleeves.reindex(z.index)).transform("mean")
            out[name] = z
        return pd.DataFrame(out)

    def _monthly_ic(self, rule: SleeveRule) -> list[tuple[int, pd.Series]]:
        """[(实现日下标, 各因子 IC)]；实现日 = 下一个月末的次日开盘。"""
        start = self.dates[min(260, len(self.dates) - 1)].date()
        months = rebalance_dates(self.dates, start, self.dates[-1].date())
        if len(months) < 3:
            return []
        forward = period_returns(self.panel, months)
        history: list[tuple[int, pd.Series]] = []
        for k, day in enumerate(forward.index):
            i = int(self.dates.searchsorted(day))
            expo = self._exposures(rule, i)
            if expo.empty:
                continue
            fwd = forward.loc[day]
            ic = pd.Series({name: rank_ic(expo[name], fwd) for name in expo.columns})
            realized = int(self.dates.searchsorted(months[k + 1])) + 1
            history.append((realized, ic))
        return history

    def weights(self, rule: SleeveRule, i: int) -> pd.Series:
        prior = pd.Series(1.0 / len(rule.factors), index=list(rule.factors))
        known = [ic for realized, ic in self._ic_history[rule.name] if realized <= i]
        known = known[-IC_LOOKBACK:]
        if len(known) < MIN_IC_HISTORY:
            return prior
        table = pd.DataFrame(known).reindex(columns=list(rule.factors))
        icir = (table.mean() / table.std().replace(0.0, np.nan)).clip(lower=0.0).fillna(0.0)
        if icir.sum() <= 0:
            return prior
        blended: pd.Series = (
            self.prior_weight * prior + (1.0 - self.prior_weight) * icir / icir.sum()
        )
        return blended

    def scores(self, rule: SleeveRule, i: int) -> pd.Series:
        expo = self._exposures(rule, i)
        if expo.empty:
            return pd.Series(dtype="float64")
        w = self.weights(rule, i).reindex(expo.columns).fillna(0.0)
        avail = expo.notna().mul(w, axis=1).sum(axis=1)
        score = expo.fillna(0.0).mul(w, axis=1).sum(axis=1) / avail.where(avail > 0)
        enough = expo.notna().sum(axis=1) >= max(1, len(rule.factors) // 2)
        return score[enough].dropna().sort_values(ascending=False)

    def rule_slots(self, i: int) -> dict[str, int]:
        """各板块名额 ∝ 当日可选股票数 × 行业层倍数，每组至少 1 个。"""
        counts = {
            name: int(self._rule_mask(rule, i).sum()) for name, rule in self.rules.items()
        }
        if self._tilt is not None:
            row = self._tilt.iloc[i]
            weighted = {name: n * float(row.get(name, 1.0)) for name, n in counts.items()}
        else:
            weighted = {name: float(n) for name, n in counts.items()}
        total = sum(weighted.values()) or 1.0
        raw = {name: self.slots * n / total for name, n in weighted.items()}
        slots = {name: max(1, int(math.floor(v))) for name, v in raw.items()}
        leftovers = sorted(raw, key=lambda n: raw[n] - math.floor(raw[n]), reverse=True)
        for name in leftovers:
            if sum(slots.values()) >= self.slots:
                break
            slots[name] += 1
        return slots

    # ---------- 逐日 ----------

    def equity(self, state: EngineState, i: int) -> float:
        value = state.cash
        for pos in state.positions.values():
            price = self.adj[i, self.col[pos.code]]
            if np.isfinite(price):
                value += pos.units * price
        return value

    def execute_open(self, state: EngineState, i: int, trades: list[Trade]) -> None:
        day = self.dates[i]
        slip = self.costs.slippage
        carry: list[Order] = []
        sells = [o for o in state.pending if o.side == "sell"]
        buys = [o for o in state.pending if o.side == "buy"]
        for order in sells:
            j = self.col[order.code]
            pos = state.positions.get(order.code)
            if pos is None:
                continue
            gap = self.open[i, j] / self.preclose[i, j] - 1.0
            if not self.trading[i, j] or not np.isfinite(gap) or gap <= -self.limit[i, j] + 0.003:
                idle = i - state.last_trading.get(order.code, pos.entry_day)
                if order.reason.startswith("长期停牌") and idle > 2 * STALE_DAYS:
                    # 退市后再无开盘价：按最后收盘价核销，避免资金永远卡住
                    state.cash += pos.units * self.adj[i, j] * (1.0 - slip)
                    trades.append(Trade(day, order.code, "sell", pos.rule, "退市核销",
                                        float(self.close[i, j]), 0, 0.0,
                                        self.adj[i, j] / pos.entry_adj - 1.0, i - pos.entry_day))
                    del state.positions[order.code]
                    continue
                carry.append(order)
                continue
            scale = self.adj[i, j] / self.close[i, j]
            shares = int(round(pos.units * scale))
            price = self.open[i, j] * (1.0 - slip)
            fee = trade_fee(self.costs, kind="stock", board=board_of_code(order.code),
                            side="sell", price_raw=price, quantity=shares)
            state.cash += shares * price - fee
            entry_raw = pos.entry_adj / scale
            trades.append(Trade(day, order.code, "sell", pos.rule, order.reason, price, shares,
                                fee, price / entry_raw - 1.0, i - pos.entry_day))
            del state.positions[order.code]
        for order in buys:
            j = self.col[order.code]
            if order.code in state.positions or self.buy_blocked[i, j]:
                continue
            price = self.open[i, j] * (1.0 + slip)
            if not np.isfinite(price) or price <= 0:
                continue
            budget = min(order.amount, state.cash)
            shares = int(budget // (price * LOT)) * LOT
            while shares > 0:
                fee = trade_fee(self.costs, kind="stock", board=board_of_code(order.code),
                                side="buy", price_raw=price, quantity=shares)
                if shares * price + fee <= state.cash:
                    break
                shares -= LOT
            if shares <= 0:
                continue
            state.cash -= shares * price + fee
            scale = self.adj[i, j] / self.close[i, j]
            state.positions[order.code] = Position(
                order.code, order.rule, shares / scale, price * scale, i
            )
            trades.append(Trade(day, order.code, "buy", order.rule, order.reason, price, shares,
                                fee))
        state.pending = carry

    def decide(self, state: EngineState, i: int) -> list[Order]:
        orders: list[Order] = []
        selling: set[str] = {o.code for o in state.pending if o.side == "sell"}
        for code, pos in state.positions.items():
            j = self.col[code]
            if self.trading[i, j]:
                state.last_trading[code] = i
            if code in selling:
                continue
            rule = self.rules.get(pos.rule)
            ret = self.adj[i, j] / pos.entry_adj - 1.0
            if i - state.last_trading.get(code, pos.entry_day) > STALE_DAYS:
                orders.append(Order(code, "sell", pos.rule, "长期停牌/退市"))
                selling.add(code)
            elif rule and rule.stop_loss is not None and ret <= -rule.stop_loss:
                orders.append(Order(code, "sell", pos.rule, f"止损 {ret:+.1%}"))
                state.cooldown_until[code] = i + COOLDOWN_DAYS
                selling.add(code)

        total = self.equity(state, i)
        slot_value = total / self.slots
        quota = float(self.quota[i]) if self.quota is not None else 1.0
        invested = total - state.cash
        planned_out = sum(
            state.positions[o.code].units * self.adj[i, self.col[o.code]]
            for o in orders
            if o.code in state.positions
        )
        exposure = invested - planned_out
        limit = quota * total
        if exposure > limit + 0.5 * slot_value:
            # 额度下调时先减进攻型板块，防御类最后动
            order = list(self.rules)[::-1]
            ranked_positions = sorted(
                (p for c, p in state.positions.items() if c not in selling),
                key=lambda p: (
                    order.index(p.rule) if p.rule in order else -1,
                    self.adj[i, self.col[p.code]] / p.entry_adj,
                ),
            )
            for pos in ranked_positions:
                if exposure <= limit + 0.5 * slot_value:
                    break
                orders.append(Order(pos.code, "sell", pos.rule, f"大盘额度 {quota:.0%}"))
                selling.add(pos.code)
                exposure -= pos.units * self.adj[i, self.col[pos.code]]
        room = limit - exposure

        slots = self.rule_slots(i)
        for name, rule in self.rules.items():
            n = slots.get(name, 0)
            held = [c for c, p in state.positions.items() if p.rule == name and c not in selling]
            last = state.last_rebalance.get(name)
            due = last is None or i - last >= rule.period
            # 未到调仓日：只在有空名额且额度允许时补仓（例如大盘额度恢复后），不做卖出
            if not due and (len(held) >= n or room < slot_value):
                continue
            if due:
                state.last_rebalance[name] = i
            ranked = self.scores(rule, i)
            if ranked.empty:
                continue
            zone = set(ranked.index[: int(math.ceil(n * rule.buffer))])
            keep = [
                c for c in held
                if not due or c in zone or i - state.positions[c].entry_day < rule.min_hold
            ]
            for code in held:
                if code not in keep:
                    orders.append(Order(code, "sell", name, "排名跌出缓冲区"))
                    selling.add(code)
                    room += state.positions[code].units * self.adj[i, self.col[code]]
            # 一手买不起的剔除；冷却期内不回补
            max_price = slot_value / LOT
            candidates = [
                c for c in ranked.index
                if c not in keep
                and state.cooldown_until.get(c, -1) < i
                and self.close[i, self.col[c]] <= max_price
            ]
            for code in candidates[: max(0, n - len(keep))]:
                if room < 0.5 * slot_value:
                    break
                orders.append(Order(code, "buy", name, f"{name} 排名入选", slot_value))
                room -= slot_value
        return orders

    def step(self, state: EngineState, i: int, trades: list[Trade]) -> float:
        self.execute_open(state, i, trades)
        value = self.equity(state, i)
        state.pending = state.pending + self.decide(state, i)
        return value


def group_indices(engine: RollingEngine) -> pd.DataFrame:
    """每个板块组的等权指数（当日在指数内且正常交易的成分），只依赖当日及以前收益。"""
    panel = engine.panel
    ret = panel.ret.where(panel.member & panel.trading)
    out = {}
    for name in engine.rules:
        cols = [c for c, r in zip(engine.codes, engine.rule_of_code, strict=True) if r == name]
        out[name] = (1.0 + ret[cols].mean(axis=1).fillna(0.0)).cumprod()
    return pd.DataFrame(out)


def momentum_tilt(
    indices: pd.DataFrame,
    *,
    lookback: int = 120,
    skip: int = 5,
    multipliers: tuple[float, ...] = (1.5, 1.15, 0.85, 0.5),
    reverse: bool = False,
) -> pd.DataFrame:
    """按板块组动量排名给名额倍数；``reverse`` 为反转（跌得多的加配）。"""
    mom = indices.shift(skip) / indices.shift(lookback) - 1.0
    ranks = mom.rank(axis=1, ascending=reverse, method="first")
    table = {float(k + 1): v for k, v in enumerate(multipliers)}
    out: pd.DataFrame = ranks.apply(lambda col: col.map(table)).fillna(1.0)
    return out


def trend_tilt(
    indices: pd.DataFrame, *, ma: int = 120, up: float = 1.3, down: float = 0.7
) -> pd.DataFrame:
    """板块组指数在 MA 之上加配、之下减配。"""
    above = indices > indices.rolling(ma, min_periods=ma).mean()
    out: pd.DataFrame = above.astype("float64") * (up - down) + down
    return out


@dataclass(frozen=True, slots=True)
class RollingResult:
    equity: pd.Series
    trades: list[Trade]
    benchmarks: dict[str, pd.Series]
    final_state: EngineState


def run_rolling(
    engine: RollingEngine,
    start: date,
    end: date,
    *,
    capital: float,
    benchmark_bars: dict[str, pd.DataFrame] | None = None,
) -> RollingResult:
    dates = engine.dates
    lo = int(dates.searchsorted(pd.Timestamp(start)))
    hi = int(dates.searchsorted(pd.Timestamp(end), side="right"))
    if hi - lo < 30:
        raise ValueError(f"{start} → {end} 交易日太少")
    state = EngineState(cash=capital)
    trades: list[Trade] = []
    values: dict[pd.Timestamp, float] = {}
    for i in range(lo, hi):
        values[dates[i]] = engine.step(state, i, trades)
    equity = pd.Series(values)

    member = engine.panel.member & engine.panel.trading
    pool = engine.panel.ret.where(member).mean(axis=1).loc[equity.index].fillna(0.0)
    benchmarks = {"股票池等权": capital * (1.0 + pool).cumprod()}
    for name, bars in (benchmark_bars or {}).items():
        if bars.empty:
            continue
        close = bars["close"].reindex(equity.index).ffill()
        benchmarks[name] = capital * close / close.iloc[0]
    return RollingResult(equity, trades, benchmarks, state)


def summarize(series: pd.Series) -> dict[str, float]:
    daily = series.pct_change().dropna()
    years = len(daily) / 244
    total = float(series.iloc[-1] / series.iloc[0] - 1.0)
    annual = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    vol = float(daily.std() * np.sqrt(244))
    drawdown = float((series / series.cummax() - 1.0).min())
    return {
        "total": total,
        "annual": annual,
        "vol": vol,
        "sharpe": annual / vol if vol > 0 else float("nan"),
        "max_dd": drawdown,
    }


def yearly_returns(series: pd.Series) -> dict[int, float]:
    out: dict[int, float] = {}
    years = pd.DatetimeIndex(series.index).year
    previous = float(series.iloc[0])
    for year in sorted(set(years)):
        last = float(series[years == year].iloc[-1])
        out[int(year)] = last / previous - 1.0
        previous = last
    return out


def render_rolling_markdown(
    result: RollingResult, engine: RollingEngine, start: date, end: date, holdout: date
) -> str:
    series = {"策略（扣成本）": result.equity, **result.benchmarks}
    lines = [
        f"# 分板块滚动决策回测  {start} → {end}",
        "",
        f"名额 {engine.slots}；逐日：开盘执行昨日订单 → 收盘估值 → 收盘决策（只用当日及以前数据）。"
        "一手 100 股、真实费率（股票最低 5 元、卖出印花税）、滑点、涨跌停/停牌限制。",
        "",
        "## 板块规则",
        "",
        "| 板块组 | 包含 | 因子（先验方向） | 调仓间隔 | 缓冲 | 最短持有 | 止损 |",
        "|---|---|---|---|---|---|---|",
    ]
    for rule in engine.rules.values():
        stop = f"−{rule.stop_loss:.0%}" if rule.stop_loss else "无"
        lines.append(
            f"| {rule.name} | {'、'.join(rule.sleeves)} | {', '.join(rule.factors)} | "
            f"{rule.period} 日 | ×{rule.buffer:g} | {rule.min_hold} 日 | {stop} |"
        )
    lines += [
        "",
        "## 全段",
        "",
        "| 序列 | 总收益 | 年化 | 波动 | 夏普 | 最大回撤 |",
        "|---|---|---|---|---|---|",
    ]
    for name, values in series.items():
        s = summarize(values.dropna())
        lines.append(
            f"| {name} | {s['total']:+.1%} | {s['annual']:+.1%} | {s['vol']:.1%} | "
            f"{s['sharpe']:.2f} | {s['max_dd']:.1%} |"
        )
    cut = pd.Timestamp(holdout)
    lines += [
        "",
        f"## 样本外（{holdout} 起；板块规则设定时参考过全样本，此段仍可能偏乐观）",
        "",
        "| 序列 | 总收益 | 年化 | 夏普 | 最大回撤 |",
        "|---|---|---|---|---|",
    ]
    for name, values in series.items():
        part = values[values.index >= cut].dropna()
        if len(part) < 30:
            continue
        s = summarize(part)
        lines.append(
            f"| {name} | {s['total']:+.1%} | {s['annual']:+.1%} | {s['sharpe']:.2f} | "
            f"{s['max_dd']:.1%} |"
        )
    years = sorted(yearly_returns(result.equity))
    lines += ["", "## 分年度", "", "| 序列 | " + " | ".join(str(y) for y in years) + " |"]
    lines.append("|---|" + "---|" * len(years))
    for name, values in series.items():
        yearly = yearly_returns(values.dropna())
        lines.append(
            f"| {name} | " + " | ".join(f"{yearly.get(y, float('nan')):+.1%}" for y in years) + " |"
        )
    sells = [t for t in result.trades if t.side == "sell" and t.pnl_pct is not None]
    lines += [
        "",
        "## 按板块的已平仓交易",
        "",
        "| 板块组 | 笔数 | 胜率 | 平均收益 | 平均持有(日) | 止损笔数 |",
        "|---|---|---|---|---|---|",
    ]
    for name in engine.rules:
        group = [t for t in sells if t.rule == name]
        if not group:
            continue
        pnl = [t.pnl_pct or 0.0 for t in group]
        hold = [t.hold_days or 0 for t in group]
        stops = sum(1 for t in group if t.reason.startswith("止损"))
        lines.append(
            f"| {name} | {len(group)} | {np.mean([p > 0 for p in pnl]):.0%} | "
            f"{np.mean(pnl):+.2%} | {np.mean(hold):.0f} | {stops} |"
        )
    fees = sum(t.fee for t in result.trades)
    traded = sum(t.price * t.shares for t in result.trades)
    avg_equity = float(result.equity.mean())
    years_span = max(len(result.equity) / 244, 1e-9)
    lines += [
        "",
        f"交易 {len(result.trades)} 笔，费用 {fees:,.0f} 元；年化双边换手 "
        f"{traded / avg_equity / years_span:.1f} 倍。",
        "",
        "## 期末持仓",
        "",
    ]
    for code, pos in sorted(result.final_state.positions.items(), key=lambda kv: kv[1].rule):
        lines.append(f"- {code}（{pos.rule}，持有 {len(engine.dates) - 1 - pos.entry_day} 日）")
    return "\n".join(lines) + "\n"
