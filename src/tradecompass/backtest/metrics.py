"""绩效指标与验收判定（SPEC §12.6）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tradecompass.backtest.types import BacktestResult, EquityPoint
from tradecompass.config.models import AcceptanceConfig

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class Metrics:
    trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    expectancy_r: float | None
    annual_return: float
    max_drawdown: float
    calmar: float
    avg_holding_days: float
    capital_utilization: float
    total_return: float


@dataclass(frozen=True, slots=True)
class AcceptanceItem:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    passed: bool
    items: tuple[AcceptanceItem, ...]

    def failed(self) -> tuple[AcceptanceItem, ...]:
        return tuple(item for item in self.items if not item.ok)


def max_drawdown(equity: tuple[EquityPoint, ...]) -> float:
    """最大回撤（正数，0.15 表示 15%）。"""
    if not equity:
        return 0.0
    values = np.array([point.equity for point in equity], dtype="float64")
    peak = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0, (peak - values) / peak, 0.0)
    return float(np.nanmax(drawdown)) if drawdown.size else 0.0


def annualized_return(equity: tuple[EquityPoint, ...]) -> float:
    if len(equity) < 2:
        return 0.0
    start_value = equity[0].equity
    end_value = equity[-1].equity
    if start_value <= 0 or end_value <= 0:
        return 0.0
    years = len(equity) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return float((end_value / start_value) ** (1.0 / years) - 1.0)


def capital_utilization(equity: tuple[EquityPoint, ...]) -> float:
    if not equity:
        return 0.0
    ratios = [
        point.position_value / point.equity if point.equity > 0 else 0.0 for point in equity
    ]
    return float(np.mean(ratios))


def compute_metrics(result: BacktestResult) -> Metrics:
    trades = result.trades
    wins = [t.net_pnl for t in trades if t.net_pnl > 0]
    losses = [t.net_pnl for t in trades if t.net_pnl <= 0]
    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    r_values = [t.r_multiple for t in trades if t.r_multiple is not None]
    expectancy = float(np.mean([t.net_pnl for t in trades])) if trades else 0.0
    expectancy_r = float(np.mean(r_values)) if r_values else None

    dd = max_drawdown(result.equity)
    annual = annualized_return(result.equity)
    total_return = (
        result.equity[-1].equity / result.equity[0].equity - 1.0 if result.equity else 0.0
    )

    return Metrics(
        trades=len(trades),
        win_rate=len(wins) / len(trades) if trades else 0.0,
        avg_win=float(np.mean(wins)) if wins else 0.0,
        avg_loss=float(np.mean(losses)) if losses else 0.0,
        profit_factor=profit_factor,
        expectancy=expectancy,
        expectancy_r=expectancy_r,
        annual_return=annual,
        max_drawdown=dd,
        calmar=(annual / dd) if dd > 0 else 0.0,
        avg_holding_days=(
            float(np.mean([t.holding_days for t in trades])) if trades else 0.0
        ),
        capital_utilization=capital_utilization(result.equity),
        total_return=total_return,
    )


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def evaluate(
    metrics: Metrics,
    acceptance: AcceptanceConfig,
    *,
    benchmark_drawdown: float | None = None,
    out_of_sample: Metrics | None = None,
    random_percentile: float | None = None,
    grid_pass_ratio: float | None = None,
) -> AcceptanceReport:
    """按 SPEC §12.6 的门槛逐项判定。

    只对"已经拿到输入"的项做判定；随机与网格相关项在 M4 才具备数据，
    传入 ``None`` 时记为该阶段未评估。
    """
    items: list[AcceptanceItem] = [
        AcceptanceItem(
            "样本内交易笔数",
            metrics.trades >= acceptance.min_trades,
            f"{metrics.trades} ≥ {acceptance.min_trades}",
        ),
        AcceptanceItem(
            "样本内利润因子",
            metrics.profit_factor >= acceptance.min_pf_in,
            f"{_fmt(metrics.profit_factor, 2)} ≥ {acceptance.min_pf_in}",
        ),
    ]

    if metrics.expectancy_r is not None:
        items.append(
            AcceptanceItem(
                "单笔期望(R)",
                metrics.expectancy_r > acceptance.min_expectancy_r,
                f"{_fmt(metrics.expectancy_r, 3)}R > {acceptance.min_expectancy_r}R",
            )
        )

    if benchmark_drawdown is not None and benchmark_drawdown > 0:
        cap = benchmark_drawdown * acceptance.max_dd_vs_bh
        items.append(
            AcceptanceItem(
                "最大回撤",
                metrics.max_drawdown <= cap,
                f"{_fmt(metrics.max_drawdown * 100, 1)}% ≤ {_fmt(cap * 100, 1)}%",
            )
        )

    if out_of_sample is not None:
        ratio = (
            out_of_sample.profit_factor / metrics.profit_factor
            if metrics.profit_factor > 0
            else 0.0
        )
        items.append(
            AcceptanceItem(
                "样本外利润因子",
                out_of_sample.profit_factor >= acceptance.min_pf_out
                and ratio >= acceptance.min_pf_out_ratio,
                f"{_fmt(out_of_sample.profit_factor, 2)} ≥ {acceptance.min_pf_out}，"
                f"比值 {_fmt(ratio, 2)} ≥ {acceptance.min_pf_out_ratio}",
            )
        )

    if random_percentile is not None:
        items.append(
            AcceptanceItem(
                "优于随机入场",
                random_percentile >= acceptance.min_random_pct,
                f"第 {_fmt(random_percentile * 100, 0)} 百分位 ≥ "
                f"{_fmt(acceptance.min_random_pct * 100, 0)}",
            )
        )

    if grid_pass_ratio is not None:
        items.append(
            AcceptanceItem(
                "参数稳健性",
                grid_pass_ratio >= acceptance.grid_pass_ratio,
                f"{_fmt(grid_pass_ratio * 100, 0)}% ≥ "
                f"{_fmt(acceptance.grid_pass_ratio * 100, 0)}%",
            )
        )

    return AcceptanceReport(passed=all(item.ok for item in items), items=tuple(items))
