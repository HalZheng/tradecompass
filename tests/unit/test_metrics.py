"""绩效指标与验收判定。"""

from __future__ import annotations

from datetime import date

import pytest

from tradecompass.backtest.metrics import (
    Metrics,
    annualized_return,
    capital_utilization,
    compute_metrics,
    evaluate,
    max_drawdown,
)
from tradecompass.backtest.types import BacktestResult, EquityPoint, Trade
from tradecompass.config.models import AcceptanceConfig


def equity(values: list[float]) -> tuple[EquityPoint, ...]:
    return tuple(
        EquityPoint(date=date(2026, 1, 5), equity=value, cash=0.0, position_value=value)
        for value in values
    )


def trade(pnl: float, r: float | None = None, holding: int = 5) -> Trade:
    return Trade(
        symbol="600000",
        entry_date=date(2026, 1, 5),
        entry_price_raw=10.0,
        exit_date=date(2026, 1, 12),
        exit_price_raw=10.0 + pnl / 100,
        quantity=100,
        gross_pnl=pnl,
        fees=0.0,
        net_pnl=pnl,
        r_multiple=r,
        holding_days=holding,
        exit_reason="test",
    )


def make_result(trades: list[Trade], values: list[float]) -> BacktestResult:
    return BacktestResult(
        strategy_id="test",
        start=date(2026, 1, 5),
        end=date(2026, 12, 31),
        initial_cash=values[0],
        trades=tuple(trades),
        equity=equity(values),
        fills=(),
        fees_total=0.0,
    )


def test_profit_factor_and_win_rate() -> None:
    result = make_result(
        [trade(300.0), trade(100.0), trade(-200.0), trade(-100.0)],
        [100_000.0, 100_100.0],
    )
    metrics = compute_metrics(result)

    assert metrics.trades == 4
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.profit_factor == pytest.approx(400.0 / 300.0)
    assert metrics.avg_win == pytest.approx(200.0)
    assert metrics.avg_loss == pytest.approx(-150.0)
    assert metrics.expectancy == pytest.approx(25.0)


def test_profit_factor_without_losses_is_infinite() -> None:
    result = make_result([trade(100.0), trade(50.0)], [100_000.0, 100_000.0])
    assert compute_metrics(result).profit_factor == float("inf")


def test_expectancy_r_averages_only_known_r() -> None:
    result = make_result(
        [trade(100.0, r=1.0), trade(200.0, r=2.0), trade(50.0, r=None)],
        [100_000.0, 100_000.0],
    )
    assert compute_metrics(result).expectancy_r == pytest.approx(1.5)


def test_max_drawdown() -> None:
    points = equity([100.0, 120.0, 90.0, 110.0, 60.0])
    assert max_drawdown(points) == pytest.approx(0.5)  # 120 → 60


def test_max_drawdown_zero_when_monotonic() -> None:
    assert max_drawdown(equity([100.0, 110.0, 120.0])) == pytest.approx(0.0)


def test_annualized_return_uses_252_day_year() -> None:
    points = equity([100.0] + [100.0] * 251 + [110.0])  # 252 个点 ≈ 1 年
    assert annualized_return(points) == pytest.approx(0.10, rel=0.02)


def test_capital_utilization() -> None:
    points = (
        EquityPoint(date(2026, 1, 5), 100.0, 50.0, 50.0),
        EquityPoint(date(2026, 1, 6), 100.0, 0.0, 100.0),
    )
    assert capital_utilization(points) == pytest.approx(0.75)


def test_acceptance_flags_failures() -> None:
    cfg = AcceptanceConfig(min_trades=30, min_pf_in=1.3, min_expectancy_r=0.3)
    metrics = Metrics(
        trades=10,
        win_rate=0.4,
        avg_win=100.0,
        avg_loss=-100.0,
        profit_factor=1.1,
        expectancy=10.0,
        expectancy_r=0.2,
        annual_return=0.05,
        max_drawdown=0.2,
        calmar=0.25,
        avg_holding_days=8.0,
        capital_utilization=0.5,
        total_return=0.05,
    )

    report = evaluate(metrics, cfg, benchmark_drawdown=0.4)

    assert report.passed is False
    failed = {item.name for item in report.failed()}
    assert failed == {"样本内交易笔数", "样本内利润因子", "单笔期望(R)"}


def test_acceptance_passes_when_all_met() -> None:
    cfg = AcceptanceConfig(min_trades=5, min_pf_in=1.3, min_expectancy_r=0.3)
    metrics = Metrics(
        trades=40,
        win_rate=0.5,
        avg_win=200.0,
        avg_loss=-100.0,
        profit_factor=2.0,
        expectancy=50.0,
        expectancy_r=0.8,
        annual_return=0.2,
        max_drawdown=0.1,
        calmar=2.0,
        avg_holding_days=8.0,
        capital_utilization=0.5,
        total_return=0.3,
    )

    report = evaluate(metrics, cfg, benchmark_drawdown=0.4)

    assert report.passed is True
    assert report.failed() == ()
