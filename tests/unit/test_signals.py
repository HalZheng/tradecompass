"""信号层：B1/B2/S1/S2 的硬条件、打分与仓位/历史否决。"""

from __future__ import annotations

from datetime import date

from tradecompass.factors.engine import FactorSnapshot, MarketContext, SectorContext
from tradecompass.signals.engine import evaluate
from tradecompass.signals.types import (
    PortfolioContext,
    PositionContext,
    SignalKind,
    SignalParams,
    SymbolHistory,
)

DAY = date(2026, 10, 22)


def market(state: str = "RISK_ON") -> MarketContext:
    quota = {"RISK_ON": 1.0, "NEUTRAL": 0.6, "RISK_OFF": 0.25, "PANIC": 0.0}[state]
    return MarketContext(
        as_of=DAY,
        state=state,
        quota=quota,
        above_weekly=state != "PANIC",
        ma20w_up=state in {"RISK_ON", "NEUTRAL"},
        ret20=0.02,
        vol_ratio20=1.0,
    )


def sector(rank: int = 1, above: bool = True) -> SectorContext:
    return SectorContext(
        code="000935",
        name="信息",
        rank=rank,
        count=10,
        momentum=0.03,
        above_ma20=above,
        vol_ratio20=1.2,
    )


def make_snapshot(
    overrides: dict[str, float | bool | None] | None = None,
    *,
    close_raw: float = 10.0,
    close_qfq: float = 10.0,
    market_context: MarketContext | None = None,
    sector_context: SectorContext | None = None,
    kind: str = "stock",
) -> FactorSnapshot:
    factors: dict[str, float | bool | None] = {
        "weekly_bull": False,
        "weekly_above_ma20w": False,
        "weekly_bull_aligned": False,
        "weekly_above_ma10w": False,
        "jw": 50.0,
        "jw_up": False,
        "kw_cross_dw": False,
        "above_ma5": False,
        "j_up": False,
        "j_cross_k": False,
        "pos_250": 50.0,
        "dist_ma20_pct": 0.0,
        "vol_ratio20": 1.0,
        "vol_ma5_gt_ma20": False,
        "hhv_close_60_prev": None,
        "hhv_close_250_prev": None,
        "macd_hist": 0.0,
        "macd_hist_up": False,
        "ma20_up": False,
        "ma_bull_aligned": False,
        "rsi6": 50.0,
        "weekly_close": 10.0,
        "ma20w": 9.0,
        "ma10w": 9.5,
        "ma20w_up": True,
        "macd_histw": 0.0,
        "macd_histw_up": False,
        "above_ma10": True,
        "atr14": 0.3,
        "llv_low_10": 9.5,
        "llv_low_10_prev": 9.5,
        "ret1": 0.0,
        "vol_price_dir": 0.0,
    }
    if overrides:
        factors.update(overrides)
    return FactorSnapshot(
        symbol="600000",
        name="测试",
        as_of=DAY,
        close_qfq=close_qfq,
        close_raw=close_raw,
        price_scale=close_raw / close_qfq if close_qfq else 1.0,
        factors=factors,
        market=market_context or market(),
        sector=sector_context if sector_context is not None else sector(),
        kind=kind,
    )


def portfolio(
    *,
    positions: dict[str, PositionContext] | None = None,
    history: dict[str, SymbolHistory] | None = None,
    cash: float = 50_000.0,
    equity: float = 100_000.0,
) -> PortfolioContext:
    return PortfolioContext(
        equity=equity,
        cash=cash,
        holdings_count=len(positions or {}),
        positions=positions or {},
        history=history or {},
    )


def position(
    *,
    close_raw: float = 10.0,
    avg_cost: float = 10.0,
    pnl_pct: float = 0.0,
    holding_days: int = 30,
) -> PositionContext:
    return PositionContext(
        symbol="600000",
        name="测试",
        quantity=1000,
        avg_cost_raw=avg_cost,
        close_raw=close_raw,
        weight=0.1,
        pnl_pct=pnl_pct,
        holding_days=holding_days,
    )


def b1_overrides() -> dict[str, float | bool | None]:
    return {
        "weekly_bull": True,
        "weekly_bull_aligned": True,
        "weekly_above_ma10w": True,
        "jw": 15.0,
        "jw_up": True,
        "kw_cross_dw": True,
        "above_ma5": True,
        "j_up": True,
        "j_cross_k": True,
        "pos_250": 30.0,
        "dist_ma20_pct": -0.02,
        "vol_ratio20": 1.0,
        "vol_ma5_gt_ma20": True,
    }


# ------------------------------------------------------------------ 买点


def test_b1_full_alignment_is_actionable() -> None:
    signal = evaluate(make_snapshot(b1_overrides()), portfolio())

    assert signal is not None
    assert signal.kind is SignalKind.B1
    assert signal.actionable is True
    assert signal.action == "买入（试仓）"
    assert signal.confidence >= 65
    assert signal.suggested_quantity == 500
    assert signal.stop_price_raw is not None


def test_b2_breakout_is_actionable() -> None:
    signal = evaluate(
        make_snapshot(
            {
                "weekly_bull": True,
                "weekly_above_ma20w": True,
                "hhv_close_60_prev": 9.5,
                "vol_ratio20": 2.0,
                "macd_hist": 0.2,
                "macd_hist_up": True,
                "ma20_up": True,
                "ma_bull_aligned": True,
                "rsi6": 65.0,
                "pos_250": 70.0,
            }
        ),
        portfolio(),
    )

    assert signal is not None
    assert signal.kind is SignalKind.B2
    assert signal.actionable is True
    assert signal.action == "买入（标准仓）"
    assert signal.confidence >= 70
    assert signal.suggested_quantity is not None


def test_low_score_buy_is_watch_only() -> None:
    signal = evaluate(
        make_snapshot(
            {
                "weekly_bull": True,
                "jw": 33.0,
                "jw_pct": 25.0,
                "jw_up": False,
                "kw_cross_dw": True,
                "above_ma5": True,
                "j_up": True,
                "j_cross_k": True,
                "pos_250": 80.0,
                "dist_ma20_pct": 0.05,
                "vol_ratio20": 2.0,
                "vol_ma5_gt_ma20": True,
            },
            market_context=market("NEUTRAL"),
            sector_context=sector(rank=5),
        ),
        portfolio(),
    )

    assert signal is not None
    assert signal.actionable is False
    assert signal.action == "观察（等待确认）"
    assert 50 <= signal.confidence < 65


# ------------------------------------------------------------------ 卖点


def test_s1_weekly_break_clears_position() -> None:
    signal = evaluate(
        make_snapshot(
            {
                "weekly_close": 8.5,
                "ma20w": 9.0,
                "ma20w_up": False,
                "ma10w": 9.2,
                "jw": 85.0,
                "jw_up": False,
                "macd_hist": -0.2,
                "macd_hist_up": False,
                "above_ma10": False,
                "pos_250": 90.0,
                "vol_price_dir": -1.0,
                "vol_ratio20": 2.0,
            },
            sector_context=sector(rank=9, above=False),
        ),
        portfolio(positions={"600000": position(pnl_pct=0.15)}),
    )

    assert signal is not None
    assert signal.kind is SignalKind.S1
    assert signal.actionable is True
    assert signal.action == "清仓"
    assert signal.suggested_quantity == 1000


def test_s1_recent_buy_is_not_actionable() -> None:
    signal = evaluate(
        make_snapshot(
            {
                "jw": 85.0,
                "jw_up": False,
                "ma10w": 10.5,
                "macd_hist": -0.2,
                "macd_hist_up": False,
                "above_ma10": False,
                "pos_250": 90.0,
                "vol_price_dir": -1.0,
                "vol_ratio20": 2.0,
            },
            sector_context=sector(rank=9, above=False),
        ),
        portfolio(positions={"600000": position(holding_days=1)}),
    )

    assert signal is not None
    assert signal.kind is SignalKind.S1
    assert signal.actionable is False
    assert "刚买入" in signal.action


def test_s2_loss_stop_always_actionable() -> None:
    signal = evaluate(
        make_snapshot(
            close_raw=8.9,
            close_qfq=8.9,
            overrides={"ret1": -0.11, "vol_ratio20": 1.0},
        ),
        portfolio(
            positions={
                "600000": position(
                    close_raw=8.9, avg_cost=10.0, pnl_pct=-0.11, holding_days=1
                )
            }
        ),
    )

    assert signal is not None
    assert signal.kind is SignalKind.S2
    assert signal.actionable is True
    assert signal.confidence == 95
    assert signal.action == "清仓（风控）"
    assert signal.suggested_quantity == 1000


# ------------------------------------------------------------------ 仓位/历史否决


def test_losing_position_cannot_average_down() -> None:
    signal = evaluate(
        make_snapshot(b1_overrides()),
        portfolio(
            positions={
                "600000": position(avg_cost=10.5, pnl_pct=-0.0476, holding_days=30)
            }
        ),
    )

    assert signal is None


def test_cooldown_after_losing_sell_blocks_buy() -> None:
    history = {
        "600000": SymbolHistory(
            symbol="600000",
            last_sell_date=date(2026, 10, 20),
            last_sell_pnl=-500.0,
        )
    }
    signal = evaluate(make_snapshot(b1_overrides()), portfolio(history=history))

    assert signal is None


def test_panic_blocks_new_buy() -> None:
    signal = evaluate(
        make_snapshot(b1_overrides(), market_context=market("PANIC")),
        portfolio(),
    )

    assert signal is None


def test_risk_off_blocks_new_buy() -> None:
    signal = evaluate(
        make_snapshot(b1_overrides(), market_context=market("RISK_OFF")),
        portfolio(),
    )

    assert signal is None


def test_dynamic_thresholds_follow_regime_and_sector() -> None:
    params = SignalParams()

    assert params.buy_threshold("B1", "RISK_ON", "neutral") == 55
    assert params.buy_threshold("B1", "NEUTRAL", "neutral") == 65
    assert params.buy_threshold("B1", "RISK_ON", "strong") == 52
    assert params.buy_threshold("B2", "NEUTRAL", "weak") == 75
    assert params.buy_threshold("B1", "RISK_OFF", "neutral") == 100
    assert params.sell_threshold("RISK_ON", "neutral") == 70
    assert params.sell_threshold("NEUTRAL", "weak") == 55
    assert params.sell_threshold("RISK_OFF", "weak") == 45


def test_s1_intervenes_at_early_loss_in_weak_market() -> None:
    signal = evaluate(
        make_snapshot(
            {
                "jw": 85.0,
                "jw_pct": 92.0,
                "jw_up": False,
                "macd_hist": -0.2,
                "macd_hist_up": False,
                "above_ma10": False,
                "pos_250": 90.0,
                "vol_price_dir": -1.0,
                "vol_ratio20": 2.0,
                "llv_low_10_prev": 8.5,
            },
            market_context=market("RISK_OFF"),
            sector_context=sector(rank=9, above=False),
        ),
        portfolio(
            positions={
                "600000": position(
                    close_raw=9.4, avg_cost=10.0, pnl_pct=-0.06, holding_days=30
                )
            }
        ),
    )

    assert signal is not None
    assert signal.kind is SignalKind.S1
    assert signal.actionable is True
    assert signal.action == "减仓 50%"
    assert any("早期预警" in reason for reason in signal.reasons)
    assert signal.threshold < signal.confidence


def test_base_etf_has_no_swing_signal() -> None:
    signal = evaluate(
        make_snapshot(b1_overrides(), kind="base_etf"),
        portfolio(positions={"600000": position()}),
    )

    assert signal is None


def test_max_positions_blocks_new_buy() -> None:
    positions = {f"60000{i}": position() for i in range(1, 6)}
    signal = evaluate(make_snapshot(b1_overrides()), portfolio(positions=positions))

    assert signal is None
