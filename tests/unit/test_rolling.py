"""分板块滚动决策引擎：无未来函数、次日开盘成交、一手约束、额度约束。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradecompass.config.models import CostsConfig
from tradecompass.research.panel import Panel, build_panel
from tradecompass.research.rolling import (
    RollingEngine,
    SleeveRule,
    group_indices,
    run_rolling,
    trend_tilt,
)
from tradecompass.research.sleeves import sleeve_of

DAYS = pd.bdate_range("2020-01-01", periods=420)
RULES = (
    SleeveRule("防御", ("大金融",), ("vol60", "ep"), period=20, buffer=2.0, min_hold=5),
    SleeveRule("科技成长", ("科技成长",), ("rev20", "ivol20"), period=10, buffer=1.5, min_hold=5),
)


def stock(seed: int, shock_from: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.02, len(DAYS))
    shock = rng.normal(-0.01, 0.05, len(DAYS))
    volume = rng.uniform(1e6, 2e6, len(DAYS))
    noise = rng.normal(0.0, 0.002, len(DAYS))
    if shock_from is not None:
        rets[shock_from:] = shock[shock_from:]
    close = 10.0 * np.cumprod(1.0 + rets)
    preclose = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open": preclose * (1.0 + noise),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "preclose": preclose,
            "volume": volume,
            "amount": volume * close,
            "turn": volume / 1e8 * 100.0,
            "tradestatus": 1.0,
            "is_st": 0.0,
            "pe_ttm": 8.0 + seed % 7,
            "pb_mrq": 1.0,
            "ps_ttm": 1.0,
            "pcf_ttm": 1.0,
        },
        index=pd.DatetimeIndex(DAYS, name="date"),
    )


def universe(shock_from: int | None = None) -> tuple[Panel, dict[str, str]]:
    codes = [f"sh.{600000 + i}" for i in range(40)]
    bars = {c: stock(i, shock_from) for i, c in enumerate(codes)}
    members = pd.DataFrame({"date": DAYS[0], "code": codes, "index": "hs300"})
    industry = {
        c: ("J66货币金融服务" if i % 2 else "I65软件和信息技术服务业")
        for i, c in enumerate(codes)
    }
    return build_panel(bars, members), industry


def engine(panel: Panel, industry: dict[str, str], quota: pd.Series | None = None) -> RollingEngine:
    return RollingEngine(panel, industry, CostsConfig(), rules=RULES, slots=6, quota=quota)


def test_sleeve_mapping() -> None:
    assert sleeve_of("C39计算机、通信和其他电子设备制造业") == "科技成长"
    assert sleeve_of("J66货币金融服务") == "大金融"
    assert sleeve_of("C27医药制造业") == "医药"
    assert sleeve_of("C26化学原料和化学制品制造业") == "周期资源"
    assert sleeve_of(None) == "其他"


def test_orders_fill_next_open_in_lots() -> None:
    panel, industry = universe()
    start, end = DAYS[260].date(), DAYS[-1].date()
    result = run_rolling(engine(panel, industry), start, end, capital=100_000)
    buys = [t for t in result.trades if t.side == "buy"]
    assert buys
    assert buys[0].day == DAYS[261]
    assert all(t.shares % 100 == 0 and t.shares > 0 for t in buys)
    assert result.final_state.cash >= -1e-6
    assert len(result.equity) == 420 - 260


def test_future_data_does_not_change_past_decisions() -> None:
    cut = 330
    base_panel, industry = universe()
    shocked_panel, _ = universe(shock_from=cut + 1)
    start, end = DAYS[260].date(), DAYS[-1].date()
    base = run_rolling(engine(base_panel, industry), start, end, capital=100_000)
    shocked = run_rolling(engine(shocked_panel, industry), start, end, capital=100_000)

    def upto(trades: list) -> list[tuple]:
        return [(t.day, t.code, t.side, t.shares) for t in trades if t.day <= DAYS[cut]]

    assert upto(base.trades) == upto(shocked.trades)
    assert base.equity.loc[: DAYS[cut]].equals(shocked.equity.loc[: DAYS[cut]])


def test_zero_quota_never_buys() -> None:
    panel, industry = universe()
    quota = pd.Series(0.0, index=panel.dates)
    result = run_rolling(
        engine(panel, industry, quota), DAYS[260].date(), DAYS[-1].date(), capital=100_000
    )
    assert not [t for t in result.trades if t.side == "buy"]


def test_trend_tilt_is_causal_and_shifts_slots() -> None:
    panel, industry = universe()
    eng = engine(panel, industry)
    idx = group_indices(eng)
    tilt = trend_tilt(idx, ma=60)
    assert set(tilt.stack().unique()) <= {0.7, 1.3}
    # 改后半段指数不应影响前半段倍数
    shifted = idx.copy()
    shifted.iloc[300:] *= 0.5
    assert trend_tilt(shifted, ma=60).iloc[:300].equals(tilt.iloc[:300])
    eng.set_tilt(pd.DataFrame({"防御": 1.5, "科技成长": 0.5}, index=panel.dates))
    slots = eng.rule_slots(300)
    assert slots["防御"] > slots["科技成长"]
    assert sum(slots.values()) == 6
