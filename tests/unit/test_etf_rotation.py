"""ETF 轮动：打分无前视、趋势破位卖出、ETF 费率（无印花税）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradecompass.config.models import CostsConfig
from tradecompass.research.etf_rotation import EtfParams, EtfRotation, slope_r2_score
from tradecompass.research.etfs import EtfInfo

DAYS = pd.bdate_range("2021-01-01", periods=400)


def etf(path: np.ndarray) -> pd.DataFrame:
    close = 1.0 * np.cumprod(1.0 + path)
    preclose = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open": preclose,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "preclose": preclose,
            "volume": 1e6,
            "amount": 1e6 * close,
            "pct_chg": (close / preclose - 1.0) * 100.0,
        },
        index=DAYS,
    )


def test_smooth_uptrend_scores_higher_than_noisy() -> None:
    rng = np.random.default_rng(1)
    smooth = pd.Series(np.exp(np.linspace(0, 0.2, 60)))
    noisy = smooth * np.exp(rng.normal(0, 0.03, 60))
    score = slope_r2_score(pd.DataFrame({"a": smooth, "b": noisy}), 25)
    assert score["a"].iloc[-1] > score["b"].iloc[-1] > -1
    changed = pd.DataFrame({"a": smooth.copy(), "b": noisy.copy()})
    changed.iloc[50:] *= 2.0
    assert slope_r2_score(changed, 25).iloc[:50].equals(score.iloc[:50])


def test_rotation_buys_trend_and_exits_on_break() -> None:
    up = np.full(len(DAYS), 0.003)
    up[330:] = -0.02
    flat = np.random.default_rng(2).normal(0.0, 0.01, len(DAYS))
    bars = {"sh.512880": etf(up), "sh.512800": etf(flat)}
    infos = {
        "sh.512880": EtfInfo("sh.512880", "证券", "行业"),
        "sh.512800": EtfInfo("sh.512800", "银行", "行业"),
    }
    model = EtfRotation(bars, infos, DAYS, CostsConfig(), EtfParams(top=1))
    equity, trades = model.run(DAYS[260].date(), DAYS[-1].date(), capital=100_000)
    buys = [t for t in trades if t.code == "sh.512880" and t.side == "buy"]
    sells = [t for t in trades if t.code == "sh.512880" and t.side == "sell"]
    assert buys and buys[0].day == DAYS[261]
    assert sells and DAYS[330] < sells[0].day <= DAYS[340]
    assert sells[0].reason.startswith("跌破")
    sell = sells[0]
    # 场内基金卖出无印花税：费用只有佣金
    assert sell.fee <= sell.price * sell.shares * 0.0003 + 1e-9
    assert all(t.shares % 100 == 0 for t in trades)
    assert len(equity) == 400 - 260


def test_share_split_does_not_look_like_a_loss() -> None:
    bars = etf(np.full(len(DAYS), 0.002))
    split = DAYS[300]
    # 1 拆 2：原始价归半，交易所前收同步调整
    for column in ("open", "high", "low", "close"):
        bars.loc[bars.index >= split, column] /= 2.0
    bars.loc[bars.index > split, "preclose"] /= 2.0
    bars.loc[split, "preclose"] = bars.loc[split, "preclose"] / 2.0
    infos = {"sh.512880": EtfInfo("sh.512880", "证券", "行业")}
    model = EtfRotation({"sh.512880": bars}, infos, DAYS, CostsConfig(), EtfParams(top=1))
    equity, _ = model.run(DAYS[260].date(), DAYS[-1].date(), capital=100_000)
    assert equity.loc[split] > equity.loc[DAYS[299]] * 0.99
    assert equity.iloc[-1] > equity.loc[DAYS[262]]
