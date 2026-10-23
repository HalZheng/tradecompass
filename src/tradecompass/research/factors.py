"""横截面候选因子：A 股文献/社区里反复验证过的量价与估值因子 + 本项目 B1/B2 的连续化版本。

``expected`` 是文献给出的方向（+1 越大越好，−1 越小越好），只用于报告对照，
不参与计算；实际方向以样本外 IC 为准。

参考来源：
- Liu, Stambaugh & Yuan (2019) *Size and Value in China*：EP 是最好的价值指标；
  异常换手（1 月 / 1 年）是 A 股情绪因子。
- 国内券商金工（华泰/东方/方正等）多因子系列：短期反转、低换手、低波动、
  最大日收益（MAX）、量价相关性在 A 股长期显著。
- Microsoft Qlib Alpha158（CSI300 基准）：CORR/ROC/STD 等量价表达式。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradecompass.research.panel import Panel


@dataclass(frozen=True, slots=True)
class FactorSpec:
    name: str
    label: str
    group: str
    expected: int
    compute: Callable[[Panel], pd.DataFrame]


def _safe_inverse(frame: pd.DataFrame) -> pd.DataFrame:
    return (1.0 / frame.where(frame != 0)).replace([np.inf, -np.inf], np.nan)


def log_frame(frame: pd.DataFrame, *, plus_one: bool = False) -> pd.DataFrame:
    values = frame.to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log1p(values) if plus_one else np.log(np.where(values > 0, values, np.nan))
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _ret(panel: Panel, n: int) -> pd.DataFrame:
    return panel.adj_close / panel.adj_close.shift(n) - 1.0


def _turn(panel: Panel) -> pd.DataFrame:
    return panel.field("turn").where(panel.trading)


def _volume(panel: Panel) -> pd.DataFrame:
    return panel.field("volume").where(panel.trading)


def ewm_rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return 100.0 - 100.0 / (1.0 + gain / loss.where(loss > 0))


def macd_hist(close: pd.DataFrame) -> pd.DataFrame:
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return 2.0 * (dif - dif.ewm(span=9, adjust=False).mean())


def position(panel: Panel, n: int) -> pd.DataFrame:
    high = panel.adj_high.rolling(n, min_periods=n).max()
    low = panel.adj_low.rolling(n, min_periods=n).min()
    span = (high - low).where(high > low)
    return (panel.adj_close - low) / span * 100.0


def vol_ratio20(panel: Panel) -> pd.DataFrame:
    volume = _volume(panel)
    return volume / volume.rolling(20, min_periods=15).mean()


def _abn_turn(panel: Panel) -> pd.DataFrame:
    turn = _turn(panel)
    return turn.rolling(20, min_periods=15).mean() / turn.rolling(250, min_periods=200).mean()


def _amihud(panel: Panel) -> pd.DataFrame:
    amount = panel.field("amount").where(panel.trading)
    illiq = panel.ret.abs() / amount.where(amount > 0) * 1e9
    return illiq.rolling(20, min_periods=15).mean()


def _pv_corr(panel: Panel) -> pd.DataFrame:
    return panel.adj_close.rolling(20, min_periods=15).corr(
        log_frame(_volume(panel), plus_one=True), pairwise=False
    )


def _roe(panel: Panel) -> pd.DataFrame:
    """ROE(TTM) ≈ PB / PE；亏损公司 PE 为负，ROE 也为负。"""
    pe = panel.field("pe_ttm")
    pb = panel.field("pb_mrq")
    return (pb / pe.where(pe != 0)).where(pb > 0)


def _ivol(panel: Panel, n: int) -> pd.DataFrame:
    """特异波动的近似：相对全样本等权收益的超额收益波动。"""
    market = panel.ret.mean(axis=1)
    return panel.ret.sub(market, axis=0).rolling(n, min_periods=int(n * 0.75)).std()


FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec("rev5", "5日收益", "反转", -1, lambda p: _ret(p, 5)),
    FactorSpec("rev20", "20日收益", "反转", -1, lambda p: _ret(p, 20)),
    FactorSpec(
        "mom_12_1", "12-1月动量", "动量", 1,
        lambda p: p.adj_close.shift(20) / p.adj_close.shift(250) - 1.0,
    ),
    FactorSpec(
        "vol20", "20日波动", "波动", -1, lambda p: p.ret.rolling(20, min_periods=15).std()
    ),
    FactorSpec(
        "max20", "20日最大日涨", "波动", -1, lambda p: p.ret.rolling(20, min_periods=15).max()
    ),
    FactorSpec(
        "turn20", "20日换手", "流动性", -1,
        lambda p: _turn(p).rolling(20, min_periods=15).mean(),
    ),
    FactorSpec("abn_turn", "异常换手 20/250", "流动性", -1, _abn_turn),
    FactorSpec("amihud20", "Amihud 非流动性", "流动性", 1, _amihud),
    FactorSpec("size", "流通市值(对数)", "规模", -1, lambda p: log_frame(p.float_mcap)),
    FactorSpec("ep", "EP=1/PE_TTM", "价值", 1, lambda p: _safe_inverse(p.field("pe_ttm"))),
    FactorSpec("bp", "BP=1/PB", "价值", 1, lambda p: _safe_inverse(p.field("pb_mrq"))),
    FactorSpec("sp", "SP=1/PS_TTM", "价值", 1, lambda p: _safe_inverse(p.field("ps_ttm"))),
    FactorSpec("cfp", "CFP=1/PCF_TTM", "价值", 1, lambda p: _safe_inverse(p.field("pcf_ttm"))),
    FactorSpec("pv_corr20", "价量相关20", "量价", -1, _pv_corr),
    FactorSpec("roe", "ROE≈PB/PE", "质量", 1, _roe),
    FactorSpec(
        "vol60", "60日波动", "波动", -1, lambda p: p.ret.rolling(60, min_periods=45).std()
    ),
    FactorSpec("ivol20", "20日特异波动", "波动", -1, lambda p: _ivol(p, 20)),
    FactorSpec("ivol60", "60日特异波动", "波动", -1, lambda p: _ivol(p, 60)),
    FactorSpec(
        "turn60", "60日换手", "流动性", -1,
        lambda p: _turn(p).rolling(60, min_periods=45).mean(),
    ),
    FactorSpec("rev60", "60日收益", "反转", -1, lambda p: _ret(p, 60)),
    FactorSpec("pos250", "250日价格位置", "技术(现系统)", 0, lambda p: position(p, 250)),
    FactorSpec(
        "bias20", "收盘/MA20−1", "技术(现系统)", 0,
        lambda p: p.adj_close / p.adj_close.rolling(20, min_periods=20).mean() - 1.0,
    ),
    FactorSpec("rsi6", "RSI6", "技术(现系统)", 0, lambda p: ewm_rsi(p.adj_close, 6)),
    FactorSpec("vol_ratio20", "量比20", "技术(现系统)", 0, vol_ratio20),
    FactorSpec(
        "breakout60", "收盘/前60日最高−1", "技术(现系统)", 0,
        lambda p: p.adj_close / p.adj_close.shift(1).rolling(60, min_periods=60).max() - 1.0,
    ),
)


def factor_by_name(name: str) -> FactorSpec:
    for spec in FACTORS:
        if spec.name == name:
            return spec
    raise KeyError(name)
