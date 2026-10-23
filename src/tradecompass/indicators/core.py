"""技术指标纯函数（SPEC §9.2）。

全部输入 ``pd.Series``、输出 ``pd.Series``，不读写外部状态、不读当前时间。
口径对齐通达信：KDJ 用 ``SMA(X, N, M)`` 递推，不是简单移动平均。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KDJ_SEED = 50.0


def ma(close: pd.Series, n: int) -> pd.Series:
    """简单移动平均，前 n−1 个为 NaN。"""
    if n <= 0:
        raise ValueError(f"n 必须为正: {n}")
    return close.astype(float).rolling(window=n, min_periods=n).mean()


def tdx_sma(x: pd.Series, n: int, m: int) -> pd.Series:
    """通达信 ``SMA(X, N, M)``：``y[t] = (m·x[t] + (n−m)·y[t−1]) / n``。

    首个有效值的前值取 50（用于 KDJ）。NaN 段沿用前值。
    """
    if n <= 0:
        raise ValueError(f"n 必须为正: {n}")
    values = x.to_numpy(dtype="float64")
    out = np.full(values.shape, np.nan, dtype="float64")
    prev = KDJ_SEED
    started = False
    for i, value in enumerate(values):
        if np.isnan(value):
            if started:
                out[i] = prev
            continue
        prev = (m * value + (n - m) * prev) / n
        started = True
        out[i] = prev
    return pd.Series(out, index=x.index, name=x.name)


def ema(x: pd.Series, n: int) -> pd.Series:
    """指数移动平均（``adjust=False``，与通达信/东财的 EMA 口径一致）。"""
    if n <= 0:
        raise ValueError(f"n 必须为正: {n}")
    return x.astype(float).ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """相对强弱指标，Wilder 平滑。

    价格完全走平时返回 50（而不是 0 或 100），这样它不会在横盘时给出
    "极度超卖/超买"的假信号。
    """
    if n <= 0:
        raise ValueError(f"n 必须为正: {n}")
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    out = out.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    out = out.mask((avg_gain == 0.0) & (avg_loss == 0.0), 50.0)
    return out.astype(float)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD，返回列 ``dif``、``dea``、``macd_hist``。

    柱体按国内软件口径 ``2 × (DIF − DEA)``。
    """
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError(f"MACD 周期必须为正: {fast}, {slow}, {signal}")
    if fast >= slow:
        raise ValueError(f"MACD fast 必须小于 slow: {fast}, {slow}")
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    hist = 2.0 * (dif - dea)
    return pd.DataFrame({"dif": dif, "dea": dea, "macd_hist": hist}, index=close.index)


def hhv(x: pd.Series, n: int) -> pd.Series:
    """滚动最高，含当期。"""
    return x.astype(float).rolling(window=n, min_periods=n).max()


def llv(x: pd.Series, n: int) -> pd.Series:
    """滚动最低，含当期。"""
    return x.astype(float).rolling(window=n, min_periods=n).min()


def ref(x: pd.Series, n: int) -> pd.Series:
    return x.astype(float).shift(n)


def slope_up(x: pd.Series, lookback: int = 5) -> pd.Series:
    """当前值是否高于 ``lookback`` 个周期之前。"""
    shifted = x.astype(float).shift(lookback)
    return (x.astype(float) > shifted).fillna(False)


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> pd.DataFrame:
    """KDJ，返回列 ``k``、``d``、``j``。

    ``RSV = (C − LLV(L,n)) / (HHV(H,n) − LLV(L,n)) × 100``，分母为 0 时取 50。
    """
    lowest = llv(low, n)
    highest = hhv(high, n)
    span = highest - lowest
    rsv = ((close.astype(float) - lowest) / span.where(span != 0)) * 100.0
    rsv = rsv.where(span != 0, 50.0)
    k = tdx_sma(rsv, m1, 1)
    d = tdx_sma(k, m2, 1)
    j = 3.0 * k - 2.0 * d
    return pd.DataFrame({"k": k, "d": d, "j": j}, index=close.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """真实波幅的 Wilder 平滑；首值为前 n 个 TR 的均值。"""
    if n <= 0:
        raise ValueError(f"n 必须为正: {n}")
    h = high.astype(float)
    lw = low.astype(float)
    c = close.astype(float)
    prev_close = c.shift(1)
    tr = pd.concat([h - lw, (h - prev_close).abs(), (lw - prev_close).abs()], axis=1).max(axis=1)
    out = pd.Series(np.nan, index=tr.index, dtype="float64")
    if len(tr) < n:
        return out
    values = tr.to_numpy(dtype="float64")
    result = out.to_numpy(copy=True)
    first = float(np.nanmean(values[:n]))
    result[n - 1] = first
    prev = first
    for i in range(n, len(values)):
        if np.isnan(values[i]):
            result[i] = prev
            continue
        prev = (prev * (n - 1) + values[i]) / n
        result[i] = prev
    return pd.Series(result, index=tr.index, name="atr")
