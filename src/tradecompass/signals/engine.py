"""B1/B2/S1/S2 信号引擎。

判断顺序：**S2 风控 → S1 趋势卖点 → B2 突破 → B1 回调**。
同一标的每个交易日只返回一条最需要被看到的信号；买点先过硬否决，
再过打分阈值，低于阈值但仍在观察区（≥50 分）的会以 ``actionable=False`` 返回。
"""

from __future__ import annotations

from tradecompass.factors.engine import FactorSnapshot
from tradecompass.signals.types import (
    PortfolioContext,
    PositionContext,
    Signal,
    SignalKind,
    SignalParams,
)


def evaluate(
    snapshot: FactorSnapshot,
    portfolio: PortfolioContext,
    params: SignalParams | None = None,
) -> Signal | None:
    """对一只标的给出当日信号；没有信号返回 ``None``。"""
    resolved = params or SignalParams()
    # 底仓 ETF 只做再平衡，不参与波段买卖信号（避免把宽基 ETF 的正常回调当成止损）。
    if snapshot.kind == "base_etf":
        return None
    position = portfolio.position_of(snapshot.symbol)

    if position is not None:
        s2 = _s2(snapshot, position, resolved)
        if s2 is not None:
            return s2
        s1 = _s1(snapshot, position, resolved)
        if s1 is not None:
            return s1

    block = _buy_block(snapshot, portfolio, position, resolved)
    if block is not None:
        return None

    b2 = _b2(snapshot, portfolio, position, resolved)
    b1 = _b1(snapshot, portfolio, position, resolved)
    if b2 is not None and b1 is not None:
        return b2 if b2.confidence >= b1.confidence else b1
    return b2 or b1


# --------------------------------------------------------------------------- 公共


def stop_levels(
    snapshot: FactorSnapshot, params: SignalParams
) -> tuple[float | None, float | None]:
    """初始止损 = max(收盘 − N×ATR, 前 10 日最低)，全部换算成不复权实际价。"""
    close = snapshot.close_raw
    atr = snapshot.num("atr14")
    prev_low = snapshot.num("llv_low_10")
    atr_raw = atr * snapshot.price_scale if atr is not None else None
    low_raw = prev_low * snapshot.price_scale if prev_low is not None else None

    candidates: list[float] = []
    if atr_raw is not None and atr_raw > 0:
        candidates.append(close - params.atr_mult * atr_raw)
    if low_raw is not None and 0 < low_raw < close:
        candidates.append(low_raw)
    if not candidates:
        return None, atr_raw
    return max(candidates), atr_raw


def _buy_block(
    snapshot: FactorSnapshot,
    portfolio: PortfolioContext,
    position: PositionContext | None,
    params: SignalParams,
) -> str | None:
    """买入的硬否决。返回原因字符串；``None`` 表示允许。"""
    if not snapshot.market.allows_new_buy:
        return f"大盘 {snapshot.market.state}，额度 {snapshot.market.quota:.0%}"

    sector = snapshot.sector
    if sector is not None and sector.above_ma20 is False:
        return f"行业 {sector.name or sector.code} 在 MA20 下方"

    if position is None:
        if portfolio.holdings_count >= params.max_positions:
            return f"持仓数已达上限 {params.max_positions}"
        if 0 < portfolio.cash < params.min_trade_amount:
            return "可用现金不足"
    else:
        if position.pnl_pct < 0:
            return "禁止亏损补仓"
        if position.weight >= params.max_weight:
            return f"单票权重 {position.weight:.0%} 已达上限"
        if (
            position.holding_days is not None
            and position.holding_days < params.min_hold_days_before_add
        ):
            return "刚买入，暂不加仓"

    history = portfolio.history_of(snapshot.symbol)
    if history is not None and history.last_sell_date is not None:
        days = (snapshot.as_of - history.last_sell_date).days
        if (
            history.last_sell_pnl is not None
            and history.last_sell_pnl < 0
            and days <= params.cooldown_days
        ):
            return f"止损冷却期（{days} 天）"
        if days <= params.profit_sell_cooldown_days:
            return f"上次卖出仅 {days} 天，避免追回"
    return None


def _buy_quantity(
    snapshot: FactorSnapshot,
    portfolio: PortfolioContext,
    params: SignalParams,
    *,
    stop_raw: float | None,
    half: bool,
) -> int | None:
    """按风险预算与单槽资金取较小者，再按 100 股取整。"""
    price = snapshot.close_raw
    if portfolio.equity <= 0 or price <= 0 or stop_raw is None or stop_raw >= price:
        return None

    risk_per_share = price - stop_raw
    risk_budget = portfolio.equity * params.risk_per_trade
    qty_risk = int(risk_budget / risk_per_share // 100) * 100

    slot = portfolio.equity * params.swing_ratio / max(params.max_positions, 1)
    qty_slot = int(slot / price // 100) * 100
    qty = min(qty_risk, qty_slot)

    if half:
        qty = int(qty // 2 // 100) * 100
    if portfolio.cash > 0:
        qty = min(qty, int(portfolio.cash / price // 100) * 100)
    if qty < 100 or qty * price < params.min_trade_amount:
        return None
    return qty


def _sell_quantity(position: PositionContext, *, half: bool) -> int:
    if half:
        qty = int(position.quantity // 2 // 100) * 100
        return qty if qty >= 100 else position.quantity
    return position.quantity


def _sector_strength(snapshot: FactorSnapshot) -> str:
    return snapshot.sector.strength if snapshot.sector is not None else "unknown"


def _position_note(position: PositionContext | None) -> str:
    if position is None:
        return "未持仓"
    return (
        f"持仓 {position.quantity} 股，成本 {position.avg_cost_raw:.2f}，"
        f"浮盈亏 {position.pnl_pct:+.1%}"
    )


def _metrics(snapshot: FactorSnapshot) -> dict[str, float | None]:
    sector_rank = snapshot.sector.rank if snapshot.sector is not None else None
    return {
        "jw": snapshot.num("jw"),
        "j": snapshot.num("j"),
        "rsi6": snapshot.num("rsi6"),
        "macd_hist": snapshot.num("macd_hist"),
        "vol_ratio20": snapshot.num("vol_ratio20"),
        "pos_250": snapshot.num("pos_250"),
        "ret5": snapshot.num("ret5"),
        "sector_rank": float(sector_rank) if sector_rank is not None else None,
    }


# --------------------------------------------------------------------------- B1


def _b1(
    snapshot: FactorSnapshot,
    portfolio: PortfolioContext,
    position: PositionContext | None,
    params: SignalParams,
) -> Signal | None:
    # 只保留两个绝对否决：位置极端（>95%）+ 止损距离过宽。
    # 市场/行业否决已经在 _buy_block 里；其余因子全部进打分，不再做 AND 门。
    pos250 = snapshot.num("pos_250")
    if pos250 is not None and pos250 >= 95:
        return None

    stop, _ = stop_levels(snapshot, params)
    if stop is None or stop >= snapshot.close_raw:
        return None
    if (snapshot.close_raw - stop) / snapshot.close_raw > params.max_stop_pct:
        return None

    score, reasons, warnings = _b1_score(snapshot)
    if score < params.watch_confidence:
        return None
    threshold = params.buy_threshold(
        "B1", snapshot.market.state, _sector_strength(snapshot)
    )
    actionable = score >= threshold
    action = ("买入（试仓）" if position is None else "加仓（试仓）") if actionable else (
        "观察（等待确认）"
    )
    return Signal(
        kind=SignalKind.B1,
        symbol=snapshot.symbol,
        name=snapshot.name,
        as_of=snapshot.as_of,
        confidence=score,
        actionable=actionable,
        action=action,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        trigger_price_raw=snapshot.close_raw,
        stop_price_raw=stop,
        suggested_quantity=(
            _buy_quantity(snapshot, portfolio, params, stop_raw=stop, half=True)
            if actionable
            else None
        ),
        position_note=_position_note(position),
        metrics=_metrics(snapshot),
        threshold=threshold,
    )


def _b1_score(snapshot: FactorSnapshot) -> tuple[int, list[str], list[str]]:
    score = 0
    reasons: list[str] = []
    warnings: list[str] = []
    jw = snapshot.num("jw")
    jw_pct = snapshot.num("jw_pct")

    # 周线趋势：25 分；周线空头不直接否决，但扣 12 分（软惩罚）。
    trend = 0
    if snapshot.flag("weekly_bull") is True:
        trend += 15
        reasons.append("周线多头（收盘>MA20w，MA20w向上）")
    elif snapshot.flag("weekly_above_ma20w") is True:
        trend += 5
        reasons.append("周收盘站上周MA20")
    if snapshot.flag("weekly_bull_aligned") is True:
        trend += 5
        reasons.append("周线多头排列")
    if snapshot.flag("weekly_above_ma10w") is True:
        trend += 3
        reasons.append("周收盘站上周MA10")
    if (
        snapshot.flag("weekly_above_ma20w") is False
        and snapshot.flag("ma20w_up") is False
    ):
        trend -= 12
        warnings.append("周线空头（降权）")
    score += max(0, min(trend, 25))

    # 周线超卖：25 分。用“该票自身历史分位”+ 绝对 J 值双重口径。
    oversold = 0
    if jw_pct is not None:
        if jw_pct <= 10.0:
            oversold += 12
            reasons.append(f"周J历史分位 {jw_pct:.0f}%（极低）")
        elif jw_pct <= 20.0:
            oversold += 9
            reasons.append(f"周J历史分位 {jw_pct:.0f}%")
        elif jw_pct <= 30.0:
            oversold += 6
        elif jw_pct <= 45.0:
            oversold += 3
    if jw is not None:
        if jw <= 20.0:
            oversold += 5
        elif jw <= 30.0:
            oversold += 3
        reasons.append(f"周J={jw:.0f}")
    if snapshot.flag("jw_up") is True:
        oversold += 4
        reasons.append("周J回升")
    if snapshot.flag("kw_cross_dw") is True:
        oversold += 4
        reasons.append("周KD上穿")
    score += max(0, min(oversold, 25))

    # 日线企稳：20 分。
    if snapshot.flag("above_ma5") is True:
        score += 8
        reasons.append("日线站上MA5")
    if snapshot.flag("j_up") is True:
        score += 6
        reasons.append("日J回升")
    if snapshot.flag("j_cross_k") is True:
        score += 6
        reasons.append("日KD上穿")

    # 位置：10 分。
    pos250 = snapshot.num("pos_250")
    if pos250 is not None:
        if pos250 < 40.0:
            score += 8
            reasons.append(f"250日位置 {pos250:.0f}%（低位）")
        elif pos250 < 60.0:
            score += 6
            reasons.append(f"250日位置 {pos250:.0f}%")
        elif pos250 < 75.0:
            score += 3
        elif pos250 >= 90.0:
            score -= 5
            warnings.append("250日位置过高")
    dist = snapshot.num("dist_ma20_pct")
    if dist is not None and -0.06 <= dist <= 0.01:
        score += 2
        reasons.append("贴近MA20")

    # 量能：10 分。
    volume = snapshot.num("vol_ratio20")
    if volume is not None:
        if 0.6 <= volume <= 1.8:
            score += 6
            reasons.append(f"量比20={volume:.1f}（温和）")
        elif volume > 2.5:
            warnings.append("放量过大")
    if snapshot.flag("vol_ma5_gt_ma20") is True:
        score += 4
        reasons.append("5日均量高于20日均量")

    score += _sector_market_score(snapshot, reasons, warnings, strong=6, neutral=3, attack=4)
    return max(0, min(score, 100)), reasons, warnings


# --------------------------------------------------------------------------- B2


def _b2(
    snapshot: FactorSnapshot,
    portfolio: PortfolioContext,
    position: PositionContext | None,
    params: SignalParams,
) -> Signal | None:
    close = snapshot.close_raw
    hhv60 = snapshot.num("hhv_close_60_prev")
    if hhv60 is None or close <= hhv60 * snapshot.price_scale:
        return None
    volume = snapshot.num("vol_ratio20")
    if volume is None or volume < params.b2_volume_mult:
        return None
    hist = snapshot.num("macd_hist")
    if hist is None or hist <= 0:
        return None
    rsi6 = snapshot.num("rsi6")
    if rsi6 is not None and rsi6 > params.b2_rsi_max:
        return None
    weekly_ok = snapshot.flag("weekly_bull") is True or (
        snapshot.flag("weekly_above_ma20w") is True
    )
    if not weekly_ok:
        return None
    pos250 = snapshot.num("pos_250")
    if pos250 is not None and pos250 >= 95:
        return None

    stop, _ = stop_levels(snapshot, params)
    if stop is None or stop >= close:
        return None
    if (close - stop) / close > params.max_stop_pct:
        return None

    score, reasons, warnings = _b2_score(snapshot, params)
    if score < params.watch_confidence:
        return None
    threshold = params.buy_threshold(
        "B2", snapshot.market.state, _sector_strength(snapshot)
    )
    actionable = score >= threshold
    action = ("买入（标准仓）" if position is None else "加仓（标准仓）") if actionable else (
        "观察（等待确认）"
    )
    return Signal(
        kind=SignalKind.B2,
        symbol=snapshot.symbol,
        name=snapshot.name,
        as_of=snapshot.as_of,
        confidence=score,
        actionable=actionable,
        action=action,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        trigger_price_raw=close,
        stop_price_raw=stop,
        suggested_quantity=(
            _buy_quantity(snapshot, portfolio, params, stop_raw=stop, half=False)
            if actionable
            else None
        ),
        position_note=_position_note(position),
        metrics=_metrics(snapshot),
        threshold=threshold,
    )


def _b2_score(
    snapshot: FactorSnapshot, params: SignalParams
) -> tuple[int, list[str], list[str]]:
    score = 0
    reasons: list[str] = []
    warnings: list[str] = []

    close = snapshot.close_raw
    scale = snapshot.price_scale
    hhv60 = snapshot.num("hhv_close_60_prev")
    if hhv60 is not None:
        high60 = hhv60 * scale
        if close > high60:
            score += 20
            reasons.append("收盘突破前60日最高")
        elif close >= high60 * 0.98:
            score += 12
            reasons.append("距前60日最高不足 2%")
        elif close >= high60 * 0.95:
            score += 6
            reasons.append("距前60日最高不足 5%")
    hhv250 = snapshot.num("hhv_close_250_prev")
    if hhv250 is not None and close > hhv250 * scale:
        score += 10
        reasons.append("同时突破前250日最高")

    volume = snapshot.num("vol_ratio20")
    if volume is not None:
        if volume > params.b2_extreme_volume:
            score += 8
            warnings.append(f"量比{volume:.1f} 极端放量，可能衰竭")
        elif volume >= 2.0:
            score += 25
        elif volume >= 1.7:
            score += 21
        elif volume >= 1.5:
            score += 18
        elif volume >= 1.2:
            score += 10
        reasons.append(f"量比20={volume:.1f}")

    if snapshot.flag("weekly_bull") is True:
        score += 10
        reasons.append("周线多头")
    if snapshot.flag("weekly_above_ma20w") is True:
        score += 4
    if snapshot.flag("ma20_up") is True:
        score += 3
        reasons.append("日MA20向上")
    if snapshot.flag("ma_bull_aligned") is True:
        score += 3
        reasons.append("日均线多头排列")
    if (
        snapshot.flag("weekly_above_ma20w") is False
        and snapshot.flag("ma20w_up") is False
    ):
        score -= 8
        warnings.append("周线空头（降权）")

    hist = snapshot.num("macd_hist")
    if hist is not None and hist > 0:
        score += 8
    if snapshot.flag("macd_hist_up") is True:
        score += 4
        reasons.append("MACD柱走高")
    rsi6 = snapshot.num("rsi6")
    if rsi6 is not None and 55.0 <= rsi6 <= 85.0:
        score += 3
        reasons.append(f"RSI6={rsi6:.0f}")
    elif rsi6 is not None and rsi6 > 85.0:
        warnings.append("RSI6 超买")

    # 250 日位置：50–70% 是回测里的甜点区，70–85% 最差。
    pos250 = snapshot.num("pos_250")
    if pos250 is not None:
        if 50.0 <= pos250 < 70.0:
            score += 8
            reasons.append(f"250日位置 {pos250:.0f}%（甜点区）")
        elif 70.0 <= pos250 < 85.0:
            score -= 8
            warnings.append(f"250日位置 {pos250:.0f}%（易追高）")
        elif pos250 >= 85.0:
            score -= 3
            warnings.append(f"250日位置 {pos250:.0f}%（偏高）")

    # 行业 strong 在 B2 里反而是拖累（D034），把加分降下来。
    score += _sector_market_score(snapshot, reasons, warnings, strong=2, neutral=4, attack=4)
    return max(0, min(score, 100)), reasons, warnings


# --------------------------------------------------------------------------- S1


def _s1(
    snapshot: FactorSnapshot,
    position: PositionContext,
    params: SignalParams,
) -> Signal | None:
    weekly_close = snapshot.num("weekly_close")
    ma20w = snapshot.num("ma20w")
    ma20w_up = snapshot.flag("ma20w_up")
    weekly_break = weekly_close is not None and ma20w is not None and weekly_close < ma20w

    score, reasons, warnings = _s1_score(snapshot, position, params)
    if score < params.watch_confidence:
        return None
    threshold = params.sell_threshold(
        snapshot.market.state, _sector_strength(snapshot)
    )
    recent_block = (
        position.holding_days is not None
        and position.holding_days < params.min_hold_days_before_s1
        and score < 80
    )
    actionable = score >= threshold and not recent_block

    clear_all = weekly_break and (ma20w_up is False or score >= 85)
    if actionable:
        action = "清仓" if clear_all else "减仓 50%"
        quantity = _sell_quantity(position, half=not clear_all)
    else:
        action = "观察（刚买入，暂不卖）" if recent_block else "观察（趋势转弱，等待确认）"
        quantity = None

    stop, _ = stop_levels(snapshot, params)
    return Signal(
        kind=SignalKind.S1,
        symbol=snapshot.symbol,
        name=snapshot.name,
        as_of=snapshot.as_of,
        confidence=score,
        actionable=actionable,
        action=action,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        trigger_price_raw=snapshot.close_raw,
        stop_price_raw=stop,
        suggested_quantity=quantity,
        position_note=_position_note(position),
        metrics=_metrics(snapshot),
        threshold=threshold,
    )


def _s1_score(
    snapshot: FactorSnapshot,
    position: PositionContext,
    params: SignalParams,
) -> tuple[int, list[str], list[str]]:
    score = 0
    reasons: list[str] = []
    warnings: list[str] = []

    weekly_close = snapshot.num("weekly_close")
    ma20w = snapshot.num("ma20w")
    ma10w = snapshot.num("ma10w")
    jw = snapshot.num("jw")
    jw_pct = snapshot.num("jw_pct")

    # 周线高位转弱：25 分。
    high = 0
    if jw_pct is not None:
        if jw_pct >= 90.0:
            high += 10
            reasons.append(f"周J历史分位 {jw_pct:.0f}%（高位）")
        elif jw_pct >= 80.0:
            high += 6
    if jw is not None and jw >= 80.0:
        high += 5
    if snapshot.flag("jw_up") is False:
        high += 5
        reasons.append("周J不再上升")
    if weekly_close is not None and ma10w is not None and weekly_close < ma10w:
        high += 8
        reasons.append("周收盘跌破周MA10")
    score += min(high, 25)

    # 周线趋势破坏：25 分。
    if weekly_close is not None and ma20w is not None and weekly_close < ma20w:
        score += 20
        reasons.append("周收盘跌破周MA20")
    if snapshot.flag("ma20w_up") is False:
        score += 5
        reasons.append("周MA20向下")

    # 日线走弱：20 分，不再要求“死叉且跌破 MA10”同时成立。
    hist = snapshot.num("macd_hist")
    if hist is not None and hist < 0 and snapshot.flag("macd_hist_up") is False:
        score += 10
        reasons.append("MACD柱走弱")
    if snapshot.flag("above_ma10") is False:
        score += 10
        reasons.append("收盘跌破MA10")

    # 位置/量能：15 分。
    pos250 = snapshot.num("pos_250")
    if pos250 is not None and pos250 > 85.0:
        score += 8
        reasons.append(f"250日位置 {pos250:.0f}%（高位）")
    volume = snapshot.num("vol_ratio20")
    direction = snapshot.num("vol_price_dir")
    if volume is not None and direction is not None and direction < 0 and volume > 1.5:
        score += 7
        reasons.append(f"放量下跌（量比 {volume:.1f}）")

    # 早期亏损：15 分。-5% 就允许 S1 介入，不等 -10% 的 S2。
    if position.pnl_pct <= params.early_loss_pct:
        score += 10
        reasons.append(f"浮亏 {position.pnl_pct:+.1%}（早期预警）")
    if position.pnl_pct <= params.deep_loss_pct:
        score += 5
        reasons.append(f"浮亏 {position.pnl_pct:+.1%}（加深）")

    # 行业：弱市更敏感，强市更宽容。
    sector = snapshot.sector
    if sector is not None and sector.strength == "weak":
        score += 5
        reasons.append("行业转弱")
    elif sector is not None and sector.strength == "strong":
        score -= 3
    elif sector is not None and sector.strength == "unknown":
        warnings.append("行业数据不足")
    return max(0, min(score, 100)), reasons, warnings


# --------------------------------------------------------------------------- S2


def _s2(
    snapshot: FactorSnapshot,
    position: PositionContext,
    params: SignalParams,
) -> Signal | None:
    close = snapshot.close_raw
    reasons: list[str] = []
    score = 0

    # 成本比较用 position.close_raw（回测里已按复权因子折算），
    # 破位/跳水用 snapshot.close_raw（当天实际不复权价）；实盘两者相同。
    cost = position.avg_cost_raw
    position_price = position.close_raw
    if cost > 0 and position_price <= cost * (1.0 + params.s2_loss_pct):
        score = 95
        reasons.append(
            f"浮亏 {position.pnl_pct:+.1%}，触及 {params.s2_loss_pct:.0%} 止损线"
        )

    volume = snapshot.num("vol_ratio20")
    prev_low = snapshot.num("llv_low_10_prev")
    if (
        prev_low is not None
        and volume is not None
        and volume >= params.s2_volume_mult
        and close < prev_low * snapshot.price_scale
    ):
        score = max(score, 90)
        reasons.append(f"跌破前10日最低，量比 {volume:.1f}")

    ret1 = snapshot.num("ret1")
    if (
        ret1 is not None
        and volume is not None
        and ret1 <= params.s2_single_day_drop
        and volume >= params.s2_volume_mult
    ):
        score = max(score, 88)
        reasons.append(f"单日 {ret1:+.1%} 放量下跌（量比 {volume:.1f}）")

    if score == 0:
        return None

    stop, _ = stop_levels(snapshot, params)
    return Signal(
        kind=SignalKind.S2,
        symbol=snapshot.symbol,
        name=snapshot.name,
        as_of=snapshot.as_of,
        confidence=score,
        actionable=True,
        action="清仓（风控）",
        reasons=tuple(reasons),
        threshold=0,
        trigger_price_raw=close,
        stop_price_raw=stop,
        suggested_quantity=position.quantity,
        position_note=_position_note(position),
        metrics=_metrics(snapshot),
    )


# --------------------------------------------------------------------------- 共用打分


def _sector_market_score(
    snapshot: FactorSnapshot,
    reasons: list[str],
    warnings: list[str],
    *,
    strong: int,
    neutral: int,
    attack: int,
) -> int:
    score = 0
    sector = snapshot.sector
    if sector is not None:
        if sector.strength == "strong":
            score += strong
            reasons.append(f"行业 {sector.name or sector.code} 强势（第{sector.rank}）")
        elif sector.strength == "neutral":
            score += neutral
        elif sector.strength == "weak":
            warnings.append("行业偏弱")
        else:
            warnings.append("行业数据不足")
    else:
        warnings.append("未映射行业")

    market = snapshot.market
    if market.state == "RISK_ON":
        score += attack
        reasons.append(f"大盘进攻（额度 {market.quota:.0%}）")
    elif market.state == "NEUTRAL":
        score += max(attack - 2, 0)
    elif market.state in ("RISK_OFF", "PANIC"):
        warnings.append(f"大盘 {market.state}")
    return score
