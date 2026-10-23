"""行情同步（SPEC §8.1、§8.2）。

流程：不复权增量 → 同源后复权 → 计算因子 → 交叉校验 → 落盘。
不复权走增量，因子走滚动窗口（默认 400 个自然日），这样历史因子被修订时能发现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from tradecompass.config.loader import AppConfig, Paths
from tradecompass.data import adjust, store, validate
from tradecompass.data.sources import registry
from tradecompass.data.sources.base import BarQuery, SourceError
from tradecompass.data.universe import index_market, limit_pct_of, to_instrument
from tradecompass.domain.models import Instrument
from tradecompass.market.sectors import INDUSTRIES

FACTOR_WINDOW_DAYS = 400  # 自然日 ≈ 270 个交易日，覆盖 SPEC 要求的最近 250 个交易日


@dataclass(frozen=True, slots=True)
class SyncResult:
    symbol: str
    market: str
    raw_rows: int
    factor_rows: int
    source: str
    added: int
    issues: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def suspect(self) -> bool:
        return bool(self.issues)


def _history_start(end: date, years: int) -> date:
    try:
        return end.replace(year=end.year - years)
    except ValueError:  # 2 月 29 日
        return end.replace(year=end.year - years, day=28)


def _ensure_factor(
    paths: Paths,
    instrument: Instrument,
    raw_all: pd.DataFrame,
    end: date,
    raw_source: str,
    *,
    cached: pd.Series | None = None,
) -> tuple[pd.Series, list[str], list[str]]:
    """从**完整历史**重建因子（同源后复权 ÷ 不复权，再归一化到最新日 = 1）。

    不做增量拼接：不同批次算出的因子绝对值不可比（后复权基期会随窗口漂移），
    拼接会引入难以察觉的尺度漂移。重建的成本是每标的多一次全历史查询，
    换来的是"同样输入必然得到同样输出"的可复现性。

    返回 ``(因子, 阻断性问题, 提示)``：因子重建本身是自愈的，所以修订只作为提示，
    不把标的标记为 ``data_suspect``。
    """
    issues: list[str] = []
    notes: list[str] = []
    path = store.factor_path(paths, instrument.market, instrument.symbol)
    cached_factor = cached if cached is not None else store.read_factor_or_empty(path)
    if raw_all.empty:
        return cached_factor, issues, notes

    query = BarQuery(
        symbol=instrument.symbol,
        market=instrument.market,
        kind=instrument.kind,
        start=raw_all.index[0].date(),
        end=end,
    )
    try:
        hfq = registry.fetch_named(query, "hfq", raw_source)
    except SourceError:
        hfq, other = registry.fetch_with_fallback(query, "hfq", [raw_source, "baostock"])
        issues.append(f"hfq 改由 {other} 提供，因子口径可能整体缩放（不影响比值）")
    if hfq.empty:
        return cached_factor, [*issues, "后复权数据为空，沿用手上的因子"], notes

    fresh = adjust.normalize_factor(adjust.compute_factor(hfq, raw_all))
    if not cached_factor.empty:
        # 只比"形状"：把两条序列各自的第一个值都拉回 1，这样仅仅因为多了一根 K 线
        # 导致的整体缩放不会被误判成除权，而真正的跳变仍然会被抓到。
        old_shape = cached_factor / cached_factor.iloc[0]
        new_shape = fresh / fresh.iloc[0]
        if adjust.factor_changed(old_shape, new_shape, tol=1e-4, detect_new_jumps=False):
            notes.append("复权因子形状有变化（除权或数据修订），已按最新数据重建")
    store.write_factor(path, fresh)
    return fresh, issues, notes


def _cross_check(
    instrument: Instrument,
    raw_window: pd.DataFrame,
    end: date,
    sources: list[str],
    primary: str,
) -> tuple[list[str], list[str]]:
    """用第二个源比对不复权收盘价，返回 ``(阻断性问题, 提示)``。

    只比价格，不比因子：实测 akshare 的 ``hfq`` 不是累积复权序列
    （同一窗口内因子先升后降、连续漂移），拿它校验 baostock 的因子只会产生假警报。
    详见 docs/DECISIONS.md D010。
    """
    others = [name for name in registry.resolve(sources) if name != primary]
    if not others or raw_window.empty:
        return [], []
    query = BarQuery(
        symbol=instrument.symbol,
        market=instrument.market,
        kind=instrument.kind,
        start=end - timedelta(days=FACTOR_WINDOW_DAYS),
        end=end,
    )
    second = others[0]
    try:
        other_raw = registry.fetch_named(query, "none", second)
    except SourceError as exc:
        return [], [f"交叉校验源 {second} 不可用: {exc}"]
    if other_raw.empty:
        return [], []

    mismatch = validate.check_cross_source(raw_window["close"], other_raw["close"])
    if mismatch is None:
        return [], []
    return [f"两源收盘价偏差（{second}）：{mismatch.detail}"], []


def sync_instrument(
    cfg: AppConfig, instrument: Instrument, *, end: date, full: bool = False
) -> SyncResult:
    """同步单个标的。``full=True`` 时重拉完整历史。"""
    paths = cfg.paths
    raw_path = store.raw_bars_path(paths, instrument.market, instrument.symbol)
    cached = store.read_bars_or_empty(raw_path)

    history_start = _history_start(end, cfg.system.data.history_years)
    if full or cached.empty:
        start = history_start
    else:
        start = cached.index[-1].date() + timedelta(days=1)

    added = 0
    merged = cached
    source = store.read_source(paths, instrument.market, instrument.symbol)
    if start <= end:
        query = BarQuery(
            symbol=instrument.symbol,
            market=instrument.market,
            kind=instrument.kind,
            start=start,
            end=end,
        )
        fetched, source = registry.fetch_with_fallback(
            query, "none", cfg.system.data.raw_sources, allow_empty=True
        )
        if not fetched.empty:
            merged = store.merge_bars(cached, fetched)
            added = len(merged) - len(cached)
            store.write_bars(raw_path, merged)
            store.write_source(paths, instrument.market, instrument.symbol, source)

    if merged.empty:
        return SyncResult(
            symbol=instrument.symbol,
            market=instrument.market,
            raw_rows=0,
            factor_rows=0,
            source=source,
            added=0,
            issues=("没有取到任何日线",),
        )

    window_start = end - timedelta(days=FACTOR_WINDOW_DAYS)
    raw_window = merged[merged.index >= pd.Timestamp(window_start)]
    factor, factor_issues, notes = _ensure_factor(paths, instrument, merged, end, source)
    issues: list[str] = list(factor_issues)
    cross_issues, cross_notes = _cross_check(
        instrument, raw_window, end, cfg.system.data.factor_sources, source
    )
    issues.extend(cross_issues)
    notes.extend(cross_notes)

    validation = validate.validate_all(
        raw_window,
        limit_pct=limit_pct_of(instrument),
        expected_last=None,
    )
    issues.extend(f"{item.code}: {item.detail}" for item in validation)

    return SyncResult(
        symbol=instrument.symbol,
        market=instrument.market,
        raw_rows=len(merged),
        factor_rows=len(factor),
        source=source,
        added=added,
        issues=tuple(issues),
        notes=tuple(notes),
    )


def to_instruments(cfg: AppConfig) -> list[Instrument]:
    return [to_instrument(item) for item in cfg.universe.instruments]


def market_index_instruments(cfg: AppConfig) -> list[Instrument]:
    """market.yaml 里用于市场环境层的指数（SPEC §8.1）。"""
    indices = cfg.market.indices
    symbols: list[str] = [indices.hs300, indices.csi1000, indices.breadth_universe]
    symbols.extend(indices.turnover)
    symbols.extend(code for code, _ in INDUSTRIES)
    seen: list[str] = []
    for symbol in symbols:
        if symbol not in seen:
            seen.append(symbol)
    return [
        Instrument(
            symbol=symbol,
            market=index_market(symbol),
            name=symbol,
            kind="index",
            board="index",
        )
        for symbol in seen
    ]
