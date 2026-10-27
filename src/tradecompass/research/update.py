"""收盘后一键更新研究数据，并给出“能不能出报告”的完整性结论。

三步：个股走新浪日线 + 东财估值（多进程）；ETF/指数走 baostock，不可用时降级腾讯；
最后做完整性检查——持仓与当前成分股都必须有期望交易日的 K 线。

完整性检查的严重程度分两档（HANDOFF §9.2）：

- **阻断**：最近还在交易、期望日却没有 K 线 → 数据没拉到，报告会基于旧价甚至输出假结论；
- **仅警告**：最近一根 K 线已早于期望日 ``STALE_TRADING_DAYS`` 个交易日以上 → 长期停牌或退市，
  这类标的不可能出现在新买清单里，不该让它永久卡住每天的提醒。

``tradecompass plan`` 在生成报告前调用同一个检查，不通过就拒绝生成。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from tradecompass.config.loader import Paths
from tradecompass.data.calendar import TradingCalendar, default_calendar
from tradecompass.data.sources.base import SourceError
from tradecompass.research import data as rdata
from tradecompass.research.etfs import etf_by_code

log = logging.getLogger(__name__)

STALE_TRADING_DAYS = 5
"""期望日与最近一根 K 线相隔超过这么多个交易日，就按“长期停牌/退市”处理，只警告不阻断。"""

HOLDING_ROLE = "持仓"
CONSTITUENT_ROLE = "成分股"


@dataclass(frozen=True, slots=True)
class Gap:
    """一个没有期望日 K 线的标的。"""

    code: str
    role: str
    last_date: date | None
    stale: bool

    @property
    def blocking(self) -> bool:
        return not self.stale


@dataclass(frozen=True, slots=True)
class Completeness:
    expected: date
    holdings: int
    constituents: int
    gaps: tuple[Gap, ...]

    @property
    def blocking(self) -> tuple[Gap, ...]:
        return tuple(gap for gap in self.gaps if gap.blocking)

    @property
    def stale(self) -> tuple[Gap, ...]:
        return tuple(gap for gap in self.gaps if gap.stale)

    @property
    def ok(self) -> bool:
        return not self.blocking


def expected_bar_date(today: date | None = None) -> date:
    """收盘后应当已经有 K 线的最近交易日（周末/节假日回退到上一个交易日）。"""
    return default_calendar().last_trading_day_on_or_before(today or date.today())


def intraday_refusal(expected: date, now: datetime | None = None) -> str | None:
    """盘中跑 ``update`` 没有意义：数据源不返回当日 K 线，跑了也是白跑。

    更要紧的是日志会显示“个股完成 1343/1405”，让人误以为数据已经更新。
    返回拒绝理由；不该拒绝时返回 None。
    """
    moment = now or datetime.now()
    if expected != moment.date() or moment.time() >= rdata.MARKET_CLOSE:
        return None
    return (
        f"现在是 {moment:%H:%M}，A 股还没收盘（{rdata.MARKET_CLOSE:%H:%M} 之后才有当日 K 线），"
        f"{expected} 的日线取不到。现在跑 update 只会白跑一遍、日志却像成功了；"
        "请收盘后再跑，或用 --as-of 指定历史日期补数。"
    )


def latest_constituent_codes(paths: Paths) -> list[str]:
    """最近一个月末快照的全部成分股。"""
    members = rdata.load_constituents(paths)
    if members.empty:
        return []
    newest = members["date"].max()
    return sorted(set(members.loc[members["date"] == newest, "code"]))


def _last_bar_date(paths: Paths, code: str) -> date | None:
    frame = (
        rdata.read_etf_bars(paths, code)
        if code in etf_by_code()
        else rdata.read_stock_bars(paths, code)
    )
    if frame.empty:
        return None
    return pd.Timestamp(frame.index.max()).date()


def check_completeness(
    paths: Paths,
    *,
    expected: date,
    holding_codes: Sequence[str],
    constituent_codes: Sequence[str],
    calendar: TradingCalendar | None = None,
) -> Completeness:
    """检查持仓与成分股是否都有 ``expected`` 当日的 K 线。

    ``holding_codes`` 既接受 6 位代码也接受 ``sh.600000`` 形式；持仓优先于成分股，
    同一只标的只报一次。
    """
    cal = calendar or default_calendar()
    roles: dict[str, str] = {}
    for code in constituent_codes:
        roles[rdata.to_bs_code(code)] = CONSTITUENT_ROLE
    for code in holding_codes:
        roles[rdata.to_bs_code(code)] = HOLDING_ROLE

    gaps: list[Gap] = []
    for code, role in sorted(roles.items()):
        last = _last_bar_date(paths, code)
        if last is not None and last >= expected:
            continue
        if last is None:
            gaps.append(Gap(code, role, None, stale=False))
            continue
        span = cal.trading_days(last + timedelta(days=1), expected)
        gaps.append(Gap(code, role, last, stale=len(span) > STALE_TRADING_DAYS))

    return Completeness(
        expected=expected,
        holdings=sum(1 for role in roles.values() if role == HOLDING_ROLE),
        constituents=sum(1 for role in roles.values() if role == CONSTITUENT_ROLE),
        gaps=tuple(gaps),
    )


def sync_stocks_fast(
    paths: Paths,
    codes: Sequence[str],
    start: date,
    *,
    workers: int = 4,
    progress: Callable[[int, int, int], None] | None = None,
) -> tuple[int, list[tuple[str, str]]]:
    """新浪日线 + 东财估值，多进程。返回（处理成功数，[(代码, 失败原因)]）。

    新浪用的 V8 不是线程安全，只能多进程；已退市标的没有新浪数据、东财也没有估值，
    会落进失败列表——它们不在当日候选池里，不影响出报告。
    """
    failures: list[tuple[str, str]] = []
    done = 0
    if not codes:
        return 0, failures
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(rdata.sync_fast, paths, code, start): code for code in codes}
        for number, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                future.result()
                done += 1
            except (SourceError, OSError) as exc:
                failures.append((code, str(exc)))
            if progress is not None:
                progress(number, len(codes), len(failures))
    return done, failures


def sync_etf_and_index(
    paths: Paths,
    *,
    etf_codes: Sequence[str],
    index_codes: Sequence[str],
    start: date,
    end: date,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[int, list[str]]:
    """ETF 与基准指数日线：baostock 优先，登录失败或单只失败时降级腾讯补数。"""
    from tradecompass.data.sources import baostock_source

    failures: list[str] = []
    targets = [(code, False) for code in etf_codes] + [(code, True) for code in index_codes]
    use_baostock = True
    try:
        baostock_source.login()
    except SourceError as exc:
        use_baostock = False
        failures.append(f"baostock 登录失败（{exc}），ETF/指数改用腾讯补数")

    success = 0
    for number, (code, is_index) in enumerate(targets, start=1):
        path = rdata.index_bars_path(paths, code) if is_index else rdata.etf_bars_path(paths, code)
        done = False
        if use_baostock:
            try:
                if is_index:
                    rdata.sync_index_bars(paths, code, start, end)
                else:
                    rdata.sync_etf_bars(paths, code, start, end)
                done = True
            except (SourceError, OSError) as exc:
                failures.append(f"{code}: baostock 失败（{exc}），改用腾讯")
        if not done:
            try:
                rdata.append_from_tencent(path, code, end, index=is_index)
                done = True
            except Exception as exc:  # noqa: BLE001 - 第三方抓取异常种类不可控
                failures.append(f"{code}: 腾讯补数失败 {exc}")
        if done:
            success += 1
        if progress is not None:
            progress(number, len(targets))
    return success, failures


def retry_via_baostock(
    paths: Paths,
    codes: Sequence[str],
    start: date,
    end: date,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[int, list[tuple[str, str]]]:
    """对 fast 通道失败的关键标的，用 baostock 逐只补数。

    新浪取不到 ≠ 退市：例如 689009（CDR）今天就是稳定的 ``No value to decode``。
    这类标的只要在当日成分股/持仓里，就必须补上，否则整份报告会被完整性闸门拦住。
    baostock 不能并发，所以只给少数关键代码用；一只约 10 秒，远好过每日卡死。
    """
    fixed = 0
    failures: list[tuple[str, str]] = []
    for number, code in enumerate(codes, start=1):
        try:
            if code in etf_by_code():
                rdata.sync_etf_bars(paths, code, start, end)
            else:
                rdata.sync_stock_bars(paths, code, start, end)
            fixed += 1
        except (SourceError, OSError) as exc:
            failures.append((code, str(exc)))
        if progress is not None:
            progress(number, len(codes))
    return fixed, failures


def describe(completeness: Completeness, limit: int = 12) -> list[str]:
    """把检查结果整理成可直接打印的几行。"""
    lines = [
        f"完整性检查（期望 {completeness.expected}）："
        f"持仓 {completeness.holdings} 只、成分股 {completeness.constituents} 只"
    ]
    if completeness.ok and not completeness.stale:
        lines.append("  全部有当日 K 线")
        return lines
    for gap in completeness.blocking[:limit]:
        last = gap.last_date.isoformat() if gap.last_date else "无数据"
        lines.append(f"  ✗ {gap.role} {gap.code} 最近一根 K 线 {last}")
    if len(completeness.blocking) > limit:
        lines.append(f"  ✗ 还有 {len(completeness.blocking) - limit} 只未列出")
    for gap in completeness.stale[:limit]:
        last = gap.last_date.isoformat() if gap.last_date else "无数据"
        lines.append(f"  · {gap.role} {gap.code} 长期停牌/退市（最近 {last}），不阻断")
    return lines
