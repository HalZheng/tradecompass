"""建议事后追踪：把历史 `plan` 的每条建议按 T+1/T+5/T+20 回填，并核对是否执行。

只读 ``reports/plan/*.json``（由 `plan` 落盘）与行情、成交流水，不参与任何买卖判断。

口径（写在这里，避免以后各说各话）：

- **入场**：建议日（``target_day``）开盘价，与回测“次日开盘成交”一致；
- **T+N**：建议日之后的第 N 个交易日收盘，价格为复权价（用 ``close/preclose`` 递推）；
- **买入命中**：T+5 相对沪深300 的超额 > 0；
- **清理建议**：条件单价位取 ``ma10``/``ma20`` 中高于现价的第一个（与报告一致），
  ``touched`` 表示建议日之后 5 个交易日内最高价是否触及该价位；
- **是否执行**：成交流水在 ``[target_day, T+5]`` 内有同代码同方向成交即为已执行；
  流水未覆盖到 T+5 时记为未知（``None``），不猜。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tradecompass.config.loader import Paths
from tradecompass.data.calendar import TradingCalendar, default_calendar
from tradecompass.research import data as rdata
from tradecompass.research.etfs import etf_by_code

HORIZONS: tuple[int, ...] = (1, 5, 20)
EXECUTION_WINDOW = 5
BENCHMARK_KEY = "hs300"
ETF_PREFIXES = frozenset({"15", "16", "18", "51", "52", "56", "58"})
"""ETF 代码前缀；静态名单 ``etf_by_code`` 之外的 ETF（如用户持仓的行业 ETF）也能读对行情。"""


@dataclass(frozen=True, slots=True)
class AdviceOutcome:
    target_day: date
    code: str
    name: str
    side: str
    action: str
    reference: float | None
    returns: dict[int, float | None]
    excess: dict[int, float | None]
    touched: bool | None
    executed: bool | None
    category: str = "stock"
    """stock / base / steady；底仓与稳健 ETF 订单单独统计。"""
    fingerprint: str = ""
    """建议所属的规则指纹；冻结后指纹变化意味着样本外失效。"""


def load_history(directory: Path, *, since: date | None = None) -> list[dict[str, Any]]:
    """读取全部已落盘的提醒，按建议日排序；坏文件跳过而不是中断。

    ``since`` 用于真样本外：只保留建议日在该日及以后的提醒。
    """
    plans: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not (isinstance(payload, dict) and payload.get("schema") == 1):
            continue
        try:
            target = date.fromisoformat(str(payload.get("target_day")))
        except ValueError:
            continue
        if since is not None and target < since:
            continue
        plans.append(payload)
    plans.sort(key=lambda item: str(item.get("target_day", "")))
    return plans


def _adjusted(frame: pd.DataFrame) -> pd.DataFrame:
    """复权开/收盘；用交易所前收递推，除权日不产生假跌幅。"""
    if frame.empty:
        return pd.DataFrame(columns=["open", "close"], dtype="float64")
    ret = (frame["close"] / frame["preclose"] - 1.0).fillna(0.0)
    adj_close = (1.0 + ret).cumprod()
    scale = adj_close / frame["close"]
    return pd.DataFrame({"open": frame["open"] * scale, "close": adj_close})


def _sessions(calendar: TradingCalendar, start: date, horizon: int) -> list[date]:
    """``start`` 起（含）的交易日序列，最多 horizon+1 个。"""
    return calendar.sessions(start, horizon + 1)


def _return_at(series: pd.Series, day: date, entry: float) -> float | None:
    if not np.isfinite(entry) or entry <= 0:
        return None
    window = series.loc[: pd.Timestamp(day)]
    if window.empty:
        return None
    exit_price = float(window.iloc[-1])
    if not np.isfinite(exit_price):
        return None
    return exit_price / entry - 1.0


def _cleanup_reference(item: dict[str, Any]) -> float | None:
    price = item.get("price")
    if not isinstance(price, int | float):
        return None
    targets = sorted(
        {
            round(float(value), 3)
            for value in (item.get("ma10"), item.get("ma20"))
            if isinstance(value, int | float) and float(value) > float(price)
        }
    )
    return targets[0] if targets else None


def _executed(
    trades: pd.DataFrame | None, code: str, side: str, start: date, end: date
) -> bool | None:
    if trades is None or trades.empty:
        return None
    dates = pd.to_datetime(trades["date"]).dt.date
    if dates.max() < end:
        return None  # 流水没覆盖到窗口末尾，不能判断
    matched = (
        (trades["code"].astype(str).str.zfill(6) == code)
        & (trades["side"] == side)
        & (dates >= start)
        & (dates <= end)
    )
    return bool(matched.any())


def _bars(paths: Paths, code: str) -> pd.DataFrame:
    bs = rdata.to_bs_code(code)
    if bs in etf_by_code() or bs.split(".")[-1][:2] in ETF_PREFIXES:
        return rdata.read_etf_bars(paths, bs)
    return rdata.read_stock_bars(paths, bs)


def _is_etf(code: str) -> bool:
    bs = rdata.to_bs_code(code)
    return bs in etf_by_code() or bs.split(".")[-1][:2] in ETF_PREFIXES


def evaluate_plan(
    plan: dict[str, Any],
    *,
    paths: Paths,
    trades: pd.DataFrame | None = None,
    calendar: TradingCalendar | None = None,
    horizons: Sequence[int] = HORIZONS,
) -> list[AdviceOutcome]:
    """把一份提醒里的每条“可执行建议”回填成结果。保留/持有类不算建议。"""
    cal = calendar or default_calendar()
    target_day = date.fromisoformat(str(plan["target_day"]))
    longest = max(horizons)
    sessions = _sessions(cal, target_day, longest)
    if not sessions or sessions[0] != target_day:
        return []

    bench = _adjusted(rdata.read_index_bars(paths, rdata.BENCHMARKS[BENCHMARK_KEY]))
    bench_entry = float(bench["open"].get(pd.Timestamp(target_day), np.nan))
    # 行情到哪天为止：晚于它的 T+N 还没发生，不能拿“现在的价”冒充未来收益
    data_end = (
        pd.Timestamp(bench.index.max()).date()
        if not bench.empty
        else date.min
    )

    outcomes: list[AdviceOutcome] = []
    fingerprint = str((plan.get("rules") or {}).get("fingerprint", ""))

    def make(
        item: dict[str, Any],
        side: str,
        action: str,
        category: str,
        reference: float | None,
    ) -> AdviceOutcome:
        code = str(item["code"])
        frame = _bars(paths, code)
        adj = _adjusted(frame)
        entry = float(adj["open"].get(pd.Timestamp(target_day), np.nan))
        returns: dict[int, float | None] = {}
        excess: dict[int, float | None] = {}
        for horizon in horizons:
            if len(sessions) <= horizon or sessions[horizon] > data_end:
                returns[horizon] = None
                excess[horizon] = None
                continue
            value = _return_at(adj["close"], sessions[horizon], entry)
            returns[horizon] = value
            bench_value = _return_at(bench["close"], sessions[horizon], bench_entry)
            excess[horizon] = (
                (1.0 + value) / (1.0 + bench_value) - 1.0
                if value is not None and bench_value is not None
                else None
            )
        touched: bool | None = None
        if (
            side == "sell"
            and category in ("stock", "etf")
            and reference is not None
            and len(sessions) > EXECUTION_WINDOW
            and sessions[EXECUTION_WINDOW] <= data_end
        ):
            window = frame["high"].loc[
                pd.Timestamp(sessions[1]) : pd.Timestamp(sessions[EXECUTION_WINDOW])
            ]
            touched = bool((window >= float(reference)).any())
        window_end = sessions[min(EXECUTION_WINDOW, len(sessions) - 1)]
        return AdviceOutcome(
            target_day=target_day,
            code=code,
            name=str(item.get("name", code)),
            side=side,
            action=action,
            reference=reference,
            returns=returns,
            excess=excess,
            touched=touched,
            executed=_executed(trades, code, side, target_day, window_end),
            category=category,
            fingerprint=fingerprint,
        )

    for item in plan.get("buys") or []:
        if item.get("action", "买入") != "买入":
            continue
        limit = item.get("limit")
        reference = float(limit) if isinstance(limit, int | float) else None
        outcomes.append(make(item, "buy", "买入", "stock", reference))
    for item in plan.get("holdings") or []:
        action = str(item.get("action", ""))
        if not action.startswith("清理"):
            continue
        category = "etf" if _is_etf(str(item["code"])) else "stock"
        outcomes.append(make(item, "sell", action, category, _cleanup_reference(item)))
    for item in plan.get("base_orders") or []:
        shares = int(item.get("shares") or 0)
        if shares <= 0:
            continue
        limit = item.get("limit")
        reference = float(limit) if isinstance(limit, int | float) and float(limit) > 0 else None
        outcomes.append(make(item, "buy", "底仓补仓", "base", reference))
    for item in plan.get("steady_orders") or []:
        lots = int(item.get("lots") or 0)
        if lots == 0:
            continue
        side = "buy" if lots > 0 else "sell"
        action = "稳健加仓" if lots > 0 else "稳健减仓"
        outcomes.append(make(item, side, action, "steady", None))
    return outcomes


def _mean(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return float(np.mean(usable)) if usable else None


def _fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:+.{digits}%}"


def _rate(values: Iterable[bool | None]) -> str:
    usable = [value for value in values if value is not None]
    if not usable:
        return "—"
    return f"{sum(usable)}/{len(usable)}（{sum(usable) / len(usable):.0%}）"


def summarize(outcomes: Sequence[AdviceOutcome]) -> str:
    """汇总 + 明细，Markdown。"""
    if not outcomes:
        return "# 建议追踪\n\n还没有可追踪的提醒（`plan` 会同时写 .json，从下一份开始累积）。\n"

    buys = [item for item in outcomes if item.category == "stock" and item.side == "buy"]
    sells = [item for item in outcomes if item.category == "stock" and item.side == "sell"]
    etf_orders = [item for item in outcomes if item.category in ("base", "steady", "etf")]
    etf_buys = [item for item in etf_orders if item.side == "buy"]
    etf_sells = [item for item in etf_orders if item.side == "sell"]
    days = sorted({item.target_day for item in outcomes})
    fingerprints = sorted({item.fingerprint for item in outcomes if item.fingerprint})
    lines = [
        "# 建议追踪",
        "",
        f"覆盖 {len(days)} 个建议日（{days[0]} ~ {days[-1]}），共 {len(outcomes)} 条可执行建议。",
    ]
    if fingerprints:
        warning = "；**指纹已变化，样本外失效**" if len(fingerprints) > 1 else ""
        lines.append(f"规则指纹：{', '.join(fingerprints)}{warning}。")
    lines.append("")
    if all(item.returns.get(1) is None for item in outcomes):
        lines += [
            "> 这些建议的 T+1 行情还没走到（数据还没到建议日之后），下面整表是 “—” "
            "不是缺数据；建议日之后第 1 个交易日起会逐日回填。",
            "",
        ]
    lines += [
        "## 一、买入建议",
        "",
    ]
    if buys:
        hits = [
            None if item.excess.get(5) is None else (item.excess[5] or 0.0) > 0
            for item in buys
        ]
        lines += [
            f"- 条数 {len(buys)}；T+5 跑赢沪深300：{_rate(hits)}",
            f"- T+5 平均超额 {_fmt(_mean(item.excess.get(5) for item in buys), 2)}，"
            f"T+20 平均超额 {_fmt(_mean(item.excess.get(20) for item in buys), 2)}",
            f"- 已执行：{_rate(item.executed for item in buys)}",
        ]
    else:
        lines.append("- 无")
    lines += ["", "## 二、清理（卖出）建议", ""]
    if sells:
        touched_rate = _rate(item.touched for item in sells)
        waited = _fmt(_mean(item.returns.get(5) for item in sells), 2)
        lines += [
            f"- 条数 {len(sells)}；5 个交易日内触及条件单价：{touched_rate}",
            f"- 已执行：{_rate(item.executed for item in sells)}",
            f"- 全部清理建议 T+5 平均涨跌 {waited}"
            "（正值表示“没卖掉”这段时间反而涨了）",
        ]
    else:
        lines.append("- 无")

    lines += [
        "",
        "## 三、ETF 建议（底仓/稳健/波段）",
        "",
    ]
    if etf_orders:
        lines += [
            f"- 条数 {len(etf_orders)}（买入 {len(etf_buys)}、减仓 {len(etf_sells)}）",
            f"- 买入 T+5 平均超额 {_fmt(_mean(item.excess.get(5) for item in etf_buys), 2)}，"
            f"T+20 平均超额 {_fmt(_mean(item.excess.get(20) for item in etf_buys), 2)}",
            f"- 减仓 T+5 平均涨跌 {_fmt(_mean(item.returns.get(5) for item in etf_sells), 2)}"
            "（正值表示“没减”这段时间反而涨了）",
            f"- 已执行：{_rate(item.executed for item in etf_orders)}",
        ]
    else:
        lines.append("- 无")

    lines += [
        "",
        "## 四、明细",
        "",
        "| 建议日 | 类别 | 代码 | 名称 | 动作 | 参考价 | T+1 | T+5 | T+20 | "
        "T+5超额 | 条件单触发 | 已执行 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    category_name = {"stock": "个股", "base": "底仓", "steady": "稳健", "etf": "ETF"}
    for item in sorted(outcomes, key=lambda o: (o.target_day, o.category, o.code)):
        lines.append(
            f"| {item.target_day} | {category_name.get(item.category, item.category)} | "
            f"{item.code} | {item.name} | {item.action} | "
            f"{'—' if item.reference is None else f'{item.reference:.3f}'} | "
            f"{_fmt(item.returns.get(1))} | {_fmt(item.returns.get(5))} | "
            f"{_fmt(item.returns.get(20))} | {_fmt(item.excess.get(5), 2)} | "
            f"{'—' if item.touched is None else ('是' if item.touched else '否')} | "
            f"{'—' if item.executed is None else ('是' if item.executed else '否')} |"
        )
    lines.append("")
    return "\n".join(lines)
