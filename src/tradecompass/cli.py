"""命令行入口（SPEC §14）。

退出码：成功 0；配置错误 2；数据错误 3；其他错误 1。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tradecompass.bootstrap import initialize
from tradecompass.config import ConfigError, load_config, repo_root, resolve_home
from tradecompass.config.loader import Paths
from tradecompass.data.sources.base import SourceError
from tradecompass.data.sync import market_index_instruments, sync_instrument, to_instruments
from tradecompass.log import configure as configure_logging
from tradecompass.storage.db import StorageError, require_ready

APP_NAME = "TradeCompass"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A 股多因子信号系统：B1/B2/S1/S2 每日提醒、风控与回测。LLM 不参与买卖判断。",
)
console = Console()

HomeOption = Annotated[
    Path | None,
    typer.Option("--home", help="TC_HOME 目录（个人配置与数据，仓库外）"),
]

Status = Literal["ok", "fail", "skip"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: Status
    detail: str


def _fail(message: str, code: int = 2) -> None:
    console.print("✗ " + escape(message), style="red")
    raise typer.Exit(code=code)


def _writable(directory: Path) -> str | None:
    """实际写一个临时文件来判断可写（Windows 下 os.access 不可靠）。"""
    if not directory.is_dir():
        return "目录不存在"
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=True):
            pass
    except OSError as exc:
        return str(exc)
    return None


def _init_target(home: Path | None) -> Path:
    """init 的目标目录：--home > TC_HOME > TC_REPO/local（SPEC §3.2）。"""
    if home is not None:
        return home.expanduser().resolve()
    env = os.environ.get("TC_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root() / "local").resolve()


@app.command()
def init(home: HomeOption = None) -> None:
    """初始化 TC_HOME：建目录、复制配置模板、建库（不覆盖已有文件）。"""
    target = _init_target(home)
    paths = Paths.from_home(target)
    report = initialize(target)

    console.print(f"[bold]{APP_NAME} 初始化完成[/bold]  {escape(str(target))}")
    console.print(f"  新建数据库: {'是' if report.db_created else '否（已存在）'}")
    if report.copied:
        console.print(f"  复制文件 {len(report.copied)} 个")
    if report.skipped:
        console.print(f"  已存在跳过 {len(report.skipped)} 个（未覆盖）")
        for item in report.skipped:
            console.print(f"    - {escape(str(item.relative_to(target)))}")

    steps = (
        f"编辑 {paths.config / 'universe.yaml'}：填自选池与底仓 target",
        f"编辑 {paths.config / 'risk.yaml'}：填 cash（可用现金）与仓位比例",
        f"编辑 {paths.config / 'system.yaml'}：核对调度时间与 notify.enabled",
        f"在 {paths.home / '.env'} 填飞书凭据（留空时推送只写本地预览）",
        f"运行 tradecompass doctor --offline --home {target}",
    )
    table = Table(title="下一步", show_header=False, box=None, padding=(0, 1))
    table.add_column("n", style="dim", width=3)
    table.add_column("task", overflow="fold")
    for idx, task in enumerate(steps, start=1):
        table.add_row(str(idx), escape(task))
    console.print(table)


@app.command()
def doctor(
    home: HomeOption = None,
    offline: Annotated[bool, typer.Option("--offline", help="跳过网络连通性检查")] = False,
) -> None:
    """校验配置、目录、数据库与数据源连通性，逐项输出 OK/FAIL/SKIP。"""
    try:
        target = resolve_home(home)
    except ConfigError as exc:
        _fail(str(exc))
        return

    results: list[CheckResult] = [CheckResult("TC_HOME", "ok", str(target))]
    paths = Paths.from_home(target)

    for label, directory in (
        ("config 目录", paths.config),
        ("data 目录", paths.data),
        ("reports 目录", paths.reports),
        ("logs 目录", paths.logs),
    ):
        problem = _writable(directory)
        results.append(
            CheckResult(f"{label}可写", "ok" if problem is None else "fail", problem or "OK")
        )

    env_file = target / ".env"
    results.append(
        CheckResult(
            ".env",
            "ok" if env_file.is_file() else "fail",
            str(env_file) if env_file.is_file() else "缺失：飞书推送相关功能不可用",
        )
    )

    try:
        config = load_config(target)
        inst_count = len(config.universe.instruments)
        base_count = sum(1 for i in config.universe.instruments if i.kind == "base_etf")
        results.append(
            CheckResult(
                "配置文件",
                "ok",
                f"{len(config.strategies)} 个策略，{inst_count} 个标的（底仓 {base_count}）",
            )
        )
    except ConfigError as exc:
        results.append(CheckResult("配置文件", "fail", str(exc)))

    try:
        require_ready(paths.db_path)
        results.append(CheckResult("数据库", "ok", paths.db_path.name))
    except (StorageError, OSError) as exc:
        results.append(CheckResult("数据库", "fail", str(exc)))

    if offline:
        results.append(CheckResult("数据源连通性", "skip", "已跳过（--offline）"))
    else:
        results.append(CheckResult("数据源连通性", "skip", "尚未接入，M1 实现后启用"))

    table = Table(show_header=True, header_style="bold")
    table.add_column("状态", width=6)
    table.add_column("检查项", width=18)
    table.add_column("详情", overflow="fold")
    marks = {"ok": "[green]OK[/green]", "fail": "[red]FAIL[/red]", "skip": "[dim]SKIP[/dim]"}
    for item in results:
        table.add_row(marks[item.status], escape(item.name), escape(item.detail))
    console.print(table)

    failed = [item for item in results if item.status == "fail"]
    if failed:
        raise typer.Exit(code=2)
    console.print(f"[green]全部检查通过[/green]（{len(results)} 项）")


data_app = typer.Typer(add_completion=False, no_args_is_help=True, help="行情同步与查看")
app.add_typer(data_app, name="data")


@data_app.command("sync")
def data_sync(
    home: HomeOption = None,
    symbols: Annotated[
        str | None, typer.Option("--symbols", help="只同步这些代码，逗号分隔")
    ] = None,
    full: Annotated[bool, typer.Option("--full", help="重拉完整历史，忽略缓存")] = False,
) -> None:
    """同步不复权日线、复权因子与市场指数。"""
    try:
        target = resolve_home(home)
        cfg = load_config(target)
    except ConfigError as exc:
        _fail(str(exc))
        return

    configure_logging(log_dir=cfg.paths.logs)
    end = date.today()
    targets = to_instruments(cfg) + market_index_instruments(cfg)
    if symbols:
        wanted = {item.strip() for item in symbols.split(",") if item.strip()}
        targets = [item for item in targets if item.symbol in wanted]
        if not targets:
            _fail(f"--symbols 没有匹配到任何标的: {sorted(wanted)}")
            return

    table = Table(show_header=True, header_style="bold")
    table.add_column("代码", width=8)
    table.add_column("名称", width=12)
    table.add_column("来源", width=9)
    table.add_column("日线", justify="right", width=6)
    table.add_column("新增", justify="right", width=6)
    table.add_column("因子", justify="right", width=6)
    table.add_column("状态", width=8)
    table.add_column("提示", overflow="fold")

    failures = 0
    suspects = 0
    for instrument in targets:
        try:
            result = sync_instrument(cfg, instrument, end=end, full=full)
        except (SourceError, OSError) as exc:
            failures += 1
            table.add_row(
                instrument.symbol, instrument.name, "-", "-", "-", "-", "[red]FAIL[/red]",
                escape(str(exc)),
            )
            continue
        status = "[yellow]SUSPECT[/yellow]" if result.suspect else "[green]OK[/green]"
        if result.suspect:
            suspects += 1
        detail = "；".join(result.issues)
        if result.notes:
            detail = (detail + "｜" if detail else "") + "；".join(result.notes)
        table.add_row(
            instrument.symbol,
            instrument.name,
            result.source,
            str(result.raw_rows),
            str(result.added),
            str(result.factor_rows),
            status,
            escape(detail),
        )
    console.print(table)
    console.print(
        f"完成 {len(targets) - failures} / {len(targets)}，"
        f"可疑 {suspects}，失败 {failures}"
    )
    if failures:
        raise typer.Exit(code=3)


@data_app.command("show")
def data_show(
    symbol: Annotated[str, typer.Argument(help="6 位代码，如 600000")],
    home: HomeOption = None,
    on_date: Annotated[
        str | None, typer.Option("--date", help="指定日期 YYYY-MM-DD，默认最新一根")
    ] = None,
) -> None:
    """打印某日的前复权价格与全部指标，用于和券商软件比对。"""
    import pandas as pd

    from tradecompass.data import adjust as adjust_mod
    from tradecompass.data import store as store_mod
    from tradecompass.data.sync import market_index_instruments, to_instruments
    from tradecompass.indicators.frame import IndicatorParams, daily_frame, weekly_frame

    try:
        target = resolve_home(home)
        cfg = load_config(target)
    except ConfigError as exc:
        _fail(str(exc))
        return

    instrument = next(
        (
            item
            for item in to_instruments(cfg) + market_index_instruments(cfg)
            if item.symbol == symbol
        ),
        None,
    )
    if instrument is None:
        _fail(f"{symbol} 不在 universe.yaml 或 market.yaml 的指数里")
        return

    raw = store_mod.read_bars_or_empty(
        store_mod.raw_bars_path(cfg.paths, instrument.market, instrument.symbol)
    )
    if raw.empty:
        _fail(f"{symbol} 还没有本地行情，先运行 tradecompass data sync")
        return
    factor = store_mod.read_factor_or_empty(
        store_mod.factor_path(cfg.paths, instrument.market, instrument.symbol)
    )
    qfq = adjust_mod.to_qfq(raw, factor) if not factor.empty else raw

    params = IndicatorParams()
    daily = daily_frame(qfq, params)
    if on_date is None:
        row_date = qfq.index[-1]
    else:
        try:
            row_date = pd.Timestamp(on_date)
        except ValueError:
            _fail(f"--date 格式应为 YYYY-MM-DD，收到 {on_date}")
            return
        if row_date not in qfq.index:
            _fail(f"{on_date} 不是 {symbol} 的交易日")
            return

    weekly = weekly_frame(qfq, row_date.date(), params)
    last_complete = weekly[weekly["is_complete"].astype(bool)].tail(1)

    title = f"{instrument.name} {instrument.symbol}  {row_date.date()}"
    table = Table(title=title, show_header=False)
    table.add_column("项目", width=18)
    table.add_column("数值", justify="right")

    def add(label: str, value: object) -> None:
        if value is None or (isinstance(value, float) and value != value):
            table.add_row(label, "-")
        elif isinstance(value, float):
            table.add_row(label, f"{value:.4f}")
        else:
            table.add_row(label, str(value))

    add("收盘_raw", float(raw.loc[row_date, "close"]))
    add("收盘_qfq", float(qfq.loc[row_date, "close"]))
    factor_value = None
    if not factor.empty and row_date in factor.index:
        base = float(factor.dropna().iloc[0])
        factor_value = float(factor.loc[row_date]) / base if base else None
    add("复权因子(首日=1)", factor_value)
    for period in params.ma_periods:
        add(f"ma{period}", float(daily.loc[row_date, f"ma{period}"]))
    for column in ("k", "d", "j"):
        add(column.upper(), float(daily.loc[row_date, column]))
    add(f"atr{params.atr_n}", float(daily.loc[row_date, f"atr{params.atr_n}"]))
    add("hhv_close_60_prev", float(daily.loc[row_date, "hhv_close_60_prev"]))
    add("llv_low_10", float(daily.loc[row_date, "llv_low_10"]))
    if not last_complete.empty:
        label = pd.Timestamp(last_complete.index[-1]).date()
        add(f"完结周线({label})", None)
        add("  周收盘_qfq", float(last_complete["close"].iloc[-1]))
        add("  ma20w", float(last_complete["ma20w"].iloc[-1]))
        add("  ma20w_up", bool(last_complete["ma20w_up"].iloc[-1]))
        add("  JW", float(last_complete["jw"].iloc[-1]))

    console.print(table)


backtest_app = typer.Typer(add_completion=False, no_args_is_help=True, help="回测")
app.add_typer(backtest_app, name="backtest")



@backtest_app.command("run")
def backtest_run(
    home: HomeOption = None,
    strategy: Annotated[str, typer.Option("--strategy", help="策略 id")] = "ma20_cross",
    symbols: Annotated[
        str | None, typer.Option("--symbols", help="只回测这些代码，逗号分隔")
    ] = None,
    start: Annotated[str | None, typer.Option("--start", help="YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option("--end", help="YYYY-MM-DD")] = None,
    cash: Annotated[float, typer.Option("--cash", help="初始资金")] = 100_000.0,
    report_html: Annotated[bool, typer.Option("--report", help="生成 HTML 报告")] = False,
    sample: Annotated[str, typer.Option("--sample", help="样本标签，仅用于记录")] = "universe",
) -> None:
    """跑一次回测并打印绩效与验收结果。"""
    import pandas as pd

    from tradecompass.backtest.baselines import benchmark_drawdown, equal_weight_buy_hold
    from tradecompass.backtest.engine import run_backtest
    from tradecompass.backtest.loader import load_backtest_data
    from tradecompass.backtest.metrics import compute_metrics, evaluate
    from tradecompass.backtest.params import params_hash, record_run, report_directory
    from tradecompass.backtest.report import report_path, write_report
    from tradecompass.data.sync import to_instruments
    from tradecompass.strategies import AVAILABLE
    from tradecompass.strategies import build as build_strategy

    try:
        target = resolve_home(home)
        cfg = load_config(target)
    except ConfigError as exc:
        _fail(str(exc))
        return

    if strategy not in cfg.strategies:
        _fail(f"策略 {strategy} 没有配置文件；已实现：{sorted(AVAILABLE)}")
        return

    end_date = pd.Timestamp(end).date() if end else date.today()
    start_date = (
        pd.Timestamp(start).date()
        if start
        else end_date.replace(year=end_date.year - cfg.backtest.years)
    )

    instruments = to_instruments(cfg)
    if symbols:
        wanted = {item.strip() for item in symbols.split(",") if item.strip()}
        instruments = [item for item in instruments if item.symbol in wanted]
    if not instruments:
        _fail("没有可回测的标的（检查 --symbols 或 universe.yaml）")
        return

    data_map = {}
    for instrument in instruments:
        data = load_backtest_data(cfg, instrument, start=start_date, end=end_date)
        if data is None:
            console.print(f"[yellow]跳过 {instrument.symbol}：没有本地行情[/yellow]")
            continue
        data_map[instrument.symbol] = data
    if not data_map:
        _fail("没有任何标的具备回测数据，先运行 tradecompass data sync")
        return

    strat = build_strategy(cfg.strategies[strategy], cfg.risk.stops)
    result = run_backtest(
        strat,
        data_map,
        start=start_date,
        end=end_date,
        costs=cfg.backtest.costs,
        initial_cash=cash,
        max_positions=cfg.risk.swing.max_positions,
        size_mode="equal_slots",
        max_gap_open=cfg.backtest.max_gap_open,
    )
    metrics = compute_metrics(result)
    benchmark = equal_weight_buy_hold(data_map, initial_cash=cash)
    bh_dd = benchmark_drawdown(benchmark) if benchmark else None
    report = evaluate(metrics, cfg.backtest.acceptance, benchmark_drawdown=bh_dd)

    summary = Table(title=f"回测 {strategy}  {start_date} → {end_date}", show_header=False)
    summary.add_column("指标", width=16)
    summary.add_column("值", justify="right")
    rows: list[tuple[str, str]] = [
        ("标的数", str(len(data_map))),
        ("交易笔数", str(metrics.trades)),
        ("胜率", f"{metrics.win_rate:.1%}"),
        ("盈亏比", f"{metrics.avg_win / abs(metrics.avg_loss):.2f}" if metrics.avg_loss else "-"),
        ("利润因子", f"{metrics.profit_factor:.2f}"),
        ("单笔期望", f"{metrics.expectancy:.2f}"),
        ("单笔期望(R)", f"{metrics.expectancy_r:.3f}" if metrics.expectancy_r is not None else "-"),
        ("总收益", f"{metrics.total_return:.1%}"),
        ("年化收益", f"{metrics.annual_return:.1%}"),
        ("最大回撤", f"{metrics.max_drawdown:.1%}"),
        ("等权买入持有回撤", f"{bh_dd:.1%}" if bh_dd is not None else "-"),
        ("资金占用率", f"{metrics.capital_utilization:.1%}"),
        ("累计费用", f"{result.fees_total:.2f}"),
    ]
    for label, value in rows:
        summary.add_row(label, value)
    console.print(summary)

    verdict = Table(show_header=True, header_style="bold")
    verdict.add_column("验收项", width=18)
    verdict.add_column("结果", width=6)
    verdict.add_column("说明", overflow="fold")
    for item in report.items:
        mark = "[green]PASS[/green]" if item.ok else "[red]FAIL[/red]"
        verdict.add_row(item.name, mark, escape(item.detail))
    console.print(verdict)
    if report.passed:
        console.print("[green]全部验收项通过[/green]")
    else:
        console.print(f"[red]未通过[/red]：{'、'.join(i.name for i in report.failed())}")

    fingerprint = params_hash(cfg.strategies[strategy].params)
    target_report = ""
    if report_html:
        path = report_path(
            report_directory(cfg.paths.reports),
            strategy,
            fingerprint,
            date.today(),
        )
        write_report(
            path,
            title=f"{strategy} 回测报告",
            subtitle=(
                f"{start_date} → {end_date} · {len(data_map)} 个标的 · "
                f"初始资金 {cash:,.0f} 元 · 参数指纹 {fingerprint}"
            ),
            result=result,
            metrics=metrics,
            acceptance=report,
            benchmark=benchmark,
        )
        target_report = str(path)
        console.print(f"报告：{escape(target_report)}")

    from tradecompass.storage.db import connect as connect_db

    conn = connect_db(cfg.paths.db_path)
    try:
        record_run(
            conn,
            strategy_id=strategy,
            fingerprint=fingerprint,
            sample=sample,
            start=str(start_date),
            end=str(end_date),
            passed=report.passed,
            failed_items=",".join(i.name for i in report.failed()),
            metrics_json=(
                f'{{"trades":{metrics.trades},"pf":{metrics.profit_factor:.4f},'
                f'"return":{metrics.total_return:.4f},"dd":{metrics.max_drawdown:.4f}}}'
            ),
            report_path=target_report,
        )
    finally:
        conn.close()


@backtest_app.command("list")
def backtest_list(
    home: HomeOption = None,
    limit: Annotated[int, typer.Option("--limit", help="显示最近 N 条")] = 20,
) -> None:
    """列出历史回测记录（写入 backtest_runs 表）。"""
    from tradecompass.backtest.params import list_runs
    from tradecompass.storage.db import connect as connect_db

    try:
        target = resolve_home(home)
        cfg = load_config(target)
    except ConfigError as exc:
        _fail(str(exc))
        return

    conn = connect_db(cfg.paths.db_path)
    try:
        rows = list_runs(conn, limit)
    finally:
        conn.close()
    if not rows:
        console.print("还没有回测记录，先运行 tradecompass backtest run")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("时间", width=17)
    table.add_column("策略", width=16)
    table.add_column("参数", width=8)
    table.add_column("样本", width=9)
    table.add_column("区间", width=22)
    table.add_column("结果", width=6)
    table.add_column("未通过项", overflow="fold")
    for row in rows:
        mark = "[green]PASS[/green]" if row["passed"] else "[red]FAIL[/red]"
        table.add_row(
            str(row["run_at"]),
            str(row["strategy_id"]),
            str(row["params_hash"]),
            str(row["sample"]),
            f"{row['start']} → {row['end']}",
            mark,
            escape(str(row["failed_items"] or "")),
        )
    console.print(table)


@backtest_app.command("signals")
def backtest_signals(
    home: HomeOption = None,
    start: Annotated[str | None, typer.Option("--start", help="YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option("--end", help="YYYY-MM-DD")] = None,
    cash: Annotated[float, typer.Option("--cash", help="初始资金")] = 100_000.0,
    random_seeds: Annotated[int, typer.Option("--random-seeds", help="随机对照组数")] = 3,
    atr_mult: Annotated[
        float | None, typer.Option("--atr-mult", help="研究用：覆盖初始止损 ATR 倍数")
    ] = None,
    max_stop_pct: Annotated[
        float | None, typer.Option("--max-stop-pct", help="研究用：覆盖最大止损距离")
    ] = None,
) -> None:
    """回测当前 B1/B2/S1/S2 信号系统（次日开盘、真实费率、组合级风控）。"""
    import dataclasses

    import pandas as pd

    from tradecompass.signals.backtest import run_signal_backtest, write_backtest_report
    from tradecompass.signals.types import SignalParams

    try:
        target = resolve_home(home)
        cfg = load_config(target)
    except ConfigError as exc:
        _fail(str(exc))
        return

    end_date = pd.Timestamp(end).date() if end else date.today()
    start_date = (
        pd.Timestamp(start).date()
        if start
        else end_date.replace(year=end_date.year - cfg.backtest.years)
    )
    params = SignalParams.from_config(cfg)
    if atr_mult is not None or max_stop_pct is not None:
        params = dataclasses.replace(
            params,
            atr_mult=atr_mult if atr_mult is not None else params.atr_mult,
            max_stop_pct=(
                max_stop_pct if max_stop_pct is not None else params.max_stop_pct
            ),
        )

    try:
        outcome = run_signal_backtest(
            cfg,
            start=start_date,
            end=end_date,
            initial_cash=cash,
            params=params,
            random_seeds=random_seeds,
        )
    except ValueError as exc:
        _fail(str(exc), code=3)
        return

    table = Table(
        title=f"信号系统回测  {start_date} → {end_date}",
        show_header=True,
        header_style="bold",
    )
    table.add_column("方案", width=20)
    table.add_column("笔数", justify="right", width=6)
    table.add_column("胜率", justify="right", width=7)
    table.add_column("利润因子", justify="right", width=8)
    table.add_column("总收益", justify="right", width=8)
    table.add_column("年化", justify="right", width=8)
    table.add_column("最大回撤", justify="right", width=9)
    table.add_column("资金占用", justify="right", width=9)
    table.add_column("平均持有", justify="right", width=9)

    def add_row(label: str, metrics: object) -> None:
        from tradecompass.backtest.metrics import Metrics

        assert isinstance(metrics, Metrics)
        pf = "inf" if metrics.profit_factor == float("inf") else f"{metrics.profit_factor:.2f}"
        table.add_row(
            label,
            str(metrics.trades),
            f"{metrics.win_rate:.1%}" if metrics.trades else "-",
            pf if metrics.trades else "-",
            f"{metrics.total_return:+.1%}",
            f"{metrics.annual_return:+.1%}",
            f"{metrics.max_drawdown:.1%}",
            f"{metrics.capital_utilization:.1%}",
            f"{metrics.avg_holding_days:.1f} 天" if metrics.trades else "-",
        )

    add_row("B1/B2/S1/S2", outcome.signal_metrics)
    add_row("等权买入持有", outcome.benchmark_metrics)
    for seed, _, metrics in outcome.random_results:
        add_row(f"随机入场 seed {seed}", metrics)
    console.print(table)
    console.print(
        f"费用合计 {outcome.signal_result.fees_total:,.0f} 元；"
        f"期末仍有 {outcome.active_signals} 只持仓；"
        f"随机入场率 {outcome.random_rate:.4f}"
    )

    if outcome.trade_breakdown:
        kind_table = Table(title="按入场信号", show_header=True, header_style="bold")
        kind_table.add_column("信号", width=6)
        kind_table.add_column("笔数", justify="right", width=6)
        kind_table.add_column("胜率", justify="right", width=7)
        kind_table.add_column("平均净盈亏", justify="right", width=11)
        kind_table.add_column("平均持有", justify="right", width=9)
        for kind, trades in sorted(outcome.trade_breakdown.items()):
            wins = [item.net_pnl for item in trades if item.net_pnl > 0]
            kind_table.add_row(
                kind,
                str(len(trades)),
                f"{len(wins) / len(trades):.1%}",
                f"{sum(item.net_pnl for item in trades) / len(trades):+.2f}",
                f"{sum(item.holding_days for item in trades) / len(trades):.1f} 天",
            )
        console.print(kind_table)
    else:
        console.print("[yellow]回测期内没有任何完成的交易[/yellow]")

    tag = f"signal_system_dynamic_{start_date}_{end_date}"
    if atr_mult is not None or max_stop_pct is not None:
        tag += f"_atr{params.atr_mult}_stop{params.max_stop_pct}"
    path = write_backtest_report(cfg, outcome, tag=tag)
    console.print(f"报告：{escape(str(path))}")
    console.print(
        "[yellow]折扣：[/yellow]自选池有幸存者偏差；回测从空仓开始，不是当前持仓的复现；"
        "随机对照只校准入场次数，不能完全消除路径差异。"
    )


@backtest_app.command("b2")
def backtest_b2(
    home: HomeOption = None,
    start: Annotated[str | None, typer.Option("--start", help="YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option("--end", help="YYYY-MM-DD")] = None,
    cash: Annotated[float, typer.Option("--cash", help="初始资金")] = 100_000.0,
    breakeven_r: Annotated[
        float | None, typer.Option("--breakeven-r", help="研究用：保本止损触发 R")
    ] = None,
    lock_r: Annotated[
        float | None, typer.Option("--lock-r", help="研究用：锁定 +0.5R 的触发 R")
    ] = None,
    trail_r: Annotated[
        float | None, typer.Option("--trail-r", help="研究用：ATR 跟踪止损触发 R")
    ] = None,
    trail_atr: Annotated[
        float | None, typer.Option("--trail-atr", help="研究用：跟踪止损 ATR 倍数")
    ] = None,
) -> None:
    """只分析 B2：亏损来自入场、退出还是仓位。"""
    import dataclasses

    import pandas as pd

    from tradecompass.signals.b2_analysis import analyze_b2, write_b2_report
    from tradecompass.signals.backtest import run_signal_backtest
    from tradecompass.signals.types import SignalParams

    try:
        target = resolve_home(home)
        cfg = load_config(target)
    except ConfigError as exc:
        _fail(str(exc))
        return

    end_date = pd.Timestamp(end).date() if end else date.today()
    start_date = (
        pd.Timestamp(start).date()
        if start
        else end_date.replace(year=end_date.year - cfg.backtest.years)
    )
    params = SignalParams.from_config(cfg)
    if any(
        value is not None
        for value in (breakeven_r, lock_r, trail_r, trail_atr)
    ):
        params = dataclasses.replace(
            params,
            breakeven_at_r=(
                breakeven_r if breakeven_r is not None else params.breakeven_at_r
            ),
            lock_half_r_at_r=lock_r if lock_r is not None else params.lock_half_r_at_r,
            trail_at_r=trail_r if trail_r is not None else params.trail_at_r,
            trail_atr_mult=(
                trail_atr if trail_atr is not None else params.trail_atr_mult
            ),
        )
    try:
        outcome = run_signal_backtest(
            cfg,
            start=start_date,
            end=end_date,
            initial_cash=cash,
            params=params,
            random_seeds=0,
        )
        analysis = analyze_b2(outcome)
    except ValueError as exc:
        _fail(str(exc), code=3)
        return

    total = analysis.total
    console.print(
        f"[bold]B2 归因 {start_date} → {end_date}[/bold]\n"
        f"已平仓 {total.count} 笔，胜率 {total.win_rate:.1%}，"
        f"PF {total.profit_factor:.2f}，平均 {total.avg_pnl:+.2f} 元/笔\n"
        f"初始止损率 {total.stop_rate:.1%}；平均 MFE {total.avg_mfe_pct:+.1%}，"
        f"平均 MAE {total.avg_mae_pct:+.1%}，平均 MFE {total.avg_mfe_r:.2f}R"
    )
    console.print(
        f"亏损单 {analysis.losers} 笔：从未盈利 {analysis.never_profit:.1%}；"
        f"曾到 +0.5R 后失败 {analysis.half_r_fail:.1%}；"
        f"曾到 +1R 后失败 {analysis.one_r_fail:.1%}\n"
        f"亏损单平均持有 {analysis.avg_loss_hold:.1f} 天，"
        f"盈利单平均持有 {analysis.avg_win_hold:.1f} 天"
    )
    if analysis.worst_symbols:
        worst = "、".join(
            f"{symbol}({pnl:+.0f})" for symbol, pnl, _ in analysis.worst_symbols[:3]
        )
        console.print(f"最亏的票：{worst}")
    path = write_b2_report(cfg, analysis, end=end_date)
    console.print(f"报告：{escape(str(path))}")


@app.command("import-trades")
def import_trades(
    file: Annotated[Path, typer.Argument(help="交割单文件（csv/txt/xlsx/xls）")],
    home: HomeOption = None,
    broker: Annotated[str | None, typer.Option("--broker", help="券商模板名")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="只解析不写库")] = False,
) -> None:
    """导入交割单。未识别的操作会列出来，不会静默丢弃。"""
    from tradecompass.account.importer import (
        ImportError_,
        import_into,
        parse_statement,
        read_statement,
    )
    from tradecompass.storage.db import connect as connect_db

    try:
        target = resolve_home(home)
        cfg = load_config(target)
    except ConfigError as exc:
        _fail(str(exc))
        return

    name = broker or cfg.broker.default
    mapping = cfg.broker.mappings.get(name)
    if mapping is None:
        _fail(f"券商模板 {name} 不存在；已配置：{sorted(cfg.broker.mappings)}")
        return

    try:
        frame = read_statement(file)
        result = parse_statement(frame, mapping)
    except ImportError_ as exc:
        _fail(str(exc))
        return

    console.print(
        f"解析到 {len(result.trades)} 笔成交，忽略 {result.ignored} 笔资金划转/利息，"
        f"未识别 {len(result.issues)} 行"
    )
    if result.issues:
        table = Table(show_header=True, header_style="bold")
        table.add_column("行号", width=6)
        table.add_column("原因", overflow="fold")
        table.add_column("原始内容", overflow="fold")
        for issue in result.issues[:20]:
            table.add_row(str(issue.row_no), escape(issue.reason), escape(issue.raw_json[:80]))
        console.print(table)

    if dry_run:
        console.print("[yellow]--dry-run：未写库[/yellow]")
        return

    conn = connect_db(cfg.paths.db_path)
    try:
        inserted, duplicated, issues = import_into(conn, result)
    finally:
        conn.close()
    console.print(
        f"[green]新增 {inserted} 笔[/green]，重复跳过 {duplicated} 笔，记录问题 {issues} 条"
    )


@app.command("checkup")
def checkup(home: HomeOption = None) -> None:
    """持仓体检：周线趋势、盈亏、止损建议与处置动作。"""
    from tradecompass.account.checkup import build_report
    from tradecompass.account.ledger import positions_from_db
    from tradecompass.storage.db import connect as connect_db

    try:
        target = resolve_home(home)
        cfg = load_config(target)
    except ConfigError as exc:
        _fail(str(exc))
        return

    conn = connect_db(cfg.paths.db_path)
    try:
        holdings = positions_from_db(conn)
    finally:
        conn.close()
    if not holdings:
        console.print("没有持仓记录，先运行 tradecompass import-trades <交割单>")
        return

    report = build_report(cfg, holdings)
    table = Table(title=f"持仓体检  {report.as_of}", show_header=True, header_style="bold")
    table.add_column("代码", width=7)
    table.add_column("名称", width=10)
    table.add_column("股数", justify="right", width=7)
    table.add_column("成本", justify="right", width=8)
    table.add_column("现价", justify="right", width=8)
    table.add_column("盈亏", justify="right", width=9)
    table.add_column("周线", width=5)
    table.add_column("止损建议", justify="right", width=9)
    table.add_column("结论", width=14)

    for row in report.rows:
        colour = "green" if row.pnl >= 0 else "red"
        weekly = "-" if row.weekly_ok is None else ("多" if row.weekly_ok else "空")
        price = f"{row.last_close:.3f}" if row.has_data else "-"
        stop = f"{row.suggested_stop:.2f}" if row.suggested_stop else "-"
        table.add_row(
            row.symbol, row.name, str(row.quantity), f"{row.avg_cost:.3f}", price,
            f"[{colour}]{row.pnl:+.0f}[/{colour}]", weekly, stop, escape(row.verdict),
        )
    console.print(table)
    console.print(
        f"市值 {report.total_value:,.0f} / 成本 {report.total_cost:,.0f} / "
        f"浮盈亏 [{('green' if report.total_pnl >= 0 else 'red')}]{report.total_pnl:+,.0f}"
        f" ({report.total_pnl_pct:+.1%})[/]"
    )
    alerts = report.alerts
    if alerts:
        console.print(f"\n[red]需要处理 {len(alerts)} 只[/red]")
        for row in alerts:
            console.print(
                f"  [bold]{row.symbol} {escape(row.name)}[/bold]  "
                f"[red]{row.pnl_pct:+.1%}[/red]  {escape(row.action)}"
            )


@app.command("regime")
def regime(
    home: HomeOption = None,
    days: Annotated[int, typer.Option("--days", help="回看最近 N 个交易日")] = 15,
) -> None:
    """市场环境：当前状态、仓位额度，以及最近的状态变化。"""
    from tradecompass.data import store as store_mod
    from tradecompass.market.regime import build_series
    from tradecompass.market.sectors import buyable, ranking

    try:
        target = resolve_home(home)
        cfg = load_config(target)
    except ConfigError as exc:
        _fail(str(exc))
        return

    index_symbol = cfg.market.indices.hs300
    bars = store_mod.read_bars_or_empty(
        store_mod.raw_bars_path(cfg.paths, "SH", index_symbol)
    )
    if bars.empty:
        _fail(f"没有沪深300（{index_symbol}）行情，先运行 data sync")
        return

    points = build_series(bars)
    if not points:
        _fail("指数历史不足，无法判定市场环境")
        return

    latest = points[-1]
    console.print(
        f"沪深300 {index_symbol}  截至 {latest.day}\n"
        f"状态 [bold]{latest.state}[/bold]  仓位额度 [bold]{latest.quota:.0%}[/bold]"
        + (f"  （{latest.note}）" if latest.note else "")
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("日期", width=12)
    table.add_column("原始", width=10)
    table.add_column("确认后", width=10)
    table.add_column("额度", justify="right", width=6)
    table.add_column("说明", overflow="fold")
    previous = None
    for point in points[-days:]:
        if point.state != previous:
            table.add_row(
                str(point.day), str(point.raw), str(point.state),
                f"{point.quota:.0%}", escape(point.note or ""),
            )
            previous = point.state
    console.print(table)

    ranks = ranking(cfg, latest.day)
    if ranks:
        top = buyable(ranks)
        sector_table = Table(title="行业动量（前 5）", show_header=True, header_style="bold")
        sector_table.add_column("排名", width=5)
        sector_table.add_column("行业", width=10)
        sector_table.add_column("20日动量", justify="right", width=10)
        sector_table.add_column("MA20", width=6)
        sector_table.add_column("可买", width=6)
        for item in ranks[:5]:
            mark = "[green]是[/green]" if item.code in top else "—"
            sector_table.add_row(
                str(item.rank), item.name, f"{item.momentum:+.2%}",
                "上" if item.above_ma20 else "下", mark,
            )
        console.print(sector_table)
        if not top:
            console.print("[yellow]当前没有行业满足买入条件，个股层应保持空仓[/yellow]")
    else:
        console.print("[yellow]没有行业指数数据：先同步 000928–000937 再运行[/yellow]")


@app.command("daily")
def daily(
    home: HomeOption = None,
    on_date: Annotated[
        str | None,
        typer.Option("--date", help="指定日期 YYYY-MM-DD（复盘用；持仓仍用当前快照）"),
    ] = None,
    min_confidence: Annotated[
        int | None, typer.Option("--min-confidence", help="覆盖买卖信号的置信度阈值")
    ] = None,
    push: Annotated[bool, typer.Option("--push", help="推送到飞书；未配置时写预览")] = False,
) -> None:
    """每日扫描：市场 → 行业 → B1/B2/S1/S2 → 行动清单。"""
    import dataclasses

    import pandas as pd

    from tradecompass.notify.feishu import send_text
    from tradecompass.signals.daily import DailyError, build_daily, render_markdown, write_report
    from tradecompass.signals.types import SignalParams

    try:
        target = resolve_home(home)
        cfg = load_config(target)
    except ConfigError as exc:
        _fail(str(exc))
        return

    as_of = None
    if on_date is not None:
        try:
            as_of = pd.Timestamp(on_date).date()
        except ValueError:
            _fail(f"--date 格式应为 YYYY-MM-DD，收到 {on_date}")
            return

    params = SignalParams.from_config(cfg)
    if min_confidence is not None:
        params = dataclasses.replace(
            params,
            min_confidence_buy=min_confidence,
            min_confidence_b2=min_confidence,
            min_confidence_sell=min_confidence,
        )

    try:
        report = build_daily(cfg, as_of=as_of, params=params)
    except DailyError as exc:
        _fail(str(exc), code=3)
        return

    market = report.market
    header = (
        f"[bold]TradeCompass 每日信号 {report.as_of}[/bold]\n"
        f"沪深300 [bold]{market.state}[/bold]  额度 [bold]{market.quota:.0%}[/bold]"
    )
    if market.ret20 is not None:
        header += f"  20日动量 {market.ret20:+.1%}"
    if market.vol_ratio20 is not None:
        header += f"  量比20 {market.vol_ratio20:.2f}"
    console.print(header)
    console.print(
        f"持仓来源：{_portfolio_source_text(report.portfolio.source)}，"
        f"权益 {report.portfolio.equity:,.0f} 元，持仓 {report.portfolio.holdings_count} 只"
    )

    if report.sectors:
        sector_table = Table(title="行业（前 5）", show_header=True, header_style="bold")
        sector_table.add_column("排名", width=5)
        sector_table.add_column("行业", width=10)
        sector_table.add_column("20日动量", justify="right", width=10)
        sector_table.add_column("量比20", justify="right", width=8)
        sector_table.add_column("MA20", width=6)
        sector_table.add_column("可买", width=6)
        for item in report.sectors:
            sector_table.add_row(
                str(item.rank),
                item.name,
                f"{item.momentum:+.1%}",
                f"{item.vol_ratio20:.2f}" if item.vol_ratio20 is not None else "-",
                "上" if item.above_ma20 else "下",
                "[green]是[/green]" if item.code in report.buyable else "—",
            )
        console.print(sector_table)

    if report.actions:
        action_table = Table(
            title=f"今日操作 {report.as_of}", show_header=True, header_style="bold"
        )
        _add_signal_columns(action_table)
        for signal in report.actions:
            _add_signal_row(action_table, signal)
        console.print(action_table)
    else:
        console.print("[yellow]今日没有达到置信度阈值的操作信号[/yellow]")

    if report.watch:
        watch_table = Table(title="观察区（不行动）", show_header=True, header_style="bold")
        _add_signal_columns(watch_table)
        for signal in report.watch:
            _add_signal_row(watch_table, signal)
        console.print(watch_table)

    if report.errors:
        console.print(f"[yellow]数据提示 {len(report.errors)} 条[/yellow]")
        for error in report.errors[:5]:
            console.print(f"  - {escape(error)}")
        if len(report.errors) > 5:
            console.print(f"  ……另有 {len(report.errors) - 5} 条")

    markdown_path, json_path = write_report(cfg, report)
    console.print(f"报告：{escape(str(markdown_path))}")
    console.print(f"数据：{escape(str(json_path))}")

    if push or cfg.system.notify.enabled:
        status = send_text(cfg, render_markdown(report), name="daily")
        console.print(f"推送：{escape(status)}")


def _add_signal_columns(table: Table) -> None:
    table.add_column("代码", width=7)
    table.add_column("名称", width=10)
    table.add_column("信号", width=5)
    table.add_column("置信/门槛", justify="right", width=8)
    table.add_column("动作", width=13)
    table.add_column("参考价", justify="right", width=8)
    table.add_column("止损", justify="right", width=8)
    table.add_column("数量", justify="right", width=7)
    table.add_column("理由", overflow="fold")


def _add_signal_row(table: Table, signal: object) -> None:
    from tradecompass.signals.types import Signal

    assert isinstance(signal, Signal)
    style = "green" if signal.kind in ("B1", "B2") else "red"
    reasons = "；".join(signal.reasons[:4])
    if signal.warnings:
        reasons += "（注意：" + "、".join(signal.warnings[:2]) + "）"
    table.add_row(
        signal.symbol,
        signal.name,
        f"[{style}]{signal.kind}[/{style}]",
        (
            f"{signal.confidence}/{signal.threshold}"
            if signal.threshold
            else str(signal.confidence)
        ),
        signal.action,
        f"{signal.trigger_price_raw:.2f}" if signal.trigger_price_raw else "-",
        f"{signal.stop_price_raw:.2f}" if signal.stop_price_raw else "-",
        str(signal.suggested_quantity) if signal.suggested_quantity else "-",
        escape(reasons),
    )


def _portfolio_source_text(source: str) -> str:
    return {
        "trades": "交割单流水",
        "opening": "期初持仓快照",
        "none": "未配置（按空仓）",
    }.get(source, source)


factors_app = typer.Typer(add_completion=False, no_args_is_help=True, help="因子快照")
app.add_typer(factors_app, name="factors")


@factors_app.command("show")
def factors_show(
    symbol: Annotated[str, typer.Argument(help="6 位代码")],
    home: HomeOption = None,
    on_date: Annotated[
        str | None, typer.Option("--date", help="指定日期 YYYY-MM-DD")
    ] = None,
) -> None:
    """打印日线 / 周线 / 行业 / 大盘的因子快照。"""
    import pandas as pd

    from tradecompass.data import store as store_mod
    from tradecompass.factors.engine import (
        build_market_context,
        build_sector_context,
        load_snapshot,
    )
    from tradecompass.market.sectors import ranking

    try:
        target = resolve_home(home)
        cfg = load_config(target)
    except ConfigError as exc:
        _fail(str(exc))
        return

    instrument = next(
        (
            item
            for item in to_instruments(cfg) + market_index_instruments(cfg)
            if item.symbol == symbol
        ),
        None,
    )
    if instrument is None:
        _fail(f"{symbol} 不在 universe.yaml 或 market.yaml 的指数里")
        return

    index_bars = store_mod.read_bars_or_empty(
        store_mod.raw_bars_path(cfg.paths, "SH", cfg.market.indices.hs300)
    )
    if index_bars.empty:
        _fail("没有沪深300 行情，先运行 data sync")
        return
    as_of = pd.Timestamp(on_date).date() if on_date else pd.Timestamp(
        index_bars.index[-1]
    ).date()
    market = build_market_context(index_bars, as_of)
    ranks = ranking(cfg, as_of)
    sector = build_sector_context(instrument.symbol, instrument.sector_sw, ranks)
    snapshot = load_snapshot(cfg, instrument, market=market, sector=sector, as_of=as_of)
    if snapshot is None:
        _fail(f"{symbol} 没有行情或历史不足，先运行 data sync")
        return

    console.print(
        f"[bold]{snapshot.name} {snapshot.symbol}[/bold]  {snapshot.as_of}\n"
        f"不复权收盘 {snapshot.close_raw:.3f}  /  前复权 {snapshot.close_qfq:.3f}  /  "
        f"周线截至 {snapshot.weekly_as_of}"
    )

    groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
        (
            "日线趋势",
            (
                ("above_ma5", "站上MA5"),
                ("above_ma10", "站上MA10"),
                ("above_ma20", "站上MA20"),
                ("above_ma60", "站上MA60"),
                ("ma20_up", "MA20向上"),
                ("ma60_up", "MA60向上"),
                ("ma_bull_aligned", "均线多头排列"),
                ("dist_ma20_pct", "距MA20"),
            ),
        ),
        (
            "日线振荡/动量",
            (
                ("j", "日J"),
                ("j_up", "日J回升"),
                ("rsi6", "RSI6"),
                ("rsi12", "RSI12"),
                ("rsi24", "RSI24"),
                ("macd_hist", "MACD柱"),
                ("macd_hist_up", "MACD柱走高"),
            ),
        ),
        (
            "位置/量能",
            (
                ("pos_60", "60日位置"),
                ("pos_250", "250日位置"),
                ("ret5", "5日涨幅"),
                ("ret20", "20日涨幅"),
                ("vol_ratio5", "量比5"),
                ("vol_ratio20", "量比20"),
                ("vol_ma5_gt_ma20", "5日均量>20日"),
                ("vol_price_dir", "量价方向"),
            ),
        ),
        (
            "周线",
            (
                ("weekly_bull", "周线多头"),
                ("weekly_bull_aligned", "周线多头排列"),
                ("jw", "周J"),
                ("jw_up", "周J回升"),
                ("ma20w_up", "周MA20向上"),
                ("rsi6w", "周RSI6"),
                ("rsi12w", "周RSI12"),
                ("macd_histw", "周MACD柱"),
                ("vol_ratio5w", "周量比5"),
                ("ret4w", "4周涨幅"),
            ),
        ),
    )
    percent_keys = {"dist_ma20_pct", "ret5", "ret20", "ret4w"}
    for title, items in groups:
        table = Table(title=title, show_header=False, padding=(0, 1))
        table.add_column("因子", width=16)
        table.add_column("值", justify="right")
        for key, label in items:
            table.add_row(label, _format_factor(snapshot.factors.get(key), key in percent_keys))
        console.print(table)

    context = Table(title="行业/大盘", show_header=False, padding=(0, 1))
    context.add_column("项目", width=16)
    context.add_column("值", justify="right")
    context.add_row("大盘状态", f"{market.state}（额度 {market.quota:.0%}）")
    context.add_row("大盘20日动量", f"{market.ret20:+.1%}" if market.ret20 is not None else "-")
    context.add_row(
        "大盘量比20", f"{market.vol_ratio20:.2f}" if market.vol_ratio20 is not None else "-"
    )
    if sector is None:
        context.add_row("行业", "未映射")
    else:
        context.add_row("行业", sector.name or sector.code or "-")
        context.add_row(
            "行业排名",
            f"{sector.rank}/{sector.count}" if sector.rank is not None else "-",
        )
        context.add_row(
            "行业动量",
            f"{sector.momentum:+.1%}" if sector.momentum is not None else "-",
        )
        context.add_row(
            "行业量比20",
            f"{sector.vol_ratio20:.2f}" if sector.vol_ratio20 is not None else "-",
        )
        context.add_row("行业强度", sector.strength)
    console.print(context)


def _format_factor(value: object, percent: bool) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:+.1%}" if percent else f"{value:.2f}"
    return str(value)
