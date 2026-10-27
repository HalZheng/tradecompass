"""收盘后一键更新：完整性检查的两档严重程度、路径选择与描述文本。"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from tradecompass.config.loader import Paths
from tradecompass.data.sources.base import SourceError
from tradecompass.research import data as rdata
from tradecompass.research.update import (
    Completeness,
    Gap,
    check_completeness,
    describe,
    expected_bar_date,
    intraday_refusal,
    latest_constituent_codes,
    retry_via_baostock,
)


def test_intraday_update_is_refused_with_a_reason() -> None:
    """盘中跑 update 拿不到当日 K 线，还会让日志看起来像成功了，必须挡在门口。"""
    day = date(2026, 10, 26)
    morning = datetime(2026, 10, 26, 10, 30)
    message = intraday_refusal(day, now=morning)
    assert message is not None
    assert "还没收盘" in message and "--as-of" in message
    # 收盘后放行
    assert intraday_refusal(day, now=datetime(2026, 10, 26, 15, 30)) is None
    # 期望日不是今天（周末/节假日）时不受影响
    assert intraday_refusal(date(2026, 10, 23), now=morning) is None


def test_drop_unclosed_bar_only_strips_today_before_close() -> None:
    """半日 K 线不落库；收盘后同一根要保留。"""
    index = pd.DatetimeIndex(["2026-10-22", "2026-10-23", "2026-10-26"], name="date")
    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=index)
    intraday = datetime(2026, 10, 26, 10, 30)
    assert len(rdata.drop_unclosed_bar(frame, now=intraday)) == 2
    assert len(rdata.drop_unclosed_bar(frame, now=datetime(2026, 10, 26, 15, 30))) == 3
    # 没有当日行时原样返回
    assert len(rdata.drop_unclosed_bar(frame.iloc[:2], now=intraday)) == 2
    assert rdata.drop_unclosed_bar(pd.DataFrame(), now=intraday).empty


EXPECTED = date(2026, 10, 23)


def _paths(tmp_path: Path) -> Paths:
    return Paths.from_home(tmp_path)


def _write_bars(path: Path, days: list[date]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"close": 1.0}, index=pd.DatetimeIndex(days, name="date"))
    frame.to_parquet(path)


def _write_stock(paths: Paths, code: str, days: list[date]) -> None:
    _write_bars(rdata.stock_bars_path(paths, rdata.to_bs_code(code)), days)


def _write_etf(paths: Paths, code: str, days: list[date]) -> None:
    _write_bars(rdata.etf_bars_path(paths, rdata.to_bs_code(code)), days)


def test_expected_bar_date_rolls_back_to_previous_session() -> None:
    assert expected_bar_date(date(2026, 10, 23)) == date(2026, 10, 23)
    assert expected_bar_date(date(2026, 10, 24)) == date(2026, 10, 23)
    assert expected_bar_date(date(2026, 10, 26)) == date(2026, 10, 26)


def test_to_bs_code_splits_shanghai_and_shenzhen() -> None:
    assert rdata.to_bs_code("600000") == "sh.600000"
    assert rdata.to_bs_code("510300") == "sh.510300"
    assert rdata.to_bs_code("688981") == "sh.688981"
    assert rdata.to_bs_code("000001") == "sz.000001"
    assert rdata.to_bs_code("159201") == "sz.159201"
    # 已带前缀的原样返回，便于调用方两种写法都传
    assert rdata.to_bs_code("sh.600000") == "sh.600000"


def test_complete_holdings_and_constituents_pass(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_stock(paths, "600000", [EXPECTED])
    _write_stock(paths, "600001", [EXPECTED])
    _write_etf(paths, "510300", [EXPECTED])
    result = check_completeness(
        paths, expected=EXPECTED, holding_codes=["600000", "510300"],
        constituent_codes=["sh.600001"],
    )
    assert result.ok and result.gaps == ()
    assert (result.holdings, result.constituents) == (2, 1)


def test_missing_bar_for_holding_blocks(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_stock(paths, "600000", [date(2026, 10, 22)])
    result = check_completeness(
        paths, expected=EXPECTED, holding_codes=["600000"], constituent_codes=[]
    )
    assert not result.ok
    gap = result.blocking[0]
    assert gap.code == "sh.600000" and gap.role == "持仓" and not gap.stale
    assert gap.last_date == date(2026, 10, 22)


def test_long_stale_constituent_only_warns(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    # 2026-09-30 距离 10-23 远超 5 个交易日：按长期停牌/退市处理
    _write_stock(paths, "600001", [date(2026, 9, 30)])
    result = check_completeness(
        paths, expected=EXPECTED, holding_codes=[], constituent_codes=["600001"]
    )
    assert result.ok
    assert [gap.code for gap in result.stale] == ["sh.600001"]


def test_recent_gap_for_constituent_blocks(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_stock(paths, "600002", [date(2026, 10, 22)])
    result = check_completeness(
        paths, expected=EXPECTED, holding_codes=[], constituent_codes=["600002"]
    )
    assert not result.ok
    assert result.blocking[0].role == "成分股"


def test_holding_role_wins_over_constituent(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_stock(paths, "600003", [date(2026, 10, 22)])
    result = check_completeness(
        paths, expected=EXPECTED, holding_codes=["600003"], constituent_codes=["600003"]
    )
    assert len(result.gaps) == 1
    assert result.gaps[0].role == "持仓"
    assert (result.holdings, result.constituents) == (1, 0)


def test_missing_file_blocks(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result = check_completeness(
        paths, expected=EXPECTED, holding_codes=["600004"], constituent_codes=[]
    )
    assert not result.ok and result.gaps[0].last_date is None


def test_latest_constituent_codes_uses_newest_snapshot(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    path = rdata.constituents_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-31", "2026-08-31", "2026-09-30"]),
            "code": ["sh.600000", "sh.600001", "sh.600002"],
            "index": ["hs300", "hs300", "hs300"],
        }
    ).to_parquet(path, index=False)
    assert latest_constituent_codes(paths) == ["sh.600002"]


def test_describe_separates_blocking_and_stale() -> None:
    result = Completeness(
        expected=EXPECTED,
        holdings=2,
        constituents=1,
        gaps=(
            Gap("sh.600000", "持仓", date(2026, 10, 22), stale=False),
            Gap("sh.600001", "成分股", date(2026, 9, 1), stale=True),
        ),
    )
    text = "\n".join(describe(result))
    assert "持仓 2 只、成分股 1 只" in text
    assert "✗ 持仓 sh.600000 最近一根 K 线 2026-10-22" in text
    assert "长期停牌/退市" in text and "不阻断" in text


def test_retry_via_baostock_fills_only_the_failed_codes(tmp_path: Path, monkeypatch) -> None:
    """新浪取不到的 CDR 用 baostock 兜底；只补传入的代码，不扩大到全池。"""
    paths = _paths(tmp_path)
    calls: list[str] = []

    def fake_sync(paths_, code, start, end):
        calls.append(code)
        return 1

    monkeypatch.setattr(rdata, "sync_stock_bars", fake_sync)
    fixed, failures = retry_via_baostock(paths, ["sh.689009"], date(2026, 9, 1), EXPECTED)
    assert fixed == 1 and failures == []
    assert calls == ["sh.689009"]


def test_retry_via_baostock_reports_codes_that_still_fail(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)

    def boom(paths_, code, start, end):
        raise SourceError("baostock 不可用")

    monkeypatch.setattr(rdata, "sync_stock_bars", boom)
    fixed, failures = retry_via_baostock(paths, ["sh.689009"], date(2026, 9, 1), EXPECTED)
    assert fixed == 0
    assert failures == [("sh.689009", "baostock 不可用")]


def test_update_falls_back_to_baostock_for_critical_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """一只关键票在新浪失败时，update 必须自动兜底而不是直接挡死每日报告。"""
    from typer.testing import CliRunner

    from tradecompass.bootstrap import initialize
    from tradecompass.cli import app
    from tradecompass.research import update as rupdate

    home = tmp_path / "home"
    initialize(home)
    monkeypatch.setattr(
        rdata,
        "load_constituents",
        lambda paths: pd.DataFrame(
            {"date": pd.to_datetime(["2026-09-30"]), "code": ["sh.689009"], "index": ["hs300"]}
        ),
    )
    monkeypatch.setattr(rupdate, "latest_constituent_codes", lambda paths: ["sh.689009"])
    monkeypatch.setattr(
        rupdate, "sync_stocks_fast", lambda *args, **kwargs: (0, [("sh.689009", "sina fail")])
    )
    monkeypatch.setattr(rupdate, "sync_etf_and_index", lambda *args, **kwargs: (47, []))
    calls: list[list[str]] = []

    def fake_retry(paths, codes, start, end, **kwargs):
        calls.append(list(codes))
        return 1, []

    monkeypatch.setattr(rupdate, "retry_via_baostock", fake_retry)
    monkeypatch.setattr(
        rupdate,
        "check_completeness",
        lambda *args, **kwargs: Completeness(
            expected=date(2026, 10, 26), holdings=0, constituents=1, gaps=()
        ),
    )

    result = CliRunner().invoke(
        app, ["update", "--home", str(home), "--as-of", "2026-10-26"]
    )

    assert result.exit_code == 0
    assert "baostock 补齐关键标的 1/1" in result.output
    assert calls == [["sh.689009"]]


def test_plan_refuses_to_write_when_holding_has_no_bar(tmp_path: Path) -> None:
    """闸门不通过时 plan 必须拒绝生成，而不是写一份看着正常的报告。"""
    from typer.testing import CliRunner

    from tradecompass.bootstrap import initialize
    from tradecompass.cli import app

    home = tmp_path / "home"
    initialize(home)
    folder = home / "portfolio" / "eastmoney"
    folder.mkdir(parents=True)
    pd.DataFrame(
        {
            "code": ["600000"],
            "name": ["浦发银行"],
            "market": ["沪市A股"],
            "quantity": [100],
            "price": [10.0],
            "cost": [9.0],
            "value": [1000.0],
            "pnl": [100.0],
            "pnl_pct": [0.11],
        }
    ).to_csv(folder / "holdings_2026-10-23.csv", index=False)
    (folder / "account_2026-10-23.yaml").write_text(
        "summary:\n  总资产: 1000.0\n  资金可用: 100.0\n", encoding="utf-8"
    )
    pd.DataFrame({"date": [], "code": [], "side": []}).to_csv(folder / "trades.csv", index=False)

    result = CliRunner().invoke(app, ["plan", "--home", str(home)])

    assert result.exit_code == 3
    assert "数据不完整" in result.output
    assert not (home / "reports" / "plan").exists() or not list(
        (home / "reports" / "plan").glob("*.md")
    )
