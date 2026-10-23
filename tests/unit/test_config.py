"""M0：配置加载与校验（SPEC §3.5、§15）。"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tradecompass.config import ConfigError, load_config


def _write(home: Path, relative: str, body: str) -> None:
    path = home / "config" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")


def test_repository_templates_load(home: Path) -> None:
    cfg = load_config(home)

    assert len(cfg.universe.instruments) == 5
    assert sum(1 for i in cfg.universe.instruments if i.kind == "base_etf") == 3
    assert set(cfg.strategies) == {
        "b1_pullback",
        "base_rebalance",
        "legacy_cleanup",
        "volume_breakout",
    }
    assert cfg.enabled_strategies().keys() == {"base_rebalance", "legacy_cleanup"}
    assert cfg.system.notify.enabled is False
    assert cfg.risk.swing.max_positions == 5
    assert cfg.backtest.costs.commission == pytest.approx(0.00025)


def test_opening_positions_are_optional_and_loaded(home: Path) -> None:
    cfg = load_config(home)

    assert cfg.opening_positions is not None
    assert [item.symbol for item in cfg.opening_positions.positions] == ["510300", "518880"]
    assert cfg.opening_positions.positions[0].avg_cost == pytest.approx(3.043)


def test_missing_file_names_the_file(home: Path) -> None:
    (home / "config" / "system.yaml").unlink()

    with pytest.raises(ConfigError) as exc:
        load_config(home)

    assert "system.yaml" in str(exc.value)
    assert "文件不存在" in str(exc.value)


def test_unknown_key_is_rejected(home: Path) -> None:
    _write(
        home,
        "risk.yaml",
        """
        cash: 1000
        swing:
          max_positions: 3
          unknown_field: 1
        """,
    )

    with pytest.raises(ConfigError) as exc:
        load_config(home)

    message = str(exc.value)
    assert "risk.yaml" in message
    assert "swing.unknown_field" in message
    assert "未知的配置键" in message


def test_invalid_clock_format_is_rejected(home: Path) -> None:
    _write(
        home,
        "system.yaml",
        """
        schedule:
          eod_push: "25:00"
        """,
    )

    with pytest.raises(ConfigError) as exc:
        load_config(home)

    message = str(exc.value)
    assert "system.yaml" in message
    assert "schedule.eod_push" in message


def test_invalid_timezone_is_rejected(home: Path) -> None:
    _write(home, "system.yaml", "schedule: {timezone: Mars/Olympus}\n")

    with pytest.raises(ConfigError) as exc:
        load_config(home)

    assert "未知时区" in str(exc.value)


def test_base_targets_must_sum_to_one(home: Path) -> None:
    _write(
        home,
        "universe.yaml",
        """
        instruments:
          - {symbol: "510300", market: SH, name: A, kind: base_etf, target: 0.5}
          - {symbol: "518880", market: SH, name: B, kind: base_etf, target: 0.3}
        """,
    )

    with pytest.raises(ConfigError) as exc:
        load_config(home)

    message = str(exc.value)
    assert "universe.yaml" in message
    assert "之和必须为 1.0" in message
    assert "0.800000" in message


def test_non_base_kind_must_not_have_target(home: Path) -> None:
    _write(
        home,
        "universe.yaml",
        """
        instruments:
          - {symbol: "510300", market: SH, name: A, kind: base_etf, target: 1.0}
          - {symbol: "600000", market: SH, name: B, kind: stock, target: 0.2}
        """,
    )

    with pytest.raises(ConfigError) as exc:
        load_config(home)

    assert "不得设置 target" in str(exc.value)


def test_base_etf_requires_target(home: Path) -> None:
    _write(
        home,
        "universe.yaml",
        """
        instruments:
          - {symbol: "510300", market: SH, name: A, kind: base_etf}
        """,
    )

    with pytest.raises(ConfigError) as exc:
        load_config(home)

    assert "必须设置 target" in str(exc.value)


def test_duplicate_symbol_is_rejected(home: Path) -> None:
    _write(
        home,
        "universe.yaml",
        """
        instruments:
          - {symbol: "510300", market: SH, name: A, kind: base_etf, target: 1.0}
          - {symbol: "510300", market: SH, name: A2, kind: stock}
        """,
    )

    with pytest.raises(ConfigError) as exc:
        load_config(home)

    assert "重复的 symbol" in str(exc.value)


def test_symbol_must_be_six_digits(home: Path) -> None:
    _write(
        home,
        "universe.yaml",
        """
        instruments:
          - {symbol: "60000", market: SH, name: A, kind: stock}
        """,
    )

    with pytest.raises(ConfigError) as exc:
        load_config(home)

    assert "6 位数字" in str(exc.value)


def test_strategy_filename_must_match_id(home: Path) -> None:
    _write(
        home,
        "strategies/wrong_name.yaml",
        """
        id: b1_pullback
        enabled: false
        """,
    )

    with pytest.raises(ConfigError) as exc:
        load_config(home)

    assert "id 必须与文件名一致" in str(exc.value)


def test_yaml_syntax_error_reports_position(home: Path) -> None:
    _write(home, "market.yaml", "levels: {attack_min: 3,\n")

    with pytest.raises(ConfigError) as exc:
        load_config(home)

    assert "market.yaml" in str(exc.value)
    assert "YAML 语法错误" in str(exc.value)


def test_market_view_sector_conflict_is_rejected(home: Path) -> None:
    _write(
        home,
        "market_view.yaml",
        """
        updated: 2026-10-21
        favor_sectors: [银行]
        avoid_sectors: [银行]
        """,
    )

    with pytest.raises(ConfigError) as exc:
        load_config(home)

    assert "不能同时出现" in str(exc.value)
