"""TC_HOME 解析与配置加载（SPEC §3.2、§3.5）。

加载失败一律抛 :class:`ConfigError`，消息形如 ``risk.yaml: swing.max_positions -> 原因``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from tradecompass.config.errors import ConfigError
from tradecompass.config.models import (
    BacktestConfig,
    BrokerConfig,
    MarketConfig,
    MarketViewConfig,
    OpeningPositionsConfig,
    RiskConfig,
    StrategyConfig,
    SystemConfig,
    UniverseConfig,
)

CONFIG_FILES: tuple[str, ...] = (
    "system.yaml",
    "universe.yaml",
    "risk.yaml",
    "market.yaml",
    "market_view.yaml",
    "backtest.yaml",
    "broker.yaml",
)


@dataclass(frozen=True, slots=True)
class Paths:
    """一个 TC_HOME 下的标准目录布局。"""

    home: Path
    config: Path
    data: Path
    reports: Path
    logs: Path

    @classmethod
    def from_home(cls, home: Path) -> Paths:
        return cls(
            home=home,
            config=home / "config",
            data=home / "data",
            reports=home / "reports",
            logs=home / "logs",
        )

    @property
    def db_path(self) -> Path:
        return self.data / "tradecompass.sqlite"

    @property
    def strategies_dir(self) -> Path:
        return self.config / "strategies"

    @property
    def bars_raw_dir(self) -> Path:
        return self.data / "bars" / "raw"

    @property
    def bars_factor_dir(self) -> Path:
        return self.data / "bars" / "factor"

    @property
    def meta_dir(self) -> Path:
        return self.data / "meta"

    @property
    def cache_dir(self) -> Path:
        return self.data / "cache"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """全部配置的聚合结果。"""

    paths: Paths
    system: SystemConfig
    universe: UniverseConfig
    risk: RiskConfig
    market: MarketConfig
    market_view: MarketViewConfig
    backtest: BacktestConfig
    broker: BrokerConfig
    strategies: dict[str, StrategyConfig]
    opening_positions: OpeningPositionsConfig | None = None

    def enabled_strategies(self) -> dict[str, StrategyConfig]:
        return {sid: cfg for sid, cfg in self.strategies.items() if cfg.enabled}


def repo_root() -> Path:
    """向上查找含 pyproject.toml 的目录，作为 TC_REPO。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ConfigError("无法定位仓库根目录：从当前模块向上找不到 pyproject.toml")


def examples_dir() -> Path:
    root = repo_root() / "config" / "examples"
    if not root.is_dir():
        raise ConfigError(f"配置模板目录不存在：{root}")
    return root


def resolve_home(cli_home: Path | None = None) -> Path:
    """按 SPEC §3.2 的顺序解析 TC_HOME。

    显式传入的 ``--home`` 与 ``TC_HOME`` 环境变量具有权威性（即使目录还不存在，
    由调用方决定是创建还是报错）；两者都没有时才回落到 ``TC_REPO/local/``。
    """
    if cli_home is not None:
        return cli_home.expanduser().resolve()
    env = os.environ.get("TC_HOME")
    if env is not None and env.strip():
        return Path(env).expanduser().resolve()
    local = repo_root() / "local"
    if local.is_dir():
        return local.resolve()
    raise ConfigError(
        "未找到 TC_HOME：请用 --home 指定，或设置 TC_HOME 环境变量，"
        "或先运行 tradecompass init --home <path>"
    )


def load_env(home: Path) -> None:
    """从 TC_HOME/.env 加载环境变量；已存在的进程环境变量优先（不覆盖）。"""
    env_file = home / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)


def _read_yaml(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ConfigError(f"{path.name}: 文件不存在（{path}），先运行 tradecompass init")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" 第 {mark.line + 1} 行第 {mark.column + 1} 列" if mark is not None else ""
        raise ConfigError(f"{path.name}: YAML 语法错误{where} -> {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name}: 顶层必须是映射（key: value），实际是 {type(raw).__name__}")
    return raw


def _format_errors(filename: str, exc: ValidationError) -> str:
    lines: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        key = loc if loc else "(文件级)"
        msg = err["msg"]
        if err["type"] == "extra_forbidden":
            msg = f"未知的配置键（拼写错误或多写了）: {msg}"
        lines.append(f"{filename}: {key} -> {msg}")
    return "\n".join(lines)


def _load[M: BaseModel](paths: Paths, filename: str, model: type[M]) -> M:
    path = paths.config / filename
    data = _read_yaml(path)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_errors(filename, exc)) from exc


def _load_strategies(directory: Path) -> dict[str, StrategyConfig]:
    result: dict[str, StrategyConfig] = {}
    if not directory.is_dir():
        return result
    for file in sorted(directory.glob("*.yaml")):
        data = _read_yaml(file)
        try:
            cfg = StrategyConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(_format_errors(file.name, exc)) from exc
        if file.stem != cfg.id:
            raise ConfigError(
                f"{file.name}: id 必须与文件名一致 -> 文件名为 {file.stem}，id 为 {cfg.id}"
            )
        result[cfg.id] = cfg
    return result


def load_config(home: Path) -> AppConfig:
    """加载并校验 TC_HOME 下的全部配置。"""
    if not home.is_dir():
        raise ConfigError(f"TC_HOME 不存在：{home}（先运行 tradecompass init --home <path>）")

    load_env(home)
    paths = Paths.from_home(home)
    if not paths.config.is_dir():
        raise ConfigError(f"配置目录不存在：{paths.config}（先运行 tradecompass init）")

    opening_path = paths.config / "opening_positions.yaml"
    opening = (
        _load(paths, "opening_positions.yaml", OpeningPositionsConfig)
        if opening_path.is_file()
        else None
    )
    return AppConfig(
        paths=paths,
        system=_load(paths, "system.yaml", SystemConfig),
        universe=_load(paths, "universe.yaml", UniverseConfig),
        risk=_load(paths, "risk.yaml", RiskConfig),
        market=_load(paths, "market.yaml", MarketConfig),
        market_view=_load(paths, "market_view.yaml", MarketViewConfig),
        backtest=_load(paths, "backtest.yaml", BacktestConfig),
        broker=_load(paths, "broker.yaml", BrokerConfig),
        strategies=_load_strategies(paths.strategies_dir),
        opening_positions=opening,
    )
