"""配置层：pydantic 模型、TC_HOME 解析、配置加载。"""

from tradecompass.config.errors import ConfigError
from tradecompass.config.loader import (
    CONFIG_FILES,
    AppConfig,
    Paths,
    examples_dir,
    load_config,
    load_env,
    repo_root,
    resolve_home,
)
from tradecompass.config.models import (
    BacktestConfig,
    BrokerConfig,
    MarketConfig,
    MarketViewConfig,
    RiskConfig,
    StrategyConfig,
    SystemConfig,
    UniverseConfig,
)

__all__ = [
    "CONFIG_FILES",
    "AppConfig",
    "BacktestConfig",
    "BrokerConfig",
    "ConfigError",
    "MarketConfig",
    "MarketViewConfig",
    "Paths",
    "RiskConfig",
    "StrategyConfig",
    "SystemConfig",
    "UniverseConfig",
    "examples_dir",
    "load_config",
    "load_env",
    "repo_root",
    "resolve_home",
]
