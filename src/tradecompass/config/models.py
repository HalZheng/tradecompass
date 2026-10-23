"""配置的 pydantic 模型（SPEC §3.5、§15）。

约定：
- 每个配置文件对应一个模型，``extra="forbid"``，拼错的键会直接报错而不是被静默忽略。
- 所有参数都有与 SPEC 一致的默认值，配置文件只需写出与默认值不同的项。
- 键名是契约，不得擅自改名（SPEC §0.1）。
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tradecompass.domain.models import Kind, Market

_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_WEEKLY = re.compile(r"^(MON|TUE|WED|THU|FRI|SAT|SUN) ([01]\d|2[0-3]):[0-5]\d$")
_SYMBOL = re.compile(r"^\d{6}$")

_STRICT = ConfigDict(extra="forbid")


class _Base(BaseModel):
    model_config = _STRICT


def _check_hhmm(value: str) -> str:
    if not _HHMM.match(value):
        raise ValueError(f"必须是 HH:MM 形式（24 小时制），收到 {value!r}")
    return value


# --------------------------------------------------------------------------- system


class DataConfig(_Base):
    raw_sources: list[str] = ["baostock", "akshare", "mootdx", "efinance"]
    factor_sources: list[str] = ["baostock", "akshare"]
    history_years: int = Field(default=6, ge=1, le=20)


class ScheduleConfig(_Base):
    timezone: str = "Asia/Shanghai"
    heartbeat: str = "09:15"
    intraday_start: str = "09:30"
    intraday_end: str = "15:00"
    intraday_poll_seconds: int = Field(default=120, ge=10, le=3600)
    eod_sync: str = "15:30"
    eod_compute: str = "16:00"
    eod_push: str = "16:30"
    weekly: str = "FRI 18:00"
    trade_file_watch_dir: str | None = None

    @field_validator("heartbeat", "intraday_start", "intraday_end")
    @classmethod
    def _v_clock(cls, v: str) -> str:
        return _check_hhmm(v)

    @field_validator("eod_sync", "eod_compute", "eod_push")
    @classmethod
    def _v_eod(cls, v: str) -> str:
        return _check_hhmm(v)

    @field_validator("weekly")
    @classmethod
    def _v_weekly(cls, v: str) -> str:
        if not _WEEKLY.match(v):
            raise ValueError(f"weekly 必须是 'FRI 18:00' 形式，收到 {v!r}")
        return v

    @field_validator("timezone")
    @classmethod
    def _v_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"未知时区 {v!r}") from exc
        return v


class NotifyConfig(_Base):
    enabled: bool = False


class LimitsConfig(_Base):
    max_non_urgent_push_per_day: int = Field(default=10, ge=1)


class SystemConfig(_Base):
    data: DataConfig = Field(default_factory=DataConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    notify: NotifyConfig = Field(default_factory=NotifyConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)


# ------------------------------------------------------------------------- universe


class InstrumentConfig(_Base):
    symbol: str
    market: Market
    name: str
    kind: Kind
    target: float | None = None
    sector_sw: str | None = None
    t0: bool = False
    is_st: bool = False

    @field_validator("symbol")
    @classmethod
    def _v_symbol(cls, v: str) -> str:
        if not _SYMBOL.match(v):
            raise ValueError(f"symbol 必须是 6 位数字字符串，收到 {v!r}")
        return v

    @field_validator("name")
    @classmethod
    def _v_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name 不能为空")
        return v


class OpeningPosition(_Base):
    """期初持仓快照的一条记录。"""

    symbol: str
    name: str
    quantity: int = Field(gt=0)
    avg_cost: float = Field(gt=0)

    @field_validator("symbol")
    @classmethod
    def _v_symbol(cls, v: str) -> str:
        if not _SYMBOL.match(v):
            raise ValueError(f"symbol 必须是 6 位数字字符串，收到 {v!r}")
        return v


class OpeningPositionsConfig(_Base):
    """没有交割单时的持仓来源；有 trades 流水后以流水为准。"""

    as_of: date
    cash: float = Field(default=0.0, ge=0.0)
    positions: list[OpeningPosition] = []

    @field_validator("positions")
    @classmethod
    def _v_unique(cls, v: list[OpeningPosition]) -> list[OpeningPosition]:
        seen: set[str] = set()
        dupes: list[str] = []
        for item in v:
            if item.symbol in seen:
                dupes.append(item.symbol)
            seen.add(item.symbol)
        if dupes:
            raise ValueError(f"重复的 symbol: {', '.join(sorted(set(dupes)))}")
        return v


class UniverseConfig(_Base):
    instruments: list[InstrumentConfig] = []

    @field_validator("instruments")
    @classmethod
    def _v_unique(cls, v: list[InstrumentConfig]) -> list[InstrumentConfig]:
        seen: set[str] = set()
        dupes: list[str] = []
        for inst in v:
            if inst.symbol in seen:
                dupes.append(inst.symbol)
            seen.add(inst.symbol)
        if dupes:
            raise ValueError(f"重复的 symbol: {', '.join(sorted(set(dupes)))}")
        return v

    @model_validator(mode="after")
    def _v_targets(self) -> UniverseConfig:
        base = [i for i in self.instruments if i.kind == "base_etf"]
        for inst in self.instruments:
            if inst.kind == "base_etf":
                if inst.target is None:
                    raise ValueError(f"{inst.symbol} 是 base_etf，必须设置 target")
                if not 0 < inst.target <= 1:
                    raise ValueError(
                        f"{inst.symbol} 的 target 必须在 (0, 1] 内，收到 {inst.target}"
                    )
            elif inst.target is not None:
                raise ValueError(
                    f"{inst.symbol} 的 kind 是 {inst.kind}，不得设置 target（只有 base_etf 能设置）"
                )
        if base:
            total = sum(i.target or 0.0 for i in base)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"base_etf 的 target 之和必须为 1.0，当前为 {total:.6f}")
        return self


# ----------------------------------------------------------------------------- risk


class AllocationConfig(_Base):
    base: float = Field(default=0.5, ge=0.0, le=1.0)
    swing: float = Field(default=0.5, ge=0.0, le=1.0)


class SwingConfig(_Base):
    max_positions: int = Field(default=5, ge=1, le=50)
    risk_per_trade: float = Field(default=0.01, gt=0.0, le=0.2)
    min_trade_amount: float = Field(default=5000.0, ge=0.0)
    cooldown_days_after_stop: int = Field(default=10, ge=0, le=250)
    circuit_breaker_drawdown: float = Field(default=0.10, gt=0.0, le=1.0)
    circuit_breaker_days: int = Field(default=10, ge=1, le=250)
    no_averaging_down: bool = True


class StopsConfig(_Base):
    atr_mult: float = Field(default=2.0, gt=0.0, le=10.0)
    max_stop_pct: float = Field(default=0.10, gt=0.0, le=0.5)


class RegimeCapsConfig(_Base):
    attack: float = Field(default=1.0, ge=0.0, le=1.0)
    neutral: float = Field(default=0.5, ge=0.0, le=1.0)
    defense: float = Field(default=0.0, ge=0.0, le=1.0)


class SectorFilterConfig(_Base):
    enabled: bool = True


class RiskConfig(_Base):
    cash: float = Field(default=0.0, ge=0.0)
    allocation: AllocationConfig = Field(default_factory=AllocationConfig)
    swing: SwingConfig = Field(default_factory=SwingConfig)
    stops: StopsConfig = Field(default_factory=StopsConfig)
    regime_caps: RegimeCapsConfig = Field(default_factory=RegimeCapsConfig)
    sector_filter: SectorFilterConfig = Field(default_factory=SectorFilterConfig)


# --------------------------------------------------------------------------- market


class IndexConfig(_Base):
    hs300: str = "000300"
    csi1000: str = "000852"
    breadth_universe: str = "000906"
    turnover: list[str] = ["000001", "399001"]


class BreadthConfig(_Base):
    above_ma: int = Field(default=20, ge=1)
    high: float = Field(default=0.5, gt=0.0, le=1.0)
    low: float = Field(default=0.3, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _v_order(self) -> BreadthConfig:
        if self.low >= self.high:
            raise ValueError(f"breadth.low({self.low}) 必须小于 breadth.high({self.high})")
        return self


class LevelsConfig(_Base):
    attack_min: int = Field(default=3, ge=-5, le=5)
    defense_max: int = Field(default=0, ge=-5, le=5)
    confirm_days: int = Field(default=2, ge=1, le=10)

    @model_validator(mode="after")
    def _v_order(self) -> LevelsConfig:
        if self.defense_max >= self.attack_min:
            raise ValueError(
                f"levels.defense_max({self.defense_max}) 必须小于 attack_min({self.attack_min})"
            )
        return self


class MarketConfig(_Base):
    indices: IndexConfig = Field(default_factory=IndexConfig)
    breadth: BreadthConfig = Field(default_factory=BreadthConfig)
    turnover_ma: int = Field(default=60, ge=2)
    limit_net_days: int = Field(default=5, ge=1)
    levels: LevelsConfig = Field(default_factory=LevelsConfig)
    market_view_stale_days: int = Field(default=14, ge=1)


class MarketViewConfig(_Base):
    updated: date
    favor_sectors: list[str] = []
    avoid_sectors: list[str] = []
    notes: list[str] = []

    @model_validator(mode="after")
    def _v_sectors(self) -> MarketViewConfig:
        both = set(self.favor_sectors) & set(self.avoid_sectors)
        if both:
            raise ValueError(f"同一行业不能同时出现在 favor/avoid 中: {', '.join(sorted(both))}")
        return self


# ------------------------------------------------------------------------- backtest


class CostsConfig(_Base):
    commission: float = Field(default=0.00025, ge=0.0, le=0.01)
    # 场内基金（ETF/LOF）与债券的佣金常与股票不同，实测中 ETF 甚至可能更高。
    # 为 None 时沿用 commission。
    commission_fund: float | None = Field(default=None, ge=0.0, le=0.01)
    commission_min: float = Field(default=5.0, ge=0.0)
    # 基金是否也有单笔最低佣金。实测券商普遍对股票收 5 元、对 ETF 免五，
    # 这一条对小单成本影响极大，必须能分开配置。
    commission_min_fund: float = Field(default=0.0, ge=0.0)
    stamp_sell: float = Field(default=0.0005, ge=0.0, le=0.01)
    transfer: float = Field(default=0.00001, ge=0.0, le=0.01)
    slippage: float = Field(default=0.001, ge=0.0, le=0.05)

    def _is_fund(self, kind: str) -> bool:
        return kind in ("base_etf", "sector_etf")

    def commission_for(self, kind: str) -> float:
        """按标的类型取佣金费率。"""
        if self._is_fund(kind) and self.commission_fund is not None:
            return self.commission_fund
        return self.commission

    def min_commission_for(self, kind: str) -> float:
        """按标的类型取单笔最低佣金。"""
        return self.commission_min_fund if self._is_fund(kind) else self.commission_min


class RandomConfig(_Base):
    size: int = Field(default=50, ge=1)
    repeats: int = Field(default=3, ge=1)
    seed: int = 42
    entry_runs: int = Field(default=100, ge=1)


class AcceptanceConfig(_Base):
    min_trades: int = Field(default=30, ge=1)
    min_pf_in: float = 1.3
    min_pf_out: float = 1.1
    min_pf_out_ratio: float = 0.6
    min_expectancy_r: float = 0.3
    max_dd_vs_bh: float = Field(default=0.7, gt=0.0, le=1.0)
    min_random_pct: float = Field(default=0.8, gt=0.0, le=1.0)
    grid_pass_ratio: float = Field(default=0.7, gt=0.0, le=1.0)
    random_sample_pass: int = Field(default=2, ge=1)


class BacktestConfig(_Base):
    years: int = Field(default=5, ge=1, le=20)
    in_sample_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    max_gap_open: float = Field(default=0.05, ge=0.0, le=0.2)
    random: RandomConfig = Field(default_factory=RandomConfig)
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)


# --------------------------------------------------------------------------- broker


class BrokerMapping(_Base):
    columns: dict[str, list[str]] = {}
    sides: dict[str, list[str]] = {}


class BrokerConfig(_Base):
    default: str = "eastmoney"
    mappings: dict[str, BrokerMapping] = {}

    @model_validator(mode="after")
    def _v_default(self) -> BrokerConfig:
        if self.mappings and self.default not in self.mappings:
            known = ", ".join(sorted(self.mappings))
            raise ValueError(f"default={self.default!r} 不在 mappings 中（已有: {known}）")
        return self


# ------------------------------------------------------------------------ strategy


class StrategyConfig(_Base):
    id: str
    enabled: bool = False
    params: dict[str, Any] = {}
    exits: dict[str, Any] = {}
    grid: dict[str, Any] = {}

    @field_validator("id")
    @classmethod
    def _v_id(cls, v: str) -> str:
        if not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError(f"策略 id 只能用小写字母、数字和下划线，收到 {v!r}")
        return v
