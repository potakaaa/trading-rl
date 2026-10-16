"""Typed loader for config/settings.yaml.

Fields marked TODO(Rald) are optional in the schema so the file can be parsed, but
`Settings.require_complete()` raises if any are still unset. The live engine and the
backtester must call it at startup (fail safe: no config, no signals).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class Instrument(BaseModel):
    name: str
    dukascopy_symbol: str
    oanda_symbol: str


class Timezones(BaseModel):
    market: str
    display: str


class SessionWindow(BaseModel):
    start: str
    end: str


class Account(BaseModel):
    prop_firm: str
    plan: str
    initial_balance: float | None = None
    risk_per_trade_pct: float
    daily_loss_limit_pct: float | None = None
    max_drawdown_pct: float | None = None


class FredRelease(BaseModel):
    fred_name: str
    title: str
    time: str


class FomcSource(BaseModel):
    calendar_url: str
    historical_url_template: str
    statement_time: str
    press_conference_time: str


class RuleEvent(BaseModel):
    business_day: int
    time: str
    title: str


class HistoricalNews(BaseModel):
    fred_api_url: str
    fred_releases: list[FredRelease]
    fomc: FomcSource
    rule_based: dict[str, RuleEvent]
    manual_csv: str


class News(BaseModel):
    currency: str
    impact: str
    blackout_minutes_before: int | None = None
    blackout_minutes_after: int | None = None
    refresh_minutes: int
    stale_after_minutes: int
    live_feed_url: str
    historical: HistoricalNews


class Dukascopy(BaseModel):
    base_url: str
    side: str
    price_divisor: float
    max_concurrency: int
    retries: int
    timeout_seconds: float


class Cleaning(BaseModel):
    spike_range_mult: float
    spike_reversion_frac: float
    spike_window: int
    gap_report_minutes: int


class Data(BaseModel):
    root: str
    history_start: str
    dukascopy: Dukascopy
    trading_day_start: str
    market_close: str
    timeframes: list[str]
    cleaning: Cleaning
    sanity_price_range: tuple[float, float]


class Oanda(BaseModel):
    practice_rest_url: str
    practice_stream_url: str
    live_rest_url: str
    live_stream_url: str
    price_component: str


class DukascopyLive(BaseModel):
    url: str
    instrument: str
    timeout_seconds: float
    max_bars_per_request: int


class Live(BaseModel):
    provider: str
    mode: Literal["paper", "live"] = "paper"
    state_source: Literal["paper", "journal"] = "paper"
    poll_seconds: int = 15
    heartbeat_seconds: int
    lookback_days: dict[str, int]
    db_path: str
    api_host: str
    api_port: int
    dukascopy_live: DukascopyLive
    oanda: Oanda


class Broker(BaseModel):
    lot_value_per_point: float | None = None
    lot_step: float
    min_lot: float
    typical_spread_points: float | None = None
    price_offset: float = 0.0
    unconfirmed: list[str] = []


class InternalBuffers(BaseModel):
    daily_loss_stop_fraction: float
    max_drawdown_stop_fraction: float
    max_signals_per_session: int
    max_signals_per_day: int
    max_consecutive_losses: int


class TimeWindow(BaseModel):
    start: str
    end: str


class Liquidity(BaseModel):
    asia: TimeWindow
    london: TimeWindow
    overnight: TimeWindow
    equal_level_atr: float
    equal_lookback_swings: int
    equal_max_age_hours: float


class Sweep(BaseModel):
    min_sweep_atr: float
    reclaim_bars: int


class Mss(BaseModel):
    disp_mult: float
    max_bars_after_sweep: int


class Fvg(BaseModel):
    min_fvg_atr: float


class OrderBlock(BaseModel):
    zone: Literal["body", "full"]
    max_lookback_bars: int


class PremiumDiscount(BaseModel):
    enabled: bool
    range_timeframe: str


class HtfBias(BaseModel):
    timeframes: list[str]
    use_prev_day_midpoint: bool
    bias_threshold: int


class Detectors(BaseModel):
    setup_timeframe: str
    atr_period: int
    swing_n: dict[str, int]
    liquidity: Liquidity
    sweep: Sweep
    mss: Mss
    fvg: Fvg
    order_block: OrderBlock
    premium_discount: PremiumDiscount
    htf_bias: HtfBias


class Setups(BaseModel):
    enabled: list[str]
    entry_mode: Literal["fvg_start", "ce"]
    ob_entry: Literal["edge", "mid"]
    target_mode: Literal["nearest", "first_meeting_min_rr", "fixed_r"]
    fixed_target_r: float = 2.0           # used when target_mode == fixed_r
    stop_buffer_atr: float
    min_rr: float = Field(ge=1.0)         # Rald's hard rule: never take a trade targeting < 1R
    entry_expiry_bars: int
    require_htf_alignment: bool
    min_minutes_to_session_end: int = 0


class Backtest(BaseModel):
    start: str
    end: str | None = None
    spread_stress_mult: float
    stop_slippage_points: float
    commission_per_lot: float
    one_position_at_a_time: bool
    news_exit: Literal["flatten_before"]
    reset_on_halt: bool


class Challenge(BaseModel):
    phase_targets_pct: list[float]
    min_trading_days: int
    max_days_per_phase: int
    n_simulations: int
    seed: int


class Research(BaseModel):
    holdout_start: str                    # data on/after this date is never used for research
    variants: dict[str, dict[str, object]]


class Settings(BaseModel):
    instrument: Instrument
    timezones: Timezones
    sessions: dict[str, SessionWindow]
    account: Account
    news: News
    data: Data
    live: Live
    broker: Broker
    internal_buffers: InternalBuffers
    detectors: Detectors
    setups: Setups
    backtest: Backtest
    challenge: Challenge
    research: Research

    def missing_todos(self) -> list[str]:
        required = {
            "account.initial_balance": self.account.initial_balance,
            "account.daily_loss_limit_pct": self.account.daily_loss_limit_pct,
            "account.max_drawdown_pct": self.account.max_drawdown_pct,
            "news.blackout_minutes_before": self.news.blackout_minutes_before,
            "news.blackout_minutes_after": self.news.blackout_minutes_after,
            "broker.lot_value_per_point": self.broker.lot_value_per_point,
            "broker.typical_spread_points": self.broker.typical_spread_points,
        }
        return [name for name, value in required.items() if value is None]

    def require_complete(self) -> Settings:
        missing = self.missing_todos()
        if missing:
            raise ValueError("settings.yaml has unset TODO(Rald) values: " + ", ".join(missing))
        return self


def data_dir(settings: Settings) -> Path:
    """Absolute path of the data root (relative paths are resolved from the project root)."""
    root = Path(settings.data.root)
    return root if root.is_absolute() else PROJECT_ROOT / root


def load_settings(path: Path | str = DEFAULT_PATH) -> Settings:
    with open(path) as f:
        return Settings.model_validate(yaml.safe_load(f))
