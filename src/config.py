"""Typed loader for config/settings.yaml.

Fields marked TODO(Rald) are optional in the schema so the file can be parsed, but
`Settings.require_complete()` raises if any are still unset. The live engine and the
backtester must call it at startup (fail safe: no config, no signals).
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


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


class News(BaseModel):
    currency: str
    impact: str
    blackout_minutes_before: int | None = None
    blackout_minutes_after: int | None = None
    refresh_minutes: int
    live_feed_url: str


class Broker(BaseModel):
    lot_value_per_point: float | None = None
    lot_step: float
    typical_spread_points: float | None = None
    price_offset: float = 0.0


class InternalBuffers(BaseModel):
    daily_loss_stop_fraction: float
    max_drawdown_stop_fraction: float
    max_signals_per_session: int
    max_signals_per_day: int
    max_consecutive_losses: int


class Settings(BaseModel):
    instrument: Instrument
    timezones: Timezones
    sessions: dict[str, SessionWindow]
    account: Account
    news: News
    broker: Broker
    internal_buffers: InternalBuffers

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


def load_settings(path: Path | str = DEFAULT_PATH) -> Settings:
    with open(path) as f:
        return Settings.model_validate(yaml.safe_load(f))
