"""Prop-firm guardrails: the hard gate every signal must pass. Shared by backtest and live engine.

The ML model cannot override this. `evaluate()` collects EVERY reason a signal is blocked (not
just the first) so the compliance log shows the full picture.

Limits (Equity Edge 2-Step, values from settings.yaml):
  daily loss limit  = daily_loss_limit_pct x initial balance, measured from the start-of-day balance
  max drawdown      = max_drawdown_pct x initial balance, static (floor = initial - limit)
Internal buffers (stricter than the firm, never weaken without Rald's approval):
  stop for the day at daily_loss_stop_fraction of the daily limit
  halt until a manual reset at max_drawdown_stop_fraction of the max drawdown
  max signals per session / per day, stop after N consecutive losses in a day
On top of that, a signal is blocked if losing it in full could breach a FIRM limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from src.config import Settings
from src.news.calendar import BlackoutStatus
from src.risk.sizing import SizeResult, position_size
from src.timeutils import session_end_utc


@dataclass(frozen=True)
class Limits:
    initial_balance: float
    daily_limit_usd: float
    daily_buffer_usd: float
    max_dd_usd: float
    dd_buffer_usd: float
    max_per_session: int
    max_per_day: int
    max_consecutive_losses: int

    @classmethod
    def from_settings(cls, s: Settings) -> Limits:
        s.require_complete()
        a, ib = s.account, s.internal_buffers
        daily = a.initial_balance * a.daily_loss_limit_pct / 100
        dd = a.initial_balance * a.max_drawdown_pct / 100
        return cls(a.initial_balance, daily, daily * ib.daily_loss_stop_fraction, dd,
                   dd * ib.max_drawdown_stop_fraction, ib.max_signals_per_session,
                   ib.max_signals_per_day, ib.max_consecutive_losses)


@dataclass
class AccountState:
    """Realized account state. Live: rebuilt from Rald's journal entries. Backtest: simulated."""
    limits: Limits
    balance: float
    trading_day: date | None = None
    day_start_balance: float = 0.0
    signals_today: int = 0
    signals_by_session: dict[str, int] = field(default_factory=dict)
    consecutive_losses: int = 0
    position_open: bool = False
    halted: bool = False
    halt_reason: str = ""

    @classmethod
    def fresh(cls, limits: Limits) -> AccountState:
        return cls(limits=limits, balance=limits.initial_balance,
                   day_start_balance=limits.initial_balance)

    @property
    def day_loss(self) -> float:
        return max(0.0, self.day_start_balance - self.balance)

    @property
    def drawdown(self) -> float:
        return max(0.0, self.limits.initial_balance - self.balance)

    def roll_day(self, trading_day: date) -> None:
        if trading_day != self.trading_day:
            self.trading_day = trading_day
            self.day_start_balance = self.balance
            self.signals_today = 0
            self.signals_by_session = {}
            self.consecutive_losses = 0

    def record_signal(self, session: str) -> None:
        self.signals_today += 1
        self.signals_by_session[session] = self.signals_by_session.get(session, 0) + 1

    def record_close(self, pnl_usd: float) -> None:
        self.balance += pnl_usd
        if pnl_usd < 0:
            self.consecutive_losses += 1
        elif pnl_usd > 0:
            self.consecutive_losses = 0
        if self.drawdown >= self.limits.dd_buffer_usd and not self.halted:
            self.halted = True
            self.halt_reason = (f"max drawdown buffer reached: down ${self.drawdown:.2f} "
                                f">= ${self.limits.dd_buffer_usd:.2f}; manual reset required")

    def manual_reset(self) -> None:
        self.halted = False
        self.halt_reason = ""


@dataclass(frozen=True)
class SignalCandidate:
    time: pd.Timestamp           # when the signal would be issued (setup armed_at)
    session: str | None
    stop_pts: float
    direction: str
    setup_type: str = ""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reasons: tuple[str, ...]
    size: SizeResult

    @property
    def reason_text(self) -> str:
        return "; ".join(self.reasons)


def evaluate(sig: SignalCandidate, state: AccountState, news: BlackoutStatus | None,
             settings: Settings, confidence: float | None = None,
             threshold: float | None = None) -> Decision:
    lim = state.limits
    reasons: list[str] = []

    if news is None:
        reasons.append("news_unavailable")
    elif not news.available:
        reasons.append("news_unavailable")
    elif news.blackout:
        reasons.append("news_blackout")
    elif news.minutes_to_next is not None and \
            news.minutes_to_next <= settings.news.blackout_minutes_before + 1:
        reasons.append("news_imminent")   # orders/positions must be gone 1 min before the window

    if sig.session is None or sig.session not in settings.sessions:
        reasons.append("outside_session")
    else:
        left = session_end_utc(sig.time, settings.sessions[sig.session].end) - sig.time
        if left < pd.Timedelta(minutes=settings.setups.min_minutes_to_session_end):
            reasons.append("too_close_to_session_end")

    if state.halted:
        reasons.append("halted_max_drawdown_buffer")
    elif state.drawdown >= lim.dd_buffer_usd:
        reasons.append("max_drawdown_buffer_reached")
    if state.day_loss >= lim.daily_buffer_usd:
        reasons.append("daily_loss_buffer_reached")
    if state.consecutive_losses >= lim.max_consecutive_losses:
        reasons.append("consecutive_losses_today")
    if state.signals_today >= lim.max_per_day:
        reasons.append("max_signals_per_day")
    if sig.session and state.signals_by_session.get(sig.session, 0) >= lim.max_per_session:
        reasons.append("max_signals_per_session")
    if state.position_open and settings.backtest.one_position_at_a_time:
        reasons.append("position_open")

    size = position_size(sig.stop_pts, settings)
    if size.blocked_reason:
        reasons.append(size.blocked_reason)
    else:
        # a full stop-out must not be able to breach a FIRM limit (slippage included)
        worst = size.risk_usd + size.lots * settings.backtest.stop_slippage_points \
            * settings.broker.lot_value_per_point
        if state.day_loss + worst > lim.daily_limit_usd:
            reasons.append("could_breach_daily_limit")
        if state.drawdown + worst > lim.max_dd_usd:
            reasons.append("could_breach_max_drawdown")

    if threshold is not None and (confidence is None or confidence < threshold):
        reasons.append("confidence_below_threshold")

    return Decision(allowed=not reasons, reasons=tuple(reasons), size=size)
