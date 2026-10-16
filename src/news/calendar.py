"""News blackout logic shared by the backtester and the live engine.

A blackout is [event - before, event + after] around every high-impact USD event.
The live calendar is FAIL-SAFE: if the feed has never loaded, or the last good load is older
than `stale_after_minutes`, `status()` reports the news as unavailable and the risk engine
must block every signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Settings
from src.news.providers import NewsProvider, events_to_frame


@dataclass
class BlackoutStatus:
    available: bool
    blackout: bool
    reason: str = ""
    event_title: str | None = None
    event_ts: pd.Timestamp | None = None
    next_event_title: str | None = None
    next_event_ts: pd.Timestamp | None = None
    minutes_to_next: float | None = None
    minutes_since_last: float | None = None

    @property
    def blocks_signals(self) -> bool:
        return self.blackout or not self.available


class NewsCalendar:
    def __init__(self, events: pd.DataFrame, minutes_before: int | None, minutes_after: int | None):
        if minutes_before is None or minutes_after is None:
            raise ValueError("news blackout window is not configured (TODO(Rald) in settings.yaml)")
        self.events = events.sort_values("ts").reset_index(drop=True)
        self._ts = self.events["ts"].to_numpy(dtype="datetime64[ns]") if len(events) else \
            np.array([], dtype="datetime64[ns]")
        self.before = pd.Timedelta(minutes=minutes_before)
        self.after = pd.Timedelta(minutes=minutes_after)

    @classmethod
    def from_settings(cls, events: pd.DataFrame, settings: Settings) -> NewsCalendar:
        return cls(events, settings.news.blackout_minutes_before, settings.news.blackout_minutes_after)

    def check(self, t: pd.Timestamp) -> BlackoutStatus:
        t = pd.Timestamp(t).tz_convert("UTC")
        tn = t.tz_localize(None).to_datetime64()
        st = BlackoutStatus(available=True, blackout=False)
        if len(self._ts) == 0:
            return st
        i = int(np.searchsorted(self._ts, tn, side="left"))   # first event >= t
        if i < len(self._ts):
            e = self.events.iloc[i]
            st.next_event_title, st.next_event_ts = e["title"], e["ts"]
            st.minutes_to_next = (e["ts"] - t).total_seconds() / 60
        if i > 0:
            st.minutes_since_last = (t - self.events["ts"].iloc[i - 1]).total_seconds() / 60
        # any event with ts - before <= t <= ts + after
        lo = int(np.searchsorted(self._ts, (t - self.after).tz_localize(None).to_datetime64(), "left"))
        hi = int(np.searchsorted(self._ts, (t + self.before).tz_localize(None).to_datetime64(), "right"))
        if hi > lo:
            e = self.events.iloc[lo]
            st.blackout = True
            st.event_title, st.event_ts = e["title"], e["ts"]
            st.reason = f"news blackout: {e['title']} at {e['ts'].tz_convert('America/New_York'):%H:%M} NY"
        return st

    def blackout_mask(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """Vectorised: True where a timestamp is inside any blackout window (for backtests)."""
        if len(self._ts) == 0:
            return np.zeros(len(idx), dtype=bool)
        t = idx.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
        lo = np.searchsorted(self._ts, t - self.after.to_timedelta64(), "left")
        hi = np.searchsorted(self._ts, t + self.before.to_timedelta64(), "right")
        return hi > lo

    def events_on(self, ny_date) -> pd.DataFrame:
        d = pd.Timestamp(ny_date).date()
        ny = self.events["ts"].dt.tz_convert("America/New_York")
        return self.events[ny.dt.date == d]


class LiveNewsCalendar:
    """Wraps a NewsProvider with refresh + staleness tracking (fail-safe)."""

    def __init__(self, provider: NewsProvider, settings: Settings):
        self.provider = provider
        self.s = settings
        self.calendar: NewsCalendar | None = None
        self.last_ok: pd.Timestamp | None = None
        self.last_error: str | None = None

    def refresh(self, now: pd.Timestamp | None = None) -> bool:
        now = now or pd.Timestamp.now(tz="UTC")
        try:
            events = self.provider.fetch()
        except Exception as e:  # any failure -> keep old data, it will go stale
            self.last_error = str(e)
            return False
        self.calendar = NewsCalendar.from_settings(events_to_frame(events), self.s)
        self.last_ok = now
        self.last_error = None
        return True

    def load_snapshot(self, fetched_at: pd.Timestamp, events) -> None:
        """Use a previously fetched snapshot. Staleness still counts from when it was FETCHED, so an
        old snapshot cannot make the system look safe."""
        self.calendar = NewsCalendar.from_settings(events_to_frame(events), self.s)
        self.last_ok = pd.Timestamp(fetched_at)

    def status(self, now: pd.Timestamp | None = None) -> BlackoutStatus:
        now = now or pd.Timestamp.now(tz="UTC")
        stale = pd.Timedelta(minutes=self.s.news.stale_after_minutes)
        if self.calendar is None or self.last_ok is None or now - self.last_ok > stale:
            why = self.last_error or "news never loaded"
            return BlackoutStatus(available=False, blackout=False,
                                  reason=f"news data unavailable ({why}) - fail-safe, no signals")
        return self.calendar.check(now)
