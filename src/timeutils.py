"""Timezone and trading-calendar helpers.

All storage is in UTC. All session / trading-day logic is done on New York *wall clock* time,
which makes it automatically correct across daylight-saving switches (March and November).
"""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

NY = "America/New_York"

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def hhmm_minutes(s: str) -> int:
    t = parse_hhmm(s)
    return t.hour * 60 + t.minute


def ny_wall(idx: pd.DatetimeIndex, tz: str = NY) -> pd.DatetimeIndex:
    """UTC (tz-aware) index -> naive New York wall-clock index."""
    return idx.tz_convert(tz).tz_localize(None)


def wall_minutes(idx: pd.DatetimeIndex, tz: str = NY) -> np.ndarray:
    w = ny_wall(idx, tz)
    return np.asarray(w.hour * 60 + w.minute)


def in_window(idx: pd.DatetimeIndex, start: str, end: str, tz: str = NY) -> np.ndarray:
    """Boolean mask: bar open time is inside [start, end) NY wall clock. Handles windows that
    cross midnight (e.g. 20:00-00:00 or 18:00-09:30)."""
    m = wall_minutes(idx, tz)
    s, e = hhmm_minutes(start), hhmm_minutes(end)
    if s < e:
        return (m >= s) & (m < e)
    return (m >= s) | (m < e)


def _day_shift(day_start: str) -> pd.Timedelta:
    # 18:00 day start -> shift wall clock by +6h so the trading day begins at "midnight"
    return pd.Timedelta(minutes=(1440 - hhmm_minutes(day_start)) % 1440)


def trading_date(idx: pd.DatetimeIndex, day_start: str = "18:00", tz: str = NY) -> pd.DatetimeIndex:
    """Trading date of each UTC timestamp. With day_start 18:00, Sunday 18:00 NY onwards belongs
    to Monday's trading date (CME / CFD convention)."""
    return (ny_wall(idx, tz) + _day_shift(day_start)).normalize()


def bucket_labels(idx: pd.DatetimeIndex, tf: str, day_start: str = "18:00",
                  tz: str = NY) -> pd.DatetimeIndex:
    """UTC open time of the `tf` bar each UTC timestamp belongs to.

    Buckets are aligned on New York wall clock, anchored at `day_start` (so 4h bars are
    18:00, 22:00, 02:00, ... and daily bars run 18:00 -> 17:00). The offset inside the bucket
    is measured on wall clock and subtracted from the real UTC time, which keeps every
    bucket correct across DST switches.
    """
    wall = ny_wall(idx, tz) + _day_shift(day_start)
    floored = wall.normalize() if tf == "1d" else wall.floor(f"{TF_MINUTES[tf]}min")
    return idx - (wall - floored)


def is_market_open(idx: pd.DatetimeIndex, close: str = "17:00", reopen: str = "18:00",
                   tz: str = NY) -> np.ndarray:
    """US index CFD/futures hours: Sunday 18:00 -> Friday 17:00 NY, with a daily break
    between `close` and `reopen`. Exchange holidays are not modelled here (data is just missing)."""
    w = ny_wall(idx, tz)
    m = np.asarray(w.hour * 60 + w.minute)
    dow = np.asarray(w.dayofweek)  # Mon=0 .. Sun=6
    c, r = hhmm_minutes(close), hhmm_minutes(reopen)
    in_break = (m >= c) & (m < r)
    closed = (dow == 5) | ((dow == 4) & (m >= c)) | ((dow == 6) & (m < r)) | in_break
    return ~closed


def session_of(idx: pd.DatetimeIndex, sessions: dict, tz: str = NY) -> pd.Series:
    """Name of the trading session (e.g. 'ny_am') for each timestamp, or None."""
    out = pd.Series([None] * len(idx), index=idx, dtype=object)
    for name, win in sessions.items():
        start = win.start if hasattr(win, "start") else win["start"]
        end = win.end if hasattr(win, "end") else win["end"]
        out[in_window(idx, start, end, tz)] = name
    return out


def session_end_utc(t: pd.Timestamp, session_end: str, tz: str = NY) -> pd.Timestamp:
    """UTC timestamp of `session_end` (NY wall time) on the NY calendar day of t (DST-safe)."""
    day = pd.Timestamp(t).tz_convert(tz).tz_localize(None).normalize()
    return (day + pd.Timedelta(minutes=hhmm_minutes(session_end))).tz_localize(tz).tz_convert("UTC")
