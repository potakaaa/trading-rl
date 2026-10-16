"""Resample 1-minute bars into higher timeframes.

Every output bar carries:
- index `ts`: bar OPEN time (UTC)
- `end`: the time the bar is COMPLETE (UTC). A bar may only be used by any detector or
  model at or after `end`. This column is how we prevent lookahead with higher timeframes.
- `n_bars`: how many 1-minute bars went into it (for data quality checks).

Alignment is on New York wall clock, anchored at the trading-day start (18:00 NY):
5m/15m/1h are normal clock buckets, 4h bars start 18:00/22:00/02:00/06:00/10:00/14:00,
daily bars run 18:00 -> 17:00 and are labelled by the bar's open time.
"""
from __future__ import annotations

import pandas as pd

from src.timeutils import NY, TF_MINUTES, bucket_labels, hhmm_minutes, trading_date


def resample(m1: pd.DataFrame, tf: str, day_start: str = "18:00",
             market_close: str = "17:00") -> pd.DataFrame:
    if tf == "1m":
        out = m1.copy()
        out["n_bars"] = 1
        out["end"] = out.index + pd.Timedelta(minutes=1)
        return out
    labels = bucket_labels(m1.index, tf, day_start)
    g = m1.groupby(labels)
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "n_bars": g["open"].count(),
    })
    out.index = pd.DatetimeIndex(out.index, name="ts")
    if tf == "1d":
        # complete at market close (17:00 NY) on the trading date
        td = trading_date(out.index, day_start)
        close_wall = td + pd.Timedelta(minutes=hhmm_minutes(market_close))
        out["end"] = close_wall.tz_localize(NY).tz_convert("UTC")
        out["trading_date"] = td.date
    else:
        out["end"] = out.index + pd.Timedelta(minutes=TF_MINUTES[tf])
    return out


def visible_at(bars: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Higher-timeframe bars that are complete at time t (no lookahead)."""
    return bars[bars["end"] <= t]
