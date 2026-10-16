"""Cleaning raw 1-minute bars.

Steps (every count is recorded so the data quality report can show exactly what changed):
1. Sort, drop duplicate timestamps.
2. Drop structurally invalid bars (high below open/close, low above, non-positive prices).
3. Drop bars outside market hours (weekends, Friday after close, the daily 17:00-18:00 break).
4. Drop "filler" bars: zero volume AND open == high == low == close (Dukascopy pads quiet periods).
5. Repair bad-tick wicks: a wick far longer than normal (> spike_range_mult x rolling median range)
   that also makes up most of the bar is clamped to the body. The body is kept.
6. Drop isolated outlier bars whose close jumps away and straight back.

Gaps are NOT filled. Inventing prices would create fake setups; gaps are reported instead.

Note: steps 5-6 look at neighbouring bars. That is fine for cleaning history (it is not a
trading feature), but it means the live feed is never "cleaned" the same way. Bad ticks are
rare on this instrument; the report shows how many were repaired.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Settings
from src.timeutils import is_market_open


@dataclass
class CleanReport:
    raw_rows: int = 0
    duplicates: int = 0
    invalid: int = 0
    outside_hours: int = 0
    filler: int = 0
    wick_spikes_repaired: int = 0
    outlier_bars_dropped: int = 0
    clean_rows: int = 0
    repaired_examples: list[str] = field(default_factory=list)


def clean_m1(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, CleanReport]:
    rep = CleanReport(raw_rows=len(df))
    cfg = settings.data.cleaning
    df = df.sort_index()

    dup = df.index.duplicated(keep="last")
    rep.duplicates = int(dup.sum())
    df = df[~dup].copy()

    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    invalid = ((df["high"] < body_hi) | (df["low"] > body_lo) |
               (df[["open", "high", "low", "close"]] <= 0).any(axis=1) |
               df[["open", "high", "low", "close"]].isna().any(axis=1))
    rep.invalid = int(invalid.sum())
    df = df[~invalid]

    open_mask = is_market_open(df.index, settings.data.market_close, settings.data.trading_day_start)
    rep.outside_hours = int((~open_mask).sum())
    df = df[open_mask]

    flat = (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    filler = flat & (df["volume"] <= 0)
    rep.filler = int(filler.sum())
    df = df[~filler].copy()

    # --- bad-tick wicks
    rng = df["high"] - df["low"]
    med = rng.rolling(cfg.spike_window, min_periods=20).median()
    med = med.fillna(rng.expanding().median()).clip(lower=1e-9)
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    up_wick = df["high"] - body_hi
    dn_wick = body_lo - df["low"]
    up_spike = (up_wick > cfg.spike_range_mult * med) & (up_wick >= cfg.spike_reversion_frac * rng)
    dn_spike = (dn_wick > cfg.spike_range_mult * med) & (dn_wick >= cfg.spike_reversion_frac * rng)
    rep.wick_spikes_repaired = int((up_spike | dn_spike).sum())
    rep.repaired_examples += [str(t) for t in df.index[up_spike | dn_spike][:20]]
    df.loc[up_spike, "high"] = body_hi[up_spike]
    df.loc[dn_spike, "low"] = body_lo[dn_spike]

    # --- isolated outlier bars (close jumps away and the next bar opens back where it was)
    prev_c = df["close"].shift(1)
    next_o = df["open"].shift(-1)
    away = (df["close"] - prev_c).abs() > cfg.spike_range_mult * med
    back = (next_o - prev_c).abs() < 2 * med
    outlier = (away & back).fillna(False)
    rep.outlier_bars_dropped = int(outlier.sum())
    rep.repaired_examples += [str(t) for t in df.index[outlier][:20]]
    df = df[~outlier]

    rep.clean_rows = len(df)
    return df, rep


def check_price_sanity(df: pd.DataFrame, settings: Settings) -> None:
    """Fail loudly if prices are outside a plausible range (usually a wrong price_divisor)."""
    if df.empty:
        return
    lo, hi = settings.data.sanity_price_range
    med = float(np.median(df["close"]))
    if not (lo <= med <= hi):
        raise ValueError(
            f"Median price {med:.3f} is outside the sanity range {lo}-{hi}. "
            "Check data.dukascopy.price_divisor in settings.yaml.")
