"""SYNTHETIC US100-like 1-minute bars, for tests and for developing the pipeline offline.

This is a random walk with realistic-ish trading hours, an intraday volatility profile and fat
tails. It contains NO real market behaviour. Never draw trading conclusions from it; every chart
or report built from it is labelled SYNTHETIC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.timeutils import is_market_open, wall_minutes

# (NY wall-clock start minute, relative volatility)
_PROFILE = [(0, 0.5), (2 * 60, 0.9), (5 * 60, 0.7), (8 * 60 + 30, 1.3), (9 * 60 + 30, 2.6),
            (10 * 60 + 30, 1.6), (11 * 60 + 30, 1.0), (13 * 60 + 30, 1.4), (15 * 60, 1.8),
            (16 * 60, 0.8), (18 * 60, 0.6)]


def synthetic_m1(start: str, end: str, seed: int = 7, start_price: float = 15000.0,
                 base_sigma: float = 2.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left", name="ts")
    idx = idx[is_market_open(idx)]
    m = wall_minutes(idx)
    starts = np.array([p[0] for p in _PROFILE])
    vols = np.array([p[1] for p in _PROFILE])
    prof = vols[np.searchsorted(starts, m, side="right") - 1]
    days = pd.Index(idx.normalize())
    day_codes, uniq = pd.factorize(days)
    regime = np.exp(rng.normal(0, 0.35, len(uniq)))[day_codes]
    sigma = base_sigma * prof * regime
    shocks = rng.standard_t(4, len(idx)) / np.sqrt(2)  # unit-variance-ish fat tails
    # mild trend persistence so structure (swings / displacement) appears
    drift = pd.Series(rng.normal(0, 0.15, len(idx))).ewm(halflife=30).mean().to_numpy()
    rets = sigma * (shocks + drift)
    close = start_price + np.cumsum(rets)
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, 0.45, (2, len(idx)))) * sigma
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    vol = np.round(rng.gamma(2.0, 50.0, len(idx)) * prof, 2)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": vol}, index=idx).round(3)
