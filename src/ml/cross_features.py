"""Round 7 cross-market features (docs/STRATEGIES_R7.md): for each other market X, its recent returns
in units of its own ATR(15m), and US100's divergence from it. Only bars complete at t are used."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings
from src.features.basics import atr

CROSS = ["US500", "US30", "US2000", "DXY", "USTBOND", "XAUUSD", "EURUSD", "USDJPY", "OIL"]
MIN_NS = 60_000_000_000


def cross_feature_names() -> list[str]:
    return [f"{x}_{k}" for x in CROSS for k in ("ret5", "ret15", "ret60", "div60")]


def market_features(df: pd.DataFrame, m1: pd.DataFrame, b15: pd.DataFrame, name: str) -> pd.DataFrame:
    m1 = m1.sort_index()
    ts = pd.DatetimeIndex(m1.index).as_unit("ns").asi8
    c = m1["close"].to_numpy(dtype=float)
    b15 = b15.sort_index()
    a15 = atr(b15, 14).to_numpy()
    a_end = pd.DatetimeIndex(b15["end"]).as_unit("ns").asi8
    t = pd.DatetimeIndex(df["t"]).as_unit("ns").asi8
    k = np.searchsorted(ts, t) - 1                       # last bar that STARTED before t (complete at t)
    fresh = (k >= 0) & (t - (ts[np.clip(k, 0, None)] + MIN_NS) <= 5 * MIN_NS)
    q = np.searchsorted(a_end, t, side="right") - 1
    A = np.where(q >= 0, a15[np.clip(q, 0, None)], np.nan)
    A = np.where(A > 0, A, np.nan)
    ck = np.where(k >= 0, c[np.clip(k, 0, None)], np.nan)
    out = {}
    for n in (5, 15, 60):
        kb = np.searchsorted(ts, t - n * MIN_NS) - 1
        cb = np.where(kb >= 0, c[np.clip(kb, 0, None)], np.nan)
        out[f"{name}_ret{n}"] = np.where(fresh, (ck - cb) / A, np.nan)
    out[f"{name}_div60"] = df["ret60"].to_numpy() - out[f"{name}_ret60"]
    return pd.DataFrame(out, index=df.index)


def add_cross_features(df: pd.DataFrame, settings: Settings, log=print) -> pd.DataFrame:
    from src.data.freeserv_history import instrument_settings
    from src.data.store import load_bars
    start = df["t"].min() - pd.Timedelta(days=5)
    end = df["t"].max() + pd.Timedelta(days=1)
    parts = [df]
    for x in CROSS:
        s = instrument_settings(settings, x)
        m1 = load_bars(s, "1m", start, end)
        b15 = load_bars(s, "15m", start - pd.Timedelta(days=5), end)
        f = market_features(df, m1, b15, x)
        log(f"  {x}: {np.isfinite(f[f'{x}_ret60']).mean():.0%} of decisions have fresh data")
        parts.append(f)
    return pd.concat(parts, axis=1)
