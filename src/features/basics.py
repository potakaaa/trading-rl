"""ATR and fractal swing points.

Conventions used by every detector in src/features:
- Bars are a DataFrame indexed by bar open time (UTC) with open/high/low/close and `end`
  (the time the bar is complete). Positional index i = the i-th bar.
- Every detection carries `known_at`: the earliest time it could have been known in real time.
  Downstream code may only use a detection at or after its `known_at`. Tests enforce this by
  checking that running on a truncated history gives identical results (no lookahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def np_utc(x) -> np.ndarray:
    """tz-aware timestamps (Index/Series/array) -> numpy datetime64[ns] in UTC, for fast searchsorted."""
    return pd.DatetimeIndex(x).tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")


def ts64(t: pd.Timestamp) -> np.datetime64:
    return pd.Timestamp(t).tz_convert("UTC").tz_localize(None).to_datetime64().astype("datetime64[ns]")


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range. Value at bar i uses bars <= i only."""
    prev_close = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"],
                    (bars["high"] - prev_close).abs(),
                    (bars["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def find_swings(bars: pd.DataFrame, n: int) -> pd.DataFrame:
    """Fractal pivots with n bars on each side.

    Swing high at bar i: high[i] > each of the n highs before it and >= each of the n highs
    after it (so the first of two equal highs is the pivot). Swing low mirrors this.
    A swing is only CONFIRMED at the close of bar i + n; `known_at` = end of that bar.
    """
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    end = pd.DatetimeIndex(bars["end"] if "end" in bars else bars.index)
    size = len(bars)
    rows = []
    if size < 2 * n + 1:
        return _empty_swings()
    hs = pd.Series(h)
    ls = pd.Series(lo)
    left_max = hs.shift(1).rolling(n).max().to_numpy()
    right_max = hs[::-1].shift(1).rolling(n).max()[::-1].to_numpy()
    left_min = ls.shift(1).rolling(n).min().to_numpy()
    right_min = ls[::-1].shift(1).rolling(n).min()[::-1].to_numpy()
    is_high = (h > left_max) & (h >= right_max)
    is_low = (lo < left_min) & (lo <= right_min)
    for i in np.flatnonzero(is_high):
        rows.append((i, "high", h[i], i + n))
    for i in np.flatnonzero(is_low):
        rows.append((i, "low", lo[i], i + n))
    if not rows:
        return _empty_swings()
    df = pd.DataFrame(rows, columns=["idx", "kind", "price", "confirm_idx"])
    df["ts"] = bars.index[df["idx"].to_numpy()]
    df["known_at"] = end[df["confirm_idx"].to_numpy()]
    return df.sort_values(["confirm_idx", "idx"]).reset_index(drop=True)


def _empty_swings() -> pd.DataFrame:
    return pd.DataFrame({"idx": pd.Series(dtype=int), "kind": pd.Series(dtype=object),
                         "price": pd.Series(dtype=float), "confirm_idx": pd.Series(dtype=int),
                         "ts": pd.Series(dtype="datetime64[ns, UTC]"),
                         "known_at": pd.Series(dtype="datetime64[ns, UTC]")})


class SwingIndex:
    """Fast lookups of the latest confirmed swing (O(log n)).

    Relies on confirm_idx == idx + n, which holds for fractal swings with fixed n.
    """

    def __init__(self, swings: pd.DataFrame, n: int):
        self.n = n
        self._d = {}
        for kind in ("high", "low"):
            s = swings[swings["kind"] == kind].sort_values("idx")
            self._d[kind] = (s["idx"].to_numpy(), s["price"].to_numpy(), np_utc(s["known_at"]) if len(s)
                             else np.array([], dtype="datetime64[ns]"))

    def latest(self, kind: str, before_idx: int, pivot_max_idx: int | None = None) -> tuple[int, float] | None:
        """Latest pivot confirmed strictly before bar `before_idx` (and pivot <= pivot_max_idx)."""
        idx, price, _ = self._d[kind]
        bound = before_idx - self.n - 1          # confirm_idx < before_idx
        if pivot_max_idx is not None:
            bound = min(bound, pivot_max_idx)
        k = int(np.searchsorted(idx, bound, "right")) - 1
        return None if k < 0 else (int(idx[k]), float(price[k]))

    def latest_at(self, kind: str, t: pd.Timestamp) -> tuple[int, float] | None:
        """Latest pivot whose confirmation is known at time t."""
        idx, price, known = self._d[kind]
        k = int(np.searchsorted(known, ts64(t), "right")) - 1
        return None if k < 0 else (int(idx[k]), float(price[k]))
