"""Liquidity pools and liquidity sweeps.

Levels (one row each):
  PDH/PDL   previous trading day high/low          known at the previous day's close (17:00 NY)
  ASIA_H/L  Asia session high/low (20:00-00:00 NY)  known when the Asia window ends
  LON_H/L   London session high/low (02:00-05:00)   known when the London window ends
  ON_H/L    overnight high/low (18:00-09:30)        known at 09:30
  EQH/EQL   equal highs/lows: two confirmed swings within equal_level_atr x ATR
Session levels stay valid until the end of their trading day; equal levels for
`equal_max_age_hours`. `side` is "high" for buy-side liquidity (above price) and "low" for
sell-side liquidity (below price).

Sweep of a high level L (mirror for lows):
  penetration bar p = first bar after L is known with high >= L + min_sweep x ATR, or that
  closes above L. It is a sweep if some bar r in [p, p + reclaim_bars] closes back below L and
  the high from p..r reached L + min_sweep x ATR. Otherwise the level is "broken".
  Either way the level is consumed at bar p. A sweep is known at the close of r.
Sweeps are NOT merged here (merging with a later sweep would leak the future into an earlier
event); simultaneous sweeps are combined later at the MSS candle, when all of them are known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings
from src.features.basics import np_utc, ts64
from src.timeutils import hhmm_minutes, in_window, trading_date

LEVEL_COLUMNS = ["kind", "side", "price", "known_at", "expires_at", "trading_date", "src_idx"]


def _session_close_utc(td: pd.Timestamp, close_hhmm: str) -> pd.Timestamp:
    return (td + pd.Timedelta(minutes=hhmm_minutes(close_hhmm))).tz_localize("America/New_York").tz_convert("UTC")


def session_levels(bars: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    liq = settings.detectors.liquidity
    day_start = settings.data.trading_day_start
    close = settings.data.market_close
    td = trading_date(bars.index, day_start)
    rows = []
    # previous-day high/low
    by_day = bars.groupby(td)
    daily = pd.DataFrame({"high": by_day["high"].max(), "low": by_day["low"].min(),
                          "end": by_day["end"].max(), "hi_idx": by_day["high"].idxmax(),
                          "lo_idx": by_day["low"].idxmin()})
    days = list(daily.index)
    pos = {t: i for i, t in enumerate(bars.index)}
    for prev, cur in zip(days[:-1], days[1:]):
        d = daily.loc[prev]
        exp = _session_close_utc(cur, close)
        rows.append(("PDH", "high", d["high"], d["end"], exp, cur, pos[d["hi_idx"]]))
        rows.append(("PDL", "low", d["low"], d["end"], exp, cur, pos[d["lo_idx"]]))
    # windowed session ranges, belonging to the trading day they fall in
    for name, win in (("ASIA", liq.asia), ("LON", liq.london), ("ON", liq.overnight)):
        mask = in_window(bars.index, win.start, win.end)
        sub = bars[mask]
        if sub.empty:
            continue
        g = sub.groupby(trading_date(sub.index, day_start))
        for day, part in g:
            exp = _session_close_utc(day, close)
            win_end = part["end"].max()
            # only a complete window defines a level; known when its last bar closes
            nominal_end = _window_end_utc(day, win.start, win.end, day_start)
            if win_end < nominal_end:
                continue
            hi_t, lo_t = part["high"].idxmax(), part["low"].idxmin()
            rows.append((f"{name}_H", "high", part["high"].max(), win_end, exp, day, pos[hi_t]))
            rows.append((f"{name}_L", "low", part["low"].min(), win_end, exp, day, pos[lo_t]))
    df = pd.DataFrame(rows, columns=LEVEL_COLUMNS)
    return df


def _window_end_utc(day: pd.Timestamp, start: str, end: str, day_start: str) -> pd.Timestamp:
    """Wall-clock end of a window on trading date `day` (windows may begin the evening before)."""
    s, e, ds = hhmm_minutes(start), hhmm_minutes(end), hhmm_minutes(day_start)
    # minutes after the trading-day start (18:00 previous evening)
    rel_end = (e - ds) % 1440 or 1440
    if (s - ds) % 1440 >= rel_end:  # degenerate; treat as ending at day end
        rel_end = 1440
    wall = day - pd.Timedelta(minutes=1440 - ds) + pd.Timedelta(minutes=rel_end)
    return wall.tz_localize("America/New_York", ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")


def equal_levels(bars: pd.DataFrame, swings: pd.DataFrame, atr_s: pd.Series,
                 settings: Settings) -> pd.DataFrame:
    liq = settings.detectors.liquidity
    a = atr_s.to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    td = trading_date(bars.index, settings.data.trading_day_start)
    rows = []
    max_age = pd.Timedelta(hours=liq.equal_max_age_hours)
    for kind, side, label in (("high", "high", "EQH"), ("low", "low", "EQL")):
        s = swings[swings["kind"] == kind].sort_values("confirm_idx")
        hist: list[pd.Series] = []
        for _, sw in s.iterrows():
            c = int(sw["confirm_idx"])
            tol = liq.equal_level_atr * a[c] if not np.isnan(a[c]) else np.nan
            for prev in reversed(hist[-liq.equal_lookback_swings:]):
                if np.isnan(tol) or abs(sw["price"] - prev["price"]) > tol:
                    continue
                i0, i1 = int(prev["idx"]), int(sw["idx"])
                if i1 - i0 < 2:
                    continue
                # intact: nothing traded beyond the level between the two pivots
                if side == "high":
                    lvl = max(sw["price"], prev["price"])
                    if high[i0 + 1:i1].max() > lvl:
                        continue
                else:
                    lvl = min(sw["price"], prev["price"])
                    if low[i0 + 1:i1].min() < lvl:
                        continue
                known = sw["known_at"]
                rows.append((label, side, lvl, known, known + max_age, td[c], i1))
                break
            hist.append(sw)
    return pd.DataFrame(rows, columns=LEVEL_COLUMNS)


def find_sweeps(bars: pd.DataFrame, levels: pd.DataFrame, atr_s: pd.Series,
                settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (levels with consumed_idx/consumed_at/outcome, sweeps)."""
    cfg = settings.detectors.sweep
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    end = pd.DatetimeIndex(bars["end"])
    starts = np_utc(bars.index)
    a_prev = atr_s.shift(1).to_numpy()
    n = len(bars)
    levels = levels.copy().reset_index(drop=True)
    consumed_idx = np.full(len(levels), -1)
    outcome = np.array([""] * len(levels), dtype=object)
    sweeps = []
    for li, lv in levels.iterrows():
        i0 = int(np.searchsorted(starts, ts64(lv["known_at"]), "left"))
        i1 = int(np.searchsorted(starts, ts64(lv["expires_at"]), "left"))
        if i0 >= min(i1, n):
            continue
        L = lv["price"]
        thr = cfg.min_sweep_atr * a_prev[i0:i1]
        if lv["side"] == "high":
            hit = (h[i0:i1] >= L + thr) | (c[i0:i1] > L)
        else:
            hit = (lo[i0:i1] <= L - thr) | (c[i0:i1] < L)
        hit &= ~np.isnan(thr)
        if not hit.any():
            continue
        p = i0 + int(np.argmax(hit))
        consumed_idx[li] = p
        last = min(p + cfg.reclaim_bars, n - 1)
        r = None
        for k in range(p, last + 1):
            if (lv["side"] == "high" and c[k] < L) or (lv["side"] == "low" and c[k] > L):
                r = k
                break
        if r is None:
            # not yet decidable if the reclaim window extends past the data we have
            outcome[li] = "pending" if p + cfg.reclaim_bars > n - 1 else "broken"
            continue
        if lv["side"] == "high":
            ext_i = p + int(np.argmax(h[p:r + 1]))
            ext = h[ext_i]
            deep = ext >= L + cfg.min_sweep_atr * a_prev[p]
        else:
            ext_i = p + int(np.argmin(lo[p:r + 1]))
            ext = lo[ext_i]
            deep = ext <= L - cfg.min_sweep_atr * a_prev[p]
        if not deep:
            outcome[li] = "broken"
            continue
        outcome[li] = "swept"
        sweeps.append({
            "level_id": li, "level_kind": lv["kind"], "side": lv["side"], "level": L,
            "direction": "short" if lv["side"] == "high" else "long",
            "pen_idx": p, "reclaim_idx": r, "extreme": ext, "extreme_idx": ext_i,
            "depth_atr": abs(ext - L) / a_prev[p], "atr": a_prev[p],
            "pen_ts": bars.index[p], "known_at": end[r],
        })
    levels["consumed_idx"] = consumed_idx
    levels["consumed_at"] = [bars.index[i] if i >= 0 else pd.NaT for i in consumed_idx]
    levels["outcome"] = outcome
    return levels, pd.DataFrame(sweeps)
