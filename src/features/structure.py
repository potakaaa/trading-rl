"""Market structure shift (MSS), fair value gaps (FVG), order blocks (OB),
premium/discount and higher-timeframe (HTF) bias."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings
from src.features.basics import SwingIndex, find_swings, np_utc, ts64

# --------------------------------------------------------------------------------------- MSS


def find_mss(bars: pd.DataFrame, sweeps: pd.DataFrame, swings: SwingIndex, atr_s: pd.Series,
             settings: Settings) -> pd.DataFrame:
    """For each sweep, look for a market structure shift.

    Short (after a high was swept): the first bar j, from the reclaim bar up to
    pen_idx + max_bars_after_sweep, that CLOSES below the most recent confirmed swing low
    (confirmed before j, pivot before the penetration) with a bearish body >= disp_mult x ATR.
    Long mirrors this. The search stops early if price closes back beyond the swept level.
    Sweeps that lead to the same MSS candle are combined into one event (all of them are
    known by that candle's close, so this is lookahead-free).
    """
    cfg = settings.detectors.mss
    if sweeps.empty:
        return pd.DataFrame()
    o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    end = pd.DatetimeIndex(bars["end"])
    a_prev = atr_s.shift(1).to_numpy()
    n = len(bars)
    found = []
    for _, s in sweeps.iterrows():
        p, r = int(s["pen_idx"]), int(s["reclaim_idx"])
        short = s["direction"] == "short"
        last = min(p + cfg.max_bars_after_sweep, n - 1)
        for j in range(r, last + 1):
            if j > r and ((short and c[j] > s["level"]) or (not short and c[j] < s["level"])):
                break  # swept level re-accepted: sweep failed
            ref = swings.latest("low" if short else "high", j, pivot_max_idx=p)
            if ref is None or np.isnan(a_prev[j]):
                continue
            ref_idx, ref_price = ref
            body = c[j] - o[j]
            broke = c[j] < ref_price if short else c[j] > ref_price
            disp = (-body if short else body) >= cfg.disp_mult * a_prev[j]
            if broke and disp:
                ext_i = p + int(np.argmax(h[p:j + 1]) if short else np.argmin(lo[p:j + 1]))
                found.append({
                    "direction": s["direction"], "mss_idx": j, "known_at": end[j],
                    "pen_idx": p, "sweep_known_at": s["known_at"],
                    "extreme": h[ext_i] if short else lo[ext_i], "extreme_idx": ext_i,
                    "level": s["level"], "level_kind": s["level_kind"], "level_side": s["side"],
                    "depth_atr": s["depth_atr"], "ref_price": ref_price, "ref_idx": ref_idx,
                    "disp_body_atr": abs(body) / a_prev[j], "atr": a_prev[j],
                })
                break
    if not found:
        return pd.DataFrame()
    df = pd.DataFrame(found)
    out = []
    for (_, _), g in df.groupby(["direction", "mss_idx"]):
        row = g.sort_values("pen_idx").iloc[0].to_dict()
        short = row["direction"] == "short"
        best = g.loc[g["extreme"].idxmax() if short else g["extreme"].idxmin()]
        row["extreme"], row["extreme_idx"] = best["extreme"], int(best["extreme_idx"])
        row["pen_idx"] = int(g["pen_idx"].min())
        row["levels_swept"] = "+".join(dict.fromkeys(g.sort_values("pen_idx")["level_kind"]))
        far = g.loc[g["level"].idxmax() if short else g["level"].idxmin()]
        row["level"], row["level_kind"] = far["level"], far["level_kind"]
        row["depth_atr"] = float(g["depth_atr"].max())
        out.append(row)
    return pd.DataFrame(out).sort_values("mss_idx").reset_index(drop=True)


# --------------------------------------------------------------------------------------- FVG


def find_fvgs(bars: pd.DataFrame, atr_s: pd.Series, settings: Settings,
              fill_horizon: int = 600) -> pd.DataFrame:
    """3-candle fair value gaps, indexed by the third candle i (known at its close).

    Bullish: high[i-2] < low[i]  -> gap [high[i-2], low[i]]
    Bearish: low[i-2] > high[i]  -> gap [high[i], low[i-2]]
    Size must be >= min_fvg_atr x ATR(i). `filled_idx` is the first later bar (within
    fill_horizon bars) that trades through the whole gap. It is FUTURE information relative to i:
    only use it through fvg_filled_by(fvg, bar_idx).
    """
    h, lo = bars["high"].to_numpy(), bars["low"].to_numpy()
    end = pd.DatetimeIndex(bars["end"])
    a = atr_s.to_numpy()
    minsz = settings.detectors.fvg.min_fvg_atr
    n = len(bars)
    if n < 3:
        return pd.DataFrame(columns=["direction", "idx", "bottom", "top", "size_atr", "filled_idx", "known_at"])
    i = np.arange(2, n)
    h2, l2, hi_, li_, ai = h[:-2], lo[:-2], h[2:], lo[2:], a[2:]
    with np.errstate(invalid="ignore"):
        bull = (h2 < li_) & ((li_ - h2) >= minsz * ai)
        bear = (l2 > hi_) & ((l2 - hi_) >= minsz * ai)
    rows = []
    for k in np.flatnonzero(bull | bear):
        ii = int(i[k])
        if bull[k]:
            bottom, top, d = h2[k], li_[k], "bull"
            seg = lo[ii + 1: ii + 1 + fill_horizon] <= bottom
        else:
            bottom, top, d = hi_[k], l2[k], "bear"
            seg = h[ii + 1: ii + 1 + fill_horizon] >= top
        f = ii + 1 + int(np.argmax(seg)) if seg.any() else -1
        rows.append((d, ii, float(bottom), float(top), float((top - bottom) / ai[k]), f))
    df = pd.DataFrame(rows, columns=["direction", "idx", "bottom", "top", "size_atr", "filled_idx"])
    df["known_at"] = end[df["idx"].to_numpy().astype(int)]
    return df


def fvg_filled_by(fvg: pd.Series, bar_idx: int) -> bool:
    return fvg["filled_idx"] >= 0 and fvg["filled_idx"] <= bar_idx


# ---------------------------------------------------------------------------------- order block


def find_order_block(bars: pd.DataFrame, mss: pd.Series, settings: Settings) -> dict | None:
    """Last opposite-coloured candle before the displacement leg that caused the MSS.

    Short: the last bullish (close > open) candle at or before the sweep extreme, searching back
    at most max_lookback_bars. Zone = open..high (body mode) or low..high (full).
    Long: last bearish candle at/before the extreme low. Zone = low..open (body) or low..high.
    """
    cfg = settings.detectors.order_block
    o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    short = mss["direction"] == "short"
    e = int(mss["extreme_idx"])
    for k in range(e, max(-1, e - cfg.max_lookback_bars - 1), -1):
        if (short and c[k] > o[k]) or (not short and c[k] < o[k]):
            if short:
                bottom = o[k] if cfg.zone == "body" else lo[k]
                top = h[k]
            else:
                bottom = lo[k]
                top = o[k] if cfg.zone == "body" else h[k]
            return {"ob_idx": k, "ob_bottom": float(bottom), "ob_top": float(top)}
    return None


# ----------------------------------------------------------------------------- premium/discount


def dealing_range_at(swings_htf: SwingIndex, t: pd.Timestamp) -> tuple[float, float] | None:
    """(low, high) of the current dealing range: most recent confirmed swing high and swing low
    on the range timeframe, known at time t."""
    hi = swings_htf.latest_at("high", t)
    lo = swings_htf.latest_at("low", t)
    if hi is None or lo is None or hi[1] <= lo[1]:
        return None
    return lo[1], hi[1]


def pd_position(price: float, rng: tuple[float, float] | None) -> float | None:
    """0 = range low, 1 = range high. < 0.5 is discount, > 0.5 premium."""
    if rng is None:
        return None
    lo, hi = rng
    return (price - lo) / (hi - lo)


# ------------------------------------------------------------------------------------ HTF bias


def bos_direction(bars: pd.DataFrame, n: int) -> pd.Series:
    """Direction of the last confirmed break of structure, as of each bar's close.

    At bar i the usable swings are those confirmed before i. +1 after a close above the latest
    usable swing high, -1 after a close below the latest usable swing low, 0 before any break.
    Each swing can only be broken once.
    """
    sw = find_swings(bars, n).sort_values("confirm_idx")
    c = bars["close"].to_numpy()
    out = np.zeros(len(bars))
    latest = {"high": None, "low": None}   # (pivot idx, price, broken?)
    rows = list(sw[["kind", "idx", "price", "confirm_idx"]].itertuples(index=False))
    k = 0
    state = 0.0
    for i in range(len(bars)):
        while k < len(rows) and rows[k].confirm_idx < i:
            r = rows[k]
            cur = latest[r.kind]
            if cur is None or r.idx > cur[0]:
                latest[r.kind] = [r.idx, r.price, False]
            k += 1
        hi, lo = latest["high"], latest["low"]
        if hi is not None and not hi[2] and c[i] > hi[1]:
            state, hi[2] = 1.0, True
        elif lo is not None and not lo[2] and c[i] < lo[1]:
            state, lo[2] = -1.0, True
        out[i] = state
    return pd.Series(out, index=bars.index)


def htf_bias_series(htf: dict[str, pd.DataFrame], settings: Settings) -> pd.DataFrame:
    """BOS direction per HTF, each stamped with the bar's `end` (when it becomes known)."""
    out = {}
    for tf in settings.detectors.htf_bias.timeframes:
        b = htf[tf]
        d = bos_direction(b, settings.detectors.swing_n[tf])
        out[tf] = pd.DataFrame({"known_at": np_utc(b["end"]), f"bos_{tf}": d.to_numpy()})
    return out


def bias_at(bias_frames: dict[str, pd.DataFrame], t: pd.Timestamp, price: float,
            prev_day_mid: float | None, settings: Settings) -> dict:
    cfg = settings.detectors.htf_bias
    score = 0
    out = {}
    for tf, f in bias_frames.items():
        k = int(np.searchsorted(f["known_at"].to_numpy(), ts64(t), "right")) - 1
        v = float(f[f"bos_{tf}"].iloc[k]) if k >= 0 else 0.0
        out[f"bos_{tf}"] = v
        score += v
    if cfg.use_prev_day_midpoint and prev_day_mid is not None:
        m = 1.0 if price > prev_day_mid else -1.0 if price < prev_day_mid else 0.0
        out["vs_pd_mid"] = m
        score += m
    out["bias_score"] = score
    out["bias"] = ("bullish" if score >= cfg.bias_threshold else
                   "bearish" if score <= -cfg.bias_threshold else "neutral")
    return out
