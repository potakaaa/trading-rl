"""Round-2 rule strategies, exactly as pre-registered in docs/STRATEGIES.md (frozen 2026-10-16).

Every strategy decides at a 5m bar CLOSE using only bars that are complete at that moment, and emits
rows in the same format as setups.detect (so the backtester, guard and live engine reuse them).
Entries are market orders at the next 1m open (`order_type = "market"`); `entry` holds the signal
bar's close as the reference price, and R is re-measured from the actual fill in the backtest.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config import Settings
from src.features.basics import atr
from src.timeutils import NY, session_end_utc, session_of

STRATEGIES = ["orb15", "orb30", "vwap_revert", "momo_open", "momo_lasthalf", "gap_fill", "pd_break"]
STOP_ATR_CAP = 3.0
FIVE = pd.Timedelta(minutes=5)


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


class Day:
    """One NY calendar day of 5m bars with helpers keyed by NY wall-clock start time."""

    def __init__(self, bars: pd.DataFrame, idx: np.ndarray):
        self.b = bars.iloc[idx]
        self.idx = idx
        ny = self.b.index.tz_convert(NY)
        self.minute = (ny.hour * 60 + ny.minute).to_numpy()
        self.pos = {m: k for k, m in enumerate(self.minute)}

    def at(self, hhmm: str):
        k = self.pos.get(_hm(hhmm))
        return None if k is None else self.b.iloc[k]

    def between(self, start: str, end: str) -> pd.DataFrame:
        """Bars whose START is in [start, end) NY."""
        m = (self.minute >= _hm(start)) & (self.minute < _hm(end))
        return self.b[m]


def generate(bars5: pd.DataFrame, daily: pd.DataFrame, settings: Settings,
             only: list[str] | None = None) -> pd.DataFrame:
    """All candidate trades for the given 5m bars. `daily` = 1d bars (trading-day 18:00-17:00)."""
    bars = bars5.copy()
    if "atr" not in bars:
        bars["atr"] = atr(bars, settings.detectors.atr_period)
    ny_dates = bars.index.tz_convert(NY).date
    groups = pd.Series(np.arange(len(bars))).groupby(ny_dates).apply(lambda s: s.to_numpy())
    dly = _daily_context(daily, settings)
    fns: dict[str, Callable] = {"orb15": _orb(15), "orb30": _orb(30), "vwap_revert": _vwap_revert,
                                "momo_open": _momo_open, "momo_lasthalf": _momo_lasthalf,
                                "gap_fill": _gap_fill, "pd_break": _pd_break}
    names = only or STRATEGIES
    rows, prev_day = [], None
    for d, idx in groups.items():
        day = Day(bars, idx)
        ctx = dly.get(pd.Timestamp(d))
        for name in names:
            for r in fns[name](day, prev_day, ctx, bars):
                rows.append(_finish(r, name, bars, settings))
        if day.at("15:55") is not None:
            prev_day = day
    cols = ["setup_type", "direction", "status", "session", "armed_at", "expires_at", "entry", "stop", "target",
            "rr", "risk_pts", "order_type", "atr", "signal_idx"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    for c in ("armed_at", "expires_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra].sort_values(["armed_at", "setup_type"]).reset_index(drop=True)


class DailyContext:
    """Previous-trading-day high/low/range and daily ATR for a NY date, using only daily bars that
    are COMPLETE before that date (the current day's bar is never needed, as in live trading)."""

    def __init__(self, daily: pd.DataFrame | None):
        if daily is None or daily.empty:
            self.dates = np.array([], dtype="datetime64[ns]")
            return
        d = daily.sort_index().copy()
        d["atr_d"] = atr(d, 14)
        td = pd.to_datetime(d["trading_date"]) if "trading_date" in d else \
            pd.Series(d.index.tz_convert(NY).normalize().tz_localize(None), index=d.index)
        self.dates = td.to_numpy(dtype="datetime64[ns]")
        self.d = d

    def get(self, date: pd.Timestamp) -> dict | None:
        k = int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(date).to_datetime64()), side="left")) - 1
        if k < 0:
            return None
        prev = self.d.iloc[k]
        a = prev["atr_d"]
        return {"pdh": float(prev["high"]), "pdl": float(prev["low"]), "pd_range": float(prev["high"] - prev["low"]),
                "atr_d": float(a) if pd.notna(a) else np.nan}


def _daily_context(daily: pd.DataFrame | None, settings: Settings) -> DailyContext:
    return DailyContext(daily)


def _trade(bar, direction: str, stop: float, target: float | None = None, r_mult: float | None = None, **extra):
    entry = float(bar["close"])
    risk = abs(entry - stop)
    if target is None:
        target = entry + r_mult * risk if direction == "long" else entry - r_mult * risk
    return {"bar": bar, "direction": direction, "entry": entry, "stop": float(stop), "target": float(target),
            **extra}


def _finish(r: dict, name: str, bars: pd.DataFrame, settings: Settings) -> dict:
    bar = r.pop("bar")
    t_end = bar["end"]
    risk = abs(r["entry"] - r["stop"])
    a = float(bar["atr"])
    sess = session_of(pd.DatetimeIndex([bar.name]), settings.sessions).iloc[0]
    rr = abs(r["target"] - r["entry"]) / risk if risk > 0 else 0.0
    status = "valid"
    wrong_side = (r["direction"] == "long" and (r["stop"] >= r["entry"] or r["target"] <= r["entry"])) or \
                 (r["direction"] == "short" and (r["stop"] <= r["entry"] or r["target"] >= r["entry"]))
    if sess is None:
        status = "outside_session"
    elif risk <= 0 or wrong_side:
        status = "bad_stop_or_target"
    elif not np.isfinite(a) or risk > STOP_ATR_CAP * a:
        status = "stop_too_wide"
    elif rr < 1.0 - 1e-9:                           # Rald's rule: never target less than 1R
        status = "target_below_1r"
    elif session_end_utc(t_end, settings.sessions[sess].end) - t_end < \
            pd.Timedelta(minutes=settings.setups.min_minutes_to_session_end):
        status = "too_close_to_session_end"
    expires = t_end + FIVE
    out = {"setup_type": name, "status": status, "session": sess, "armed_at": t_end, "expires_at": expires,
           "rr": rr, "risk_pts": risk, "order_type": "market", "atr": a,
           "signal_idx": int(bars.index.get_loc(bar.name)), **r}
    return out


# --------------------------------------------------------------------------- strategies

def _orb(minutes: int):
    def run(day: Day, prev, ctx, bars):
        start, or_end = "09:30", "09:45" if minutes == 15 else "10:00"
        rng = day.between(start, or_end)
        if len(rng) < minutes // 5:
            return []
        hi, lo = rng["high"].max(), rng["low"].min()
        for _, bar in day.between(or_end, "11:00").iterrows():
            if bar["close"] > hi:
                return [_trade(bar, "long", lo, r_mult=1.5, or_high=hi, or_low=lo)]
            if bar["close"] < lo:
                return [_trade(bar, "short", hi, r_mult=1.5, or_high=hi, or_low=lo)]
        return []
    return run


def _vwap_revert(day: Day, prev, ctx, bars):
    sess = day.between("09:30", "16:00")
    if len(sess) < 8:
        return []
    tp = (sess["high"] + sess["low"] + sess["close"]) / 3
    vol = sess["volume"].clip(lower=0).to_numpy()
    if vol.sum() <= 0:
        vol = np.ones(len(sess))
    cum_v = np.cumsum(vol)
    vwap = np.cumsum(tp.to_numpy() * vol) / np.where(cum_v > 0, cum_v, 1)
    dev = sess["close"].to_numpy() - vwap
    sd = np.array([dev[: k + 1].std(ddof=0) if k >= 5 else np.nan for k in range(len(dev))])
    up, dn = vwap + 2 * sd, vwap - 2 * sd
    ny = sess.index.tz_convert(NY)
    mins = ny.hour * 60 + ny.minute
    out, used = [], set()
    windows = [(_hm("10:00"), _hm("11:15"), "am"), (_hm("13:45"), _hm("15:30"), "pm")]
    c = sess["close"].to_numpy()
    for k in range(1, len(sess)):
        if not np.isfinite(sd[k - 1]) or not np.isfinite(sd[k]):
            continue
        m = mins[k]
        win = next((w for a, b, w in windows if a <= m < b), None)
        if win is None or win in used:
            continue
        bar = sess.iloc[k]
        lo3 = sess["low"].iloc[max(0, k - 2): k + 1].min()
        hi3 = sess["high"].iloc[max(0, k - 2): k + 1].max()
        a = float(bar["atr"]) if np.isfinite(bar["atr"]) else 0.0
        if c[k - 1] > up[k - 1] and c[k] <= up[k]:
            out.append(_trade(bar, "short", hi3 + 0.1 * a, target=vwap[k], vwap=vwap[k], vwap_sd=sd[k]))
            used.add(win)
        elif c[k - 1] < dn[k - 1] and c[k] >= dn[k]:
            out.append(_trade(bar, "long", lo3 - 0.1 * a, target=vwap[k], vwap=vwap[k], vwap_sd=sd[k]))
            used.add(win)
    return out


def _momo_open(day: Day, prev, ctx, bars):
    first, b10 = day.at("09:30"), day.at("09:55")      # 09:55 bar closes at 10:00
    if first is None or b10 is None or ctx is None or not np.isfinite(ctx["pd_range"]):
        return []
    move = b10["close"] - first["open"]
    rng = day.between("09:30", "10:00")
    if abs(move) < 0.3 * ctx["pd_range"] or move == 0:
        return []
    if move > 0:
        return [_trade(b10, "long", rng["low"].min(), r_mult=2.0, move=move)]
    return [_trade(b10, "short", rng["high"].max(), r_mult=2.0, move=move)]


def _momo_lasthalf(day: Day, prev, ctx, bars):
    b10, b1530 = day.at("09:55"), day.at("15:25")      # closes at 10:00 and 15:30
    if b10 is None or b1530 is None or prev is None or prev.at("15:55") is None:
        return []
    prev_close = prev.at("15:55")["close"]
    r1 = b10["close"] - prev_close
    if r1 == 0 or not np.isfinite(b1530["atr"]):
        return []
    d = "long" if r1 > 0 else "short"
    e = b1530["close"]
    stop = e - 1.5 * b1530["atr"] if d == "long" else e + 1.5 * b1530["atr"]
    return [_trade(b1530, d, stop, r_mult=1.0, first_half_hour=r1)]


def _gap_fill(day: Day, prev, ctx, bars):
    first = day.at("09:30")                              # closes at 09:35
    if first is None or prev is None or prev.at("15:55") is None or ctx is None or not np.isfinite(ctx["atr_d"]):
        return []
    prev_close = prev.at("15:55")["close"]
    gap = first["open"] - prev_close
    g = abs(gap) / ctx["atr_d"] if ctx["atr_d"] > 0 else np.nan
    if not (0.25 <= g <= 1.0):
        return []
    a = float(first["atr"]) if np.isfinite(first["atr"]) else 0.0
    if gap > 0:
        return [_trade(first, "short", first["high"] + 0.1 * a, target=prev_close, gap_atr=g)]
    return [_trade(first, "long", first["low"] - 0.1 * a, target=prev_close, gap_atr=-g)]


def _pd_break(day: Day, prev, ctx, bars):
    if ctx is None:
        return []
    out, done = [], set()
    sess = pd.concat([day.between("09:30", "11:30"), day.between("13:30", "16:00")])
    for t, bar in sess.iterrows():
        i = bars.index.get_loc(t)
        if i == 0:
            continue
        pc = bars["close"].iloc[i - 1]
        a = float(bar["atr"]) if np.isfinite(bar["atr"]) else 0.0
        if "pdh" not in done and pc <= ctx["pdh"] < bar["close"]:
            out.append(_trade(bar, "long", bar["low"] - 0.1 * a, r_mult=2.0, level=ctx["pdh"]))
            done.add("pdh")
        elif "pdl" not in done and pc >= ctx["pdl"] > bar["close"]:
            out.append(_trade(bar, "short", bar["high"] + 0.1 * a, r_mult=2.0, level=ctx["pdl"]))
            done.add("pdl")
    return out
