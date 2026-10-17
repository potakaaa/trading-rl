"""Round-5 published setups (docs/STRATEGIES_R5.md, frozen 2026-10-17), simulated on 1m BID bars.

Execution conventions (same as every other backtest here):
  market entry  : next 1m open; long pays the spread (ask = bid + spread)
  stop entry    : long buy-stop triggers when ask high >= level, fills at max(level, ask open) + slippage;
                  short sell-stop when bid low <= level, fills at min(level, bid open) - slippage
  protective stop: long when bid low <= stop -> min(stop, open) - slip; short when ask high >= stop -> ...
  target (limit): long when bid high >= target; short when ask low <= target; stop wins a same-bar tie
  end of day    : flat at the close of the 15:59 bar (long sells at bid, short buys at ask)
Decisions only use bars that are complete at the decision time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategies.rules import DailyContext
from src.timeutils import NY

SETUPS = ["noise_momentum", "orb5", "crabel_nr7", "holy_grail", "williams_vbo"]
CHECKS = [600 + 30 * k for k in range(12)]           # 10:00 .. 15:30 as minute-of-day


@dataclass(frozen=True)
class Cost:
    spread: float
    slip: float


@dataclass
class Day:
    date: pd.Timestamp
    o: np.ndarray
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    v: np.ndarray
    minute: np.ndarray           # minute of day (NY) of each bar's START
    ts: np.ndarray               # bar start, ns UTC
    prev_close: float | None
    ctx: dict | None             # previous trading day's pdh/pdl/pd_range/atr_d

    def idx(self, minute: int) -> int | None:
        k = np.flatnonzero(self.minute == minute)
        return int(k[0]) if len(k) else None

    @property
    def last(self) -> int:
        k = np.flatnonzero(self.minute <= 959)
        return int(k[-1])


def iter_days(m1: pd.DataFrame, d1: pd.DataFrame):
    m1 = m1.sort_index()
    ny = m1.index.tz_convert(NY)
    mins = (ny.hour * 60 + ny.minute).to_numpy()
    dates = ny.normalize().tz_localize(None)
    sess = (mins >= 570) & (mins <= 959)
    df = m1[sess]
    mins, dates = mins[sess], dates[sess]
    dctx = DailyContext(d1)
    prev_close = None
    for d, g in df.groupby(dates, sort=True):
        k = np.flatnonzero(dates == d)
        m = mins[k]
        if m[0] != 570 or m[-1] < 959:                 # need a full 09:30-16:00 session
            prev_close = float(g["close"].iloc[-1]) if m[-1] >= 959 else prev_close
            continue
        yield Day(pd.Timestamp(d), g["open"].to_numpy(), g["high"].to_numpy(), g["low"].to_numpy(),
                  g["close"].to_numpy(), g["volume"].to_numpy(dtype=float), m,
                  pd.DatetimeIndex(g.index).as_unit("ns").asi8, prev_close, dctx.get(pd.Timestamp(d)))
        prev_close = float(g["close"].iloc[-1])


# ------------------------------------------------------------------------------ position engine

def run_position(day: Day, i: int, side: str, entry: float, stop: float | None, target: float | None,
                 cost: Cost, stop_active_from_entry_bar: bool = True) -> tuple[int, float, str, float]:
    """Hold from bar i (entry happened during bar i) to stop / target / end of day.
    Returns (exit_bar, exit_price, reason, worst_adverse_pts)."""
    sp, sl = cost.spread, cost.slip
    mae = 0.0
    for j in range(i, day.last + 1):
        o, h, lo = day.o[j], day.h[j], day.lo[j]
        if side == "long":
            adverse = entry - lo
            stop_hit = stop is not None and lo <= stop and (j > i or stop_active_from_entry_bar)
            tgt_hit = target is not None and h >= target and j > i
        else:
            adverse = (h + sp) - entry
            stop_hit = stop is not None and h + sp >= stop and (j > i or stop_active_from_entry_bar)
            tgt_hit = target is not None and lo + sp <= target and j > i
        if stop_hit:
            px = (min(stop, o) - sl) if side == "long" else (max(stop, o + sp) + sl)
            if j == i:
                px = stop - sl if side == "long" else stop + sl
            return j, px, "sl", max(mae, (entry - px) if side == "long" else (px - entry))
        mae = max(mae, adverse)
        if tgt_hit:
            return j, target, "tp", mae
    j = day.last
    px = day.c[j] if side == "long" else day.c[j] + sp
    return j, px, "eod", mae


def trade_row(setup, day: Day, side, i_entry, entry, stop0, j, exit_px, reason, mae) -> dict | None:
    risk = abs(entry - stop0)
    if risk <= 0:
        return None
    pnl = (exit_px - entry) if side == "long" else (entry - exit_px)
    return {"setup": setup, "date": day.date, "side": side, "entry_time": pd.Timestamp(day.ts[i_entry], tz="UTC"),
            "exit_time": pd.Timestamp(day.ts[j], tz="UTC") + pd.Timedelta(minutes=1), "entry": entry,
            "stop0": stop0, "exit": exit_px, "reason": reason, "risk_pts": risk, "pnl_pts": pnl, "r": pnl / risk,
            "mae_r": mae / risk}


def stop_entry(day: Day, i0: int, i1: int, long_lvl: float | None, short_lvl: float | None, cost: Cost):
    """First trigger of a buy-stop / sell-stop between bars i0..i1. Returns (bar, side, fill) or None.
    If both trigger inside one bar we cannot know which came first: pessimistically take the long and let
    the engine evaluate the short level as its stop in that same bar."""
    sp, sl = cost.spread, cost.slip
    for j in range(i0, i1 + 1):
        o, h, lo = day.o[j], day.h[j], day.lo[j]
        up = long_lvl is not None and h + sp >= long_lvl
        dn = short_lvl is not None and lo <= short_lvl
        if up:
            return j, "long", max(long_lvl, o + sp) + sl
        if dn:
            return j, "short", min(short_lvl, o) - sl
    return None


# --------------------------------------------------------------------------------- setups

def orb5(day: Day, cost: Cost) -> list[dict]:
    i0, i5 = day.idx(570), day.idx(575)
    if i0 is None or i5 is None:
        return []
    o, c = day.o[i0], day.c[i5 - 1]
    if c == o:
        return []
    side = "long" if c > o else "short"
    hi, lo = day.h[i0:i5].max(), day.lo[i0:i5].min()
    entry = day.o[i5] + (cost.spread if side == "long" else 0.0)
    stop = lo if side == "long" else hi
    risk = abs(entry - stop)
    if (side == "long" and entry <= stop) or (side == "short" and entry >= stop):
        return []
    target = entry + 10 * risk if side == "long" else entry - 10 * risk
    j, px, why, mae = run_position(day, i5, side, entry, stop, target, cost)
    r = trade_row("orb5", day, side, i5, entry, stop, j, px, why, mae)
    return [r] if r else []


def williams_vbo(day: Day, cost: Cost) -> list[dict]:
    if day.ctx is None or not np.isfinite(day.ctx["pd_range"]):
        return []
    op = day.o[0]
    k = 0.5 * day.ctx["pd_range"]
    hit = stop_entry(day, 0, day.last, op + k, op - k, cost)
    if hit is None:
        return []
    j0, side, fill = hit
    j, px, why, mae = run_position(day, j0, side, fill, op, None, cost)
    r = trade_row("williams_vbo", day, side, j0, fill, op, j, px, why, mae)
    return [r] if r else []


def crabel_nr7(day: Day, cost: Cost, d1: pd.DataFrame) -> list[dict]:
    hist = d1[d1["ny_date"] < day.date].tail(10)
    if len(hist) < 10:
        return []
    rng = (hist["high"] - hist["low"]).to_numpy()
    if not rng[-1] < rng[-7:-1].min():                 # yesterday = narrowest of the last 7
        return []
    stretch = float(np.minimum(hist["open"] - hist["low"], hist["high"] - hist["open"]).mean())
    if stretch <= 0:
        return []
    op = day.o[0]
    hit = stop_entry(day, 0, day.last, op + stretch, op - stretch, cost)
    if hit is None:
        return []
    j0, side, fill = hit
    stop = op - stretch if side == "long" else op + stretch
    j, px, why, mae = run_position(day, j0, side, fill, stop, None, cost)
    r = trade_row("crabel_nr7", day, side, j0, fill, stop, j, px, why, mae)
    return [r] if r else []


def noise_momentum(day: Day, cost: Cost, sigma: dict) -> list[dict]:
    """sigma[minute] for this day (mean |move from open| over the previous 14 days)."""
    if day.prev_close is None or not sigma:
        return []
    op = day.o[0]
    up_ref, dn_ref = max(op, day.prev_close), min(op, day.prev_close)
    tp = (day.h + day.lo + day.c) / 3
    vol = np.where(day.v > 0, day.v, 0.0)
    cpv, cv = np.cumsum(tp * vol), np.cumsum(vol)
    trades, pos = [], None                           # pos = (side, entry_bar, entry_px, stop0, mae)
    for m in CHECKS:
        s = sigma.get(m)
        k = day.idx(m)                               # bar starting at the check time -> fills at its open
        if s is None or not np.isfinite(s) or k is None or k == 0:
            continue
        close = day.c[k - 1]                         # close of the bar ending at the check time
        ub, lb = up_ref * (1 + s), dn_ref * (1 - s)
        vwap = cpv[k - 1] / cv[k - 1] if cv[k - 1] > 0 else close
        if pos is not None:
            side, ib, ep, st0, _ = pos
            trail = max(ub, vwap) if side == "long" else min(lb, vwap)
            if (side == "long" and close < trail) or (side == "short" and close > trail):
                px = day.o[k] if side == "long" else day.o[k] + cost.spread
                mae = _mae(day, ib, k, side, ep, cost)
                t = trade_row("noise_momentum", day, side, ib, ep, st0, k - 1, px, "trail", mae)
                if t:
                    t["exit_time"] = pd.Timestamp(day.ts[k], tz="UTC")
                    trades.append(t)
                pos = None
        if pos is None:
            if close > ub:
                ep = day.o[k] + cost.spread
                pos = ("long", k, ep, max(ub, vwap), 0.0)
            elif close < lb:
                ep = day.o[k]
                pos = ("short", k, ep, min(lb, vwap), 0.0)
            if pos is not None and ((pos[0] == "long" and pos[2] <= pos[3])
                                    or (pos[0] == "short" and pos[2] >= pos[3])):
                pos = None                           # already through the stop at entry: no defined risk
    if pos is not None:
        side, ib, ep, st0, _ = pos
        j = day.last
        px = day.c[j] if side == "long" else day.c[j] + cost.spread
        t = trade_row("noise_momentum", day, side, ib, ep, st0, j, px, "eod", _mae(day, ib, j + 1, side, ep, cost))
        if t:
            trades.append(t)
    return trades


def _mae(day: Day, i0: int, i1: int, side: str, entry: float, cost: Cost) -> float:
    if i1 <= i0:
        return 0.0
    return float(entry - day.lo[i0:i1].min()) if side == "long" else float(day.h[i0:i1].max() + cost.spread - entry)


def noise_sigmas(days: list[Day]) -> dict[pd.Timestamp, dict[int, float]]:
    """For each day: sigma per check minute = mean over the PREVIOUS 14 days of |close_m / open - 1|."""
    moves = []
    for d in days:
        row = {}
        for m in CHECKS:
            k = d.idx(m)
            row[m] = abs(d.c[k - 1] / d.o[0] - 1) if k else np.nan
        moves.append(row)
    mv = pd.DataFrame(moves, index=[d.date for d in days])
    sig = mv.rolling(14, min_periods=14).mean().shift(1)
    return {dt: {m: float(v) for m, v in r.items()} for dt, r in sig.iterrows()}


def holy_grail(day: Day, cost: Cost, ind: dict) -> list[dict]:
    """ind: arrays of the continuous 5m series (ts ns = bar start, high, low, close, ema, adx)."""
    ts5 = ind["ts"]
    p0 = int(np.searchsorted(ts5, day.ts[0]))
    p1 = int(np.searchsorted(ts5, day.ts[0] + 390 * 60_000_000_000))
    trades, busy_until, n = [], -1, 0
    for q in range(max(p0, 21), p1):
        if n >= 2:
            break
        mod = int(pd.Timestamp(ts5[q], tz="UTC").tz_convert(NY).hour * 60
                  + pd.Timestamp(ts5[q], tz="UTC").tz_convert(NY).minute)
        if not (600 <= mod + 5 <= 930):               # bar closes 10:00 .. 15:30
            continue
        a, a_prev = ind["adx"][q], ind["adx"][q - 1]
        if not (a > 30 and a > a_prev):
            continue
        e, e_prev = ind["ema"][q], ind["ema"][q - 1]
        hi, lo, cl = ind["high"][q], ind["low"][q], ind["close"][q]
        up = e > e_prev and lo <= e < cl
        dn = e < e_prev and hi >= e > cl
        if not (up or dn):
            continue
        i_close = int(np.searchsorted(day.ts, ts5[q] + 5 * 60_000_000_000))
        if i_close <= busy_until or i_close > day.last:
            continue
        if up:
            target, stop = ind["high"][q - 20:q].max(), lo
            hit = stop_entry(day, i_close, min(i_close + 15, day.last), hi, None, cost)
        else:
            target, stop = ind["low"][q - 20:q].min(), hi
            hit = stop_entry(day, i_close, min(i_close + 15, day.last), None, lo, cost)
        if hit is None:
            continue
        j0, side, fill = hit
        risk = abs(fill - stop)
        if risk <= 0 or (side == "long" and (target - fill) < risk) or (side == "short" and (fill - target) < risk):
            continue
        j, px, why, mae = run_position(day, j0, side, fill, stop, target, cost)
        tr = trade_row("holy_grail", day, side, j0, fill, stop, j, px, why, mae)
        if tr:
            trades.append(tr)
            busy_until = j
            n += 1
    return trades


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, lo, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -lo.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    a = 1 / n
    atr_ = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=a, adjust=False).mean() / atr_
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=a, adjust=False).mean() / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()


def run_all(m1: pd.DataFrame, b5: pd.DataFrame, d1: pd.DataFrame, cost: Cost, only=None, log=print) -> pd.DataFrame:
    d1 = d1.sort_index().copy()
    d1["ny_date"] = pd.DatetimeIndex(d1["end"]).tz_convert(NY).normalize().tz_localize(None)
    days = list(iter_days(m1, d1))
    sig = noise_sigmas(days)
    b5 = b5.sort_index()
    ind = {"ts": pd.DatetimeIndex(b5.index).as_unit("ns").asi8, "high": b5["high"].to_numpy(),
           "low": b5["low"].to_numpy(), "close": b5["close"].to_numpy(),
           "ema": b5["close"].ewm(span=20, adjust=False).mean().to_numpy(), "adx": adx(b5).to_numpy()}
    rows = []
    for d in days:
        for name in (only or SETUPS):
            if name == "orb5":
                rows += orb5(d, cost)
            elif name == "williams_vbo":
                rows += williams_vbo(d, cost)
            elif name == "crabel_nr7":
                rows += crabel_nr7(d, cost, d1)
            elif name == "noise_momentum":
                rows += noise_momentum(d, cost, sig.get(d.date, {}))
            elif name == "holy_grail":
                rows += holy_grail(d, cost, ind)
    log(f"  {len(days)} full sessions, {len(rows)} trades")
    return pd.DataFrame(rows)
