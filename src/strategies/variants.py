"""Round 9 (docs/STRATEGIES_R9.md): parameter grids of the three small-edge US100 setups and
walk-forward selection (variants chosen on the previous 3 years only, traded on the next fold)."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from src.strategies.published import Cost, Day, iter_days, run_position, stop_entry, trade_row

NOISE_MARKS = [600 + 15 * k for k in range(24)]          # 10:00 .. 15:45 every 15 minutes


def grid() -> dict[str, list[dict]]:
    return {
        "williams": [dict(k=k, stop=s, exit=e) for k, s, e in
                     product([0.25, 0.40, 0.50, 0.65, 0.80], ["open", "half"], ["eod", 2.0, 3.0])],
        "crabel": [dict(filt=f, mult=m, exit=e) for f, m, e in
                   product(["NR7", "NR4", "none"], [1.0, 1.5, 2.0], ["eod", 2.0, 3.0])],
        "noise": [dict(band=b, every=ev, trail=tr) for b, ev, tr in
                  product([1.0, 1.25, 1.5], [30, 15], ["vwap", "band"])],
    }


def vid(setup: str, p: dict) -> str:
    return setup + "|" + ",".join(f"{k}={v}" for k, v in p.items())


def _target(side, entry, stop, exit_):
    if exit_ == "eod":
        return None
    r = abs(entry - stop)
    return entry + exit_ * r if side == "long" else entry - exit_ * r


# --------------------------------------------------------------------------------- setups

def williams(day: Day, cost: Cost, k: float, stop: str, exit) -> list[dict]:
    if day.ctx is None or not np.isfinite(day.ctx["pd_range"]):
        return []
    op = day.o[0]
    dist = k * day.ctx["pd_range"]
    hit = stop_entry(day, 0, day.last, op + dist, op - dist, cost)
    if hit is None:
        return []
    j0, side, fill = hit
    st = op if stop == "open" else (op + dist / 2 if side == "long" else op - dist / 2)
    if (side == "long" and fill <= st) or (side == "short" and fill >= st):
        return []
    j, px, why, mae = run_position(day, j0, side, fill, st, _target(side, fill, st, exit), cost)
    r = trade_row("williams", day, side, j0, fill, st, j, px, why, mae)
    return [r] if r else []


def crabel(day: Day, cost: Cost, d1: pd.DataFrame, filt: str, mult: float, exit) -> list[dict]:
    hist = d1[d1["ny_date"] < day.date].tail(10)
    if len(hist) < 10:
        return []
    rng = (hist["high"] - hist["low"]).to_numpy()
    if filt == "NR7" and not rng[-1] < rng[-7:-1].min():
        return []
    if filt == "NR4" and not rng[-1] < rng[-4:-1].min():
        return []
    stretch = mult * float(np.minimum(hist["open"] - hist["low"], hist["high"] - hist["open"]).mean())
    if stretch <= 0:
        return []
    op = day.o[0]
    hit = stop_entry(day, 0, day.last, op + stretch, op - stretch, cost)
    if hit is None:
        return []
    j0, side, fill = hit
    st = op - stretch if side == "long" else op + stretch
    j, px, why, mae = run_position(day, j0, side, fill, st, _target(side, fill, st, exit), cost)
    r = trade_row("crabel", day, side, j0, fill, st, j, px, why, mae)
    return [r] if r else []


def noise(day: Day, cost: Cost, sig: dict, band: float, every: int, trail: str) -> list[dict]:
    if day.prev_close is None or not sig:
        return []
    marks = set(range(600, 960, every))
    op = day.o[0]
    up_ref, dn_ref = max(op, day.prev_close), min(op, day.prev_close)
    tp = (day.h + day.lo + day.c) / 3
    vol = np.where(day.v > 0, day.v, 0.0)
    cpv, cv = np.cumsum(tp * vol), np.cumsum(vol)
    sp, sl = cost.spread, cost.slip
    trades, pos = [], None                # pos: dict(side, i, entry, stop, mae)
    for k in range(1, day.last + 1):
        m = int(day.minute[k])
        # 1) intrabar hard stop on the open position (bar k)
        if pos is not None:
            side = pos["side"]
            if side == "long":
                pos["mae"] = max(pos["mae"], pos["entry"] - day.lo[k])
                hit = day.lo[k] <= pos["stop"]
                px = min(pos["stop"], day.o[k]) - sl
            else:
                pos["mae"] = max(pos["mae"], day.h[k] + sp - pos["entry"])
                hit = day.h[k] + sp >= pos["stop"]
                px = max(pos["stop"], day.o[k] + sp) + sl
            if hit:
                t = trade_row("noise", day, side, pos["i"], pos["entry"], pos["stop"], k, px, "sl", pos["mae"])
                if t:
                    trades.append(t)
                pos = None
        # 2) decisions at check marks, on the close of the bar that ended at the mark, filled at bar k's open
        if m not in marks or m > 945:
            continue
        s = sig.get(m)
        if s is None or not np.isfinite(s) or s <= 0:
            continue
        close = day.c[k - 1]
        ub, lb = up_ref * (1 + band * s), dn_ref * (1 - band * s)
        vwap = cpv[k - 1] / cv[k - 1] if cv[k - 1] > 0 else close
        if pos is not None:
            side = pos["side"]
            tr = (max(ub, vwap) if side == "long" else min(lb, vwap)) if trail == "vwap" else (ub if side == "long"
                                                                                                 else lb)
            if (side == "long" and close < tr) or (side == "short" and close > tr):
                px = day.o[k] if side == "long" else day.o[k] + sp
                t = trade_row("noise", day, side, pos["i"], pos["entry"], pos["stop"], k, px, "trail", pos["mae"])
                if t:
                    t["exit_time"] = pd.Timestamp(day.ts[k], tz="UTC")
                    trades.append(t)
                pos = None
        if pos is None and (close > ub or close < lb):
            side = "long" if close > ub else "short"
            entry = day.o[k] + (sp if side == "long" else 0.0)
            dist = band * s * entry
            pos = {"side": side, "i": k, "entry": entry, "stop": entry - dist if side == "long" else entry + dist,
                   "mae": 0.0}
    if pos is not None:
        j = day.last
        px = day.c[j] if pos["side"] == "long" else day.c[j] + sp
        t = trade_row("noise", day, pos["side"], pos["i"], pos["entry"], pos["stop"], j, px, "eod", pos["mae"])
        if t:
            trades.append(t)
    return trades


def noise_sigmas(days: list[Day]) -> dict:
    moves = []
    for d in days:
        row = {}
        for m in NOISE_MARKS:
            k = d.idx(m)
            row[m] = abs(d.c[k - 1] / d.o[0] - 1) if k else np.nan
        moves.append(row)
    mv = pd.DataFrame(moves, index=[d.date for d in days])
    sig = mv.rolling(14, min_periods=14).mean().shift(1)
    return {dt: {m: float(v) for m, v in r.items()} for dt, r in sig.iterrows()}


# ----------------------------------------------------------------------------- running

def run_variants(m1: pd.DataFrame, d1: pd.DataFrame, cost: Cost, log=print) -> pd.DataFrame:
    d1 = d1.sort_index().copy()
    from src.timeutils import NY
    d1["ny_date"] = pd.DatetimeIndex(d1["end"]).tz_convert(NY).normalize().tz_localize(None)
    days = list(iter_days(m1, d1))
    sig = noise_sigmas(days)
    rows = []
    for setup, plist in grid().items():
        for p in plist:
            v = vid(setup, p)
            n0 = len(rows)
            for d in days:
                if setup == "williams":
                    out = williams(d, cost, **p)
                elif setup == "crabel":
                    out = crabel(d, cost, d1, **p)
                else:
                    out = noise(d, cost, sig.get(d.date, {}), **p)
                for t in out:
                    t["variant"] = v
                    t["family"] = setup
                rows += out
            log(f"  {v}: {len(rows) - n0} trades")
    return pd.DataFrame(rows)


def walk_forward(t: pd.DataFrame, folds: list[tuple[str, str]], lookback_years: int = 3,
                 min_trades: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (chosen OOS trades, per-fold choice table with the PBO indicator)."""
    t = t.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    chosen, rows = [], []
    for start, end in folds:
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        train = t[(t["entry_time"] >= s - pd.DateOffset(years=lookback_years)) & (t["entry_time"] < s)]
        test = t[(t["entry_time"] >= s) & (t["entry_time"] < e)]
        for fam, g in train.groupby("family"):
            stats = g.groupby("variant")["r"].agg(["mean", "size"])
            stats = stats[stats["size"] >= min_trades]
            if stats.empty:
                continue
            best = stats["mean"].idxmax()
            oos = test[test["variant"] == best]
            oos_all = test[test["family"] == fam].groupby("variant")["r"].mean()
            rows.append({"fold": f"{start[:7]}..{end[:10]}", "family": fam, "chosen": best,
                         "is_mean_r": float(stats.loc[best, "mean"]), "oos_mean_r": float(oos["r"].mean())
                         if len(oos) else np.nan, "oos_trades": len(oos),
                         "below_median": bool(len(oos) and oos["r"].mean() < oos_all.median())})
            chosen.append(oos.assign(fold=f"{start[:7]}..{end[:10]}"))
    return (pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame()), pd.DataFrame(rows)


def portfolio(chosen: pd.DataFrame, max_per_day: int = 3) -> pd.DataFrame:
    c = chosen.sort_values("entry_time")
    return c.groupby(c["entry_time"].dt.tz_convert("America/New_York").dt.date, group_keys=False).head(max_per_day)
