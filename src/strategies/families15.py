"""Round 15 families (docs/STRATEGIES_R15.md). All decisions use bars complete at the decision time."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from src.strategies.published import Cost, Day, iter_days, run_position, stop_entry, trade_row
from src.strategies.variants import opening_drive


def _tgt(side, entry, stop, t):
    if t == "eod":
        return None
    r = abs(entry - stop)
    return entry + t * r if side == "long" else entry - t * r


def _hist(d1, day, n):
    return d1[d1["ny_date"] < day.date].tail(n)


def _stretch(h):
    return float(np.minimum(h["open"] - h["low"], h["high"] - h["open"]).mean())


def orb(day: Day, cost: Cost, d1, n: int, target, filt: str):
    h = _hist(d1, day, 10)
    if len(h) < 10:
        return []
    if filt == "NR4":
        rng = (h["high"] - h["low"]).to_numpy()
        if not rng[-1] < rng[-4:-1].min():
            return []
    k = day.idx(570 + n)
    end = day.idx(690)
    if k is None or end is None:
        return []
    hi, lo = day.h[:k].max(), day.lo[:k].min()
    hit = stop_entry(day, k, end - 1, hi, lo, cost)
    if hit is None:
        return []
    j0, side, fill = hit
    stop = lo if side == "long" else hi
    if (side == "long" and fill <= stop) or (side == "short" and fill >= stop):
        return []
    j, px, why, mae = run_position(day, j0, side, fill, stop, _tgt(side, fill, stop, target), cost)
    r = trade_row("orb", day, side, j0, fill, stop, j, px, why, mae)
    return [r] if r else []


def inside(day: Day, cost: Cost, d1, target):
    h = _hist(d1, day, 2)
    if len(h) < 2:
        return []
    y, p = h.iloc[-1], h.iloc[-2]
    if not (y["high"] < p["high"] and y["low"] > p["low"]):
        return []
    hit = stop_entry(day, 0, day.last, float(y["high"]), float(y["low"]), cost)
    if hit is None:
        return []
    j0, side, fill = hit
    stop = float(y["low"]) if side == "long" else float(y["high"])
    if (side == "long" and fill <= stop) or (side == "short" and fill >= stop):
        return []
    j, px, why, mae = run_position(day, j0, side, fill, stop, _tgt(side, fill, stop, target), cost)
    r = trade_row("inside", day, side, j0, fill, stop, j, px, why, mae)
    return [r] if r else []


def pm_trend(day: Day, cost: Cost, d1, m: float, target):
    h = _hist(d1, day, 10)
    k, k13 = day.idx(840), day.idx(780)
    if len(h) < 10 or k is None or k13 is None:
        return []
    st = _stretch(h)
    move = day.c[k - 1] - day.o[0]
    if st <= 0 or abs(move) < m * st:
        return []
    side = "long" if move > 0 else "short"
    entry = day.o[k] + (cost.spread if side == "long" else 0.0)
    stop = day.lo[k13:k].min() if side == "long" else day.h[k13:k].max()
    if (side == "long" and entry <= stop) or (side == "short" and entry >= stop):
        return []
    j, px, why, mae = run_position(day, k, side, entry, stop, _tgt(side, entry, stop, target), cost)
    r = trade_row("pm_trend", day, side, k, entry, stop, j, px, why, mae)
    return [r] if r else []


def gap_go(day: Day, cost: Cost, d1, atr_d: dict, g: float, target):
    a = atr_d.get(day.date)
    k = day.idx(575)
    if day.prev_close is None or not a or k is None:
        return []
    gap = day.o[0] - day.prev_close
    if abs(gap) / a < g:
        return []
    c5 = day.c[k - 1] - day.o[0]
    if np.sign(c5) != np.sign(gap) or c5 == 0:
        return []
    side = "long" if gap > 0 else "short"
    entry = day.o[k] + (cost.spread if side == "long" else 0.0)
    stop = day.lo[:k].min() if side == "long" else day.h[:k].max()
    if (side == "long" and entry <= stop) or (side == "short" and entry >= stop):
        return []
    j, px, why, mae = run_position(day, k, side, entry, stop, _tgt(side, entry, stop, target), cost)
    r = trade_row("gap_go", day, side, k, entry, stop, j, px, why, mae)
    return [r] if r else []


def grid():
    return {
        "orb": [dict(n=n, target=t, filt=f) for n, t, f in product([15, 30, 60], [2.0, 3.0, 5.0], ["none", "NR4"])],
        "inside": [dict(target=t) for t in [2.0, 3.0, "eod"]],
        "pm_trend": [dict(m=m, target=t) for m, t in product([0.5, 1.0], [2.0, 3.0])],
        "gap_go": [dict(g=g, target=t) for g, t in product([0.3, 0.6], [2.0, 3.0])],
        "drive": [dict(min_move=m, target_r=t, max_risk=r) for m, t, r in
                  product([0.3, 0.5, 0.75], [2.0, 3.0, 5.0], [1.0, 1.5])],
    }


def run(m1: pd.DataFrame, d1: pd.DataFrame, cost: Cost, log=print) -> pd.DataFrame:
    from src.features.basics import atr
    from src.timeutils import NY
    d1 = d1.sort_index().copy()
    d1["ny_date"] = pd.DatetimeIndex(d1["end"]).tz_convert(NY).normalize().tz_localize(None)
    a = atr(d1, 14).shift(1)
    atr_d = dict(zip(d1["ny_date"], a, strict=False))
    days = list(iter_days(m1, d1))
    rows = []
    for fam, plist in grid().items():
        for p in plist:
            v = fam + "|" + ",".join(f"{k}={x}" for k, x in p.items())
            for d in days:
                if fam == "orb":
                    out = orb(d, cost, d1, **p)
                elif fam == "inside":
                    out = inside(d, cost, d1, **p)
                elif fam == "pm_trend":
                    out = pm_trend(d, cost, d1, **p)
                elif fam == "gap_go":
                    out = gap_go(d, cost, d1, atr_d, **p)
                else:
                    out = opening_drive(d, cost, d1, **p)
                for t in out:
                    t["variant"], t["family"] = v, fam
                rows += out
        log(f"  {fam}: done")
    return pd.DataFrame(rows)
