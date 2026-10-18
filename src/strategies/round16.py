"""Round 16 (docs/STRATEGIES_R16.md): equity-curve filter on walk-forward-chosen variants + stricter bar."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_SHADOW = 20


def equity_filter(chosen: pd.DataFrame, all_trades: pd.DataFrame, n: int = N_SHADOW) -> pd.DataFrame:
    """Keep a chosen trade only if its variant's last n shadow trades CLOSED before its entry sum to > 0."""
    keep = []
    by_v = {v: g.sort_values("exit_time") for v, g in all_trades.groupby("variant")}
    for _, t in chosen.iterrows():
        g = by_v[t["variant"]]
        ex = pd.DatetimeIndex(pd.to_datetime(g["exit_time"], utc=True)).as_unit("ns").asi8
        m = int(np.searchsorted(ex, pd.Timestamp(t["entry_time"]).value, side="left"))   # closed strictly before entry
        past = g["r"].to_numpy()[max(0, m - n):m]
        keep.append(len(past) == n and past.sum() > 0)
    return chosen[np.array(keep, dtype=bool)]


def bar(x: pd.DataFrame, risk_pct: float) -> dict:
    r = x["r"].to_numpy()
    if len(r) < 3:
        return {"trades": len(r), "pass": False}
    ts = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    c = np.concatenate([[0], np.cumsum(r)])
    dd = float((np.maximum.accumulate(c) - c).max())
    pos = int((x.groupby("fold")["r"].sum() > 0).sum())
    recent = x[pd.to_datetime(x["entry_time"], utc=True) >= pd.Timestamp("2024-04-16", tz="UTC")]["r"]
    need_r, max_dd = (0.20, 8.0) if risk_pct == 1.0 else (0.40, 16.0)
    ok = (r.mean() >= need_r and len(r) >= 100 and x["r_2x"].mean() > 0 and ts >= 2.5 and pos >= 7 and dd < max_dd
          and len(recent) > 0 and recent.mean() > 0)
    risk_usd = 2500 * risk_pct / 100
    return {"trades": len(r), "mean_r": round(float(r.mean()), 3), "usd": round(float(r.mean() * risk_usd), 2),
            "r_2x": round(float(x["r_2x"].mean()), 3), "t": round(float(ts), 2), "pos": f"{pos}/11",
            "dd_r": round(dd, 1), "dd_usd": round(dd * risk_usd),
            "last2y_r": round(float(recent.mean()), 3) if len(recent) else None, "pass": bool(ok)}
