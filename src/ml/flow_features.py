"""Round 8 order-flow features (docs/STRATEGIES_R8.md, amended): NQ futures trades in the 3 minutes before
each decision time (10:00 and 10:30 NY). Relative measures compare with the previous 20 days only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.databento_flow import load_day

NY = "America/New_York"
FLOW_FEATURES = ["of_delta3", "of_vol3_rel", "of_trades3_rel", "of_big_delta3", "of_avg_size3", "of_vwap_dist3"]
BIG = 10


def window_stats(tr: pd.DataFrame, t: pd.Timestamp, atr15: float) -> dict:
    w = tr[(tr.index >= t - pd.Timedelta(minutes=3)) & (tr.index < t)]
    if w.empty:
        return {"vol": np.nan, "n": np.nan, "avg": np.nan, "of_delta3": np.nan, "of_big_delta3": np.nan,
                "of_vwap_dist3": np.nan}
    sz = w["size"].astype(float)
    buy, sell = sz[w["side"] == "B"].sum(), sz[w["side"] == "A"].sum()
    big = sz >= BIG                       # sizes are UNSIGNED ints in Databento data: use the float copy
    bb = float(sz[big & (w["side"] == "B")].sum())
    bs = float(sz[big & (w["side"] == "A")].sum())
    vwap = float((w["price"] * sz).sum() / sz.sum())
    return {"vol": float(sz.sum()), "n": float(len(w)), "avg": float(sz.mean()),
            "of_delta3": float((buy - sell) / sz.sum()),
            "of_big_delta3": float((bb - bs) / (bb + bs)) if bb + bs > 0 else 0.0,
            "of_vwap_dist3": float((w["price"].iloc[-1] - vwap) / atr15) if atr15 > 0 else np.nan}


def add_flow_features(df: pd.DataFrame, log=print) -> pd.DataFrame:
    """df: round-4 decision rows. Keeps only the 10:00 and 10:30 decisions, adds the 6 features."""
    d = df[df["slot"].isin(["10:00", "10:30"])].copy().sort_values("t")
    stats = []
    cache: dict = {}
    for _, r in d.iterrows():
        day = str(r["date"].date())
        if day not in cache:
            cache.clear()
            cache[day] = load_day(day)
        tr = cache[day]
        if tr is None:
            stats.append({k: np.nan for k in ("vol", "n", "avg", "of_delta3", "of_big_delta3", "of_vwap_dist3")})
            continue
        if tr.index.tz is None:
            tr.index = tr.index.tz_localize("UTC")
        stats.append(window_stats(tr, r["t"], r["atr15"]))
    s = pd.DataFrame(stats, index=d.index)
    d = pd.concat([d, s], axis=1)
    for col, out in (("vol", "of_vol3_rel"), ("n", "of_trades3_rel"), ("avg", "of_avg_size3")):
        # median of the same slot over the PREVIOUS 20 days (shift: today's value is never in its own baseline)
        base = d.groupby("slot")[col].transform(lambda x: x.shift(1).rolling(20, min_periods=10).median())
        d[out] = d[col] / base
    log(f"  order-flow features for {d['of_delta3'].notna().sum()} of {len(d)} decisions")
    return d.drop(columns=["vol", "n", "avg"])
