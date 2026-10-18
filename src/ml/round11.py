"""Round 11 (docs/STRATEGIES_R11.md): first trade of the day whose meta-score clears tau, across 3 setups."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ml import round10 as r10

VARIANTS = {"crabel|filt=none,mult=1.0,exit=3.0": 0, "williams|k=0.5,stop=half,exit=3.0": 1,
            "noise|band=1.25,every=30,trail=band": 2}
FEATS = r10.FEATS + ["family", "stop_atr"]


def build(trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for v, code in VARIANTS.items():
        t = r10.build(trades.assign(variant=trades["variant"].replace({v: r10.VARIANT}))
                      .loc[trades["variant"] == v], daily)
        t["family"] = code
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    key = pd.DatetimeIndex(t["date"]).normalize()
    t["stop_atr"] = t["risk_pts"].to_numpy() / daily["atr20"].reindex(key).to_numpy()
    return t.sort_values("entry_time").reset_index(drop=True)


def _fit(df, seed):
    m = lgb.LGBMRegressor(random_state=seed, **r10.PARAMS)
    m.fit(df[FEATS].to_numpy(float), df["r"].to_numpy())
    return m


def first_qualifying(df: pd.DataFrame, pred: np.ndarray, tau: float) -> pd.DataFrame:
    d = df.assign(pred=pred)
    d = d[d["pred"] > tau].sort_values("entry_time")
    return d.groupby(pd.DatetimeIndex(d["date"]).normalize(), group_keys=False).head(1)


def walk_forward(t: pd.DataFrame, folds):
    rows, taken = [], []
    for k, (start, end) in enumerate(folds):
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        train, test = t[t["entry_time"] < s], t[(t["entry_time"] >= s) & (t["entry_time"] < e)]
        cut = s - pd.DateOffset(months=12)
        inner, val = train[train["entry_time"] < cut], train[train["entry_time"] >= cut]
        pv = _fit(inner, k).predict(val[FEATS].to_numpy(float))
        tau, best = r10.TAUS[0], -np.inf
        for th in r10.TAUS:
            sel = first_qualifying(val, pv, th)
            if len(sel) >= 30 and sel["r"].mean() > best:
                tau, best = th, sel["r"].mean()
        p = _fit(train, k).predict(test[FEATS].to_numpy(float))
        tk = first_qualifying(test, p, tau).assign(fold=f"{start[:7]}..{end[:10]}")
        taken.append(tk)
        rows.append({"fold": f"{start[:7]}..{end[:10]}", "tau": tau, "taken": len(tk), "R": float(tk["r"].sum()),
                     "mean": float(tk["r"].mean()) if len(tk) else np.nan})
    return pd.concat(taken, ignore_index=True), pd.DataFrame(rows)
