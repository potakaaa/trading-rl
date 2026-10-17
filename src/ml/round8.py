"""Round 8 runner: round-4 model at 10:00/10:30 with and without NQ order-flow features (no gate)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.flow_features import FLOW_FEATURES
from src.ml.intraday_dataset import FEATURES
from src.ml.intraday_gbm import Models, _chooser, folds, pick_tau
from src.ml.sequencing import sequence

RISK_USD, WIN_R = 12.5, 0.4


def run(df: pd.DataFrame, log=print) -> dict:
    arms = {"base_am": FEATURES, "flow": FEATURES + FLOW_FEATURES}
    out = {a: [] for a in arms}
    gains = []
    for k, (start, end) in enumerate(folds()):
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        train, test = df[df["t"] < s], df[(df["t"] >= s) & (df["t"] < e)]
        cut = s - pd.DateOffset(months=12)
        for arm, feats in arms.items():
            inner = Models(train[train["t"] < cut], k, feats)
            tau = pick_tau(inner, train[train["t"] >= cut], use_gate=False)
            full = Models(train, k, feats)
            tr = sequence(test, _chooser(full.predict(test), tau, False))
            out[arm].append((f"{start[:7]}..{end[:10]}", tau, tr))
            if arm == "flow":
                gains.append(pd.Series(full.long.booster_.feature_importance("gain")
                                       + full.short.booster_.feature_importance("gain"), index=feats))
            log(f"  {start[:7]} {arm:7s} tau {tau:.2f} trades {len(tr):4d} R {tr['r'].sum():+7.1f}")
    res = {}
    for arm, fl in out.items():
        allt = pd.concat([t for _, _, t in fl], ignore_index=True)
        r = allt["r"].to_numpy()
        ts = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 else np.nan
        pos = sum(t["r"].sum() > 0 for _, _, t in fl)
        res[arm] = {"trades": len(r), "avg_usd": float(r.mean() * RISK_USD) if len(r) else None,
                    "mean_r": float(r.mean()) if len(r) else None,
                    "mean_r_2x": float(allt["r_2x"].mean()) if len(r) else None, "t_stat": float(ts),
                    "win_rate_over_5usd": float((r > WIN_R).mean()) if len(r) else None,
                    "positive_folds": f"{pos}/{len(fl)}",
                    "PASS": bool(len(r) >= 150 and r.mean() > WIN_R and allt["r_2x"].mean() > 0 and ts >= 3
                                 and pos >= 7),
                    "folds": [(f, tau, len(t), float(t["r"].sum())) for f, tau, t in fl]}
    g = pd.concat(gains, axis=1).mean(axis=1)
    res["flow_gain_share"] = float(g[FLOW_FEATURES].sum() / g.sum())
    res["flow_gain"] = g[FLOW_FEATURES].round(0).to_dict()
    return res
