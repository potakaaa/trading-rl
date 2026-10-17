"""Round-4 models and walk-forward evaluation, exactly as pre-registered in docs/STRATEGIES_R4.md."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ml.intraday_dataset import FEATURES
from src.ml.sequencing import MAX_PER_DAY, MAX_PER_SESSION, sequence  # noqa: F401

PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
TAUS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
RISK_USD = 12.5                 # 0.5% of $2,500
WIN_R = 5.0 / RISK_USD          # Rald's rule: a trade is a success only if it nets > $5 (= 0.4 R)


def _fit(X, y, seed):
    m = lgb.LGBMRegressor(random_state=seed, **PARAMS)
    m.fit(X, y)
    return m


class Models:
    def __init__(self, train: pd.DataFrame, seed: int = 0):
        tr = train.dropna(subset=["r_long", "r_short", "range60"])
        X = tr[FEATURES].to_numpy(dtype=float)
        self.vol = _fit(X, np.log(tr["range60"].clip(lower=1e-3)), seed)
        self.long = _fit(X, tr["r_long"].to_numpy(), seed + 1)
        self.short = _fit(X, tr["r_short"].to_numpy(), seed + 2)
        self.gate_threshold = float(np.median(self.vol.predict(X)))   # median of predictions on its training window

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES].to_numpy(dtype=float)
        out = pd.DataFrame(index=df.index)
        out["vol_pred"] = self.vol.predict(X)
        out["gate"] = out["vol_pred"] > self.gate_threshold
        out["p_long"] = self.long.predict(X)
        out["p_short"] = self.short.predict(X)
        return out


def _chooser(pred: pd.DataFrame, tau: float, use_gate: bool):
    def choose(r):
        p = pred.loc[r.name]
        if use_gate and not p["gate"]:
            return None
        side, val = ("long", p["p_long"]) if p["p_long"] >= p["p_short"] else ("short", p["p_short"])
        return side if val > tau else None
    return choose


def pick_tau(models: Models, val: pd.DataFrame, use_gate: bool) -> float:
    pred = models.predict(val)
    best, best_mean = 0.0, -np.inf
    for tau in TAUS:
        t = sequence(val, _chooser(pred, tau, use_gate))
        if len(t) >= 100 and t["r"].mean() > best_mean:
            best, best_mean = tau, t["r"].mean()
    return best


def run_fold(df: pd.DataFrame, start: str, end: str, seed: int) -> dict:
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    train = df[df["t"] < s]
    test = df[(df["t"] >= s) & (df["t"] < e)]
    inner_cut = s - pd.DateOffset(months=12)
    inner = Models(train[train["t"] < inner_cut], seed)
    val = train[train["t"] >= inner_cut]
    tau_g, tau_ng = pick_tau(inner, val, True), pick_tau(inner, val, False)
    full = Models(train, seed)
    pred = full.predict(test)
    rng = np.random.default_rng(seed)
    res = {
        "gbm_gated": sequence(test, _chooser(pred, tau_g, True)),
        "gbm_no_gate": sequence(test, _chooser(pred, tau_ng, False)),
        "gate_random_dir": sequence(test, lambda r: (rng.choice(["long", "short"]) if pred.loc[r.name, "gate"]
                                                     else None)),
    }
    return {"fold": f"{start[:7]}..{end[:10]}", "tau_gated": tau_g, "tau_no_gate": tau_ng,
            "train_rows": len(train), "test_rows": len(test), "trades": res}


def folds() -> list[tuple[str, str]]:
    out = [(f"{y}-{m:02d}-01", f"{y if m == 1 else y + 1}-{7 if m == 1 else 1:02d}-01")
           for y in range(2021, 2026) for m in (1, 7)]
    return out + [("2026-01-01", "2026-04-16")]


def summarize(results: list[dict]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for f in results:
        row = {"fold": f["fold"], "tau_gated": f["tau_gated"]}
        for k, t in f["trades"].items():
            row[f"{k}_n"] = len(t)
            row[f"{k}_R"] = float(t["r"].sum()) if len(t) else 0.0
            row[f"{k}_R_2x"] = float(t["r_2x"].sum()) if len(t) else 0.0
        rows.append(row)
    table = pd.DataFrame(rows)
    verdicts = {}
    for k in ("gbm_gated", "gbm_no_gate", "gate_random_dir"):
        allt = pd.concat([f["trades"][k] for f in results], ignore_index=True)
        r = allt["r"].to_numpy()
        tstat = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 else np.nan
        pos = int((table[f"{k}_R"] > 0).sum())
        verdicts[k] = {"trades": len(r), "mean_r": float(r.mean()) if len(r) else np.nan,
                       "mean_r_2x": float(allt["r_2x"].mean()) if len(r) else np.nan, "t_stat": float(tstat),
                       "win_rate": float((r > 0).mean()) if len(r) else np.nan,
                       "win_rate_over_5usd": float((r > WIN_R).mean()) if len(r) else np.nan,
                       "avg_usd_per_trade": float(r.mean() * RISK_USD) if len(r) else np.nan,
                       "total_r": float(r.sum()), "positive_folds": f"{pos}/{len(table)}",
                       "PASS": bool(len(r) >= 150 and allt["r_2x"].mean() > 0 and tstat >= 3.0 and pos >= 7
                                    and r.mean() > WIN_R)}
    return table, verdicts
