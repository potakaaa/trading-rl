"""Round 10 (docs/STRATEGIES_R10.md): meta-labelled stretch breakout on US100, walk-forward."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.basics import atr

NY = "America/New_York"
VARIANT = "crabel|filt=none,mult=1.0,exit=3.0"
FEATS = ["nr_rank7", "inside_day", "yrange_atr", "ret5_atr", "yclose_loc", "gap_atr", "on_range_atr",
         "trig_minute", "dir", "with_trend", "stretch_atr", "atr_pct", "weekday", "news_day"]
PARAMS = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=50, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
TAUS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def daily_table(d1: pd.DataFrame, m1: pd.DataFrame, news: pd.DataFrame | None) -> pd.DataFrame:
    """One row per NY date D with features known BEFORE D's 09:30 open (plus the open itself)."""
    d = d1.sort_index().copy()
    d["ny_date"] = pd.DatetimeIndex(d["end"]).tz_convert(NY).normalize().tz_localize(None)
    d["atr20"] = atr(d, 20)
    rng = d["high"] - d["low"]
    d["nr_rank7"] = rng.rolling(7).apply(lambda x: (x < x[-1]).sum() + 1, raw=True)
    d["inside"] = ((d["high"] < d["high"].shift()) & (d["low"] > d["low"].shift())).astype(float)
    d["yrange_atr"] = rng / d["atr20"]
    d["ret5_atr"] = (d["close"] - d["close"].shift(5)) / d["atr20"]
    d["yclose_loc"] = (d["close"] - d["low"]) / rng.replace(0, np.nan)
    d["stretch"] = np.minimum(d["open"] - d["low"], d["high"] - d["open"]).rolling(10).mean()
    d["atr_pct"] = d["atr20"].rolling(252, min_periods=60).rank(pct=True)
    # shift by one row: the values for trading date D come from the day BEFORE D
    cols = ["atr20", "nr_rank7", "inside", "yrange_atr", "ret5_atr", "yclose_loc", "stretch", "atr_pct"]
    prev = d[cols].shift(1)
    prev["ny_date"] = d["ny_date"].values
    prev = prev.set_index("ny_date")
    # 09:30 open, previous 16:00 close, overnight range (18:00 prev .. 09:30)
    ny = m1.index.tz_convert(NY)
    mins = ny.hour * 60 + ny.minute
    date = ny.normalize().tz_localize(None)
    o930 = m1["open"][mins == 570].groupby(date[mins == 570]).first()
    c1559 = m1["close"][mins == 959].groupby(date[mins == 959]).last()
    on_mask = (mins >= 18 * 60) | (mins < 570)
    on_date = pd.Series(date, index=m1.index).where(mins < 570, pd.Series(date + pd.Timedelta(days=1), index=m1.index))
    onr = (m1["high"][on_mask].groupby(on_date[on_mask].values).max()
           - m1["low"][on_mask].groupby(on_date[on_mask].values).min())
    out = prev.copy()
    out["open930"] = o930.reindex(out.index)
    out["prev_close"] = c1559.shift(1).reindex(out.index)      # previous session's 15:59 close
    out["on_range"] = onr.reindex(out.index)
    if news is not None and len(news):
        nd = pd.DatetimeIndex(news["ts"]).tz_convert(NY).normalize().tz_localize(None)
        out["news_day"] = out.index.isin(set(nd)).astype(float)
    else:
        out["news_day"] = 0.0
    return out


def build(trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    t = trades[trades["variant"] == VARIANT].copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    key = pd.DatetimeIndex(t["date"]).normalize()
    f = daily.reindex(key)
    f.index = t.index
    a = f["atr20"]
    t["nr_rank7"], t["inside_day"], t["yrange_atr"] = f["nr_rank7"], f["inside"], f["yrange_atr"]
    t["ret5_atr"], t["yclose_loc"] = f["ret5_atr"], f["yclose_loc"]
    t["gap_atr"] = (f["open930"] - f["prev_close"]) / a
    t["on_range_atr"] = f["on_range"] / a
    ny = t["entry_time"].dt.tz_convert(NY)
    t["trig_minute"] = ny.dt.hour * 60 + ny.dt.minute - 570
    t["dir"] = np.where(t["side"] == "long", 1.0, -1.0)
    t["with_trend"] = t["dir"] * np.sign(t["ret5_atr"])
    t["stretch_atr"] = f["stretch"] / a
    t["atr_pct"], t["news_day"] = f["atr_pct"], f["news_day"]
    t["weekday"] = ny.dt.weekday
    return t.sort_values("entry_time").reset_index(drop=True)


def _fit(df, seed):
    m = lgb.LGBMRegressor(random_state=seed, **PARAMS)
    m.fit(df[FEATS].to_numpy(float), df["r"].to_numpy())
    return m


def walk_forward(t: pd.DataFrame, folds) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, taken = [], []
    for k, (start, end) in enumerate(folds):
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        train, test = t[t["entry_time"] < s], t[(t["entry_time"] >= s) & (t["entry_time"] < e)]
        cut = s - pd.DateOffset(months=12)
        inner, val = train[train["entry_time"] < cut], train[train["entry_time"] >= cut]
        m_in = _fit(inner, k)
        pv = m_in.predict(val[FEATS].to_numpy(float))
        tau, best = TAUS[0], -np.inf
        for th in TAUS:
            sel = val["r"][pv > th]
            if len(sel) >= 30 and sel.mean() > best:
                tau, best = th, sel.mean()
        m = _fit(train, k)
        p = m.predict(test[FEATS].to_numpy(float))
        tk = test[p > tau].assign(fold=f"{start[:7]}..{end[:10]}", pred=p[p > tau])
        taken.append(tk)
        rows.append({"fold": f"{start[:7]}..{end[:10]}", "tau": tau, "candidates": len(test), "taken": len(tk),
                     "R_taken": float(tk["r"].sum()), "mean_taken": float(tk["r"].mean()) if len(tk) else np.nan,
                     "mean_all": float(test["r"].mean())})
    return pd.concat(taken, ignore_index=True), pd.DataFrame(rows)


def verdict(x: pd.DataFrame, months: float) -> dict:
    r = x["r"].to_numpy()
    if len(r) < 3:
        return {"trades": len(r)}
    ts = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    c = np.concatenate([[0], np.cumsum(r)])
    dd = float((np.maximum.accumulate(c) - c).max())
    pos = int((x.groupby("fold")["r"].sum() > 0).sum()) if "fold" in x else None
    trusted = len(r) >= 100 and x["r_2x"].mean() > 0 and ts >= 2.5 and (pos or 0) >= 7
    return {"trades": len(r), "per_month": len(r) / months, "mean_r": r.mean(), "mean_r_2x": x["r_2x"].mean(),
            "usd_0.5pct": r.mean() * 12.5, "usd_1pct": r.mean() * 25, "t": ts, "pos_folds": pos,
            "max_dd_r": dd, "max_dd_usd_0.5pct": dd * 12.5, "max_dd_usd_1pct": dd * 25,
            "goal_1pct": bool(r.mean() >= 0.20 and trusted), "goal_0.5pct": bool(r.mean() >= 0.40 and trusted),
            "trusted": bool(trusted)}
