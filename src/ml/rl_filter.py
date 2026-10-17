"""RL trade filter: offline reinforcement learning with fitted Q-iteration (docs/STRATEGIES.md).

Environment (one episode = one trading day):
  state  = market features known at the candidate's signal time + account state within the day
           (R so far today, trades taken today)
  action = 0 (skip), 0.5 (half size), 1 (full size)
  reward = a*R - 0.25*a*max(0, -R)  and  -1 once if the day's R falls to <= -2 (daily-limit pressure)
  next state = the day's next candidate with the updated account state; the day's end is terminal.
Each candidate's raw outcome R does not depend on our action, so rewards for every action are known
and the day state can be explored with random behaviour rollouts. Q(s, a) is a LightGBM regressor
refit for 3 iterations (discount 1 within the day). The policy picks argmax Q and trades only if
that Q > 0. Training/evaluation is walk-forward; nothing from a test fold is used to train for it.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import Settings
from src.timeutils import NY

ACTIONS = np.array([0.0, 0.5, 1.0])
LOSS_AVERSION = 0.25
DAY_STOP_R, DAY_STOP_PENALTY = -2.0, 1.0
N_ITER, N_ROLLOUTS = 3, 4
LGB_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)

MARKET_FEATURES = ["strategy_code", "dir", "is_pm", "minute", "weekday", "risk_atr", "rr", "atr_pct",
                   "ret_1h_atr", "ret_day_atr", "ema_dist_atr", "pos_in_pd_range", "pd_range_atrd",
                   "gap_atrd", "news_to_min", "news_since_min", "strat_recent_r"]
STATE_FEATURES = ["day_r", "day_trades"]


# ------------------------------------------------------------------------------- features

def build_features(c: pd.DataFrame, bars5: pd.DataFrame, daily: pd.DataFrame, news: pd.DataFrame | None,
                   settings: Settings) -> pd.DataFrame:
    """Market features for each candidate, computed only from data complete at its armed_at."""
    from src.features.basics import atr
    from src.strategies.rules import DailyContext
    b = bars5.sort_index().copy()
    if "atr" not in b:
        b["atr"] = atr(b, settings.detectors.atr_period)
    b["atr_pct"] = b["atr"].rolling(20 * 78, min_periods=200).rank(pct=True)
    b["ema20"] = b["close"].ewm(span=20, adjust=False).mean()
    ny = b.index.tz_convert(NY)
    b["ny_date"] = ny.normalize()
    first_open = b[(ny.hour == 9) & (ny.minute == 30)].groupby("ny_date")["open"].first()
    ends = pd.DatetimeIndex(b["end"]).as_unit("ns").asi8
    dctx = DailyContext(daily)
    c = c.copy()
    t = pd.DatetimeIndex(c["armed_at"]).as_unit("ns").asi8
    k = np.searchsorted(ends, t, side="right") - 1            # last 5m bar complete at armed_at
    k = np.clip(k, 0, len(b) - 1)
    row = b.iloc[k]
    a = row["atr"].to_numpy()
    close = row["close"].to_numpy()
    close_1h = b["close"].shift(12).iloc[k].to_numpy()
    c["strategy_code"] = c["setup_type"].astype("category").cat.codes
    c["dir"] = np.where(c["direction"] == "long", 1.0, -1.0)
    c["is_pm"] = (c["session"] == "ny_pm").astype(float)
    ny_t = pd.DatetimeIndex(c["armed_at"]).tz_convert(NY)
    c["minute"] = ny_t.hour * 60 + ny_t.minute
    c["weekday"] = ny_t.weekday
    c["risk_atr"] = c["risk_pts"].to_numpy(dtype=float) / a
    c["atr_pct"] = row["atr_pct"].to_numpy()
    c["ret_1h_atr"] = c["dir"] * (close - close_1h) / a
    day_open = first_open.reindex(row["ny_date"]).to_numpy()
    c["ret_day_atr"] = c["dir"] * (close - day_open) / a
    c["ema_dist_atr"] = c["dir"] * (close - row["ema20"].to_numpy()) / a
    pos, rng_d, gap = [], [], []
    for date, cl in zip(ny_t.normalize().tz_localize(None), close, strict=False):
        x = dctx.get(date)
        if x is None or not x["pd_range"]:
            pos.append(np.nan), rng_d.append(np.nan), gap.append(np.nan)
            continue
        pos.append((cl - x["pdl"]) / x["pd_range"])
        rng_d.append(x["pd_range"] / x["atr_d"] if x["atr_d"] else np.nan)
        gap.append(np.nan)   # previous 16:00 close is strategy-specific; use the gap feature when present
    c["pos_in_pd_range"] = pos
    c["pd_range_atrd"] = rng_d
    c["gap_atrd"] = c["gap_atr"] if "gap_atr" in c else np.nan
    if news is not None and len(news):
        ev = np.sort(pd.DatetimeIndex(news["ts"]).as_unit("ns").asi8)
        j = np.searchsorted(ev, t, side="left")
        nxt = np.where(j < len(ev), (ev[np.minimum(j, len(ev) - 1)] - t) / 6e10, np.nan)
        prv = np.where(j > 0, (t - ev[np.maximum(j - 1, 0)]) / 6e10, np.nan)
        c["news_to_min"] = np.clip(nxt, 0, 720)
        c["news_since_min"] = np.clip(prv, 0, 720)
    else:
        c["news_to_min"] = c["news_since_min"] = 720.0
    c["strat_recent_r"] = _recent_strategy_r(c)
    return c


def _recent_strategy_r(c: pd.DataFrame, n: int = 20) -> np.ndarray:
    """Mean R of the strategy's last n trades that had CLOSED before this candidate's signal time."""
    out = np.full(len(c), np.nan)
    for _, g in c.groupby("setup_type"):
        done = g[g["filled"].astype(bool) & g["exit_time"].notna()].sort_values("exit_time")
        ex = pd.DatetimeIndex(done["exit_time"]).as_unit("ns").asi8
        rs = done["r"].to_numpy(dtype=float)
        csum = np.concatenate([[0.0], np.cumsum(rs)])
        t = pd.DatetimeIndex(g["armed_at"]).as_unit("ns").asi8
        m = np.searchsorted(ex, t, side="right")          # trades closed strictly up to signal time
        lo = np.maximum(m - n, 0)
        cnt = m - lo
        val = np.where(cnt > 0, (csum[m] - csum[lo]) / np.maximum(cnt, 1), np.nan)
        out[c.index.get_indexer(g.index)] = val
    return out


# ------------------------------------------------------------------------------------ env

def reward(a: float, r: float, day_before: float) -> tuple[float, float]:
    day_after = day_before + a * r
    rew = a * r - LOSS_AVERSION * a * max(0.0, -r)
    if day_before > DAY_STOP_R >= day_after:
        rew -= DAY_STOP_PENALTY
    return rew, day_after


def _episodes(c: pd.DataFrame, episode_col: str = "trading_day"):
    c = c.sort_values("armed_at")
    for _, g in c.groupby(episode_col, sort=True):
        yield g


@dataclass
class QPolicy:
    model: lgb.LGBMRegressor | None
    features: list[str]

    def q(self, X: np.ndarray) -> np.ndarray:
        """Q-values for every action, shape (n, 3)."""
        if self.model is None:
            return np.zeros((len(X), len(ACTIONS)))
        out = np.empty((len(X), len(ACTIONS)))
        for k, a in enumerate(ACTIONS):
            out[:, k] = self.model.predict(np.column_stack([X, np.full(len(X), a)]))
        return out

    def act(self, x: np.ndarray) -> tuple[float, float]:
        q = self.q(x[None, :])[0]
        k = int(np.argmax(q))
        return (float(ACTIONS[k]), float(q[k])) if q[k] > 0 and ACTIONS[k] > 0 else (0.0, float(q[k]))


def _row_features(g: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return g[feats].to_numpy(dtype=float)


def _times(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Decision times and exit times (ns). A candidate without an exit (unfilled) 'closes' at its decision."""
    t = pd.DatetimeIndex(pd.to_datetime(g["armed_at"], utc=True)).as_unit("ns").asi8
    if "exit_time" in g:
        ex = pd.to_datetime(g["exit_time"], utc=True)
        ex = ex.fillna(pd.to_datetime(g["armed_at"], utc=True))
        e = pd.DatetimeIndex(ex).as_unit("ns").asi8
    else:
        e = t.copy()
    return t, np.maximum(e, t)


def known_r(taken_r: np.ndarray, exits: np.ndarray, upto: int, t_now: int) -> float:
    """R of the earlier candidates (0..upto-1) whose trades had CLOSED by t_now. Open trades' outcomes
    are unknown at decision time and must not leak into the state (no lookahead)."""
    if upto <= 0:
        return 0.0
    m = exits[:upto] <= t_now
    return float(taken_r[:upto][m].sum())


def train(c: pd.DataFrame, seed: int = 0, market_features: list[str] | None = None,
          episode_col: str = "trading_day") -> QPolicy:
    """Fitted Q-iteration on the candidates in `c` (must contain market features, r, episode_col)."""
    rng = np.random.default_rng(seed)
    mf = market_features or MARKET_FEATURES
    feats = mf + STATE_FEATURES
    # behaviour rollouts with random actions -> visited states; every action is then evaluated at every
    # visited state (each candidate's outcome R does not depend on our action)
    S, A, R, NEXT, TERM = [], [], [], [], []
    for g in _episodes(c, episode_col):
        Xm = _row_features(g, mf)
        rs = g["r"].to_numpy(dtype=float)
        tt, ex = _times(g)
        n = len(g)
        for _ in range(N_ROLLOUTS):
            acts = rng.choice(ACTIONS, size=n)
            taken = acts * rs
            committed = np.concatenate([[0.0], np.cumsum(taken)])       # incl. still-open trades (reward only)
            trades = np.concatenate([[0], np.cumsum(acts > 0)])
            for i in range(n):
                s = np.concatenate([Xm[i], [known_r(taken, ex, i, tt[i]), trades[i]]])
                for a in ACTIONS:
                    rew, _ = reward(a, rs[i], committed[i])
                    S.append(s)
                    A.append(a)
                    R.append(rew)
                    if i + 1 < n:
                        k_next = known_r(taken, ex, i, tt[i + 1]) + (a * rs[i] if ex[i] <= tt[i + 1] else 0.0)
                        NEXT.append(np.concatenate([Xm[i + 1], [k_next, trades[i] + (a > 0)]]))
                        TERM.append(False)
                    else:
                        NEXT.append(s)
                        TERM.append(True)
    S, A, R, NEXT, TERM = (np.array(x) for x in (S, A, R, NEXT, TERM))
    if len(S) < 200:
        return QPolicy(None, feats)
    model = None
    for it in range(N_ITER):
        if model is None:
            y = R
        else:
            nq = np.max(np.column_stack([model.predict(np.column_stack([NEXT, np.full(len(NEXT), a)]))
                                         for a in ACTIONS]), axis=1)
            nq = np.maximum(nq, 0.0)          # skipping the rest of the episode is always available (Q >= 0)
            y = R + np.where(TERM, 0.0, nq)
        model = lgb.LGBMRegressor(random_state=seed + it, **LGB_PARAMS)
        model.fit(np.column_stack([S, A]), y)
    return QPolicy(model, feats)


def run_policy(policy: QPolicy | str, c: pd.DataFrame, r_col: str = "r", market_features: list[str] | None = None,
               episode_col: str = "trading_day") -> pd.DataFrame:
    """Greedy rollout over candidates (in time order). policy 'all' = take every candidate at 1x.
    The account state at each decision only counts trades that had closed by then."""
    rows = []
    mf = market_features or MARKET_FEATURES
    for g in _episodes(c, episode_col):
        Xm = _row_features(g, mf)
        rs = g[r_col].to_numpy(dtype=float)
        tt, ex = _times(g)
        taken = np.zeros(len(g))
        trades = 0
        for i, (idx, cand) in enumerate(g.iterrows()):
            if policy == "all":
                a, q = 1.0, np.nan
            else:
                a, q = policy.act(np.concatenate([Xm[i], [known_r(taken, ex, i, tt[i]), trades]]))
            taken[i] = a * rs[i]
            trades += a > 0
            rows.append({"idx": idx, "setup_type": cand["setup_type"], "armed_at": cand["armed_at"], "action": a,
                         "q": q, "r": rs[i], "r_taken": a * rs[i], "filled": bool(cand["filled"])})
    return pd.DataFrame(rows)


def fold_summary(res: pd.DataFrame) -> dict:
    t = res[(res["action"] > 0) & res["filled"]]
    return {"taken": len(t), "size_units": float(t["action"].sum()), "total_r": float(t["r_taken"].sum()),
            "r_per_unit": float(t["r_taken"].sum() / t["action"].sum()) if len(t) else np.nan,
            "win_rate": float((t["r"] > 0).mean()) if len(t) else np.nan}


def walk_forward(c: pd.DataFrame, folds: list[tuple[str, str]], log=print, market_features: list[str] | None = None,
                 episode_col: str = "trading_day") -> pd.DataFrame:
    """Expanding-window walk-forward. Returns one row per fold with RL and take-all results."""
    rows = []
    c = c.copy()
    c["armed_at"] = pd.to_datetime(c["armed_at"], utc=True)
    for k, (start, end) in enumerate(folds):
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        tr = c[pd.to_datetime(c["exit_time"], utc=True).fillna(c["armed_at"]) < s]
        tr = tr[tr["armed_at"] < s]
        te = c[(c["armed_at"] >= s) & (c["armed_at"] < e)]
        kw = {"market_features": market_features, "episode_col": episode_col}
        pol = train(tr, seed=k, **kw)
        rl, rl2 = run_policy(pol, te, **kw), run_policy(pol, te, "r_2x", **kw)
        allr = run_policy("all", te, **kw)
        a, b, b2 = fold_summary(allr), fold_summary(rl), fold_summary(rl2)
        rows.append({"fold": f"{start[:7]}..{end[:7]}", "train_candidates": len(tr), "test_candidates": len(te),
                     "all_taken": a["taken"], "all_total_r": a["total_r"], "all_r_per_trade": a["r_per_unit"],
                     "rl_taken": b["taken"], "rl_size_units": b["size_units"], "rl_total_r": b["total_r"],
                     "rl_r_per_unit": b["r_per_unit"], "rl_win_rate": b["win_rate"],
                     "rl_total_r_2x": b2["total_r"], "rl_r_per_unit_2x": b2["r_per_unit"],
                     "rl_beats_all": b["total_r"] > a["total_r"]})
        log(f"  fold {rows[-1]['fold']}: all {a['total_r']:+.1f}R ({a['taken']}) | RL {b['total_r']:+.1f}R "
            f"({b['taken']} trades, {b['r_per_unit']:+.3f}R/unit) | RL 2x {b2['total_r']:+.1f}R")
    return pd.DataFrame(rows)


def rl_passes(wf: pd.DataFrame) -> tuple[bool, str]:
    taken = wf["rl_taken"].sum()
    units = wf["rl_size_units"].sum()
    per_unit_2x = wf["rl_total_r_2x"].sum() / units if units else np.nan
    beats = int(wf["rl_beats_all"].sum())
    total = wf["rl_total_r"].sum()
    ok = bool(taken > 0 and per_unit_2x > 0 and beats >= 5 and total > 0)
    why = (f"{taken} OOS trades, {per_unit_2x:+.3f} R per unit with 2x spread, beats take-all in {beats}/"
           f"{len(wf)} folds, total OOS R {total:+.1f}")
    return ok, why
