"""Backtest statistics. Everything is in R (multiples of the planned risk) unless named *_usd/*_pct."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _streak(losses: np.ndarray) -> int:
    best = cur = 0
    for x in losses:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def max_drawdown(cum: np.ndarray) -> float:
    """Largest peak-to-trough fall of a cumulative series (starting from 0)."""
    if len(cum) == 0:
        return 0.0
    c = np.concatenate([[0.0], cum])
    return float(np.max(np.maximum.accumulate(c) - c))


def summary(trades: pd.DataFrame, months: float | None = None) -> dict:
    """Core stats over filled trades. `months` = length of the test period (for trades/month)."""
    n_sig = len(trades)
    f = trades[trades["filled"].astype(bool)] if n_sig else trades
    r = f["r"].to_numpy(dtype=float) if len(f) else np.array([])
    wins, losses = r[r > 0], r[r < 0]
    out = {
        "signals": n_sig,
        "filled": len(r),
        "fill_rate": len(r) / n_sig if n_sig else np.nan,
        "win_rate": len(wins) / len(r) if len(r) else np.nan,
        "avg_win_r": wins.mean() if len(wins) else np.nan,
        "avg_loss_r": losses.mean() if len(losses) else np.nan,
        "expectancy_r": r.mean() if len(r) else np.nan,
        "total_r": r.sum(),
        "profit_factor": wins.sum() / -losses.sum() if len(losses) and losses.sum() < 0 else np.nan,
        "max_drawdown_r": max_drawdown(np.cumsum(r)),
        "longest_losing_streak": _streak(r < 0),
        "hit_1r": f["hit_1r"].mean() if len(f) else np.nan,
        "hit_2r": f["hit_2r"].mean() if len(f) else np.nan,
        "hit_3r": f["hit_3r"].mean() if len(f) else np.nan,
    }
    if months:
        out["trades_per_month"] = len(r) / months
    if len(r) >= 2:
        # t-statistic of the mean R: rough "is it distinguishable from zero" check (>2 ~ yes)
        out["t_stat"] = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if r.std(ddof=1) > 0 else np.nan
    return out


def breakdown(trades: pd.DataFrame, by) -> pd.DataFrame:
    f = trades[trades["filled"].astype(bool)]
    if f.empty:
        return pd.DataFrame()
    g = f.groupby(by)["r"]
    return pd.DataFrame({
        "trades": g.size(),
        "win_rate": g.apply(lambda x: (x > 0).mean()),
        "expectancy_r": g.mean(),
        "total_r": g.sum(),
    })


def outcome_counts(trades: pd.DataFrame) -> pd.Series:
    return trades["outcome"].value_counts()


def daily_limit_breach_rate(trades: pd.DataFrame, risk_pct: float, daily_limit_pct: float) -> float:
    """Share of trading days (with >=1 filled trade) whose losses alone would exceed the firm's
    daily limit if every trade risked `risk_pct` (no internal buffers). Uses the intraday worst
    point: cumulative R before each trade minus that trade's adverse excursion."""
    f = trades[trades["filled"].astype(bool)].sort_values("fill_time")
    if f.empty:
        return 0.0
    breached = []
    for _, d in f.groupby("trading_day"):
        cum, worst = 0.0, 0.0
        for _, t in d.iterrows():
            worst = min(worst, cum - t["mae_r"], cum + t["r"])
            cum += t["r"]
        breached.append(-worst * risk_pct > daily_limit_pct)
    return float(np.mean(breached))


def equity_drawdown_pct(balance: pd.Series, initial: float) -> float:
    """Max peak-to-trough drawdown of a balance series, in % of the initial balance."""
    if balance.empty:
        return 0.0
    b = np.concatenate([[initial], balance.to_numpy(dtype=float)])
    return float(np.max(np.maximum.accumulate(b) - b) / initial * 100)
