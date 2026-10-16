"""Monte Carlo estimate of passing the Equity Edge 2-Step challenge.

Method: block-bootstrap of whole session days from the guarded backtest (days with no trade
included, so time-to-pass is realistic). Intraday rules (daily buffer, max signals, consecutive
losses) were already applied inside each historical day, so sampling whole days keeps them.
Each simulated phase starts at the initial balance and ends at the first of:
  pass   : balance >= initial x (1 + target) and at least `min_trading_days` days traded
  breach : firm daily loss limit or static max drawdown hit (intraday worst equity)
  halt   : our internal max-drawdown buffer reached (the system stops; not a firm breach)
  timeout: no result within `max_days_per_phase` session days (the firm has no time limit)
Limitations: days are sampled independently (losing streaks that cluster in time are
under-represented), and the future need not look like the past.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Settings
from src.risk.guard import Limits


def day_samples(trades: pd.DataFrame, n_session_days: int) -> pd.DataFrame:
    """One row per session day: pnl_usd, worst intraday equity vs day start (<= 0), traded flag.
    Days without a filled trade are added as zeros up to `n_session_days`."""
    f = trades[trades["filled"].astype(bool)].sort_values("fill_time") if len(trades) else trades
    rows = []
    for _, d in (f.groupby("trading_day") if len(f) else []):
        cum, worst, low = 0.0, 0.0, 0.0
        for _, t in d.iterrows():
            mae_usd = t["mae_r"] * t["risk_usd"]
            worst = min(worst, cum - mae_usd, cum + t["pnl_usd"])
            cum += t["pnl_usd"]
            low = min(low, cum)
        rows.append({"pnl": cum, "worst": worst, "realized_low": low, "traded": True})
    zeros = max(0, n_session_days - len(rows))
    rows += [{"pnl": 0.0, "worst": 0.0, "realized_low": 0.0, "traded": False}] * zeros
    return pd.DataFrame(rows)


@dataclass
class PhaseResult:
    target_pct: float
    p_pass: float
    p_breach: float
    p_halt: float
    p_timeout: float
    median_days_to_pass: float


def simulate_phase(days: pd.DataFrame, lim: Limits, target_pct: float, min_days: int,
                   horizon: int, n: int, rng: np.random.Generator) -> PhaseResult:
    if "realized_low" not in days:
        days = days.assign(realized_low=np.minimum(days["pnl"], 0.0))
    cols = ("pnl", "worst", "realized_low", "traded")
    pnl, worst, low, traded = (days[k].to_numpy(dtype=float) for k in cols)
    idx = rng.integers(0, len(days), size=(n, horizon))
    P, W, L, T = pnl[idx], worst[idx], low[idx], traded[idx]
    after = lim.initial_balance + np.cumsum(P, axis=1)
    before = after - P
    floor = lim.initial_balance - lim.max_dd_usd
    breach = (W < -lim.daily_limit_usd) | (before + W < floor)
    # the engine halts right after the losing close, so use the day's lowest realized balance
    halt = (lim.initial_balance - (before + L)) >= lim.dd_buffer_usd
    passed = (after >= lim.initial_balance * (1 + target_pct / 100)) & (np.cumsum(T, axis=1) >= min_days)

    def first(m):
        return np.where(m.any(axis=1), m.argmax(axis=1), horizon + 1)

    fb, fh, fp = first(breach), first(halt), first(passed)
    # breach and halt happen intraday, so both win a tie with a same-day pass
    is_breach = (fb <= fh) & (fb <= fp) & (fb <= horizon)
    is_halt = ~is_breach & (fh <= fp) & (fh <= horizon)
    is_pass = ~is_breach & ~is_halt & (fp <= horizon)
    return PhaseResult(target_pct, is_pass.mean(), is_breach.mean(), is_halt.mean(),
                       1 - is_pass.mean() - is_breach.mean() - is_halt.mean(),
                       float(np.median(fp[is_pass] + 1)) if is_pass.any() else float("nan"))


def challenge_pass_probability(trades: pd.DataFrame, n_session_days: int, settings: Settings) -> dict:
    c = settings.challenge
    lim = Limits.from_settings(settings)
    days = day_samples(trades, n_session_days)
    rng = np.random.default_rng(c.seed)
    phases = [simulate_phase(days, lim, t, c.min_trading_days, c.max_days_per_phase,
                             c.n_simulations, rng) for t in c.phase_targets_pct]
    return {
        "phases": phases,
        "p_pass_all": float(np.prod([p.p_pass for p in phases])),
        "session_days_sampled": len(days),
        "days_with_trades": int(days["traded"].sum()),
    }
