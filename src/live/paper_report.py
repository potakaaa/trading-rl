"""Phase 7: paper-trading monitor. Compares live (paper) results with what the backtest expects.

Expected range: if live trades come from the same distribution as the backtest trades, the mean R
of n live trades should fall inside  mean_bt ± 2 · sd_bt / sqrt(n)  about 95% of the time. Outside
that band = DRIFT (live behaves differently from the backtest: data, execution, or market change).
Signal frequency is compared the same way (Poisson band on weekly signal counts).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, Settings
from src.live.store import Store


@dataclass
class Expectation:
    mean_r: float
    sd_r: float
    fills_per_week: float
    signals_per_week: float


def backtest_expectation(settings: Settings, raw_trades: pd.DataFrame | None = None) -> Expectation:
    if raw_trades is None:
        raw_trades = pd.read_csv(PROJECT_ROOT / "reports" / "backtest" / "raw_trades.csv", parse_dates=["armed_at"])
    t = raw_trades
    f = t[t["filled"].astype(bool)]
    span_weeks = max((pd.to_datetime(t["armed_at"], utc=True).max()
                      - pd.to_datetime(t["armed_at"], utc=True).min()).days / 7, 1)
    return Expectation(float(f["r"].mean()), float(f["r"].std(ddof=1)), len(f) / span_weeks, len(t) / span_weeks)


def paper_trades(store: Store) -> pd.DataFrame:
    rows = store._all("SELECT id, created_at, trading_day, setup_type, direction, status, paper_outcome, "
                      "paper_r, paper_pnl_usd, action, pnl_usd FROM signals ORDER BY created_at")
    df = pd.DataFrame(rows)
    if len(df):
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df["week"] = df["created_at"].dt.tz_convert("America/New_York").dt.strftime("%G-W%V")
    return df


def compare(store: Store, settings: Settings, exp: Expectation | None = None) -> dict:
    exp = exp or backtest_expectation(settings)
    df = paper_trades(store)
    if df.empty:
        return {"weeks": [], "n_signals": 0, "n_closed": 0, "verdict": "no signals yet", "drift": False,
                "expectation": exp.__dict__}
    closed = df[df["paper_r"].notna()]
    n = len(closed)
    mean = float(closed["paper_r"].mean()) if n else float("nan")
    band = 2 * exp.sd_r / np.sqrt(n) if n else float("nan")
    lo, hi = exp.mean_r - band, exp.mean_r + band
    weeks = []
    for wk, g in df.groupby("week"):
        c = g[g["paper_r"].notna()]
        lam = exp.signals_per_week
        weeks.append({"week": wk, "signals": len(g), "filled": len(c), "total_r": float(c["paper_r"].sum()),
                      "mean_r": float(c["paper_r"].mean()) if len(c) else None,
                      "signals_expected": lam,
                      "signal_count_unusual": bool(abs(len(g) - lam) > 2 * np.sqrt(max(lam, 1e-9)) + 1)})
    drift = bool(n >= 5 and not (lo <= mean <= hi))
    if n < 5:
        verdict = f"{n} closed paper trades: too few to judge (need >= 5; ideally 30+)"
    elif drift:
        verdict = (f"DRIFT: live mean {mean:+.2f}R is outside the expected range [{lo:+.2f}, {hi:+.2f}] "
                   f"for {n} trades")
    else:
        verdict = f"in line with the backtest: live mean {mean:+.2f}R within [{lo:+.2f}, {hi:+.2f}] for {n} trades"
    by_setup = {}
    for name, g in closed.groupby("setup_type"):
        r = g["paper_r"].to_numpy(dtype=float)
        by_setup[name] = {"closed": len(r), "mean_r": float(r.mean()),
                          "usd_per_trade_0.5pct": float(r.mean() * 12.5), "usd_per_trade_1pct": float(r.mean() * 25),
                          "goal_5usd_at_1pct": "on track" if r.mean() >= 0.2 else "below",
                          "goal_5usd_at_0.5pct": "on track" if r.mean() >= 0.4 else "below",
                          "note": "too few trades to judge (need 30+)" if len(r) < 30 else ""}
    return {"weeks": weeks, "n_signals": len(df), "n_closed": n, "mean_r": mean, "expected_range": [lo, hi],
            "by_setup": by_setup,
            "verdict": verdict, "drift": drift, "expectation": exp.__dict__,
            "taken": int((df["action"] == "taken").sum()), "skipped": int((df["action"] == "skipped").sum())}


def write_html(result: dict, path) -> None:
    from src.backtest.report import CSS, df_table, kv_table
    weeks = pd.DataFrame(result["weeks"])
    cls = "bad" if result["drift"] else "good" if result.get("n_closed", 0) >= 5 else "warn"
    page = (f"<html><head><meta charset=utf-8><title>Paper trading</title><style>{CSS}</style></head><body>"
            "<h1>Paper trading vs backtest</h1>"
            f"<div class={cls}><b>{result['verdict']}</b></div>"
            + kv_table({k: v for k, v in result.items() if k not in ("weeks", "expectation", "verdict", "by_setup")})
            + "<h3>By setup ($5/trade goal: R >= 0.2 at 1% risk, R >= 0.4 at 0.5% risk)</h3>"
            + df_table(pd.DataFrame(result.get("by_setup", {})).T)
            + "<h3>Backtest expectation</h3>" + kv_table(result["expectation"])
            + "<h3>By week</h3>" + df_table(weeks, index=False) + "</body></html>")
    path.write_text(page)
