"""Round-2 research (docs/STRATEGIES.md): run every pre-registered strategy on the research period,
score it against the frozen pass criteria, and save every candidate with its raw outcome (the RL
filter's training data). The final period (>= research.holdout_start) is never loaded here unless
`final=True`, which is meant to be called exactly once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import metrics
from src.backtest.engine import M1, costs_from_settings, run_raw
from src.config import PROJECT_ROOT, Settings
from src.strategies.rules import STRATEGIES, generate

CRITERIA = {"min_trades": 60, "min_t_stat": 2.0, "min_positive_years": 4, "years_total": 6}


def period(settings: Settings, final: bool) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    h = pd.Timestamp(settings.research.holdout_start, tz="UTC")
    return (h, None) if final else (pd.Timestamp(settings.backtest.start, tz="UTC"), h)


def candidates_with_outcomes(settings: Settings, final: bool = False, include_ict: bool = True,
                             log=print) -> pd.DataFrame:
    """Every candidate (valid ones simulated) for the period, base costs and 2x-spread R."""
    from src.data.store import load_bars
    start, end = period(settings, final)
    warm = start - pd.Timedelta(days=60)
    b5 = load_bars(settings, "5m", warm, end)
    d1 = load_bars(settings, "1d", warm - pd.Timedelta(days=60), end)
    m1 = M1.from_frame(load_bars(settings, "1m", start, end))
    log("  generating round-2 candidates ...")
    c = generate(b5, d1, settings)
    c = c[c["armed_at"] >= start]
    if not final:
        assert c["armed_at"].max() < end, "final-period data leaked into research"
    parts = [c]
    if include_ict:
        log("  ICT setups ...")
        from src.backtest.runner import load_detections
        ict = load_detections(settings, start, end).setups
        ict = ict.assign(setup_type="ict_" + ict["setup_type"], order_type="limit")
        parts.append(ict[ict["armed_at"] >= start])
    allc = pd.concat(parts, ignore_index=True)
    valid = allc[allc["status"] == "valid"].reset_index(drop=True)
    log(f"  simulating {len(valid)} valid candidates (base + 2x spread) ...")
    base = run_raw(valid, m1, settings, costs_from_settings(settings))
    stress = run_raw(valid, m1, settings, costs_from_settings(settings, settings.backtest.spread_stress_mult))
    # run_raw sorts by armed_at; align on setup_id (= index of `valid`)
    base = base.set_index("setup_id")
    stress = stress.set_index("setup_id")
    out = valid.copy()
    for col in ("outcome", "filled", "r", "mfe_r", "mae_r", "fill_time", "exit_time", "fill_price", "exit_price",
                "trading_day"):
        out[col] = base[col].reindex(out.index).to_numpy()
    out["r_2x"] = stress["r"].reindex(out.index).to_numpy()
    return out


def score(c: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    rows = []
    for name, g in c.groupby("setup_type"):
        f = g[g["filled"].astype(bool)]
        r = f["r"].to_numpy(dtype=float)
        by_year = f.groupby(pd.DatetimeIndex(f["armed_at"]).year)["r"].sum().reindex(years, fill_value=0.0)
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 and r.std(ddof=1) > 0 else np.nan
        exp2 = f["r_2x"].mean() if len(f) else np.nan
        pos_years = int((by_year > 0).sum())
        passed = (len(r) >= CRITERIA["min_trades"] and exp2 > 0 and t >= CRITERIA["min_t_stat"]
                  and pos_years >= CRITERIA["min_positive_years"])
        rows.append({"strategy": name, "filled": len(r), "trades_per_year": len(r) / max(len(years), 1),
                     "win_rate": float((r > 0).mean()) if len(r) else np.nan,
                     "expectancy_r": float(r.mean()) if len(r) else np.nan, "expectancy_r_2x_spread": float(exp2),
                     "t_stat": float(t),
                     "profit_factor": float(r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else np.nan,
                     "total_r": float(r.sum()), "max_drawdown_r": metrics.max_drawdown(np.cumsum(r)),
                     "positive_years": f"{pos_years}/{len(years)}",
                     "PASS": bool(passed) if not name.startswith("ict_") else None,
                     **{f"r_{y}": float(by_year[y]) for y in years}})
    return pd.DataFrame(rows).set_index("strategy").sort_values("expectancy_r", ascending=False)


def run(settings: Settings, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings.require_complete()
    c = candidates_with_outcomes(settings, log=log)
    out = PROJECT_ROOT / "reports" / "round2"
    out.mkdir(parents=True, exist_ok=True)
    c.to_parquet(out / "candidates_research.parquet")
    start, end = period(settings, False)
    years = list(range(start.year, end.year))
    table = score(c, years)
    table.to_csv(out / "strategies_research.csv")
    assert set(STRATEGIES) <= set(table.index) | set(), "every pre-registered strategy must be reported"
    return c, table


def write_report(table: pd.DataFrame, gross: pd.Series | None, wf: pd.DataFrame | None, rl_verdict: tuple | None,
                 settings: Settings) -> str:
    from datetime import UTC, datetime

    from src.backtest.report import CSS, df_table
    passed = table.index[table["PASS"].fillna(False).astype(bool)].tolist()
    t = table.copy()
    if gross is not None:
        t.insert(4, "expectancy_r_no_costs", gross.reindex(t.index))
    head = ("<div class=good><b>Passing strategies: " + ", ".join(passed) + "</b></div>") if passed else \
        ("<div class=bad><b>No strategy passed the pre-registered criteria</b> (2019-2024, base and 2x-spread "
         "costs). Per docs/STRATEGIES.md the final 2025-26 test is not run for failed strategies.</div>")
    parts = [f"<html><head><meta charset=utf-8><title>Strategy research round 2</title><style>{CSS}</style>"
             "</head><body><h1>Strategy research round 2 (pre-registered, 2019-2024)</h1>",
             f"<p class=muted>generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC · rules frozen in "
             "docs/STRATEGIES.md (git commit de60517, before any test) · pass = >=60 trades, expectancy > 0 "
             "with 2x spread, t-stat >= 2, positive in >= 4/6 years</p>", head,
             "<h2>All strategies</h2>", df_table(t.round(3)),
             "<p class=muted>win_rate is shown for information; it is not a pass criterion. "
             "expectancy_r_no_costs is a diagnostic only (zero spread/slippage).</p>"]
    if wf is not None:
        ok, why = rl_verdict
        parts += ["<h2>RL trade filter (fitted Q-iteration, walk-forward)</h2>",
                  f"<div class={'good' if ok else 'bad'}><b>{'PASS' if ok else 'FAIL'}</b>: {why}</div>",
                  df_table(wf.round(3), index=False)]
    parts.append("</body></html>")
    return "\n".join(parts)
