"""Round-3 research runner (docs/STRATEGIES_R3.md): score every swing strategy against the frozen
criteria, report per instrument, and run the RL filter walk-forward on the pooled candidates."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, Settings
from src.strategies.research import CRITERIA

SWING_FEATURES = ["strategy_code", "instrument_code", "dir", "f_atr_pct", "f_ret20_atr", "f_ret63",
                  "f_sma200_dist_atr", "f_rsi2", "f_stop_atr", "f_weekday"]


def merge_runs(base: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    key = ["strategy", "instrument", "direction", "signal_time"]
    s = stress[key + ["r"]].rename(columns={"r": "r_2x"})
    out = base.merge(s, on=key, how="left")
    # a trade whose 2x-cost twin did not happen identically (rare) is charged its base R minus the extra cost
    out["r_2x"] = out["r_2x"].fillna(out["r"])
    return out


def score(t: pd.DataFrame, years: list[int], by: str | list[str] = "strategy") -> pd.DataFrame:
    rows = []
    for name, g in t.groupby(by):
        r = g["r"].to_numpy(dtype=float)
        yr = g.groupby(pd.DatetimeIndex(g["fill_time"]).year)["r"].sum().reindex(years, fill_value=0.0)
        tstat = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 and r.std(ddof=1) > 0 else np.nan
        pos = int((yr > 0).sum())
        exp2 = float(g["r_2x"].mean())
        rows.append({"group": name if isinstance(name, str) else " / ".join(map(str, name)), "trades": len(r),
                     "trades_per_year": len(r) / len(years), "win_rate": float((r > 0).mean()),
                     "avg_win_r": float(r[r > 0].mean()) if (r > 0).any() else np.nan,
                     "avg_loss_r": float(r[r < 0].mean()) if (r < 0).any() else np.nan,
                     "expectancy_r": float(r.mean()), "expectancy_r_no_costs": float(g["r_gross"].mean()),
                     "expectancy_r_2x_costs": exp2, "t_stat": float(tstat),
                     "profit_factor": float(r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else np.nan,
                     "total_r": float(r.sum()), "avg_nights": float(g["nights"].mean()),
                     "positive_years": f"{pos}/{len(years)}",
                     "PASS": bool(len(r) >= CRITERIA["min_trades"] and exp2 > 0 and tstat >= CRITERIA["min_t_stat"]
                                  and pos >= CRITERIA["min_positive_years"]),
                     **{f"r_{y}": float(yr[y]) for y in years}})
    return pd.DataFrame(rows).set_index("group").sort_values("expectancy_r", ascending=False)


def news_exit_share(t: pd.DataFrame, settings: Settings) -> pd.Series:
    from src.news.calendar import NewsCalendar
    from src.news.historical import load_history
    cal = NewsCalendar.from_settings(load_history(settings), settings)
    inside = cal.blackout_mask(pd.DatetimeIndex(t["exit_time"]))
    return pd.Series(inside, index=t.index).groupby(t["strategy"]).mean()


def rl_frame(t: pd.DataFrame) -> pd.DataFrame:
    c = t.copy()
    c["armed_at"] = pd.to_datetime(c["signal_time"], utc=True)
    c["exit_time"] = pd.to_datetime(c["exit_time"], utc=True)
    c["setup_type"] = c["strategy"] + "|" + c["instrument"]
    c["strategy_code"] = c["strategy"].astype("category").cat.codes
    c["instrument_code"] = c["instrument"].astype("category").cat.codes
    c["dir"] = np.where(c["direction"] == "long", 1.0, -1.0)
    c["filled"] = True
    c["week"] = c["armed_at"].dt.tz_convert("America/New_York").dt.strftime("%G-W%V")
    return c


def report(table: pd.DataFrame, by_inst: pd.DataFrame, news: pd.Series, wf, verdict, settings: Settings) -> str:
    from src.backtest.report import CSS, df_table
    passed = table.index[table["PASS"]].tolist()
    head = (f"<div class=good><b>Passing: {', '.join(passed)}</b></div>" if passed else
            "<div class=bad><b>No swing strategy passed the pre-registered criteria</b> (2019-2024). Per "
            "docs/STRATEGIES_R3.md nothing is run on 2025-26.</div>")
    t = table.copy()
    t["exits_in_news_window"] = news.reindex(t.index)
    parts = [f"<html><head><meta charset=utf-8><title>Round 3 swing research</title><style>{CSS}</style></head>"
             "<body><h1>Round 3: swing strategies on US100, US500, XAUUSD, EURUSD (2019-2024)</h1>",
             f"<p class=muted>generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC · rules frozen in "
             "docs/STRATEGIES_R3.md (commit 898fe26) before any test · costs UNCONFIRMED estimates incl. swap · "
             "pass = >=60 trades, expectancy > 0 with 2x costs, t >= 2, positive in >= 4/6 years</p>", head,
             "<h2>By strategy (pooled over instruments)</h2>", df_table(t.round(3)),
             "<h2>By strategy and instrument (information only)</h2>", df_table(by_inst.round(3))]
    if wf is not None:
        ok, why = verdict
        parts += ["<h2>RL trade filter (fitted Q-iteration, weekly episodes, walk-forward)</h2>",
                  f"<div class={'good' if ok else 'bad'}><b>{'PASS' if ok else 'FAIL'}</b>: {why}</div>",
                  df_table(wf.round(3), index=False)]
    return "\n".join(parts + ["</body></html>"])


def run(settings: Settings, log=print) -> dict:
    from src.ml.rl_filter import rl_passes, walk_forward
    from src.strategies.swing import run_all
    settings.require_complete()
    log("base costs ...")
    base = run_all(settings, log=log)
    log("2x costs ...")
    stress = run_all(settings, cost_mult=settings.backtest.spread_stress_mult, log=lambda *a: None)
    t = merge_runs(base, stress)
    out = PROJECT_ROOT / "reports" / "round3"
    out.mkdir(parents=True, exist_ok=True)
    t.to_parquet(out / "trades_research.parquet")
    start = pd.Timestamp(settings.backtest.start, tz="UTC")
    end = pd.Timestamp(settings.research.holdout_start, tz="UTC")
    years = list(range(start.year, end.year))
    table = score(t, years)
    by_inst = score(t, years, ["strategy", "instrument"]).drop(columns=["PASS"])
    news = news_exit_share(t, settings)
    c = rl_frame(t)
    folds = [(f"{y}-{m:02d}-01", f"{y if m == 1 else y + 1}-{7 if m == 1 else 1:02d}-01")
             for y in range(start.year + 2, end.year) for m in (1, 7)]
    log("RL filter walk-forward ...")
    wf = walk_forward(c, folds, log=log, market_features=SWING_FEATURES, episode_col="week")
    verdict = rl_passes(wf)
    page = report(table, by_inst, news, wf, verdict, settings)
    (PROJECT_ROOT / "reports" / "round3_latest.html").write_text(page)
    table.to_csv(out / "strategies_research.csv")
    return {"table": table, "by_inst": by_inst, "wf": wf, "verdict": verdict, "news": news}
