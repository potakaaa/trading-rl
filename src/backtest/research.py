"""Rule-variant research with a locked holdout (Phase 3b).

Rules that keep this honest:
  * The variant list lives in settings.yaml and was written before any variant was run.
  * Only data BEFORE `research.holdout_start` is loaded. A guard asserts nothing later slipped in.
  * Every variant is reported, including the bad ones (no cherry-picking), with its t-stat.
  * The holdout is evaluated once, later, on the single variant Rald chooses (`evaluate_holdout`).
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import metrics
from src.backtest.engine import M1, costs_from_settings, run_raw
from src.backtest.report import CSS, df_table
from src.config import PROJECT_ROOT, Settings


def apply_overrides(settings: Settings, overrides: dict) -> Settings:
    """Return a validated copy of settings with dotted-path overrides applied."""
    data = settings.model_dump()
    for path, value in overrides.items():
        node = data
        *parents, leaf = path.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"unknown setting in research variant: {path}")
        node[leaf] = value
    return type(settings).model_validate(data)     # re-validates (e.g. min_rr >= 1)


def _window(settings: Settings, holdout: bool) -> tuple[str, str | None]:
    h = settings.research.holdout_start
    return (h, settings.backtest.end) if holdout else (settings.backtest.start, h)


def evaluate_variant(settings: Settings, name: str, holdout: bool = False) -> tuple[dict, pd.DataFrame]:
    from src.backtest.runner import load_detections
    from src.data.store import load_bars
    s = apply_overrides(settings, settings.research.variants[name])
    start, end = _window(s, holdout)
    det = load_detections(s, start, end)
    cut = pd.Timestamp(s.research.holdout_start, tz="UTC")
    if not holdout and len(det.bars) and det.bars.index.max() >= cut:
        raise AssertionError("holdout data leaked into research")
    m1 = M1.from_frame(load_bars(s, "1m", start, end))
    trades = run_raw(det.setups, m1, s, costs_from_settings(s))
    stress = run_raw(det.setups, m1, s, costs_from_settings(s, s.backtest.spread_stress_mult))
    years = (det.bars.index[-1] - det.bars.index[0]).days / 365.25
    sm = metrics.summary(trades, years * 12)
    f = trades[trades["filled"].astype(bool)]
    by_year = f.groupby(pd.DatetimeIndex(f["armed_at"]).year)["r"].sum() if len(f) else pd.Series(dtype=float)
    row = {
        "variant": name,
        "overrides": ", ".join(f"{k}={v}" for k, v in settings.research.variants[name].items()) or "-",
        "valid_setups": int((det.setups["status"] == "valid").sum()),
        "filled": sm["filled"], "fills_per_year": sm["filled"] / years if years else np.nan,
        "win_rate": sm["win_rate"], "expectancy_r": sm["expectancy_r"],
        "expectancy_r_2x_spread": metrics.summary(stress)["expectancy_r"],
        "t_stat": sm.get("t_stat", np.nan), "profit_factor": sm["profit_factor"],
        "total_r": sm["total_r"], "max_drawdown_r": sm["max_drawdown_r"],
        "positive_years": f"{int((by_year > 0).sum())}/{len(by_year)}",
    }
    return row, trades


def run_research(settings: Settings, out_dir: Path | None = None, log=print) -> pd.DataFrame:
    settings.require_complete()
    rows = []
    for name in settings.research.variants:
        log(f"  {name} ...")
        row, _ = evaluate_variant(settings, name)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("variant")
    out_dir = out_dir or PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "research_variants.csv")
    n = len(table)
    page = (f"<html><head><meta charset=utf-8><title>Rule research</title><style>{CSS}</style></head><body>"
            f"<h1>Rule-variant research (data {settings.backtest.start} to {settings.research.holdout_start}, "
            f"holdout locked)</h1><p class=muted>generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. "
            f"Raw setups, base costs (2x-spread column = stress). {n} variants were tested, so roughly "
            f"one in {max(n, 1)} would look 'good' by luck alone at a t-stat near 1.7. "
            "Treat a t-stat below ~2.5 as noise.</p>"
            + df_table(table) + "</body></html>")
    (out_dir / "research_latest.html").write_text(page)
    log(f"report: {out_dir / 'research_latest.html'}")
    return table


def evaluate_holdout(settings: Settings, name: str) -> dict:
    """ONE-TIME evaluation of the chosen variant on the untouched holdout period."""
    row, _ = evaluate_variant(settings, name, holdout=True)
    return row

