"""Data quality checks + HTML report (reports/data_quality.html)."""
from __future__ import annotations

import html
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import Settings
from src.data.cleaning import CleanReport
from src.timeutils import NY, in_window, is_market_open, ny_wall


def _expected_grid(m1: pd.DataFrame, settings: Settings) -> pd.DatetimeIndex:
    start = m1.index.min().floor("D")
    end = m1.index.max().ceil("D")
    grid = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    return grid[is_market_open(grid, settings.data.market_close, settings.data.trading_day_start)]


def session_coverage(m1: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Per NY weekday: minutes present vs expected inside each configured session."""
    grid = _expected_grid(m1, settings)
    rows = {}
    present = m1.index
    for name, win in settings.sessions.items():
        g = grid[in_window(grid, win.start, win.end)]
        p = present[in_window(present, win.start, win.end)]
        exp = pd.Series(1, index=ny_wall(g).normalize()).groupby(level=0).sum()
        got = pd.Series(1, index=ny_wall(p).normalize()).groupby(level=0).sum()
        rows[f"{name}_expected"] = exp
        rows[f"{name}_present"] = got
    df = pd.DataFrame(rows).fillna(0).astype(int)
    exp_cols = [c for c in df if c.endswith("_expected")]
    got_cols = [c for c in df if c.endswith("_present")]
    df["coverage"] = df[got_cols].sum(axis=1) / df[exp_cols].sum(axis=1).replace(0, np.nan)
    df.index.name = "ny_date"
    return df


def session_gaps(m1: pd.DataFrame, settings: Settings, min_minutes: int) -> pd.DataFrame:
    """Runs of missing minutes inside NY sessions longer than `min_minutes`."""
    grid = _expected_grid(m1, settings)
    mask = np.zeros(len(grid), dtype=bool)
    for win in settings.sessions.values():
        mask |= in_window(grid, win.start, win.end)
    grid = grid[mask]
    missing = grid[~grid.isin(m1.index)]
    if missing.empty:
        return pd.DataFrame(columns=["start_ny", "end_ny", "minutes"])
    # group consecutive missing minutes
    brk = np.diff(missing.asi8) != 60 * 10**9
    run_id = np.concatenate([[0], np.cumsum(brk)])
    out = []
    for _, run in pd.Series(missing, index=missing).groupby(run_id):
        if len(run) > min_minutes:
            out.append({"start_ny": run.iloc[0].tz_convert(NY), "end_ny": run.iloc[-1].tz_convert(NY),
                        "minutes": len(run)})
    return pd.DataFrame(out)


def monthly_coverage(m1: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    grid = _expected_grid(m1, settings)
    exp = pd.Series(1, index=grid).groupby(grid.strftime("%Y-%m")).sum()
    got = pd.Series(1, index=m1.index).groupby(m1.index.strftime("%Y-%m")).sum()
    df = pd.DataFrame({"expected_minutes": exp, "present_minutes": got}).fillna(0).astype(int)
    df["coverage"] = (df["present_minutes"] / df["expected_minutes"]).round(4)
    df.index.name = "month"
    return df


def dst_check(m1: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """UTC time of the first NY-AM bar around each DST switch. Should be 14:30 UTC in winter
    and 13:30 UTC in summer; if it is not, the timezone handling is wrong."""
    am = settings.sessions[next(iter(settings.sessions))]
    p = m1.index[in_window(m1.index, am.start, am.end)]
    if p.empty:
        return pd.DataFrame()
    first = pd.Series(p, index=ny_wall(p).normalize()).groupby(level=0).min()
    off = first.dt.tz_convert(NY).apply(lambda t: t.utcoffset().total_seconds() / 3600)
    switch = off.ne(off.shift())
    switch.iloc[0] = False
    keep = np.zeros(len(off), dtype=bool)
    for i in np.flatnonzero(switch.to_numpy()):
        keep[max(0, i - 2): i + 2] = True
    out = pd.DataFrame({"first_bar_utc": first.dt.strftime("%H:%M"),
                        "first_bar_ny": first.dt.tz_convert(NY).dt.strftime("%H:%M"),
                        "utc_offset_h": off})[keep]
    return out


def write_report(m1: pd.DataFrame, bars: dict[str, pd.DataFrame], clean: CleanReport,
                 settings: Settings, path: Path, synthetic: bool = False) -> dict:
    cov = session_coverage(m1, settings)
    gaps = session_gaps(m1, settings, settings.data.cleaning.gap_report_minutes)
    monthly = monthly_coverage(m1, settings)
    dst = dst_check(m1, settings)
    full_days = cov[cov[[c for c in cov if c.endswith("_present")]].sum(axis=1) > 0]
    low_cov = full_days[full_days["coverage"] < 0.95]
    no_data_days = cov[(cov[[c for c in cov if c.endswith("_present")]].sum(axis=1) == 0)]
    no_data_days = no_data_days[pd.DatetimeIndex(no_data_days.index).dayofweek < 5]

    daily = bars.get("1d")
    summary = {
        "first_bar_utc": str(m1.index.min()),
        "last_bar_utc": str(m1.index.max()),
        "rows_1m": len(m1),
        "years": round((m1.index.max() - m1.index.min()).days / 365.25, 2),
        "median_close": float(np.median(m1["close"])),
        "session_days": len(full_days),
        "session_days_below_95pct": len(low_cov),
        "weekdays_without_session_data": len(no_data_days),
        "session_gaps_listed": len(gaps),
        "mean_session_coverage": round(float(full_days["coverage"].mean()), 4) if len(full_days) else None,
        **{f"bars_{k}": len(v) for k, v in bars.items()},
    }

    figs = []
    if daily is not None and len(daily):
        f = go.Figure(go.Scatter(x=daily.index, y=daily["close"], mode="lines", name="daily close"))
        f.update_layout(title="Daily close", template="plotly_dark", height=320,
                        margin=dict(l=40, r=20, t=40, b=30))
        figs.append(f)
    f2 = go.Figure(go.Bar(x=monthly.index, y=monthly["coverage"], name="coverage"))
    f2.update_layout(title="Market-hours coverage by month", template="plotly_dark", height=280,
                     yaxis=dict(range=[0, 1.05]), margin=dict(l=40, r=20, t=40, b=30))
    figs.append(f2)
    if daily is not None and len(daily):
        rng = (daily["high"] - daily["low"])
        f3 = go.Figure(go.Histogram(x=rng, nbinsx=60))
        f3.update_layout(title="Daily range (points)", template="plotly_dark", height=280,
                         margin=dict(l=40, r=20, t=40, b=30))
        figs.append(f3)
    fig_html = "".join(f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
                       for i, f in enumerate(figs))

    def table(df: pd.DataFrame, n: int = 60) -> str:
        if df is None or df.empty:
            return "<p class='muted'>none</p>"
        more = f"<p class='muted'>showing {n} of {len(df)}</p>" if len(df) > n else ""
        return df.head(n).to_html(classes="t", border=0) + more

    banner = ("<div class='warn'>SYNTHETIC DATA: this report was generated from a random-walk "
              "test series, not real market data.</div>" if synthetic else "")
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>Data quality</title>
<style>body{{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;margin:24px;max-width:1100px}}
h1,h2{{font-weight:600}} .t{{border-collapse:collapse;font-size:13px}} .t td,.t th{{padding:3px 10px;
border-bottom:1px solid #30363d;text-align:right}} .muted{{color:#8b949e}}
.warn{{background:#5a1e02;padding:10px 14px;border-radius:6px;margin:10px 0}}</style></head><body>
<h1>Data quality report: {html.escape(settings.instrument.name)} ({html.escape(settings.instrument.dukascopy_symbol)})</h1>
<p class="muted">generated {datetime.now().isoformat(timespec="seconds")}</p>{banner}
<h2>Summary</h2>{table(pd.DataFrame([summary]).T.rename(columns={0: "value"}), 100)}
<h2>Cleaning</h2>{table(pd.DataFrame([{k: v for k, v in asdict(clean).items() if k != "repaired_examples"}]).T.rename(columns={0: "bars"}))}
<p class="muted">repaired/dropped examples (UTC): {html.escape(", ".join(clean.repaired_examples[:20])) or "none"}</p>
{fig_html}
<h2>DST check (first NY-AM bar around each switch)</h2>
<p class="muted">Expect 14:30 UTC in winter (offset -5) and 13:30 UTC in summer (offset -4); NY time must always be the session start.</p>{table(dst)}
<h2>Monthly coverage of market hours</h2>{table(monthly, 200)}
<h2>Session days below 95% coverage</h2>{table(low_cov)}
<h2>Weekdays with no session data (holidays or missing files)</h2>{table(no_data_days)}
<h2>Gaps inside NY sessions longer than {settings.data.cleaning.gap_report_minutes} min</h2>{table(gaps)}
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return summary
