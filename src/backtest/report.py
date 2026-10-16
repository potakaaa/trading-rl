"""Self-contained HTML backtest report (plotly charts + tables)."""
from __future__ import annotations

import html
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0f1115;color:#e6e6e6;margin:24px;max-width:1200px}
h1,h2,h3{color:#fff} table{border-collapse:collapse;margin:8px 0 20px;font-size:13px}
td,th{border:1px solid #333;padding:4px 8px;text-align:right} th{background:#1b1f27}
td:first-child,th:first-child{text-align:left}
.warn{background:#3a2a10;border:1px solid #a87b2c;padding:10px 14px;border-radius:6px;margin:12px 0}
.bad{background:#3a1414;border:1px solid #b44;padding:10px 14px;border-radius:6px;margin:12px 0}
.good{background:#123018;border:1px solid #3a8a4a;padding:10px 14px;border-radius:6px;margin:12px 0}
.muted{color:#999;font-size:13px}
"""


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "-"
        return f"{v:,.3f}" if abs(v) < 100 else f"{v:,.1f}"
    return html.escape(str(v))


def kv_table(d: dict) -> str:
    return "<table>" + "".join(f"<tr><td>{html.escape(str(k))}</td><td>{_fmt(v)}</td></tr>"
                               for k, v in d.items()) + "</table>"


def df_table(df: pd.DataFrame, index=True) -> str:
    if df is None or df.empty:
        return "<p class=muted>none</p>"
    return df.to_html(index=index, float_format=lambda x: f"{x:,.3f}", border=0, escape=True)


def fig_html(fig: go.Figure, first: list[bool]) -> str:
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=40, r=20, t=40, b=30))
    inc = "cdn" if first[0] else False
    first[0] = False
    return fig.to_html(full_html=False, include_plotlyjs=inc)


def equity_r_figure(series: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for name, t in series.items():
        f = t[t["filled"].astype(bool)].sort_values("exit_time")
        fig.add_trace(go.Scatter(x=f["exit_time"], y=f["r"].cumsum(), mode="lines", name=name))
    fig.update_layout(title="Cumulative R (raw setups, every valid setup taken)", yaxis_title="R")
    return fig


def balance_figure(trades: pd.DataFrame, initial: float, floor: float, buffer_floor: float) -> go.Figure:
    f = trades[trades["filled"].astype(bool)].sort_values("exit_time")
    fig = go.Figure()
    for acc, g in f.groupby("account"):
        fig.add_trace(go.Scatter(x=g["exit_time"], y=g["balance_after"], mode="lines", name=f"account {acc}"))
    fig.add_hline(y=initial, line_dash="dot", line_color="#888")
    fig.add_hline(y=floor, line_dash="dash", line_color="#d55", annotation_text="firm max-DD floor")
    fig.add_hline(y=buffer_floor, line_dash="dot", line_color="#da5", annotation_text="internal halt")
    fig.update_layout(title="Guarded account balance ($)", yaxis_title="$")
    return fig


def build(ctx: dict) -> str:
    first = [True]
    s = ctx["settings"]
    parts = [f"<html><head><meta charset=utf-8><title>US100 backtest</title><style>{CSS}</style></head><body>",
             "<h1>US100 ICT setups: backtest report</h1>",
             f"<p class=muted>generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC · data {ctx['period']} · "
             f"setup timeframe {s.detectors.setup_timeframe} · 1m execution</p>"]
    if s.broker.unconfirmed:
        parts.append("<div class=warn><b>Unconfirmed inputs:</b> " +
                     ", ".join(f"{k} = {getattr(s.broker, k)}" for k in s.broker.unconfirmed) +
                     f" · news window {s.news.blackout_minutes_before}/{s.news.blackout_minutes_after} min "
                     "(stricter of the published values). R-based results do not depend on lot value.</div>")
    g = ctx["gate"]
    verdict = "POSITIVE" if g["pass"] else "NO POSITIVE"
    parts.append(f"<div class={'good' if g['pass'] else 'bad'}><b>Gate (raw setups, base costs): "
                 f"{verdict} expectancy after costs.</b> {html.escape(g['text'])}</div>")

    parts.append("<h2>1. Raw setups (no prop/news rules)</h2>")
    parts.append(fig_html(equity_r_figure({"base costs": ctx["raw"], "2x spread": ctx["raw_stress"]}), first))
    comp = pd.DataFrame({"base costs": ctx["raw_summary"], "2x spread stress": ctx["raw_stress_summary"]})
    parts.append(df_table(comp))
    parts.append("<h3>Outcomes</h3>" + df_table(ctx["raw"]["outcome"].value_counts().to_frame("count")))
    for title, key in [("By setup", "setup_type"), ("By session", "session"), ("By direction", "direction"),
                       ("By weekday", "weekday"), ("By year", "year")]:
        parts.append(f"<h3>{title}</h3>" + df_table(ctx["raw_breakdowns"][key]))
    parts.append(f"<p>Share of trading days whose losses alone would breach the firm's "
                 f"{s.account.daily_loss_limit_pct}% daily limit at {s.account.risk_per_trade_pct}% risk "
                 f"per trade (no buffers): <b>{ctx['raw_daily_breach']:.1%}</b></p>")

    gr = ctx["guarded"]
    parts.append("<h2>2. Guarded account (all prop, news and safety rules, exactly as live)</h2>")
    parts.append(fig_html(balance_figure(gr.trades, ctx["limits"].initial_balance,
                                         ctx["limits"].initial_balance - ctx["limits"].max_dd_usd,
                                         ctx["limits"].initial_balance - ctx["limits"].dd_buffer_usd), first))
    parts.append(kv_table(ctx["guarded_summary"]))
    parts.append("<h3>Outcomes</h3>" + df_table(gr.trades["outcome"].value_counts().to_frame("count")
                                                if len(gr.trades) else None))
    parts.append("<h3>Blocked signals (compliance log, counts by reason)</h3>" + df_table(ctx["block_counts"]))
    parts.append("<h3>Account events (halts, resets, firm breaches)</h3>" + df_table(gr.events, index=False))

    mc = ctx["mc"]
    parts.append("<h2>3. Challenge pass probability (Monte Carlo)</h2>")
    parts.append(f"<p class=muted>{mc['session_days_sampled']} session days bootstrapped "
                 f"({mc['days_with_trades']} with trades), {s.challenge.n_simulations:,} simulations per phase, "
                 f"horizon {s.challenge.max_days_per_phase} session days.</p>")
    ph = pd.DataFrame([{"phase": i + 1, "target_%": p.target_pct, "P(pass)": p.p_pass,
                        "P(firm breach)": p.p_breach, "P(internal halt)": p.p_halt,
                        "P(no result in horizon)": p.p_timeout, "median days to pass": p.median_days_to_pass}
                       for i, p in enumerate(mc["phases"])])
    parts.append(df_table(ph, index=False))
    parts.append(f"<p><b>P(pass both phases) ≈ {mc['p_pass_all']:.1%}</b></p>")
    parts.append("<h2>Method notes</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in ctx["notes"]) + "</ul>")
    parts.append("</body></html>")
    return "\n".join(parts)
