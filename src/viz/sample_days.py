"""Annotated sample-day charts for visually checking the ICT detectors.

    uv run python -m src.cli charts sample --n 20

Writes reports/samples/<date>.html (interactive, zoomable) and reports/samples/index.html.
Every overlay is a legend entry: click a legend item to hide/show that layer.
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import PROJECT_ROOT, Settings
from src.features.structure import bias_at, find_order_block
from src.setups.detect import Detections, detect_all
from src.timeutils import NY, hhmm_minutes, trading_date

COL = {
    "bg": "#0d1117", "grid": "#21262d", "up": "#26a69a", "down": "#ef5350",
    "buy_liq": "#f0b429", "sell_liq": "#4fc3f7", "fvg_bull": "rgba(38,166,154,0.18)",
    "fvg_bear": "rgba(239,83,80,0.18)", "ob": "rgba(186,104,200,0.30)", "mss": "#e6edf3",
    "entry": "#58a6ff", "stop": "#ff7b72", "target": "#3fb950", "swing": "#8b949e",
}
SETUP_SHORT = {"sweep_mss_fvg": "S1 sweep→MSS→FVG", "ob_retest": "S2 OB retest"}


def _ny(t) -> pd.Timestamp | pd.DatetimeIndex:
    return pd.DatetimeIndex(t).tz_convert(NY).tz_localize(None) if not isinstance(t, pd.Timestamp) \
        else t.tz_convert(NY).tz_localize(None)


def _window_span(day_ny: pd.Timestamp, start: str, end: str, day_start: str):
    """Wall-clock (start, end) of a session window within trading date `day_ny`."""
    s, e, ds = hhmm_minutes(start), hhmm_minutes(end), hhmm_minutes(day_start)
    s_t = day_ny + pd.Timedelta(minutes=s) - (pd.Timedelta(days=1) if s >= ds else pd.Timedelta(0))
    e_t = s_t + pd.Timedelta(minutes=(e - s) % 1440 or 1440)
    return s_t, e_t


def day_chart(d: Detections, day: pd.Timestamp, settings: Settings, synthetic: bool = False) -> go.Figure:
    """Chart for trading date `day` (18:00 NY the evening before -> 17:00 NY)."""
    b = d.bars
    td = trading_date(b.index, settings.data.trading_day_start)
    pos = np.flatnonzero(td == day)
    if len(pos) == 0:
        raise ValueError(f"no bars for {day.date()}")
    i0, i1 = int(pos[0]), int(pos[-1])
    win = b.iloc[i0:i1 + 1]
    x = _ny(win.index)
    t_start, t_end = win.index[0], win["end"].iloc[-1]
    fig = go.Figure()
    manila = win.index.tz_convert(settings.timezones.display).strftime("%H:%M")
    fig.add_trace(go.Candlestick(
        x=x, open=win.open, high=win.high, low=win.low, close=win.close, name="US100 5m",
        increasing_line_color=COL["up"], decreasing_line_color=COL["down"],
        increasing_fillcolor=COL["up"], decreasing_fillcolor=COL["down"], line_width=1,
        text=[f"Manila {m}" for m in manila], hoverinfo="x+y+text"))

    def xt(t):  # timestamp -> chart x
        return _ny(pd.Timestamp(t))

    def seg(name, group, x0, x1, y, color, dash="solid", width=1.5, text=None, show=False, opacity=1.0):
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y, y], mode="lines", name=name, legendgroup=group,
                                 showlegend=show, line=dict(color=color, dash=dash, width=width),
                                 opacity=opacity, hovertext=text, hoverinfo="text" if text else "y"))

    def rect(name, group, x0, x1, y0, y1, fill, show=False, text=None):
        fig.add_trace(go.Scatter(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0], fill="toself",
                                 fillcolor=fill, mode="lines", line=dict(width=0), name=name,
                                 legendgroup=group, showlegend=show, hovertext=text, hoverinfo="text"))

    # ---- sessions (shading)
    shade = {"ny_am": "rgba(88,166,255,0.07)", "ny_pm": "rgba(88,166,255,0.07)"}
    liq = settings.detectors.liquidity
    windows = [(k, w.start, w.end, shade.get(k, "rgba(255,255,255,0.04)")) for k, w in settings.sessions.items()]
    windows += [("asia", liq.asia.start, liq.asia.end, "rgba(240,180,41,0.04)"),
                ("london", liq.london.start, liq.london.end, "rgba(79,195,247,0.04)")]
    day_ny = pd.Timestamp(day).normalize()
    for name, s_, e_, color in windows:
        s_t, e_t = _window_span(day_ny, s_, e_, settings.data.trading_day_start)
        fig.add_vrect(x0=s_t, x1=e_t, fillcolor=color, line_width=0, layer="below",
                      annotation_text=name.upper().replace("_", " "), annotation_position="top left",
                      annotation_font_size=9, annotation_font_color="#8b949e")

    # ---- liquidity levels alive during the window
    lv = d.levels
    lv = lv[(lv["known_at"] < t_end) & (lv["expires_at"] > t_start)]
    first = {"high": True, "low": True}
    for _, L in lv.iterrows():
        x0 = xt(max(L["known_at"], t_start))
        consumed = L["consumed_at"] if pd.notna(L["consumed_at"]) else None
        stop_t = min(consumed, L["expires_at"], t_end) if consumed is not None else min(L["expires_at"], t_end)
        if stop_t <= max(L["known_at"], t_start):
            continue
        color = COL["buy_liq"] if L["side"] == "high" else COL["sell_liq"]
        dash = "dot" if L["kind"].startswith("EQ") else "dash"
        txt = f"{L['kind']} {L['price']:.1f} ({L['outcome'] or 'untouched'})"
        grp = "buy-side liquidity" if L["side"] == "high" else "sell-side liquidity"
        seg(grp, grp, x0, xt(stop_t), L["price"], color, dash=dash, text=txt, show=first[L["side"]])
        first[L["side"]] = False
        fig.add_annotation(x=xt(stop_t), y=L["price"], text=L["kind"], showarrow=False, xanchor="right",
                           yanchor="bottom", font=dict(size=9, color=color))

    # ---- swings (pivot markers; hover shows when they were confirmed)
    sw = d.swings[(d.swings["idx"] >= i0) & (d.swings["idx"] <= i1)]
    for kind, sym in (("high", "triangle-down"), ("low", "triangle-up")):
        s = sw[sw["kind"] == kind]
        if s.empty:
            continue
        off = (win["high"] - win["low"]).median() * 0.6 * (1 if kind == "high" else -1)
        fig.add_trace(go.Scatter(
            x=_ny(s["ts"]), y=s["price"] + off, mode="markers", name="swings", legendgroup="swings",
            showlegend=kind == "high", marker=dict(symbol=sym, size=7, color=COL["swing"]),
            hovertext=[f"swing {kind} {p:.1f}<br>confirmed {k:%H:%M} NY" for p, k in
                       zip(s["price"], s["known_at"].dt.tz_convert(NY))], hoverinfo="text"))

    # ---- FVGs (all, faint) – drawn until filled or for 36 bars
    fv = d.fvgs[(d.fvgs["idx"] >= i0) & (d.fvgs["idx"] <= i1)]
    shown = {"bull": False, "bear": False}
    for _, f in fv.iterrows():
        a = int(f["idx"]) - 2
        z = int(f["filled_idx"]) if f["filled_idx"] >= 0 else int(f["idx"]) + 24
        z = min(z, int(f["idx"]) + 24, i1)
        grp = f"FVG {f['direction']}"
        rect(grp, grp, xt(b.index[a]), xt(b["end"].iloc[z]), f["bottom"], f["top"],
             COL[f"fvg_{f['direction']}"], show=not shown[f["direction"]],
             text=f"{grp} {f['bottom']:.1f}-{f['top']:.1f} ({f['size_atr']:.2f} ATR)")
        shown[f["direction"]] = True

    # ---- sweeps
    sp = d.sweeps[(d.sweeps["pen_idx"] >= i0) & (d.sweeps["pen_idx"] <= i1)] if len(d.sweeps) else d.sweeps
    if len(sp):
        fig.add_trace(go.Scatter(
            x=_ny(b.index[sp["extreme_idx"].to_numpy()]), y=sp["extreme"], mode="markers",
            name="sweeps", legendgroup="sweeps", marker=dict(symbol="x", size=10, color="#ffa657"),
            hovertext=[f"sweep of {k} {lv_:.1f}<br>depth {dp:.2f} ATR<br>known {t:%H:%M} NY"
                       for k, lv_, dp, t in zip(sp["level_kind"], sp["level"], sp["depth_atr"],
                                                sp["known_at"].dt.tz_convert(NY))],
            hoverinfo="text"))
        for _, w in sp.iterrows():
            up = w["side"] == "high"
            fig.add_annotation(x=xt(b.index[int(w["extreme_idx"])]), y=w["extreme"], text=f"sweep {w['level_kind']}",
                               showarrow=True, arrowhead=0, arrowcolor="#ffa657", ax=0, ay=-26 if up else 26,
                               font=dict(size=9, color="#ffa657"))

    # ---- MSS + order blocks
    ms = d.mss[(d.mss["mss_idx"] >= i0) & (d.mss["mss_idx"] <= i1)] if len(d.mss) else d.mss
    first_m = True
    for _, m in ms.iterrows():
        j = int(m["mss_idx"])
        seg("MSS", "MSS", xt(b.index[max(int(m["ref_idx"]), i0)]), xt(b["end"].iloc[j]), m["ref_price"],
            COL["mss"], width=1.2, show=first_m,
            text=f"MSS {m['direction']} through {m['ref_price']:.1f}<br>displacement {m['disp_body_atr']:.2f} ATR"
                 f"<br>after sweep of {m['levels_swept']}")
        fig.add_annotation(x=xt(b.index[j]), y=m["ref_price"], text="MSS", showarrow=False,
                           yanchor="top" if m["direction"] == "short" else "bottom",
                           font=dict(size=10, color=COL["mss"]))
        ob = find_order_block(b, m, settings)
        if ob is not None:
            rect("order blocks", "OB", xt(b.index[ob["ob_idx"]]), xt(b["end"].iloc[min(j + 24, i1)]),
                 ob["ob_bottom"], ob["ob_top"], COL["ob"], show=first_m,
                 text=f"OB {ob['ob_bottom']:.1f}-{ob['ob_top']:.1f}")
        first_m = False

    # ---- setups
    st = d.setups
    st = st[(st["mss_idx"] >= i0) & (st["mss_idx"] <= i1)]
    lines = []
    for k, (_, s) in enumerate(st.iterrows()):
        label = f"{SETUP_SHORT.get(s['setup_type'], s['setup_type'])} {s['direction']}"
        valid = s["status"] == "valid"
        if pd.notna(s["armed_at"]) and pd.notna(s.get("entry")):
            x0 = xt(s["armed_at"])
            x1 = xt(min(s["armed_at"] + pd.Timedelta(hours=3), t_end))
            xe = xt(s["expires_at"]) if pd.notna(s["expires_at"]) else x1
            grp = f"setup {k + 1}"
            dash = "solid" if valid else "dot"
            op = 1.0 if valid else 0.45
            seg(f"#{k + 1} {label} [{s['status']}]", grp, x0, xe, s["entry"], COL["entry"], dash, 2,
                f"#{k + 1} entry {s['entry']:.1f} (expires {xe:%H:%M})", show=True, opacity=op)
            seg(grp, grp, x0, x1, s["stop"], COL["stop"], dash, 1.5, f"stop {s['stop']:.1f}", opacity=op)
            if pd.notna(s["target"]):
                seg(grp, grp, x0, x1, s["target"], COL["target"], dash, 1.5,
                    f"target {s['target']:.1f} ({s['target_kind']}) R:R {s['rr']:.2f}", opacity=op)
            fig.add_annotation(x=x0, y=s["entry"], text=f"#{k + 1}", showarrow=True, arrowhead=0,
                               ax=-18, ay=-18 if s["direction"] == "long" else 18,
                               font=dict(size=10, color=COL["entry"]))
        rr = f"R:R {s['rr']:.2f}" if pd.notna(s.get("rr")) else ""
        pdp = f"P/D {s['pd_pos']:.2f}" if pd.notna(s.get("pd_pos")) else ""
        armed = f"{_ny(s['armed_at']):%H:%M}" if pd.notna(s["armed_at"]) else "-"
        lines.append(f"#{k + 1} {label} armed {armed} → <b>{s['status']}</b> {rr} {pdp} bias {s.get('bias') or '-'}")

    # ---- HTF bias at the NY open
    open_t = (day_ny + pd.Timedelta(hours=9, minutes=30)).tz_localize(NY).tz_convert("UTC")
    pdl = d.levels[(d.levels["trading_date"] == day) & d.levels["kind"].isin(["PDH", "PDL"])]
    pd_mid = pdl["price"].mean() if len(pdl) == 2 else None
    k_open = b.index.searchsorted(open_t) - 1
    bias_txt = ""
    if d.bias_frames and k_open >= 0:
        bz = bias_at(d.bias_frames, open_t, float(b["close"].iloc[k_open]), pd_mid, settings)
        parts = " ".join(f"{k.replace('bos_', '')} {v:+.0f}" for k, v in bz.items()
                         if k.startswith("bos_") or k == "vs_pd_mid")
        bias_txt = f" | HTF bias at 09:30: <b>{bz['bias']}</b> ({parts})"

    title = (f"{'SYNTHETIC DATA · ' if synthetic else ''}{settings.instrument.name} "
             f"{settings.detectors.setup_timeframe} · trading day {pd.Timestamp(day):%a %Y-%m-%d} (NY time){bias_txt}")
    view0, view1 = day_ny + pd.Timedelta(hours=7), day_ny + pd.Timedelta(hours=16, minutes=30)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)), template="plotly_dark", height=820,
        paper_bgcolor=COL["bg"], plot_bgcolor=COL["bg"], xaxis_rangeslider_visible=False,
        hovermode="closest", legend=dict(font=dict(size=10), orientation="v"),
        margin=dict(l=50, r=20, t=60, b=40 + 16 * max(1, len(lines))),
        xaxis=dict(gridcolor=COL["grid"], range=[view0, view1]),
        yaxis=dict(gridcolor=COL["grid"], side="right", tickformat=",.0f", fixedrange=False))
    vis = win[(x >= view0) & (x <= view1)]
    if len(vis):
        pad = (vis["high"].max() - vis["low"].min()) * 0.08
        fig.update_yaxes(range=[vis["low"].min() - pad, vis["high"].max() + pad])
    fig.add_annotation(xref="paper", yref="paper", x=0, y=-0.04, yanchor="top", xanchor="left", align="left",
                       showarrow=False, font=dict(size=11),
                       text="<br>".join(lines) if lines else "no setups this day")
    return fig


def pick_days(d: Detections, settings: Settings, n: int, seed: int) -> list[pd.Timestamp]:
    b = d.bars
    td = pd.Series(trading_date(b.index, settings.data.trading_day_start), index=b.index)
    in_sess = b["session"].notna()
    expected = sum((pd.Timestamp(f"2000-01-01 {w.end}") - pd.Timestamp(f"2000-01-01 {w.start}")).seconds // 60
                   for w in settings.sessions.values()) // 5
    cov = in_sess.groupby(td).sum() / expected
    days = cov[cov >= 0.95].index
    days = days[pd.DatetimeIndex(days).dayofweek < 5]
    # need some history before the day for levels/bias
    days = days[days >= td.iloc[0] + pd.Timedelta(days=10)]
    rng = np.random.default_rng(seed)
    with_setups = set()
    if len(d.setups):
        armed = d.setups.dropna(subset=["armed_at"])
        v = armed[armed["status"] == "valid"]
        with_setups = set(trading_date(pd.DatetimeIndex(v["armed_at"]), settings.data.trading_day_start))
    a = [x for x in days if x in with_setups]
    rest = [x for x in days if x not in with_setups]
    k = min(len(a), n // 2)
    chosen = list(rng.choice(a, k, replace=False)) if k else []
    m = min(len(rest), n - k)
    chosen += list(rng.choice(rest, m, replace=False)) if m else []
    return sorted(pd.Timestamp(x) for x in chosen)


def generate_samples(settings: Settings, n: int = 20, seed: int = 1, synthetic: bool = False,
                     out_dir: Path | None = None) -> list[Path]:
    from src.data.resample import resample
    from src.data.store import load_bars

    tf = settings.detectors.setup_timeframe
    need = sorted({tf, settings.detectors.premium_discount.range_timeframe, *settings.detectors.htf_bias.timeframes})
    if synthetic:
        from src.data.synthetic import synthetic_m1
        m1 = synthetic_m1("2024-01-01", "2024-07-01", seed=seed)
        frames = {t: resample(m1, t) for t in need}
    else:
        frames = {t: load_bars(settings, t) for t in need}
    d = detect_all(frames[tf], {t: frames[t] for t in need if t != tf}, settings)
    days = pick_days(d, settings, n, seed)
    out_dir = out_dir or (PROJECT_ROOT / "reports" / ("samples_SYNTHETIC" if synthetic else "samples"))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, rows = [], []
    td_all = trading_date(d.bars.index, settings.data.trading_day_start)
    for day in days:
        fig = day_chart(d, day, settings, synthetic=synthetic)
        p = out_dir / f"{day:%Y-%m-%d}.html"
        fig.write_html(p, include_plotlyjs="directory", full_html=True)
        paths.append(p)
        lo, hi = np.flatnonzero(td_all == day)[[0, -1]]
        s = d.setups[(d.setups["mss_idx"] >= lo) & (d.setups["mss_idx"] <= hi)]
        sw = d.sweeps[(d.sweeps["pen_idx"] >= lo) & (d.sweeps["pen_idx"] <= hi)] if len(d.sweeps) else d.sweeps
        ms = d.mss[(d.mss["mss_idx"] >= lo) & (d.mss["mss_idx"] <= hi)] if len(d.mss) else d.mss
        rows.append((day, len(sw), len(ms), int((s["status"] == "valid").sum()), len(s)))
    _write_index(out_dir, rows, settings, synthetic)
    return paths


def _write_index(out_dir: Path, rows, settings: Settings, synthetic: bool):
    trs = "".join(
        f"<tr><td><a href='{d:%Y-%m-%d}.html'>{d:%a %Y-%m-%d}</a></td><td>{sw}</td><td>{ms}</td>"
        f"<td>{v} / {t}</td><td><input type='checkbox'> matches how I'd mark it</td><td contenteditable='true' "
        f"style='min-width:240px;border-bottom:1px solid #30363d'></td></tr>" for d, sw, ms, v, t in rows)
    det = settings.detectors
    params = html.escape(
        f"swing N={det.swing_n} · sweep ≥{det.sweep.min_sweep_atr}×ATR, reclaim ≤{det.sweep.reclaim_bars} bars · "
        f"MSS body ≥{det.mss.disp_mult}×ATR within {det.mss.max_bars_after_sweep} bars · "
        f"FVG ≥{det.fvg.min_fvg_atr}×ATR · OB zone={det.order_block.zone} · "
        f"equal levels within {det.liquidity.equal_level_atr}×ATR · entry={settings.setups.entry_mode} · "
        f"min R:R {settings.setups.min_rr}")
    banner = ("<p style='background:#5a1e02;padding:8px'>SYNTHETIC DATA: random walk for testing the chart "
              "tool only. Do not judge the detectors on these.</p>" if synthetic else "")
    (out_dir / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Detector review</title><style>body{{background:#0d1117;color:#e6edf3;font-family:system-ui;margin:24px}}
a{{color:#58a6ff}} td,th{{padding:4px 12px;border-bottom:1px solid #21262d;text-align:left}}</style></head><body>
<h1>Detector visual review ({len(rows)} days)</h1>{banner}
<p>For each day, check that swings, liquidity levels (PDH/PDL, Asia, London, overnight, equal highs/lows),
sweeps, MSS, FVGs, order blocks and setups are where <b>you</b> would mark them. Click legend items to toggle
layers; drag to zoom; double-click to reset. Note anything that looks wrong next to the day.</p>
<p style="color:#8b949e">Parameters: {params}</p>
<table><tr><th>Day</th><th>Sweeps</th><th>MSS</th><th>Valid setups / candidates</th><th>OK?</th><th>Notes</th></tr>
{trs}</table></body></html>""")
