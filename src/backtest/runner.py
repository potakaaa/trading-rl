"""Phase 3 entry point: detect -> raw + guarded backtests (base and 2x spread) -> Monte Carlo -> report."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.backtest import metrics, montecarlo, report
from src.backtest.engine import M1, costs_from_settings, run_guarded, run_raw
from src.config import PROJECT_ROOT, Settings
from src.news.calendar import NewsCalendar
from src.news.historical import load_history
from src.risk.guard import Limits
from src.setups.detect import detect_all
from src.timeutils import trading_date


def load_detections(settings: Settings, start=None, end=None):
    from src.data.store import load_bars
    tf = settings.detectors.setup_timeframe
    need = sorted({tf, settings.detectors.premium_discount.range_timeframe, *settings.detectors.htf_bias.timeframes})
    frames = {t: load_bars(settings, t, start, end) for t in need}
    return detect_all(frames[tf], {t: frames[t] for t in need if t != tf}, settings)


def session_days(bars: pd.DataFrame, settings: Settings) -> int:
    sess = bars["session"].notna().to_numpy()
    return int(pd.Index(trading_date(bars.index[sess], settings.data.trading_day_start)).nunique())


def with_calendar_cols(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    ny = pd.DatetimeIndex(t["armed_at"]).tz_convert("America/New_York")
    t["weekday"] = ny.day_name().str[:3]
    t["year"] = ny.year
    return t


def run(settings: Settings, out_dir: Path | None = None, log=print) -> dict:
    settings.require_complete()
    bt = settings.backtest
    from src.data.store import load_bars
    log("detecting setups ...")
    det = load_detections(settings, bt.start, bt.end)
    setups = det.setups
    log(f"  {int((setups.status == 'valid').sum())} valid setups of {len(setups)} candidates")
    log("loading 1m bars ...")
    m1 = M1.from_frame(load_bars(settings, "1m", bt.start, bt.end))
    events = load_history(settings)
    cal = NewsCalendar.from_settings(events, settings)
    base, stress = costs_from_settings(settings), costs_from_settings(settings, bt.spread_stress_mult)

    log("raw backtests ...")
    raw = with_calendar_cols(run_raw(setups, m1, settings, base))
    raw_stress = with_calendar_cols(run_raw(setups, m1, settings, stress))
    first, last = det.bars.index[0], det.bars.index[-1]
    months = (last - first).days / 30.44
    raw_sum, raw_stress_sum = metrics.summary(raw, months), metrics.summary(raw_stress, months)

    log("guarded backtest ...")
    guarded = run_guarded(setups, m1, settings, base, cal)
    g_tr = guarded.trades
    lim = Limits.from_settings(settings)
    g_sum = metrics.summary(g_tr, months) if len(g_tr) else {}
    if len(g_tr):
        f = g_tr[g_tr["filled"].astype(bool)].sort_values("exit_time")
        g_sum["net_pnl_usd"] = f["pnl_usd"].sum()
        g_sum["max_drawdown_pct_of_initial"] = max(
            (metrics.equity_drawdown_pct(a["balance_after"], lim.initial_balance) for _, a in f.groupby("account")),
            default=0.0)
        g_sum["accounts_used (halts -> reset)"] = guarded.accounts
        g_sum["days_breaching_firm_daily_limit"] = int(
            (guarded.events["event"] == "FIRM_DAILY_LIMIT_BREACH").sum()) if len(guarded.events) else 0
    block_counts = pd.Series(dtype=int)
    if len(guarded.blocks):
        block_counts = guarded.blocks["reasons"].str.split("; ").explode().value_counts()

    log("Monte Carlo ...")
    mc = montecarlo.challenge_pass_probability(g_tr, session_days(det.bars, settings), settings)

    exp = raw_sum["expectancy_r"]
    gate_pass = bool(exp > 0)
    gate_text = (f"Raw expectancy {exp:+.3f} R over {raw_sum['filled']} filled trades "
                 f"(t-stat {raw_sum.get('t_stat', float('nan')):.2f}); with 2x spread "
                 f"{raw_stress_sum['expectancy_r']:+.3f} R.")
    if gate_pass and raw_sum.get("t_stat", 0) < 2:
        gate_text += " The t-stat is below 2: the edge is NOT statistically distinguishable from zero."

    notes = [
        "Prices are Dukascopy BID; buys pay the configured spread. Stops slip by "
        f"{bt.stop_slippage_points} pt. Same-bar stop and target -> stop first. "
        "On the fill bar only the stop is checked.",
        "Positions are closed at session end and 1 minute before a news blackout starts; pending "
        "orders are cancelled at the same time (Equity Edge forbids opening or closing inside the window).",
        "Guarded run: one order/position at a time; R and $ use the planned stop distance. "
        "After an internal max-drawdown halt the backtest starts a fresh account the next day "
        "(live trading needs a manual reset).",
        "Daily loss is measured from the start-of-day balance with intraday worst equity "
        "(entry minus maximum adverse excursion).",
        "Monte Carlo samples whole session days independently; clustered losing streaks are "
        "under-represented, so treat the pass rate as optimistic.",
        "No parameter was tuned on these results.",
    ]
    raw_b = {k: metrics.breakdown(raw, k) for k in ("setup_type", "session", "direction", "weekday", "year")}
    ctx = dict(settings=settings, period=f"{first:%Y-%m-%d} to {last:%Y-%m-%d}", raw=raw, raw_stress=raw_stress,
               raw_summary=raw_sum, raw_stress_summary=raw_stress_sum, raw_breakdowns=raw_b,
               raw_daily_breach=metrics.daily_limit_breach_rate(raw, settings.account.risk_per_trade_pct,
                                                                settings.account.daily_loss_limit_pct),
               guarded=guarded, guarded_summary=g_sum, block_counts=block_counts.to_frame("signals"),
               limits=lim, mc=mc, gate={"pass": gate_pass, "text": gate_text}, notes=notes)

    out_dir = out_dir or PROJECT_ROOT / "reports"
    bt_dir = out_dir / "backtest"
    bt_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    path = out_dir / f"backtest_{stamp}.html"
    page = report.build(ctx)
    path.write_text(page)
    (out_dir / "backtest_latest.html").write_text(page)
    raw.to_csv(bt_dir / "raw_trades.csv", index=False)
    g_tr.to_csv(bt_dir / "guarded_trades.csv", index=False)
    guarded.blocks.to_csv(bt_dir / "blocks.csv", index=False)
    guarded.events.to_csv(bt_dir / "events.csv", index=False)
    log(f"report: {path}")
    return dict(report=path, raw_summary=raw_sum, raw_stress_summary=raw_stress_sum, guarded_summary=g_sum,
                mc=mc, gate=ctx["gate"], block_counts=block_counts, events=guarded.events,
                raw_breakdowns=raw_b, raw_daily_breach=ctx["raw_daily_breach"])
