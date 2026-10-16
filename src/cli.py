"""Command line entry point.

    uv run python -m src.cli data download [--start 2019-01-01] [--end 2026-10-15]
    uv run python -m src.cli data build
    uv run python -m src.cli data check-live [--seconds 15]
    uv run python -m src.cli data cross-check [--days 5]
    uv run python -m src.cli news check-live
    uv run python -m src.cli news build-history
    uv run python -m src.cli charts sample [--n 20] [--seed 1]
    uv run python -m src.cli backtest run
    uv run python -m src.cli backtest research
    uv run python -m src.cli backtest round2     # 7 pre-registered strategies + RL filter
    uv run python -m src.cli live run            # API + live paper engine (one command)
    uv run python -m src.cli live check-feed
    uv run python -m src.cli live paper-report   # Phase 7: paper vs backtest
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta

from dotenv import load_dotenv

from src.config import PROJECT_ROOT, load_settings


def _date(s: str) -> date:
    return date.fromisoformat(s)


def cmd_data_download(s, a):
    from src.data.dukascopy import download
    start = a.start or _date(s.data.history_start)
    end = a.end or (datetime.now(UTC).date() - timedelta(days=1))
    print(f"Downloading {s.instrument.dukascopy_symbol} {s.data.dukascopy.side} 1m candles "
          f"{start} -> {end} (resumable; re-run to continue)")
    problems = download(s, start, end, force=a.force)
    if problems:
        print("\nPROBLEMS (re-run the command to retry):\n  " + "\n  ".join(problems))
        return 1
    print("done")


def cmd_data_build(s, a):
    from src.data.pipeline import build
    print(json.dumps(build(s), indent=1, default=str))


def cmd_data_check_live(s, a):
    from src.data.oanda import check_live
    print(json.dumps(check_live(s, a.seconds), indent=1, default=str))


def cmd_data_cross_check(s, a):
    import pandas as pd

    from src.data.oanda import OandaProvider, cross_check
    from src.data.store import load_bars
    end = pd.Timestamp.now(tz="UTC").floor("D")
    start = end - pd.Timedelta(days=a.days)
    ref = load_bars(s, "1m", start, end)
    if ref.empty:
        print("No processed Dukascopy data in that window; run data download + data build first")
        return 1
    other = OandaProvider(s).fetch_candles(start, end)
    print(json.dumps(cross_check(ref, other), indent=1))


def cmd_news_check_live(s, a):
    from src.news.forexfactory import ForexFactoryProvider
    ev = ForexFactoryProvider(s).fetch()
    print(f"OK: {len(ev)} {s.news.currency} {s.news.impact}-impact events this week")
    for e in ev:
        ny = e.ts.tz_convert(s.timezones.market)
        mnl = e.ts.tz_convert(s.timezones.display)
        print(f"  {ny:%a %Y-%m-%d %H:%M} NY | {mnl:%a %H:%M} Manila | {e.title}")


def cmd_news_build_history(s, a):
    from src.news.historical import build_history
    df = build_history(s)
    print(df.groupby("title").size().sort_values(ascending=False).to_string())


def cmd_charts_sample(s, a):
    from src.viz.sample_days import generate_samples
    paths = generate_samples(s, n=a.n, seed=a.seed, synthetic=a.synthetic)
    print(f"wrote {len(paths)} charts to {paths[0].parent if paths else '-'}")


def cmd_backtest_run(s, a):
    from src.backtest.runner import run
    r = run(s)
    print(json.dumps({"gate": r["gate"], "raw": r["raw_summary"], "raw_2x_spread": r["raw_stress_summary"],
                      "guarded": r["guarded_summary"],
                      "p_pass_both_phases": r["mc"]["p_pass_all"]}, indent=1, default=float))


def cmd_research_run(s, a):
    from src.backtest.research import run_research
    t = run_research(s)
    with __import__("pandas").option_context("display.width", 200, "display.max_columns", 20):
        print(t.drop(columns="overrides").round(3).to_string())


def build_engine(s):
    from src.alerts.notify import Notifier
    from src.data.dukascopy_live import DukascopyLiveProvider
    from src.live.engine import LiveEngine, default_db_path
    from src.live.store import Store
    from src.news.calendar import LiveNewsCalendar
    from src.news.forexfactory import ForexFactoryProvider
    if s.live.provider != "dukascopy_live":
        raise SystemExit(f"live.provider={s.live.provider} is not wired; use dukascopy_live")
    return LiveEngine(s, DukascopyLiveProvider(s), LiveNewsCalendar(ForexFactoryProvider(s), s),
                      Store(default_db_path(s)), Notifier())


def cmd_live_run(s, a):
    import os
    import platform
    import subprocess

    import uvicorn

    from src.api.app import create_app
    from src.live.checklist import format_checklist, run_checklist
    if a.port:
        s.live.api_port = a.port
    print("Startup checklist:\n" + format_checklist(run_checklist(s)))
    if s.live.mode != "paper":
        print("\n!! live.mode is not 'paper'. Only Rald may change this, after paper results justify it.")
    if platform.system() == "Darwin" and not a.no_caffeinate:
        # keep the Mac awake (idle sleep) for as long as this process lives
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        print("  caffeinate: idle sleep disabled while the engine runs (closing the lid still sleeps the Mac)")
    engine = build_engine(s)
    app = create_app(s, engine)
    print(f"\nAPI + live engine on http://{s.live.api_host}:{s.live.api_port}"
          "  (dashboard: cd dashboard && npm run dev)")
    uvicorn.run(app, host=s.live.api_host, port=s.live.api_port, log_level="info")


def cmd_live_check_feed(s, a):
    import pandas as pd

    from src.data.dukascopy_live import DukascopyLiveProvider, FeedError
    now = pd.Timestamp.now(tz="UTC")
    try:
        df = DukascopyLiveProvider(s).fetch_before(now, 5)
    except FeedError as e:
        print(f"FEED FAILED: {e}")
        return 1
    if df.empty:
        print("feed reachable but returned no bars (market closed?)")
        return 1
    last = df.index[-1]
    print(f"OK: newest completed 1m bar {last} close {df['close'].iloc[-1]:.2f}; "
          f"lag {(now - last - pd.Timedelta(minutes=1)).total_seconds():.0f}s after bar close")


def cmd_live_paper_report(s, a):
    from src.live.engine import default_db_path
    from src.live.paper_report import compare, write_html
    from src.live.store import Store
    res = compare(Store(default_db_path(s)), s)
    write_html(res, PROJECT_ROOT / "reports" / "paper_latest.html")
    print(json.dumps({k: v for k, v in res.items() if k != "weeks"}, indent=1, default=float))
    print(f"report: {PROJECT_ROOT / 'reports' / 'paper_latest.html'}")


def cmd_backtest_round2(s, a):
    """Round-2 research: 7 pre-registered strategies + RL trade filter (docs/STRATEGIES.md)."""
    import pandas as pd

    from src.backtest.engine import M1, run_raw
    from src.backtest.fills import Costs
    from src.data.store import load_bars
    from src.ml.rl_filter import build_features, rl_passes, walk_forward
    from src.news.historical import load_history
    from src.strategies.research import run, write_report
    c, table = run(s)
    start, end = pd.Timestamp(s.backtest.start, tz="UTC"), pd.Timestamp(s.research.holdout_start, tz="UTC")
    m1 = M1.from_frame(load_bars(s, "1m", start, end))
    z = run_raw(c, m1, s, Costs(0, 0, 0)).set_index("setup_id")
    c["r_gross"] = z["r"].reindex(c.index).to_numpy()
    gross = c[c["filled"].astype(bool)].groupby("setup_type")["r_gross"].mean()
    b5 = load_bars(s, "5m", start - pd.Timedelta(days=60), end)
    d1 = load_bars(s, "1d", start - pd.Timedelta(days=120), end)
    feats = build_features(c, b5, d1, load_history(s), s)
    folds = [(f"{y}-{m:02d}-01", f"{y if m == 1 else y + 1}-{7 if m == 1 else 1:02d}-01")
             for y in range(start.year + 2, end.year) for m in (1, 7)]
    wf = walk_forward(feats, folds)
    verdict = rl_passes(wf)
    out = PROJECT_ROOT / "reports" / "round2_latest.html"
    out.write_text(write_report(table, gross, wf, verdict, s))
    print(table[["filled", "win_rate", "expectancy_r", "expectancy_r_2x_spread", "t_stat", "positive_years",
                 "PASS"]].round(3).to_string())
    print("RL filter:", "PASS" if verdict[0] else "FAIL", "-", verdict[1])
    print(f"report: {out}")


def cmd_data_swing_download(s, a):
    """Round-3 instruments: paced download via the chart endpoint + build processed bars."""
    from src.data.freeserv_history import build, download
    keys = a.only or list(s.instruments)
    for key in keys:
        print(f"== {key}", flush=True)
        probs = download(s, key, s.swing_history_start, log=lambda m: print(m, flush=True))
        if probs:
            print("   PROBLEMS (re-run to retry): " + "; ".join(probs), flush=True)
        build(s, key, log=lambda m: print(m, flush=True))
    print("ALL DONE", flush=True)


def main(argv=None):
    load_dotenv(PROJECT_ROOT / ".env")
    p = argparse.ArgumentParser(prog="src.cli")
    sub = p.add_subparsers(dest="group", required=True)

    d = sub.add_parser("data").add_subparsers(dest="cmd", required=True)
    x = d.add_parser("download")
    x.add_argument("--start", type=_date)
    x.add_argument("--end", type=_date)
    x.add_argument("--force", action="store_true", help="re-download complete months")
    x.set_defaults(fn=cmd_data_download)
    d.add_parser("build").set_defaults(fn=cmd_data_build)
    x = d.add_parser("swing-download")
    x.add_argument("--only", nargs="*", help="instrument keys (default: all in settings.instruments)")
    x.set_defaults(fn=cmd_data_swing_download)
    x = d.add_parser("check-live")
    x.add_argument("--seconds", type=int, default=15)
    x.set_defaults(fn=cmd_data_check_live)
    x = d.add_parser("cross-check")
    x.add_argument("--days", type=int, default=5)
    x.set_defaults(fn=cmd_data_cross_check)

    n = sub.add_parser("news").add_subparsers(dest="cmd", required=True)
    n.add_parser("check-live").set_defaults(fn=cmd_news_check_live)
    n.add_parser("build-history").set_defaults(fn=cmd_news_build_history)

    c = sub.add_parser("charts").add_subparsers(dest="cmd", required=True)
    x = c.add_parser("sample")
    x.add_argument("--n", type=int, default=20)
    x.add_argument("--seed", type=int, default=1)
    x.add_argument("--synthetic", action="store_true", help="use a synthetic series (dev only)")
    x.set_defaults(fn=cmd_charts_sample)

    b = sub.add_parser("backtest").add_subparsers(dest="cmd", required=True)
    b.add_parser("run").set_defaults(fn=cmd_backtest_run)
    b.add_parser("research").set_defaults(fn=cmd_research_run)
    b.add_parser("round2").set_defaults(fn=cmd_backtest_round2)

    lv = sub.add_parser("live").add_subparsers(dest="cmd", required=True)
    x = lv.add_parser("run")
    x.add_argument("--no-caffeinate", action="store_true")
    x.add_argument("--port", type=int, help="override live.api_port")
    x.set_defaults(fn=cmd_live_run)
    lv.add_parser("check-feed").set_defaults(fn=cmd_live_check_feed)
    lv.add_parser("paper-report").set_defaults(fn=cmd_live_paper_report)

    a = p.parse_args(argv)
    return a.fn(load_settings(), a) or 0


if __name__ == "__main__":
    sys.exit(main())
