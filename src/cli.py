"""Command line entry point.

    uv run python -m src.cli data download [--start 2019-01-01] [--end 2026-10-15]
    uv run python -m src.cli data build
    uv run python -m src.cli data check-live [--seconds 15]
    uv run python -m src.cli data cross-check [--days 5]
    uv run python -m src.cli news check-live
    uv run python -m src.cli news build-history
    uv run python -m src.cli charts sample [--n 20] [--seed 1]
    uv run python -m src.cli backtest run
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

    a = p.parse_args(argv)
    return a.fn(load_settings(), a) or 0


if __name__ == "__main__":
    sys.exit(main())
