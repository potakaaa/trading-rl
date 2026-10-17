"""History download through Dukascopy's chart endpoint (the live feed), month by month, resumable.

Used for the round-3 instruments while the .bi5 history server rate-limits us (HTTP 429). The bars
are the same Dukascopy BID feed (checked bar-for-bar for US100). Requests are paced (one every
`pause` seconds) so the live engine sharing this endpoint is never starved or rate-limited.
"""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, Settings
from src.data.dukascopy_live import DukascopyLiveProvider, FeedError


def instrument_settings(settings: Settings, key: str) -> Settings:
    """A copy of settings pointed at one round-3 instrument (data root, sanity range, feed name)."""
    spec = _spec(settings, key)
    s = settings.model_copy(deep=True)
    s.data.root = spec.root
    s.data.sanity_price_range = spec.sanity
    s.live.dukascopy_live.instrument = spec.feed
    s.instrument.name = key
    return s


def _spec(settings: Settings, key: str):
    return settings.instruments[key] if key in settings.instruments else settings.cross_markets[key]


def raw_dir(settings: Settings, key: str) -> Path:
    root = Path(_spec(settings, key).root)
    root = root if root.is_absolute() else PROJECT_ROOT / root
    return root / "raw" / "freeserv" / key


def download(settings: Settings, key: str, start: str, end: str | None = None, pause: float = 0.6,
             log=print) -> list[str]:
    s = instrument_settings(settings, key)
    prov = DukascopyLiveProvider(s)
    out = raw_dir(settings, key)
    out.mkdir(parents=True, exist_ok=True)
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC").floor("D")
    months = pd.date_range(pd.Timestamp(start, tz="UTC"), end_ts, freq="MS")
    problems = []
    for m0 in months:
        m1 = min(m0 + pd.offsets.MonthBegin(1), end_ts)
        f, man = out / f"{m0:%Y-%m}.parquet", out / f"{m0:%Y-%m}.json"
        if man.exists() and json.loads(man.read_text()).get("complete"):
            continue
        parts, cursor, ok = [], m1, True
        while cursor > m0:
            try:
                df = prov.fetch_before(cursor, 5000)
            except FeedError as e:
                log(f"  {key} {m0:%Y-%m}: {e}; retrying in 30 s")
                time.sleep(30)
                try:
                    df = prov.fetch_before(cursor, 5000)
                except FeedError as e2:
                    problems.append(f"{key} {m0:%Y-%m}: {e2}")
                    ok = False
                    break
            time.sleep(pause)
            df = df[df.index >= m0]
            if df.empty:
                break
            parts.append(df)
            cursor = df.index[0]
        bars = pd.concat(parts).sort_index() if parts else pd.DataFrame()
        if len(bars):
            bars = bars[~bars.index.duplicated(keep="last")]
        bars.to_parquet(f)
        complete = ok and m1 < end_ts
        man.write_text(json.dumps({"bars": len(bars), "complete": complete,
                                   "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds")}))
        log(f"  {key} {m0:%Y-%m}: {len(bars):>6} bars{'' if complete else ' (incomplete)'}")
    prov.close()
    return problems


def load_raw(settings: Settings, key: str) -> pd.DataFrame:
    files = sorted(raw_dir(settings, key).glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    return df[~df.index.duplicated(keep="last")]


def build(settings: Settings, key: str, log=print) -> dict:
    """raw -> clean 1m -> all timeframes, saved under the instrument's data root."""
    from src.data.cleaning import check_price_sanity, clean_m1
    from src.data.resample import resample
    from src.data.store import save_bars
    s = instrument_settings(settings, key)
    raw = load_raw(settings, key)
    m1, rep = clean_m1(raw, s)
    check_price_sanity(m1, s)
    save_bars(s, "1m", m1)
    for tf in s.data.timeframes:
        save_bars(s, tf, resample(m1, tf, s.data.trading_day_start, s.data.market_close))
    info = {"instrument": key, "raw": len(raw), "clean": len(m1), "first": str(m1.index[0]),
            "last": str(m1.index[-1]), "median_close": float(m1["close"].median()),
            "wick_spikes_repaired": rep.wick_spikes_repaired, "outliers_dropped": rep.outlier_bars_dropped}
    log(json.dumps(info))
    return info
