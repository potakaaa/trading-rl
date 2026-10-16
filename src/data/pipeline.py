"""raw 1m -> clean 1m -> all timeframes -> processed parquet + data quality report."""
from __future__ import annotations

from src.config import PROJECT_ROOT, Settings
from src.data.cleaning import check_price_sanity, clean_m1
from src.data.dukascopy import load_raw
from src.data.quality import write_report
from src.data.resample import resample
from src.data.store import save_bars


def build(settings: Settings, raw=None, synthetic: bool = False, log=print) -> dict:
    raw = load_raw(settings) if raw is None else raw
    log(f"raw 1m bars: {len(raw):,}")
    m1, rep = clean_m1(raw, settings)
    check_price_sanity(m1, settings)
    log(f"clean 1m bars: {len(m1):,} (dropped {rep.raw_rows - rep.clean_rows:,}, "
        f"repaired wicks {rep.wick_spikes_repaired})")
    bars = {"1m": resample(m1, "1m")}
    save_bars(settings, "1m", m1)
    for tf in settings.data.timeframes:
        b = resample(m1, tf, settings.data.trading_day_start, settings.data.market_close)
        save_bars(settings, tf, b)
        bars[tf] = b
        log(f"{tf:>4}: {len(b):,} bars")
    name = "data_quality_SYNTHETIC.html" if synthetic else "data_quality.html"
    path = PROJECT_ROOT / "reports" / name
    summary = write_report(m1, bars, rep, settings, path, synthetic=synthetic)
    log(f"report: {path}")
    return summary
