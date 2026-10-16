"""Processed bar storage: data/processed/{INSTRUMENT}/{tf}/{YYYY}.parquet"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Settings, data_dir


def processed_dir(settings: Settings, tf: str) -> Path:
    return data_dir(settings) / "processed" / settings.instrument.name / tf


def save_bars(settings: Settings, tf: str, df: pd.DataFrame) -> None:
    d = processed_dir(settings, tf)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.parquet"):
        old.unlink()
    for year, part in df.groupby(df.index.year):
        part.to_parquet(d / f"{year}.parquet")


def load_bars(settings: Settings, tf: str, start=None, end=None) -> pd.DataFrame:
    """Load processed bars with start <= ts < end (UTC if naive)."""
    d = processed_dir(settings, tf)
    files = sorted(d.glob("*.parquet"))
    if start is not None:
        start = _utc(start)
        files = [f for f in files if int(f.stem) >= start.year]
    if end is not None:
        end = _utc(end)
        files = [f for f in files if int(f.stem) <= end.year]
    if not files:
        raise FileNotFoundError(
            f"No processed {tf} bars in {d}. Run: uv run python -m src.cli data build")
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index < end]
    return df


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
