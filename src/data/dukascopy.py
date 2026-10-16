"""Dukascopy historical 1-minute candles.

Dukascopy publishes one file per instrument per UTC day:
    {base}/{SYMBOL}/{YYYY}/{MM-1}/{DD}/{SIDE}_candles_min_1.bi5      (month is 0-based!)
A .bi5 file is LZMA-compressed binary. Each candle is 24 bytes, big-endian:
    uint32 seconds-since-midnight-UTC, uint32 open, uint32 close, uint32 low, uint32 high,
    float32 volume
Prices are integers; divide by `price_divisor` (1000 for USATECHIDXUSD) to get index points.
The divisor is checked by a sanity range on load so a wrong value fails loudly.
"""
from __future__ import annotations

import asyncio
import json
import lzma
import struct
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from src.config import Settings, data_dir
from src.data.providers import BAR_COLUMNS, DataProvider, empty_bars

RECORD = struct.Struct(">IIIIIf")


def day_url(base: str, symbol: str, side: str, day: date) -> str:
    return f"{base}/{symbol}/{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/{side}_candles_min_1.bi5"


def decode_candles(raw: bytes, day: date, divisor: float) -> pd.DataFrame:
    """Decode one day's .bi5 payload into a 1-minute bar frame (UTC index)."""
    if not raw:
        return empty_bars()
    data = lzma.decompress(raw)
    n = len(data) // RECORD.size
    if n == 0:
        return empty_bars()
    arr = np.frombuffer(data[: n * RECORD.size], dtype=np.dtype([
        ("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4"),
    ]))
    day_start = pd.Timestamp(day, tz="UTC")
    idx = pd.DatetimeIndex(day_start + pd.to_timedelta(arr["t"].astype("int64"), unit="s"),
                           name="ts")
    df = pd.DataFrame({
        "open": arr["o"] / divisor,
        "high": arr["h"] / divisor,
        "low": arr["l"] / divisor,
        "close": arr["c"] / divisor,
        "volume": arr["v"].astype("float64"),
    }, index=idx)
    return df[BAR_COLUMNS]


class DukascopyProvider(DataProvider):
    name = "dukascopy"

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.s = settings
        self.cfg = settings.data.dukascopy
        self.symbol = settings.instrument.dukascopy_symbol
        self._transport = transport  # injectable for tests

    async def _fetch_day(self, client: httpx.AsyncClient, day: date) -> bytes | None:
        """Raw bytes for a day; b"" if Dukascopy has no data; None if every retry failed."""
        url = day_url(self.cfg.base_url, self.symbol, self.cfg.side, day)
        for attempt in range(self.cfg.retries + 1):
            try:
                r = await client.get(url)
                if r.status_code == 404:
                    return b""
                if r.status_code == 200:
                    return r.content
            except httpx.HTTPError:
                pass
            await asyncio.sleep(min(2 ** attempt, 16))
        return None

    async def fetch_days(self, days: list[date]) -> dict[date, pd.DataFrame | None]:
        sem = asyncio.Semaphore(self.cfg.max_concurrency)
        out: dict[date, pd.DataFrame | None] = {}
        async with httpx.AsyncClient(timeout=self.cfg.timeout_seconds,
                                     transport=self._transport) as client:
            async def one(d: date):
                async with sem:
                    raw = await self._fetch_day(client, d)
                out[d] = None if raw is None else decode_candles(raw, d, self.cfg.price_divisor)
            await asyncio.gather(*(one(d) for d in days))
        return out

    def fetch_history(self, start: datetime, end: datetime) -> pd.DataFrame:
        days = list(_days(start.date() if isinstance(start, datetime) else start,
                          end.date() if isinstance(end, datetime) else end))
        frames = asyncio.run(self.fetch_days(days))
        failed = [d for d, f in frames.items() if f is None]
        if failed:
            raise RuntimeError(f"Dukascopy download failed for {len(failed)} day(s): {failed[:5]}")
        parts = [f for f in frames.values() if f is not None and len(f)]
        return pd.concat(parts).sort_index() if parts else empty_bars()


def _days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ---------------------------------------------------------------- monthly raw storage

def raw_dir(settings: Settings) -> Path:
    s = settings
    return data_dir(s) / "raw" / "dukascopy" / f"{s.instrument.dukascopy_symbol}_{s.data.dukascopy.side}"


def month_paths(settings: Settings, year: int, month: int) -> tuple[Path, Path]:
    d = raw_dir(settings) / f"{year:04d}"
    return d / f"{year:04d}-{month:02d}.parquet", d / f"{year:04d}-{month:02d}.manifest.json"


def download(settings: Settings, start: date, end: date, force: bool = False,
             transport: httpx.AsyncBaseTransport | None = None, log=print) -> list[str]:
    """Download [start, end] month by month into data/raw. Months whose manifest says complete
    are skipped (so the command is resumable). Returns a list of problems."""
    provider = DukascopyProvider(settings, transport)
    problems: list[str] = []
    today = datetime.now(UTC).date()
    for year, month in _months(start, end):
        pq_path, man_path = month_paths(settings, year, month)
        if not force and man_path.exists():
            if json.loads(man_path.read_text()).get("complete"):
                continue
        m_start = max(date(year, month, 1), start)
        m_end = min(_month_end(year, month), end, today - timedelta(days=1))
        if m_start > m_end:
            continue
        days = list(_days(m_start, m_end))
        frames = asyncio.run(provider.fetch_days(days))
        failed = sorted(str(d) for d, f in frames.items() if f is None)
        parts = [f for f in frames.values() if f is not None and len(f)]
        df = pd.concat(parts).sort_index() if parts else empty_bars()
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq_path)
        manifest = {
            "symbol": settings.instrument.dukascopy_symbol,
            "side": settings.data.dukascopy.side,
            "days": {str(d): (None if f is None else len(f)) for d, f in sorted(frames.items())},
            "failed_days": failed,
            # complete = whole calendar month downloaded with no failures
            "complete": not failed and m_end == _month_end(year, month) and m_start.day == 1,
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        man_path.write_text(json.dumps(manifest, indent=1))
        log(f"{year}-{month:02d}: {len(df):>6} bars, {len(failed)} failed day(s)")
        if failed:
            problems.append(f"{year}-{month:02d}: failed days {failed}")
    return problems


def load_raw(settings: Settings, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    files = sorted(raw_dir(settings).glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No raw data in {raw_dir(settings)}. Run: uv run python -m src.cli data download")
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    return df


def _months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _month_end(y: int, m: int) -> date:
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return nxt - timedelta(days=1)
