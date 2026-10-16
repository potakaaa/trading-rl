"""Live 1-minute US100 bars from Dukascopy's public chart endpoint (freeserv).

Why this source: it is the SAME BID feed as the historical .bi5 files (verified bar-for-bar,
including volume), so live detection sees exactly what the backtest saw. Completed bars appear
~10-20 s after each minute closes; the bar still forming is not served.
Caveat: the endpoint is unofficial and may change or rate-limit. Any failure raises FeedError, and
the live engine treats that as "no data -> no signals + alert" (fail safe).

Response format (JSONP): _cb([[ms, open, high, low, close, volume], ...]) newest first.
"""
from __future__ import annotations

import json

import httpx
import pandas as pd

from src.config import Settings
from src.data.providers import BAR_COLUMNS, DataProvider, empty_bars


class FeedError(RuntimeError):
    pass


def parse_jsonp(text: str) -> pd.DataFrame:
    try:
        rows = json.loads(text[text.index("(") + 1: text.rindex(")")])
    except (ValueError, json.JSONDecodeError) as e:
        raise FeedError(f"unexpected live feed response: {text[:120]!r}") from e
    if not isinstance(rows, list):
        raise FeedError("live feed payload is not a list")
    rows = [r for r in rows if isinstance(r, list) and len(r) >= 6]
    if not rows:
        return empty_bars()
    df = pd.DataFrame([r[:6] for r in rows], columns=["t", *BAR_COLUMNS])
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("t"), unit="ms", utc=True), name="ts").as_unit("ns")
    df = df.astype(float).sort_index()
    return df[~df.index.duplicated(keep="last")]


class DukascopyLiveProvider(DataProvider):
    name = "dukascopy_live"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.cfg = settings.live.dukascopy_live
        self._transport = transport

    def fetch_before(self, end: pd.Timestamp, limit: int) -> pd.DataFrame:
        """Up to `limit` completed 1m bars starting before `end` (UTC)."""
        params = {"path": "chart/json3", "instrument": self.cfg.instrument, "offer_side": "B",
                  "interval": "1MIN", "splits": "true", "stocks": "true",
                  "limit": str(min(limit, self.cfg.max_bars_per_request)), "time_direction": "P",
                  "timestamp": str(int(pd.Timestamp(end).timestamp() * 1000)), "jsonp": "_cb"}
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.dukascopy.com/"}
        try:
            with httpx.Client(timeout=self.cfg.timeout_seconds, transport=self._transport) as c:
                r = c.get(self.cfg.url, params=params, headers=headers)
        except httpx.HTTPError as e:
            raise FeedError(f"live feed unreachable: {e}") from e
        if r.status_code != 200:
            raise FeedError(f"live feed HTTP {r.status_code}")
        df = parse_jsonp(r.text)
        return df[df.index < pd.Timestamp(end)]

    def fetch_history(self, start, end) -> pd.DataFrame:
        """All bars in [start, end), paging backwards."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        parts, cursor = [], end
        while cursor > start:
            need = int((cursor - start) / pd.Timedelta(minutes=1)) + 1
            df = self.fetch_before(cursor, need)
            if df.empty:
                break
            parts.append(df)
            if df.index[0] >= cursor:   # defensive: no progress
                break
            cursor = df.index[0]
            if len(df) < min(need, self.cfg.max_bars_per_request):
                break                   # reached the start of available data / weekend
        if not parts:
            return empty_bars()
        out = pd.concat(parts).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out[(out.index >= start) & (out.index < end)]
