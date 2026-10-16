"""Live economic calendar: ForexFactory weekly JSON feed (faireconomy.media mirror).

Each successful fetch is archived to data/news/ff_archive/ so that, over time, we build our own
history of exactly the events the live system saw (useful for future backtests and audits).
The feed is rate-limited by its host; we fetch every 30 minutes, which is well within limits.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from src.config import Settings, data_dir
from src.news.providers import NewsEvent, NewsProvider


class NewsFeedError(RuntimeError):
    pass


def parse_ff(payload, currency: str = "USD", impact: str = "High") -> list[NewsEvent]:
    if not isinstance(payload, list):
        raise NewsFeedError("unexpected ForexFactory payload (not a list); feed may be rate-limiting")
    if not payload:
        raise NewsFeedError("ForexFactory payload is empty; refusing to treat it as 'no news'")
    out = []
    for e in payload:
        if e.get("country") != currency or e.get("impact") != impact:
            continue
        ts = pd.Timestamp(e["date"])  # ISO with offset, e.g. 2024-01-05T08:30:00-05:00
        if ts.tzinfo is None:
            raise NewsFeedError(f"event without timezone: {e}")
        out.append(NewsEvent(ts=ts.tz_convert("UTC"), title=e.get("title", ""), currency=currency,
                             impact=impact, source="forexfactory"))
    return out


class ForexFactoryProvider(NewsProvider):
    name = "forexfactory"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None,
                 archive: bool = True):
        self.s = settings
        self._transport = transport
        self.archive = archive

    def fetch(self) -> list[NewsEvent]:
        try:
            with httpx.Client(timeout=20, transport=self._transport,
                              headers={"User-Agent": "ict-signal-assistant/0.1"}) as c:
                r = c.get(self.s.news.live_feed_url)
        except httpx.HTTPError as e:
            raise NewsFeedError(f"news feed unreachable: {e}") from e
        if r.status_code != 200:
            raise NewsFeedError(f"news feed HTTP {r.status_code}")
        try:
            payload = r.json()
        except ValueError as e:
            raise NewsFeedError("news feed returned non-JSON (likely rate-limited)") from e
        events = parse_ff(payload, self.s.news.currency, self.s.news.impact)
        if self.archive:
            d = archive_dir(self.s)
            d.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M")
            (d / f"ff_{stamp}.json").write_text(json.dumps(payload))
        return events


def archive_dir(settings: Settings) -> Path:
    return data_dir(settings) / "news" / "ff_archive"


def load_archive(settings: Settings) -> list[NewsEvent]:
    events: dict[tuple, NewsEvent] = {}
    for f in sorted(archive_dir(settings).glob("ff_*.json")):
        try:
            for e in parse_ff(json.loads(f.read_text()), settings.news.currency, settings.news.impact):
                events[(e.ts, e.title)] = e
        except (NewsFeedError, ValueError):
            continue
    return list(events.values())
