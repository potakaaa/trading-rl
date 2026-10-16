"""News event model and the pluggable NewsProvider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import pandas as pd

EVENT_COLUMNS = ["ts", "title", "currency", "impact", "source"]


@dataclass(frozen=True)
class NewsEvent:
    ts: pd.Timestamp       # UTC
    title: str
    currency: str = "USD"
    impact: str = "High"
    source: str = ""


class NewsProvider(ABC):
    name: str

    @abstractmethod
    def fetch(self) -> list[NewsEvent]:
        """Return events. Must raise on any failure (callers go fail-safe)."""


def events_to_frame(events: list[NewsEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in EVENT_COLUMNS}).astype(
            {"ts": "datetime64[ns, UTC]"})
    df = pd.DataFrame([asdict(e) for e in events])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df[EVENT_COLUMNS].sort_values("ts").reset_index(drop=True)


def ny_time_to_utc(day, hhmm: str) -> pd.Timestamp:
    """A New York wall-clock time on a given date -> UTC timestamp (DST aware)."""
    return (pd.Timestamp(f"{pd.Timestamp(day).date()} {hhmm}")
            .tz_localize("America/New_York").tz_convert("UTC"))
