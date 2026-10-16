"""Pluggable data-provider interface.

Every source of prices (Dukascopy history, OANDA live, anything we add later) implements the
same small interface, so the rest of the system never cares where bars come from.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class PriceTick:
    time: pd.Timestamp        # UTC
    bid: float
    ask: float | None = None

    @property
    def mid(self) -> float:
        return self.bid if self.ask is None else (self.bid + self.ask) / 2


class DataProvider(ABC):
    """Source of historical 1-minute bars. Returned frames are indexed by bar open time (UTC,
    tz-aware) and have the columns in BAR_COLUMNS."""

    name: str

    @abstractmethod
    def fetch_history(self, start: datetime, end: datetime) -> pd.DataFrame: ...


class StreamingProvider(DataProvider):
    """A provider that can also stream live prices."""

    @abstractmethod
    def stream_prices(self) -> AsyncIterator[PriceTick]: ...


def empty_bars() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}, index=idx)
