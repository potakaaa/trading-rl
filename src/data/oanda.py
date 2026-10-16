"""OANDA v20 REST + streaming provider (practice/demo account, NAS100_USD).

Credentials come from .env: OANDA_API_TOKEN, OANDA_ACCOUNT_ID, OANDA_ENV (practice|live).
Docs: https://developer.oanda.com/rest-live-v20/introduction/
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd

from src.config import Settings
from src.data.providers import BAR_COLUMNS, PriceTick, StreamingProvider, empty_bars

MAX_CANDLES = 5000  # OANDA per-request limit


class OandaConfigError(RuntimeError):
    pass


class OandaProvider(StreamingProvider):
    name = "oanda"

    def __init__(self, settings: Settings, token: str | None = None, account_id: str | None = None,
                 env: str | None = None, transport: httpx.BaseTransport | None = None,
                 async_transport: httpx.AsyncBaseTransport | None = None):
        self.s = settings
        self.token = token or os.getenv("OANDA_API_TOKEN", "")
        self.account_id = account_id or os.getenv("OANDA_ACCOUNT_ID", "")
        self.env = (env or os.getenv("OANDA_ENV", "practice")).lower()
        o = settings.live.oanda
        live = self.env == "live"
        self.rest_url = o.live_rest_url if live else o.practice_rest_url
        self.stream_url = o.live_stream_url if live else o.practice_stream_url
        self.instrument = settings.instrument.oanda_symbol
        self.price = o.price_component
        self._transport = transport
        self._async_transport = async_transport

    def _require_creds(self):
        if not self.token:
            raise OandaConfigError("OANDA_API_TOKEN is not set (copy .env.example to .env)")

    @property
    def _headers(self):
        return {"Authorization": f"Bearer {self.token}", "Accept-Datetime-Format": "RFC3339"}

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.rest_url, headers=self._headers, timeout=30,
                            transport=self._transport)

    # ------------------------------------------------------------------ history
    def fetch_candles(self, start: datetime, end: datetime, granularity: str = "M1",
                      include_incomplete: bool = False) -> pd.DataFrame:
        self._require_creds()
        start = _utc(start)
        end = _utc(end)
        key = {"B": "bid", "A": "ask", "M": "mid"}[self.price]
        rows = []
        cursor = start
        with self._client() as c:
            while cursor < end:
                r = c.get(f"/v3/instruments/{self.instrument}/candles", params={
                    "granularity": granularity, "price": self.price,
                    "from": cursor.isoformat().replace("+00:00", "Z"), "count": MAX_CANDLES,
                })
                _raise_for(r)
                candles = r.json().get("candles", [])
                if not candles:
                    break
                for k in candles:
                    if not k.get("complete", True) and not include_incomplete:
                        continue
                    p = k[key]
                    rows.append((pd.Timestamp(k["time"]).tz_convert("UTC"), float(p["o"]),
                                 float(p["h"]), float(p["l"]), float(p["c"]), float(k["volume"])))
                last = pd.Timestamp(candles[-1]["time"]).tz_convert("UTC")
                nxt = last + pd.Timedelta(seconds=1)
                if nxt <= cursor or len(candles) < MAX_CANDLES:
                    break
                cursor = nxt
        if not rows:
            return empty_bars()
        df = pd.DataFrame(rows, columns=["ts", *BAR_COLUMNS]).set_index("ts")
        df = df[(df.index >= start) & (df.index < end)]
        return df[~df.index.duplicated()].sort_index()

    def fetch_history(self, start: datetime, end: datetime) -> pd.DataFrame:
        return self.fetch_candles(start, end, "M1")

    # ------------------------------------------------------------------ streaming
    async def stream_prices(self) -> AsyncIterator[PriceTick | None]:
        """Yields PriceTick for every price update and None for every heartbeat (so callers can
        tell 'connected but quiet' from 'disconnected')."""
        self._require_creds()
        if not self.account_id:
            raise OandaConfigError("OANDA_ACCOUNT_ID is not set")
        url = f"{self.stream_url}/v3/accounts/{self.account_id}/pricing/stream"
        async with httpx.AsyncClient(headers=self._headers, timeout=httpx.Timeout(30, read=None),
                                     transport=self._async_transport) as c:
            async with c.stream("GET", url, params={"instruments": self.instrument}) as r:
                if r.status_code != 200:
                    await r.aread()
                    _raise_for(r)
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    if msg.get("type") == "HEARTBEAT":
                        yield None
                    elif msg.get("type") == "PRICE":
                        yield parse_price(msg)


def parse_price(msg: dict) -> PriceTick:
    bid = float(msg["bids"][0]["price"]) if msg.get("bids") else float(msg["closeoutBid"])
    ask = float(msg["asks"][0]["price"]) if msg.get("asks") else None
    return PriceTick(time=pd.Timestamp(msg["time"]).tz_convert("UTC"), bid=bid, ask=ask)


def _raise_for(r: httpx.Response):
    if r.status_code >= 400:
        try:
            detail = r.json().get("errorMessage", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"OANDA HTTP {r.status_code}: {detail}")


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ---------------------------------------------------------------------- checks

def check_live(settings: Settings, stream_seconds: int = 15, log=print) -> dict:
    """Connection test: recent candles + a short burst of the live price stream."""
    import asyncio

    p = OandaProvider(settings)
    out: dict = {"env": p.env, "instrument": p.instrument}
    t0 = time.monotonic()
    end = datetime.now(UTC)
    df = p.fetch_candles(end - timedelta(hours=6), end)
    out["rest_ok"] = True
    out["rest_latency_s"] = round(time.monotonic() - t0, 2)
    out["recent_candles"] = len(df)
    if len(df):
        out["last_candle_utc"] = str(df.index[-1])
        out["last_close"] = float(df["close"].iloc[-1])
    log(f"REST ok: {len(df)} M1 candles in last 6h; last={out.get('last_candle_utc')}")

    async def burst():
        ticks, beats = 0, 0
        last = None
        start = time.monotonic()
        async for tick in p.stream_prices():
            if tick is None:
                beats += 1
            else:
                ticks += 1
                last = tick
            if time.monotonic() - start > stream_seconds:
                break
        return ticks, beats, last

    ticks, beats, last = asyncio.run(burst())
    out.update(stream_ok=True, stream_ticks=ticks, stream_heartbeats=beats,
               last_bid=None if last is None else last.bid,
               last_spread=None if last is None or last.ask is None else round(last.ask - last.bid, 2))
    log(f"Stream ok: {ticks} prices, {beats} heartbeats in {stream_seconds}s; "
        f"last bid={out['last_bid']} spread={out['last_spread']}")
    if ticks == 0:
        log("No prices received. Normal when the market is closed (weekend / 17:00-18:00 NY).")
    return out


def cross_check(reference: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Compare two 1m bar sets on overlapping timestamps (e.g. Dukascopy vs OANDA)."""
    j = reference[["close"]].join(other[["close"]], how="inner", lsuffix="_ref", rsuffix="_other")
    if j.empty:
        return {"overlap_bars": 0}
    diff = j["close_other"] - j["close_ref"]
    r1 = j["close_ref"].diff()
    r2 = j["close_other"].diff()
    resid = diff - diff.median()
    return {
        "overlap_bars": len(j),
        "median_offset_points": round(float(diff.median()), 3),
        "offset_std_points": round(float(diff.std()), 3),
        "p95_abs_residual_points": round(float(resid.abs().quantile(0.95)), 3),
        "return_correlation": round(float(r1.corr(r2)), 4),
        "bars_off_by_gt_5pts": int((resid.abs() > 5).sum()),
        "ref_only_bars": len(reference.index.difference(other.index)),
        "other_only_bars": len(other.index.difference(reference.index)),
    }
