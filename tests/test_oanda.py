import asyncio
import json

import httpx
import pandas as pd

from src.data.oanda import OandaProvider, cross_check, parse_price


def candle(t, o, h, l, c, complete=True):
    return {"time": t, "volume": 10, "complete": complete,
            "bid": {"o": str(o), "h": str(h), "l": str(l), "c": str(c)}}


def test_fetch_candles_parses_and_pages(settings, monkeypatch):
    monkeypatch.setattr("src.data.oanda.MAX_CANDLES", 2)
    pages = {
        "2024-06-10T13:30:00Z": [candle("2024-06-10T13:30:00.000000000Z", 1, 3, 0.5, 2),
                                 candle("2024-06-10T13:31:00.000000000Z", 2, 4, 1.5, 3)],
        "2024-06-10T13:31:01Z": [candle("2024-06-10T13:32:00.000000000Z", 3, 5, 2.5, 4, complete=False)],
    }
    seen = []

    def handler(req: httpx.Request):
        assert req.headers["Authorization"] == "Bearer tok"
        assert req.url.params["price"] == "B"
        seen.append(req.url.params["from"])
        return httpx.Response(200, json={"candles": pages[req.url.params["from"]]})

    p = OandaProvider(settings, token="tok", account_id="1", transport=httpx.MockTransport(handler))
    df = p.fetch_candles(pd.Timestamp("2024-06-10 13:30", tz="UTC"), pd.Timestamp("2024-06-10 14:00", tz="UTC"))
    assert seen == ["2024-06-10T13:30:00Z", "2024-06-10T13:31:01Z"]
    assert len(df) == 2  # incomplete candle skipped
    assert df.iloc[1].tolist() == [2.0, 4.0, 1.5, 3.0, 10.0]


def test_http_error_is_raised(settings):
    t = httpx.MockTransport(lambda r: httpx.Response(401, json={"errorMessage": "Insufficient authorization"}))
    p = OandaProvider(settings, token="bad", account_id="1", transport=t)
    try:
        p.fetch_candles(pd.Timestamp("2024-06-10", tz="UTC"), pd.Timestamp("2024-06-11", tz="UTC"))
        raise AssertionError("should raise")
    except RuntimeError as e:
        assert "401" in str(e)


def test_stream_yields_prices_and_heartbeats(settings):
    lines = "\n".join(json.dumps(m) for m in [
        {"type": "HEARTBEAT", "time": "2024-06-10T13:30:00Z"},
        {"type": "PRICE", "time": "2024-06-10T13:30:01.5Z", "bids": [{"price": "19000.1"}],
         "asks": [{"price": "19001.3"}]},
    ]) + "\n"
    t = httpx.MockTransport(lambda r: httpx.Response(200, content=lines.encode()))
    p = OandaProvider(settings, token="tok", account_id="acc", async_transport=t)

    async def collect():
        return [x async for x in p.stream_prices()]
    got = asyncio.run(collect())
    assert got[0] is None
    assert got[1].bid == 19000.1 and round(got[1].ask - got[1].bid, 1) == 1.2


def test_parse_price_utc():
    tick = parse_price({"time": "2024-06-10T13:30:01Z", "bids": [{"price": "1"}], "asks": []})
    assert tick.time.tz is not None and tick.ask is None


def test_cross_check_detects_offset():
    idx = pd.date_range("2024-06-10 13:30", periods=100, freq="1min", tz="UTC")
    a = pd.DataFrame({"close": 19000 + pd.Series([i * i % 17 for i in range(100)], index=idx).astype(float)})
    b = a + 2.5
    r = cross_check(a, b)
    assert r["overlap_bars"] == 100 and r["median_offset_points"] == 2.5
    assert r["return_correlation"] == 1.0 and r["bars_off_by_gt_5pts"] == 0
