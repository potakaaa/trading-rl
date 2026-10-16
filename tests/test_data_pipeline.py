import lzma
from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import check_price_sanity, clean_m1
from src.data.dukascopy import RECORD, day_url, decode_candles, download, load_raw
from src.data.resample import resample, visible_at
from src.data.synthetic import synthetic_m1


def make_bi5(rows):
    """rows: (sec, o, c, l, h, v) with integer prices"""
    return lzma.compress(b"".join(RECORD.pack(*r) for r in rows), format=lzma.FORMAT_ALONE)


def test_day_url_month_is_zero_based():
    u = day_url("https://x", "USATECHIDXUSD", "BID", date(2024, 1, 2))
    assert u == "https://x/USATECHIDXUSD/2024/00/02/BID_candles_min_1.bi5"


def test_decode_candles():
    raw = make_bi5([(0, 17000500, 17001000, 16999000, 17002000, 1.5),
                    (60, 17001000, 17000000, 16998000, 17003000, 2.0)])
    df = decode_candles(raw, date(2024, 1, 2), 1000)
    assert list(df.index) == [pd.Timestamp("2024-01-02 00:00", tz="UTC"),
                              pd.Timestamp("2024-01-02 00:01", tz="UTC")]
    r = df.iloc[0]
    assert (r.open, r.high, r.low, r.close, r.volume) == (17000.5, 17002.0, 16999.0, 17001.0, 1.5)
    assert decode_candles(b"", date(2024, 1, 2), 1000).empty


def test_download_resumable_with_mock_transport(settings):
    calls = []

    def handler(req: httpx.Request):
        calls.append(req.url.path)
        if "/2024/00/06/" in req.url.path:   # Saturday: no data
            return httpx.Response(404)
        sec = 14 * 3600 + 30 * 60
        return httpx.Response(200, content=make_bi5([(sec, 17000000, 17001000, 16999000, 17002000, 1.0)]))

    t = httpx.MockTransport(handler)
    problems = download(settings, date(2024, 1, 1), date(2024, 1, 31), transport=t, log=lambda *_: None)
    assert problems == []
    assert len(calls) == 31
    df = load_raw(settings)
    assert len(df) == 30 and df["close"].iloc[0] == 17001.0
    # second run skips the complete month
    download(settings, date(2024, 1, 1), date(2024, 1, 31), transport=t, log=lambda *_: None)
    assert len(calls) == 31


def test_download_records_failures(settings, monkeypatch):
    settings.data.dukascopy.retries = 0
    import src.data.dukascopy as dk

    async def no_sleep(_):
        return None
    monkeypatch.setattr(dk.asyncio, "sleep", no_sleep)
    t = httpx.MockTransport(lambda req: httpx.Response(503))
    problems = download(settings, date(2024, 1, 1), date(2024, 1, 3), transport=t, log=lambda *_: None)
    assert problems and "failed days" in problems[0]


def test_cleaning_drops_weekend_filler_and_repairs_spike(settings):
    m1 = synthetic_m1("2024-06-10", "2024-06-12")
    i = 600
    spike_t = m1.index[i]
    m1.loc[spike_t, "high"] = m1.loc[spike_t, ["open", "close"]].max() + 500
    filler_t = m1.index[700]
    p = m1.loc[filler_t, "open"]
    m1.loc[filler_t, ["high", "low", "close", "volume"]] = [p, p, p, 0.0]
    weekend = pd.DataFrame({"open": [1.0e4], "high": [1.0e4], "low": [1.0e4], "close": [1.0e4],
                            "volume": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp("2024-06-08 12:00", tz="UTC")]))
    dup = m1.iloc[[5]]
    raw = pd.concat([m1, weekend, dup])
    clean, rep = clean_m1(raw, settings)
    assert rep.duplicates == 1 and rep.outside_hours == 1 and rep.filler == 1
    assert rep.wick_spikes_repaired >= 1
    assert clean.loc[spike_t, "high"] == clean.loc[spike_t, ["open", "close"]].max()
    assert filler_t not in clean.index


def test_price_sanity_catches_wrong_divisor(settings):
    df = synthetic_m1("2024-06-10", "2024-06-11") / 1000
    with pytest.raises(ValueError, match="price_divisor"):
        check_price_sanity(df, settings)


def test_resample_ohlc_and_end_times():
    m1 = synthetic_m1("2024-06-10 13:30", "2024-06-10 14:30")
    b = resample(m1, "5m")
    first = m1.iloc[:5]
    r = b.iloc[0]
    assert b.index[0] == pd.Timestamp("2024-06-10 13:30", tz="UTC")
    assert r.open == first.open.iloc[0] and r.close == first.close.iloc[-1]
    assert r.high == first.high.max() and r.low == first.low.min()
    assert r["end"] == pd.Timestamp("2024-06-10 13:35", tz="UTC") and r.n_bars == 5


def test_daily_bar_end_is_17_ny_both_seasons():
    m1 = synthetic_m1("2024-03-06", "2024-03-13")
    d = resample(m1, "1d")
    ends = d["end"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    assert set(ends) == {"17:00"}
    assert set(d.index.tz_convert("America/New_York").strftime("%H:%M")) == {"18:00"}


def test_htf_bar_not_visible_before_it_closes():
    m1 = synthetic_m1("2024-06-10 13:00", "2024-06-10 17:00")
    h1 = resample(m1, "1h")
    t = pd.Timestamp("2024-06-10 14:59", tz="UTC")
    vis = visible_at(h1, t)
    assert vis.index.max() == pd.Timestamp("2024-06-10 13:00", tz="UTC")
    assert len(visible_at(h1, pd.Timestamp("2024-06-10 15:00", tz="UTC"))) == 2


def test_resample_volume_sums():
    m1 = synthetic_m1("2024-06-10 13:00", "2024-06-10 15:00")
    for tf in ["15m", "1h"]:
        assert np.isclose(resample(m1, tf)["volume"].sum(), m1["volume"].sum())
