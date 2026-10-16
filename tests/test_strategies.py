"""Round-2 strategies: hand-built days with known answers + no-lookahead (truncation) test."""
import numpy as np
import pandas as pd
import pytest

from src.data.resample import resample
from src.data.synthetic import synthetic_m1
from src.strategies.rules import STRATEGIES, generate

NY = "America/New_York"


def day_bars(rows, date="2024-06-11", start="09:30"):
    idx = pd.date_range(pd.Timestamp(f"{date} {start}", tz=NY), periods=len(rows), freq="5min").tz_convert("UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df.index.name = "ts"
    df["volume"] = 1.0
    df["end"] = df.index + pd.Timedelta(minutes=5)
    df["atr"] = 10.0
    return df


def test_orb15_long_breakout(settings):
    rows = [(100, 104, 98, 102), (102, 103, 99, 101), (101, 103, 100, 102),   # OR 98-104
            (102, 103.5, 101, 103), (103, 107, 102.5, 106)]                     # 09:50 closes 106 > 104
    c = generate(day_bars(rows), None, settings, only=["orb15"])
    r = c.iloc[0]
    assert r.status == "valid" and r.direction == "long" and r.stop == 98 and r.entry == 106
    assert r.armed_at == pd.Timestamp("2024-06-11 09:55", tz=NY) and r.target == pytest.approx(106 + 1.5 * 8)
    assert r.order_type == "market"


def test_orb_stop_cap(settings):
    rows = [(100, 140, 98, 102), (102, 103, 99, 101), (101, 103, 100, 102), (102, 150, 101, 145)]
    c = generate(day_bars(rows), None, settings, only=["orb15"])
    assert c.iloc[0].status == "stop_too_wide"          # risk 47 > 3 x ATR(10)


def test_gap_fill_short_toward_prev_close(settings):
    prev = day_bars([(100, 101, 99, 100)] * 78, date="2024-06-10")              # 09:30..15:55, close 100
    today = day_bars([(110, 111, 106, 107)] + [(107, 108, 106, 107)] * 5)
    bars = pd.concat([prev, today])
    daily = pd.DataFrame({"open": [100.0] * 20, "high": [130.0] * 20, "low": [100.0] * 20, "close": [110.0] * 20},
                         index=pd.date_range("2024-05-20", periods=20, freq="B", tz="UTC"))
    daily["trading_date"] = [d.date() for d in daily.index.tz_convert(NY)]
    c = generate(bars, daily, settings, only=["gap_fill"])
    r = c.iloc[-1]
    assert r.direction == "short" and r.target == 100 and r.stop == pytest.approx(111 + 1.0)
    assert r.armed_at == pd.Timestamp("2024-06-11 09:35", tz=NY)


@pytest.mark.parametrize("cut", ["2024-02-14 10:47", "2024-02-21 15:12"])
def test_no_lookahead_truncation(settings, cut):
    """Signals armed at or before T must be identical whether or not data after T exists."""
    m1 = synthetic_m1("2024-01-01", "2024-03-01", seed=5)
    b5, d1 = resample(m1, "5m"), resample(m1, "1d")
    full = generate(b5, d1, settings)
    t = pd.Timestamp(cut, tz=NY).tz_convert("UTC")
    part = generate(b5[b5["end"] <= t], d1[d1["end"] <= t], settings)
    keep = ["setup_type", "direction", "status", "armed_at", "entry", "stop", "target"]
    a = full[full["armed_at"] <= t][keep].reset_index(drop=True)
    b = part[part["armed_at"] <= t][keep].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) > 10 and set(full["setup_type"]) <= set(STRATEGIES)


def test_all_valid_targets_at_least_1r(settings):
    m1 = synthetic_m1("2024-01-01", "2024-03-01", seed=5)
    c = generate(resample(m1, "5m"), resample(m1, "1d"), settings)
    v = c[c.status == "valid"]
    assert len(v) and (v["rr"] >= 1.0 - 1e-9).all() and np.isfinite(v["risk_pts"]).all()
