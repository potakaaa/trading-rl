"""Round-3 swing engine: indicators, swap, execution timing and no-lookahead."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.fills import M1
from src.data.resample import resample
from src.data.synthetic import synthetic_m1
from src.strategies.swing import SWING_STRATEGIES, Clock, _nights, rsi, run_strategy, swap_pts

NY = "America/New_York"


@pytest.fixture(scope="module")
def data():
    m1 = synthetic_m1("2023-01-02", "2024-04-01", seed=11)
    return m1, resample(m1, "1d"), M1.from_frame(m1)


def test_rsi_extremes():
    up = pd.Series(np.arange(1.0, 30.0))
    assert rsi(up, 2).iloc[-1] == pytest.approx(100)
    down = pd.Series(np.arange(30.0, 1.0, -1))
    assert rsi(down, 2).iloc[-1] == pytest.approx(0)


def test_swap_and_nights(settings):
    spec = settings.instruments["US500"]
    assert swap_pts(spec, 5000.0, 3) == pytest.approx(3 * 5000 * 0.075 / 360)
    assert swap_pts(settings.instruments["EURUSD"], 1.1, 2, mult=2) == pytest.approx(0.0004)
    fri = pd.Timestamp("2024-06-07 13:30", tz="UTC")       # Fri 09:30 NY
    assert _nights(fri, pd.Timestamp("2024-06-10 13:30", tz="UTC")) == 3
    assert _nights(fri, fri + pd.Timedelta(hours=5)) == 0


@pytest.mark.parametrize("name", [n for n in SWING_STRATEGIES])
def test_entries_at_0930_next_day_and_r_includes_swap(settings, data, name):
    m1, d1, M = data
    spec = settings.instruments["US500"]
    tr = run_strategy(name, "US500", spec, d1, M, Clock.from_m1(M), pd.Timestamp("2023-06-01", tz="UTC"), None)
    for t in tr:
        ny_in = t["fill_time"].tz_convert(NY)
        if name == "overnight":
            assert ny_in.strftime("%H:%M") == "15:55"
            assert t["exit_time"].tz_convert(NY).strftime("%H:%M") <= "09:35" or t["outcome"] == "sl"
        else:
            assert ny_in.strftime("%H:%M") == "09:30" and ny_in.date() > t["signal_time"].tz_convert(NY).date()
        assert t["r"] == pytest.approx((t["pnl_pts"] - t["swap_pts"]) / t["risk_pts"])
        assert t["fill_time"] < t["exit_time"]
    # one position at a time per strategy/instrument
    for a, b in zip(tr, tr[1:], strict=False):
        assert b["fill_time"] >= a["exit_time"]


@pytest.mark.parametrize("name", ["donchian", "rsi2_revert", "tsmom", "ma_trend"])
def test_no_lookahead_entries(settings, data, name):
    """Entries decided before T are identical whether or not daily/1m data after T exists."""
    m1, d1, M = data
    spec = settings.instruments["US500"]
    T = pd.Timestamp("2024-01-20", tz="UTC")
    start = pd.Timestamp("2023-03-01", tz="UTC")
    full = run_strategy(name, "US500", spec, d1, M, Clock.from_m1(M), start, None)
    m1c = m1[m1.index < T]
    Mc = M1.from_frame(m1c)
    cut = run_strategy(name, "US500", spec, d1[d1["end"] <= T], Mc, Clock.from_m1(Mc), start, None)
    key = lambda t: (t["signal_time"], t["direction"], t["fill_time"], round(t["stop"], 6))  # noqa: E731
    a = [key(t) for t in full if t["exit_time"] < T - pd.Timedelta(days=3)]
    b = [key(t) for t in cut if t["exit_time"] < T - pd.Timedelta(days=3)]
    assert a == b


def test_ma_trend_cross_signal_and_exit():
    from src.strategies.swing import _entry_signal, _rule_exit_k, daily_indicators
    closes = np.concatenate([np.linspace(200, 100, 260), np.linspace(100, 220, 140), np.linspace(220, 120, 150)])
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="B", tz="UTC")
    d1 = pd.DataFrame({"open": closes, "high": closes + 1, "low": closes - 1, "close": closes, "volume": 1.0},
                      index=idx)
    d1["end"] = d1.index + pd.Timedelta(hours=21)
    d = daily_indicators(d1)
    sig = [(k, _entry_signal("ma_trend", d, k, "US500")) for k in range(len(d))]
    longs = [k for k, s in sig if s == "long"]
    shorts = [k for k, s in sig if s == "short"]
    assert len(longs) == 1 and len(shorts) == 1 and longs[0] < shorts[0]
    k = longs[0]
    assert d["sma50"].iloc[k - 1] <= d["sma200"].iloc[k - 1] and d["sma50"].iloc[k] > d["sma200"].iloc[k]
    assert _rule_exit_k("ma_trend", d, longs[0], "long") == shorts[0]
