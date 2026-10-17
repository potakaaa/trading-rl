"""Round 8: order-flow window stats use only trades strictly before t."""
import numpy as np
import pandas as pd
import pytest

from src.ml.flow_features import window_stats


def test_window_stats_exclude_trades_at_or_after_t():
    t = pd.Timestamp("2024-03-13 14:00", tz="UTC")
    idx = pd.DatetimeIndex([t - pd.Timedelta(seconds=s) for s in (170, 90, 30)] + [t, t + pd.Timedelta(seconds=5)])
    tr = pd.DataFrame({"price": [100.0, 101.0, 102.0, 500.0, 500.0], "size": [2, 10, 4, 99, 99],
                       "side": ["B", "A", "B", "B", "B"]}, index=idx)
    s = window_stats(tr, t, atr15=10.0)
    assert s["vol"] == 16 and s["n"] == 3
    assert s["of_delta3"] == pytest.approx((2 + 4 - 10) / 16)
    assert s["of_big_delta3"] == pytest.approx(-1.0)                       # only the size-10 sell is "big"
    vwap = (100 * 2 + 101 * 10 + 102 * 4) / 16
    assert s["of_vwap_dist3"] == pytest.approx((102 - vwap) / 10)
    assert np.isnan(window_stats(tr.iloc[:0], t, 10.0)["of_delta3"])


def test_unsigned_sizes_do_not_overflow():
    """Databento sizes are uint32: sell-heavy windows must give a NEGATIVE delta, not a wrapped-around number."""
    t = pd.Timestamp("2024-03-13 14:00", tz="UTC")
    idx = pd.DatetimeIndex([t - pd.Timedelta(seconds=s) for s in (100, 50)])
    tr = pd.DataFrame({"price": [100.0, 99.0], "size": np.array([10, 30], dtype=np.uint32),
                       "side": ["B", "A"]}, index=idx)
    s = window_stats(tr, t, atr15=10.0)
    assert s["of_big_delta3"] == pytest.approx(-0.5) and s["of_delta3"] == pytest.approx(-0.5)
