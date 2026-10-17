"""Round 7: cross-market features only use the other market's bars complete at t."""
import numpy as np
import pandas as pd

from src.data.resample import resample
from src.data.synthetic import synthetic_m1
from src.ml.cross_features import market_features


def test_cross_features_ignore_present_and_future_bars():
    x = synthetic_m1("2024-01-01", "2024-02-01", seed=4)
    b15 = resample(x, "15m")
    t = pd.DatetimeIndex([x.index[3000], x.index[9000]])
    df = pd.DataFrame({"t": t, "ret60": [0.5, -0.2]})
    a = market_features(df, x, b15, "X")
    x2 = x.copy()
    for tt in t:
        x2.loc[x2.index >= tt, ["open", "high", "low", "close"]] += 500.0   # change present + future
    b2 = resample(x2, "15m")
    b2 = b2[b2["end"] <= t.max()]
    b = market_features(df, x2, b2, "X")
    pd.testing.assert_frame_equal(a, b)
    assert np.isfinite(a.to_numpy()).all()
    assert a.loc[0, "X_div60"] == 0.5 - a.loc[0, "X_ret60"]


def test_stale_market_gives_missing_features():
    x = synthetic_m1("2024-01-01", "2024-01-05", seed=4)
    b15 = resample(x, "15m")
    t = x.index[2000] + pd.Timedelta(minutes=30)
    xs = x[(x.index < t - pd.Timedelta(minutes=20)) | (x.index >= t)]        # 20-minute hole before t
    f = market_features(pd.DataFrame({"t": [t], "ret60": [0.0]}), xs, b15, "X")
    assert np.isnan(f["X_ret5"].iloc[0])
