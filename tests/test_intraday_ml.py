"""Round-4 intraday ML: features use only past data; labels match the fill simulator."""
import numpy as np
import pandas as pd
import pytest

from src.data.resample import resample
from src.data.synthetic import synthetic_m1
from src.ml.intraday_dataset import FEATURES, build


@pytest.fixture(scope="module")
def synth():
    m1 = synthetic_m1("2024-01-01", "2024-03-15", seed=21)
    return m1, resample(m1, "15m"), resample(m1, "1d")


def test_features_identical_when_future_is_removed(settings, synth):
    m1, b15, d1 = synth
    full = build(m1, b15, d1, None, settings, log=lambda *a: None)
    T = pd.Timestamp("2024-02-21 14:40", tz="America/New_York").tz_convert("UTC")
    cut = build(m1[m1.index < T], b15[b15["end"] <= T], d1[d1["end"] <= T], None, settings, log=lambda *a: None)
    a = full[full["t"] <= T].set_index("t")[FEATURES]
    b = cut[cut["t"] <= T].set_index("t")[FEATURES]
    assert len(a) > 200 and a.index.equals(b.index)
    pd.testing.assert_frame_equal(a, b)


def test_labels_are_fixed_trade_outcomes(settings, synth):
    m1, b15, d1 = synth
    df = build(m1, b15, d1, None, settings, log=lambda *a: None).dropna(subset=["r_long", "r_short"])
    assert df["r_long"].max() <= 1.5 + 1e-9 and df["r_short"].max() <= 1.5 + 1e-9    # 1.5R target cap
    assert (df["r_long_2x"] <= df["r_long"] + 1e-9).all()                            # higher costs never help
    ny = df["t"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    assert set(ny) <= {"09:45", "10:00", "10:15", "10:30", "13:45", "14:00", "14:15", "14:30", "14:45", "15:00"}
    ex = pd.to_datetime(df["exit_long"], utc=True)
    assert ((ex - df["t"]) <= pd.Timedelta(minutes=61)).all()                        # 60-min time barrier
    assert np.isfinite(df[FEATURES].drop(columns=["vol_ratio30", "gap_atrd", "prev_ret_atrd", "d_pdh", "d_pdl",
                                                  "atr15_atrd", "day_range_atrd"]).to_numpy()).all()
