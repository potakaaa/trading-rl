"""Round-6 CNN: sequences use only bars before t; the pipeline can learn a planted pattern."""
import numpy as np
import pandas as pd

from src.data.synthetic import synthetic_m1
from src.ml.cnn import SEQ, Ensemble, build_sequences
from src.ml.intraday_dataset import FEATURES


def test_sequences_only_use_bars_before_t():
    m1 = synthetic_m1("2024-01-01", "2024-01-10", seed=3)
    t = m1.index[500]
    df = pd.DataFrame({"t": [t], "atr15": [10.0]})
    a = build_sequences(df, m1)
    m2 = m1.copy()
    m2.loc[m2.index >= t, ["open", "high", "low", "close"]] *= 3.0     # change the present and future
    b = build_sequences(df, m2)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (1, 4, SEQ) and np.isfinite(a).all()


def test_cnn_learns_a_planted_sequence_pattern():
    """y_long = 1 when the last 10 bars rose; the ensemble must find it out of sample."""
    rng = np.random.default_rng(0)
    n = 3000
    seq = rng.normal(size=(n, 4, SEQ)).astype(np.float32)
    trend = rng.choice([-1.0, 1.0], size=n)
    seq[:, 0, -10:] += (trend[:, None] * np.linspace(0, 2, 10)).astype(np.float32)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    df["r_long"] = np.where(trend > 0, 1.5, -1.0)
    df["r_short"] = np.where(trend < 0, 1.5, -1.0)
    tr, te = slice(0, 2400), slice(2400, n)
    ens = Ensemble().fit(df.iloc[tr], seq[tr], df.iloc[2000:2400], seq[2000:2400])
    p = ens.predict(df.iloc[te], seq[te])
    acc = ((p["p_long"] > p["p_short"]).to_numpy() == (trend[te] > 0)).mean()
    assert acc > 0.9, acc
