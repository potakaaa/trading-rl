"""Round 9: variants reproduce round-5 trades where identical; walk-forward selection never sees the test fold."""
import numpy as np
import pandas as pd

from src.data.resample import resample
from src.data.synthetic import synthetic_m1
from src.strategies.published import Cost, run_all
from src.strategies.variants import run_variants, walk_forward

NOCOST = Cost(0.0, 0.0)


def test_williams_variant_matches_round5_published_rule(settings):
    m1 = synthetic_m1("2024-01-01", "2024-03-01", seed=9)
    b5, d1 = resample(m1, "5m"), resample(m1, "1d")
    r5 = run_all(m1, b5, d1, NOCOST, only=["williams_vbo"], log=lambda *a: None)
    v = run_variants(m1, d1, NOCOST, log=lambda *a: None)
    v = v[v["variant"] == "williams|k=0.5,stop=open,exit=eod"]
    assert len(r5) > 10
    np.testing.assert_allclose(sorted(r5["r"]), sorted(v["r"]))


def test_walk_forward_selects_on_past_only():
    rng = np.random.default_rng(0)
    rows = []
    for v, good_until in (("A", "2022-01-01"), ("B", "2030-01-01")):
        for d in pd.date_range("2019-01-01", "2023-01-01", freq="B"):
            r = 1.0 if d < pd.Timestamp(good_until) else -1.0
            rows.append({"family": "f", "variant": v, "entry_time": d.tz_localize("UTC"), "r": r + rng.normal(0, .01)})
    t = pd.DataFrame(rows)
    _, tab = walk_forward(t, [("2022-01-01", "2022-07-01")])
    # both looked equally good before 2022 -> choice is made on the past; the test period never leaks in
    assert tab.iloc[0]["is_mean_r"] > 0.9
    t2 = t[t["entry_time"] < pd.Timestamp("2022-01-01", tz="UTC")]
    _, tab2 = walk_forward(pd.concat([t2, t[t["entry_time"] >= pd.Timestamp("2022-01-01", tz="UTC")].assign(
        r=-5.0)]), [("2022-01-01", "2022-07-01")])
    assert tab2.iloc[0]["chosen"] == tab.iloc[0]["chosen"]          # changing the future never changes the pick
