"""Round 15 families: no lookahead (whole-day truncation) on synthetic data."""
import pandas as pd
import pytest

from src.data.resample import resample
from src.data.synthetic import synthetic_m1
from src.strategies.families15 import run
from src.strategies.published import Cost

NY = "America/New_York"


@pytest.mark.parametrize("fam", ["orb", "inside", "pm_trend", "gap_go", "drive"])
def test_no_lookahead(fam, monkeypatch):
    import src.strategies.families15 as f
    g = f.grid()
    monkeypatch.setattr(f, "grid", lambda: {fam: g[fam]})
    m1 = synthetic_m1("2024-01-01", "2024-04-01", seed=13)
    d1 = resample(m1, "1d")
    full = run(m1, d1, Cost(0, 0), log=lambda *a: None)
    T = pd.Timestamp("2024-03-12 00:00", tz=NY).tz_convert("UTC")
    cut = run(m1[m1.index < T], d1[d1["end"] <= T], Cost(0, 0), log=lambda *a: None)
    cols = ["variant", "date", "side", "entry_time", "entry", "stop0", "r"]
    a = full[pd.to_datetime(full["entry_time"], utc=True) < T][cols].reset_index(drop=True) if len(full) else full
    b = cut[cols].reset_index(drop=True) if len(cut) else cut
    assert len(a) == len(b)
    if len(a):
        pd.testing.assert_frame_equal(a, b)
