"""Round-5 published setups: hand-built sessions with known answers + no-lookahead."""
import numpy as np
import pandas as pd
import pytest

from src.data.resample import resample
from src.data.synthetic import synthetic_m1
from src.strategies.published import Cost, Day, noise_momentum, orb5, run_all, run_position, williams_vbo

NY = "America/New_York"
NOCOST = Cost(0.0, 0.0)


def mkday(path, prev_close=100.0, pd_range=10.0, date="2024-06-11"):
    """path: list of (o,h,l,c) for each minute from 09:30, padded flat to 16:00."""
    rows = list(path) + [path[-1][3:] * 4] * (390 - len(path))
    rows = [r if len(r) == 4 else (r[0],) * 4 for r in rows]
    a = np.array(rows, dtype=float)
    idx = pd.date_range(pd.Timestamp(f"{date} 09:30", tz=NY), periods=390, freq="1min").tz_convert("UTC")
    mins = np.arange(570, 960)
    ctx = {"pdh": 105.0, "pdl": 95.0, "pd_range": pd_range, "atr_d": 10.0}
    return Day(pd.Timestamp(date), a[:, 0], a[:, 1], a[:, 2], a[:, 3], np.ones(390), mins,
               idx.as_unit("ns").asi8, prev_close, ctx)


def test_orb5_long_target_10r_or_eod():
    path = [(100, 101, 99.5, 100.5)] + [(100.5, 101, 100, 100.8)] * 4 + [(101, 101.2, 100.6, 101)] + \
           [(101, 102, 100.9, 101.9)] * 10
    t = orb5(mkday(path), NOCOST)[0]
    assert t["side"] == "long" and t["entry"] == 101 and t["stop0"] == 99.5
    assert t["reason"] == "eod" and t["r"] == pytest.approx((101.9 - 101) / 1.5)


def test_orb5_doji_no_trade():
    assert orb5(mkday([(100, 101, 99, 100.2)] + [(100.2, 100.5, 99.8, 100)] * 4), NOCOST) == []


def test_williams_breakout_stop_at_open():
    # open 100, range 10 -> buy stop 105; rally to 106 then falls to 99 -> stopped at the open (100)
    path = [(100, 101, 99.5, 100.5)] * 3 + [(104, 106, 104, 105.5)] + [(105, 105, 99, 99.5)]
    t = williams_vbo(mkday(path), Cost(0.0, 0.5))[0]
    assert t["side"] == "long" and t["entry"] == pytest.approx(105.5)       # level 105 + 0.5 slippage
    assert t["reason"] == "sl" and t["exit"] == pytest.approx(99.5)         # stop 100 - 0.5 slippage
    assert t["r"] == pytest.approx(-6 / 5.5)


def test_run_position_stop_wins_same_bar():
    d = mkday([(100, 100, 100, 100), (100, 110, 90, 100)])
    j, px, why, _ = run_position(d, 1, "long", 100, 95, 105, NOCOST)
    assert why == "sl" and px == 95


def test_noise_momentum_enters_above_boundary_and_trails_vwap():
    # flat until 10:00, then a strong rally -> long at the 10:00 check, exit at 16:00 (never below trail)
    path = [(100, 100.1, 99.9, 100)] * 30 + [(100 + k * 0.1, 100 + k * 0.1 + 0.05, 100 + k * 0.1 - 0.05,
                                                100 + k * 0.1) for k in range(1, 300)]
    d = mkday(path)
    sig = {m: 0.0005 for m in range(600, 960, 30)}
    tr = noise_momentum(d, NOCOST, sig)
    assert tr and tr[0]["side"] == "long" and tr[0]["reason"] == "eod" and tr[0]["r"] > 0


@pytest.mark.parametrize("name", ["orb5", "williams_vbo", "crabel_nr7", "noise_momentum", "holy_grail"])
def test_no_lookahead_whole_days(settings, name):
    m1 = synthetic_m1("2024-01-01", "2024-04-01", seed=9)
    b5, d1 = resample(m1, "5m"), resample(m1, "1d")
    full = run_all(m1, b5, d1, NOCOST, only=[name], log=lambda *a: None)
    T = pd.Timestamp("2024-03-12 00:00", tz=NY).tz_convert("UTC")
    cut = run_all(m1[m1.index < T], b5[b5["end"] <= T], d1[d1["end"] <= T], NOCOST, only=[name],
                  log=lambda *a: None)
    cols = ["date", "side", "entry_time", "entry", "stop0", "exit", "r"]
    a = full[full["entry_time"] < T][cols].reset_index(drop=True) if len(full) else full
    b = cut[cols].reset_index(drop=True) if len(cut) else cut
    assert len(a) == len(b)
    if len(a):
        pd.testing.assert_frame_equal(a, b)
