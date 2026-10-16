"""ICT detector tests: hand-built bar sequences with known answers, plus no-lookahead checks."""
import numpy as np
import pandas as pd
import pytest

from src.data.resample import resample
from src.data.synthetic import synthetic_m1
from src.features.basics import SwingIndex, atr, find_swings
from src.features.liquidity import find_sweeps, session_levels
from src.features.structure import (
    bias_at,
    bos_direction,
    dealing_range_at,
    find_fvgs,
    find_mss,
    find_order_block,
    pd_position,
)
from src.setups.detect import detect_all

NY = "America/New_York"


def mk(rows, start="2024-06-11 09:30", tf_min=5, tz=NY):
    """rows: list of (o, h, l, c). start is New York wall time."""
    idx = pd.date_range(pd.Timestamp(start, tz=tz), periods=len(rows), freq=f"{tf_min}min").tz_convert("UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df.index.name = "ts"
    df["volume"] = 1.0
    df["end"] = df.index + pd.Timedelta(minutes=tf_min)
    return df


def flat(n, price=100.0, rng=1.0):
    return [(price, price + rng, price - rng, price)] * n


# ------------------------------------------------------------------------------- ATR / swings

def test_atr_wilder():
    b = mk([(10, 12, 9, 11), (11, 13, 10, 12), (12, 12.5, 8, 9)], start="2024-06-11 09:30")
    a = atr(b, period=2)
    # TR = 3, 3, max(4.5, 0.5, 4) = 4.5 ; Wilder alpha=1/2: 3 -> 3 -> 3.75
    assert np.isnan(a.iloc[0]) and a.iloc[1] == pytest.approx(3.0) and a.iloc[2] == pytest.approx(3.75)


def test_swing_confirmed_n_bars_later():
    highs = [5, 6, 7, 10, 8, 7, 6]
    b = mk([(h - 1, h, h - 2, h - 1) for h in highs])
    s = find_swings(b, 2)
    hi = s[s.kind == "high"]
    assert list(hi.idx) == [3] and list(hi.confirm_idx) == [5]
    assert hi.known_at.iloc[0] == b["end"].iloc[5]


def test_swing_not_reported_without_right_side():
    b = mk([(h - 1, h, h - 2, h - 1) for h in [5, 6, 7, 10, 8]])
    assert (find_swings(b, 2).kind == "high").sum() == 0   # needs 2 bars after the pivot


def test_equal_highs_first_is_pivot():
    highs = [5, 6, 10, 10, 8, 7]
    b = mk([(h - 1, h, h - 2, h - 1) for h in highs])
    hi = find_swings(b, 2).query("kind == 'high'")
    assert list(hi.idx) == [2]


def test_swing_index_lookup():
    highs = [5, 6, 9, 6, 5, 6, 8, 6, 5]
    b = mk([(h - 1, h, h - 2, h - 1) for h in highs])
    si = SwingIndex(find_swings(b, 2), 2)
    assert si.latest("high", before_idx=4) is None            # pivot 2 confirms at bar 4 ...
    assert si.latest("high", before_idx=5)[0] == 2            # ... so it is usable from bar 5
    assert si.latest("high", before_idx=9)[0] == 6
    assert si.latest("high", before_idx=9, pivot_max_idx=5)[0] == 2


# ------------------------------------------------------------------------------------- FVG

def test_fvg_bull_and_bear_with_min_size(settings):
    rows = flat(14) + [(100, 101, 99, 100.5), (100.5, 106, 100.5, 105.5), (105.5, 107, 103, 106)]
    rows += [(106, 106.5, 104, 104.5), (104.5, 104.5, 99, 99.5), (99.5, 100, 97, 98)]
    b = mk(rows)
    f = find_fvgs(b, atr(b, 14), settings)
    bull = f[f.direction == "bull"].iloc[0]
    assert bull.idx == 16 and bull.bottom == 101 and bull.top == 103
    bear = f[f.direction == "bear"].iloc[0]
    assert bear.idx == 19 and bear.bottom == 100 and bear.top == 104
    assert bull.filled_idx == 18  # bar 18 low 99 trades through 101
    settings.detectors.fvg.min_fvg_atr = 5.0
    assert find_fvgs(b, atr(b, 14), settings).empty


# ------------------------------------------------------------------------------ levels / sweeps

def overnight_session(settings):
    """18:00 -> 11:30 NY with known overnight/Asia/London extremes, then an NY sweep of ON_H."""
    rows = []
    t0 = pd.Timestamp("2024-06-10 18:00", tz=NY)
    n_pre = int((pd.Timestamp("2024-06-11 09:30", tz=NY) - t0) / pd.Timedelta("5min"))
    cyc = [19000, 19003, 19000, 18997]
    prev = 19000.0
    for i in range(n_pre):
        c = cyc[i % 4]
        o = prev
        h, lo = max(o, c) + 3, min(o, c) - 3
        t = t0 + pd.Timedelta(minutes=5 * i)
        if t.strftime("%H:%M") == "19:00":
            h = 19010.0                       # overnight high
        if t.strftime("%H:%M") == "21:00":
            lo = 18930.0                      # Asia (and overnight) low
        rows.append((o, h, lo, c))
        prev = c
    rows += [
        (prev, 19003, 18998, 19001),     # 09:30
        (19001, 19002, 18997, 18998),    # 09:35
        (18998, 18999, 18994, 18996),    # 09:40 swing low 18994
        (18996, 19001, 18995, 19000),    # 09:45
        (19000, 19004, 18999, 19003),    # 09:50
        (19003, 19016, 19002, 19004),    # 09:55 sweep of ON_H 19010, closes back below
        (19004, 19005, 18975, 18978),    # 10:00 bearish displacement, closes below 18994 = MSS
        (18978, 18990, 18970, 18975),    # 10:05 completes bearish FVG [18990, 19002]
        (18975, 18980, 18968, 18972),    # 10:10
        (18972, 18978, 18966, 18970),    # 10:15
    ]
    return mk(rows, start="2024-06-10 18:00")


def test_session_levels_values_and_known_times(settings):
    b = overnight_session(settings)
    lv = session_levels(b, settings).set_index("kind")
    assert lv.loc["ON_H", "price"] == 19010 and lv.loc["ON_L", "price"] == 18930
    assert lv.loc["ASIA_L", "price"] == 18930
    assert lv.loc["ON_H", "known_at"] == pd.Timestamp("2024-06-11 09:30", tz=NY)
    assert lv.loc["ASIA_H", "known_at"] == pd.Timestamp("2024-06-11 00:00", tz=NY)
    assert lv.loc["LON_L", "known_at"] == pd.Timestamp("2024-06-11 05:00", tz=NY)
    assert lv.loc["ON_H", "expires_at"] == pd.Timestamp("2024-06-11 17:00", tz=NY)
    assert "PDH" not in lv.index  # only one trading day of data


def test_previous_day_levels(settings):
    m1 = synthetic_m1("2024-06-10", "2024-06-13")
    b = resample(m1, "5m")
    lv = session_levels(b, settings)
    pdh = lv[lv.kind == "PDH"].iloc[0]
    td = pd.Timestamp(pdh.trading_date)
    prev_day = b[(b.index >= pd.Timestamp(td - pd.Timedelta(days=1) - pd.Timedelta(hours=6), tz=NY))
                 & (b.index < pd.Timestamp(td - pd.Timedelta(hours=6), tz=NY))]
    assert pdh.price == prev_day.high.max()
    assert pdh.known_at.tz_convert(NY).strftime("%H:%M") == "17:00"


def test_sweep_detected_and_level_consumed(settings):
    b = overnight_session(settings)
    a = atr(b, 14)
    lv = session_levels(b, settings)
    lv2, sw = find_sweeps(b, lv, a, settings)
    on_h = sw[sw.level_kind == "ON_H"].iloc[0]
    t955 = pd.Timestamp("2024-06-11 09:55", tz=NY)
    assert b.index[on_h.pen_idx] == t955 and on_h.reclaim_idx == on_h.pen_idx
    assert on_h.extreme == 19016 and on_h.direction == "short"
    assert on_h.known_at == t955 + pd.Timedelta(minutes=5)
    lon_l = lv2[lv2.kind == "LON_L"].iloc[0]
    assert lon_l.outcome == "broken"   # displacement closed through it and never came back


def test_shallow_poke_is_not_a_sweep(settings):
    settings.detectors.sweep.min_sweep_atr = 5.0   # demand a huge sweep
    b = overnight_session(settings)
    _, sw = find_sweeps(b, session_levels(b, settings), atr(b, 14), settings)
    assert sw.empty or "ON_H" not in set(sw.level_kind)


def test_mss_and_order_block(settings):
    b = overnight_session(settings)
    a = atr(b, 14)
    _, sw = find_sweeps(b, session_levels(b, settings), a, settings)
    si = SwingIndex(find_swings(b, 2), 2)
    m = find_mss(b, sw, si, a, settings)
    m = m[m.direction == "short"].iloc[0]
    assert b.index[m.mss_idx] == pd.Timestamp("2024-06-11 10:00", tz=NY)
    assert m.ref_price == 18994 and m.extreme == 19016
    ob = find_order_block(b, m, settings)
    assert b.index[ob["ob_idx"]] == pd.Timestamp("2024-06-11 09:55", tz=NY)
    assert (ob["ob_bottom"], ob["ob_top"]) == (19003, 19016)


def test_no_mss_without_displacement(settings):
    settings.detectors.mss.disp_mult = 10.0
    b = overnight_session(settings)
    a = atr(b, 14)
    _, sw = find_sweeps(b, session_levels(b, settings), a, settings)
    assert find_mss(b, sw, SwingIndex(find_swings(b, 2), 2), a, settings).empty


def test_full_sweep_mss_fvg_setup(settings):
    settings.detectors.premium_discount.enabled = False   # tested separately below
    b = overnight_session(settings)
    htf = {tf: resample(b.drop(columns="end"), tf) for tf in ("15m", "4h", "1d")}
    d = detect_all(b, htf, settings)
    s = d.setups[(d.setups.setup_type == "sweep_mss_fvg")].iloc[0]
    a_prev = atr(b, 14).shift(1)
    j = b.index.get_loc(pd.Timestamp("2024-06-11 10:00", tz=NY))
    assert s.status == "valid", s.status
    assert s.direction == "short" and "ON_H" in s.levels_swept
    assert s.entry == pytest.approx((18990 + 19002) / 2)              # 50% CE of the FVG
    assert s.stop == pytest.approx(19016 + 0.10 * a_prev.iloc[j])
    assert s.target == 18930 and s.target_kind in ("ASIA_L", "ON_L")
    assert s.armed_at == pd.Timestamp("2024-06-11 10:10", tz=NY)      # FVG completes at 10:05 close
    assert s.rr == pytest.approx((s.entry - 18930) / (s.stop - s.entry))
    assert s.expires_at == s.armed_at + pd.Timedelta(minutes=5 * settings.setups.entry_expiry_bars)
    assert s.session == "ny_am"


def test_min_rr_rejects(settings):
    settings.detectors.premium_discount.enabled = False
    settings.setups.min_rr = 10
    b = overnight_session(settings)
    htf = {tf: resample(b.drop(columns="end"), tf) for tf in ("15m", "4h", "1d")}
    s = detect_all(b, htf, settings).setups
    assert s[s.setup_type == "sweep_mss_fvg"].iloc[0].status == "rr_below_min"


# ----------------------------------------------------------------- premium/discount and bias

def test_premium_discount_position():
    rows = [(h - 1, h, lo, h - 1) for h, lo in
            [(10, 8), (11, 9), (20, 15), (12, 9), (11, 2), (12, 5), (13, 6), (14, 7)]]
    b = mk(rows, tf_min=15)
    si = SwingIndex(find_swings(b, 2), 2)
    t = b["end"].iloc[-1]
    rng = dealing_range_at(si, t)
    assert rng == (2.0, 20.0)
    assert pd_position(15.5, rng) == pytest.approx(0.75)   # premium
    assert dealing_range_at(si, b["end"].iloc[3]) is None   # nothing confirmed yet


def test_bos_direction():
    closes = [10, 11, 15, 11, 10, 12, 16, 12, 9, 7, 5]
    b = mk([(c, c + 0.5, c - 0.5, c) for c in closes])
    d = bos_direction(b, 2).tolist()
    # swing high at idx 2 (15.5) confirmed at 4 -> close 16 at idx 6 breaks it -> bullish
    assert d[5] == 0 and d[6] == 1
    # swing low at idx 4 (9.5) confirmed at idx 6 -> close 9 at idx 8 -> bearish
    assert d[8] == -1 and d[10] == -1


def test_bias_combination(settings):
    frames = {"1d": pd.DataFrame({"known_at": np.array(["2024-06-10T21:00"], dtype="datetime64[ns]"),
                                  "bos_1d": [1.0]}),
              "4h": pd.DataFrame({"known_at": np.array(["2024-06-11T12:00"], dtype="datetime64[ns]"),
                                  "bos_4h": [-1.0]})}
    t = pd.Timestamp("2024-06-11 14:00", tz="UTC")
    assert bias_at(frames, t, 105, 100, settings)["bias"] == "neutral"   # +1 -1 +1 = 1
    frames["4h"]["bos_4h"] = [1.0]
    assert bias_at(frames, t, 105, 100, settings)["bias"] == "bullish"
    early = pd.Timestamp("2024-06-10 20:00", tz="UTC")                    # daily not known yet
    assert bias_at(frames, early, 95, 100, settings)["bos_1d"] == 0


# --------------------------------------------------------------------------- no lookahead

def _key(df, cols):
    return sorted(map(tuple, df[cols].round(6).astype(str).to_numpy())) if len(df) else []


@pytest.mark.parametrize("cut", ["2024-03-06 15:00", "2024-03-12 14:35", "2024-03-20 19:05",
                                 "2024-03-27 16:20"])
def test_detections_identical_on_truncated_history(settings, cut):
    """Everything known at time T must be identical whether or not the future is in the data."""
    m1 = synthetic_m1("2024-02-01", "2024-04-01", seed=11)
    cut = pd.Timestamp(cut, tz="UTC")

    def run(m):
        bars = resample(m, "5m")
        htf = {tf: resample(m, tf) for tf in ("15m", "4h", "1d")}
        return detect_all(bars, htf, settings)

    full = run(m1)
    part = run(m1[m1.index < cut])

    def known(df, col="known_at"):
        return df[df[col] <= cut] if len(df) else df

    assert _key(known(full.swings), ["idx", "kind", "price"]) == _key(known(part.swings), ["idx", "kind", "price"])
    assert _key(known(full.sweeps), ["pen_idx", "reclaim_idx", "level", "extreme"]) == \
        _key(known(part.sweeps), ["pen_idx", "reclaim_idx", "level", "extreme"])
    assert _key(known(full.mss), ["mss_idx", "extreme", "ref_price", "levels_swept"]) == \
        _key(known(part.mss), ["mss_idx", "extreme", "ref_price", "levels_swept"])
    assert _key(known(full.fvgs), ["idx", "bottom", "top"]) == _key(known(part.fvgs), ["idx", "bottom", "top"])
    cols = ["setup_type", "armed_idx", "status", "entry", "stop", "target", "bias", "pd_pos"]
    fs = full.setups[full.setups["armed_at"] <= cut]
    ps = part.setups[part.setups["armed_at"] <= cut]
    assert _key(fs, cols) == _key(ps, cols)


def test_synthetic_run_produces_detections(settings):
    m1 = synthetic_m1("2024-01-01", "2024-04-01", seed=3)
    d = detect_all(resample(m1, "5m"), {tf: resample(m1, tf) for tf in ("15m", "4h", "1d")}, settings)
    assert len(d.swings) and len(d.levels) and len(d.sweeps) and len(d.fvgs)
    assert len(d.mss) > 0 and len(d.setups) > 0
    v = d.valid_setups()
    if len(v):
        short = v.direction == "short"
        assert (v.loc[short, "stop"] > v.loc[short, "entry"]).all()
        assert (v.loc[short, "target"] < v.loc[short, "entry"]).all()
        assert (v.loc[~short, "stop"] < v.loc[~short, "entry"]).all()
        assert (v.rr >= settings.setups.min_rr).all()


def test_setups_appear_exactly_when_armed(settings):
    """Cut the history exactly at each setup's arming time: the setup must already exist, with
    identical numbers. Cut one bar earlier: it must not exist yet."""
    m1 = synthetic_m1("2024-02-01", "2024-03-15", seed=11)

    def run(m):
        return detect_all(resample(m, "5m"), {tf: resample(m, tf) for tf in ("15m", "4h", "1d")},
                          settings).setups

    full = run(m1)
    armed = full[full["armed_at"].notna()].head(6)
    assert len(armed) >= 3
    cols = ["setup_type", "direction", "status", "entry", "stop", "target", "bias"]
    for _, s in armed.iterrows():
        at = run(m1[m1.index < s.armed_at])
        match = at[(at.armed_idx == s.armed_idx) & (at.setup_type == s.setup_type)]
        assert len(match) == 1, f"setup armed at {s.armed_at} missing when history ends there"
        assert _key(match, cols) == _key(full.loc[[s.name]], cols)
        before = run(m1[m1.index < s.armed_at - pd.Timedelta(minutes=5)])
        early = before[(before.armed_idx == s.armed_idx) & (before.setup_type == s.setup_type)]
        assert early.empty, f"setup armed at {s.armed_at} visible one bar early"
