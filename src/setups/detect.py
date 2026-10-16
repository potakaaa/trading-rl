"""Rule-based setup detection: the single entry point shared by backtest, charts and live engine.

detect_all(bars, htf, settings) runs every detector on the setup timeframe and returns all
intermediate detections plus a `setups` table. Every setup candidate is kept, with
status == "valid" or the reason it was rejected, so charts and logs can show why.

Setup 1, sweep_mss_fvg: liquidity sweep inside a session -> MSS with displacement -> limit entry
  in the fair value gap created by the displacement leg (FVG start or 50% CE).
Setup 2, ob_retest: same sweep -> MSS chain -> limit entry on a return to the order block, only
  in the correct premium/discount half and aligned with HTF bias.
Both: stop beyond the sweep extreme + buffer; target = nearest opposing liquidity pool,
  minimum R:R from config. Orders expire after entry_expiry_bars or at session end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Settings
from src.features.basics import SwingIndex, atr, find_swings
from src.features.liquidity import equal_levels, find_sweeps, session_levels
from src.features.structure import (
    bias_at,
    dealing_range_at,
    find_fvgs,
    find_mss,
    find_order_block,
    htf_bias_series,
    pd_position,
)
from src.timeutils import TF_MINUTES, session_end_utc, session_of, trading_date

SETUP_COLUMNS = [
    "setup_type", "direction", "status", "session", "armed_idx", "armed_at", "expires_at",
    "entry", "stop", "target", "target_kind", "rr", "risk_pts",
    "levels_swept", "sweep_level_kind", "sweep_level", "sweep_session", "depth_atr",
    "pen_idx", "extreme", "extreme_idx", "mss_idx", "mss_ref_price", "disp_body_atr", "atr",
    "fvg_idx", "fvg_top", "fvg_bottom", "fvg_size_atr", "ob_idx", "ob_top", "ob_bottom", "ob_size_atr",
    "zone_top", "zone_bottom", "pd_pos", "bias", "bias_score", "bos_1d", "bos_4h", "vs_pd_mid",
]


@dataclass
class Detections:
    bars: pd.DataFrame
    swings: pd.DataFrame
    levels: pd.DataFrame
    sweeps: pd.DataFrame
    mss: pd.DataFrame
    fvgs: pd.DataFrame
    setups: pd.DataFrame
    htf_swings: pd.DataFrame = field(default_factory=pd.DataFrame)
    bias_frames: dict = field(default_factory=dict)

    def valid_setups(self) -> pd.DataFrame:
        return self.setups[self.setups["status"] == "valid"] if len(self.setups) else self.setups


def _session_end_utc(t: pd.Timestamp, session_end: str) -> pd.Timestamp:
    return session_end_utc(t, session_end)


def detect_all(bars: pd.DataFrame, htf: dict[str, pd.DataFrame], settings: Settings) -> Detections:
    det = settings.detectors
    tf = det.setup_timeframe
    bars = bars.copy()
    a = atr(bars, det.atr_period)
    bars["atr"] = a
    bars["session"] = session_of(bars.index, settings.sessions).to_numpy()

    swings = find_swings(bars, det.swing_n[tf])
    sidx = SwingIndex(swings, det.swing_n[tf])
    levels = pd.concat([session_levels(bars, settings), equal_levels(bars, swings, a, settings)],
                       ignore_index=True)
    levels, sweeps = find_sweeps(bars, levels, a, settings)
    if len(sweeps):
        in_sess = bars["session"].to_numpy()[sweeps["pen_idx"].to_numpy()]
        sweeps["session"] = in_sess
        sess_sweeps = sweeps[pd.notna(sweeps["session"])]
    else:
        sess_sweeps = sweeps
    mss = find_mss(bars, sess_sweeps, sidx, a, settings)
    fvgs = find_fvgs(bars, a, settings)

    rng_tf = det.premium_discount.range_timeframe
    htf_sw = find_swings(htf[rng_tf], det.swing_n[rng_tf]) if rng_tf in htf else pd.DataFrame()
    htf_sidx = SwingIndex(htf_sw, det.swing_n[rng_tf]) if len(htf_sw) else None
    bias_frames = htf_bias_series(htf, settings)

    setups = _build_setups(bars, mss, fvgs, levels, htf_sidx, bias_frames, settings)
    return Detections(bars=bars, swings=swings, levels=levels, sweeps=sweeps, mss=mss, fvgs=fvgs,
                      setups=setups, htf_swings=htf_sw, bias_frames=bias_frames)


def _build_setups(bars, mss, fvgs, levels, htf_sidx, bias_frames, settings: Settings) -> pd.DataFrame:
    cfg = settings.setups
    det = settings.detectors
    if mss is None or mss.empty:
        return pd.DataFrame(columns=SETUP_COLUMNS)
    h, lo, c = (bars[k].to_numpy() for k in ("high", "low", "close"))
    end = pd.DatetimeIndex(bars["end"])
    sess = [x if isinstance(x, str) else None for x in bars["session"]]
    n = len(bars)
    bar_len = pd.Timedelta(minutes=TF_MINUTES[det.setup_timeframe])
    td_bars = trading_date(bars.index, settings.data.trading_day_start)
    pd_levels = levels[levels["kind"].isin(["PDH", "PDL"])].set_index(["trading_date", "kind"])["price"]
    rows = []

    for _, m in mss.iterrows():
        short = m["direction"] == "short"
        j = int(m["mss_idx"])
        base = {
            "direction": m["direction"], "mss_idx": j, "pen_idx": int(m["pen_idx"]),
            "extreme": float(m["extreme"]), "extreme_idx": int(m["extreme_idx"]),
            "levels_swept": m["levels_swept"], "sweep_level_kind": m["level_kind"],
            "sweep_level": float(m["level"]), "depth_atr": float(m["depth_atr"]),
            "disp_body_atr": float(m["disp_body_atr"]), "mss_ref_price": float(m["ref_price"]),
            "atr": float(m["atr"]), "sweep_session": sess[int(m["pen_idx"])],
        }
        stop = m["extreme"] + (1 if short else -1) * cfg.stop_buffer_atr * m["atr"]
        ob = find_order_block(bars, m, settings)

        candidates = []
        if "sweep_mss_fvg" in cfg.enabled:
            want = "bear" if short else "bull"
            f = fvgs[(fvgs["direction"] == want) & (fvgs["idx"] >= m["extreme_idx"] + 2) & (fvgs["idx"] <= j + 1)]
            f = f[f["idx"] < n]
            row = dict(base, setup_type="sweep_mss_fvg")
            if f.empty:
                row["status"] = "no_fvg"
                candidates.append((row, None, None))
            else:
                # Decide with what is known at the MSS close: prefer the FVG completed by the MSS
                # candle, else the latest one in the leg. Only if the leg has none yet, use the FVG
                # completed by the next candle (MSS candle in the middle), decided one bar later.
                now = f[f["idx"] <= j]
                if len(now):
                    at_mss = now[now["idx"] == j]
                    fv = at_mss.loc[at_mss["size_atr"].idxmax()] if len(at_mss) else now.loc[now["idx"].idxmax()]
                else:
                    fv = f.loc[f["size_atr"].idxmax()]
                near = fv["bottom"] if short else fv["top"]
                entry = near if cfg.entry_mode == "fvg_start" else (fv["top"] + fv["bottom"]) / 2
                armed = max(j, int(fv["idx"]))
                row.update(fvg_idx=int(fv["idx"]), fvg_top=float(fv["top"]), fvg_bottom=float(fv["bottom"]),
                           fvg_size_atr=float(fv["size_atr"]), zone_top=float(fv["top"]),
                           zone_bottom=float(fv["bottom"]))
                # price must not have reached the entry between FVG formation and arming
                seg = slice(int(fv["idx"]) + 1, armed + 1)
                touched = (h[seg] >= entry).any() if short else (lo[seg] <= entry).any()
                row["status"] = "entry_touched_before_armed" if touched else "pending"
                candidates.append((row, float(entry), armed))

        if "ob_retest" in cfg.enabled:
            row = dict(base, setup_type="ob_retest")
            if ob is None:
                row["status"] = "no_order_block"
                candidates.append((row, None, None))
            else:
                edge = ob["ob_bottom"] if short else ob["ob_top"]
                entry = edge if cfg.ob_entry == "edge" else (ob["ob_top"] + ob["ob_bottom"]) / 2
                row.update(ob_idx=ob["ob_idx"], ob_top=ob["ob_top"], ob_bottom=ob["ob_bottom"],
                           ob_size_atr=(ob["ob_top"] - ob["ob_bottom"]) / m["atr"],
                           zone_top=ob["ob_top"], zone_bottom=ob["ob_bottom"])
                beyond = c[j] < entry if short else c[j] > entry
                row["status"] = "pending" if beyond else "price_not_beyond_ob"
                candidates.append((row, float(entry), j))

        for row, entry, armed in candidates:
            if row["status"] != "pending":
                rows.append(row)
                continue
            armed_at = end[armed]
            row.update(armed_idx=armed, armed_at=armed_at, entry=entry, stop=float(stop),
                       risk_pts=abs(entry - stop), session=sess[armed] if armed < n else None)
            if ob is not None and "ob_size_atr" not in row:
                row["ob_size_atr"] = (ob["ob_top"] - ob["ob_bottom"]) / m["atr"]
            # context: premium/discount + HTF bias at arming time
            rng = dealing_range_at(htf_sidx, armed_at) if htf_sidx is not None else None
            row["pd_pos"] = pd_position(entry, rng)
            td = td_bars[armed]
            try:
                pdh, pdl = pd_levels.loc[(td, "PDH")], pd_levels.loc[(td, "PDL")]
                pd_mid = (pdh + pdl) / 2
            except KeyError:
                pd_mid = None
            row.update(bias_at(bias_frames, armed_at, c[armed], pd_mid, settings))
            row["status"] = _validate(row, short, entry, stop, armed, armed_at, levels, settings)
            # expiry
            sess_name = row["session"]
            if sess_name is not None:
                s_end = _session_end_utc(armed_at, settings.sessions[sess_name].end)
                row["expires_at"] = min(armed_at + cfg.entry_expiry_bars * bar_len, s_end)
            rows.append(row)

    df = pd.DataFrame(rows).reindex(columns=SETUP_COLUMNS)
    for col in ("armed_at", "expires_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df.sort_values(["mss_idx", "setup_type"]).reset_index(drop=True)


def _validate(row: dict, short: bool, entry: float, stop: float, armed: int, armed_at: pd.Timestamp,
              levels: pd.DataFrame, settings: Settings) -> str:
    cfg = settings.setups
    if row["session"] is None:
        return "armed_outside_session"
    s_end = _session_end_utc(armed_at, settings.sessions[row["session"]].end)
    if s_end - armed_at < pd.Timedelta(minutes=cfg.min_minutes_to_session_end):
        return "too_close_to_session_end"
    if (short and stop <= entry) or (not short and stop >= entry):
        return "bad_stop"
    pdc = settings.detectors.premium_discount
    if pdc.enabled:
        p = row.get("pd_pos")
        if p is None or (np.isnan(p) if isinstance(p, float) else False):
            return "no_dealing_range"
        if (short and p <= 0.5) or (not short and p >= 0.5):
            return "wrong_premium_discount"
    if row["setup_type"] == "ob_retest" and cfg.require_htf_alignment:
        if row["bias"] != ("bearish" if short else "bullish"):
            return "htf_bias_not_aligned"
    if cfg.target_mode == "fixed_r":
        k = cfg.fixed_target_r
        risk = abs(entry - stop)
        row.update(target=float(entry - k * risk if short else entry + k * risk), target_kind="fixed_r", rr=k)
        return "rr_below_min" if k < cfg.min_rr else "valid"
    # target: opposing liquidity pools that exist and are untouched at arming time
    side = "low" if short else "high"
    lv = levels[(levels["side"] == side) & (levels["known_at"] <= armed_at) & (levels["expires_at"] > armed_at)]
    untouched = lv["consumed_at"].isna() | (lv["consumed_at"] >= armed_at)
    lv = lv[untouched]
    lv = lv[lv["price"] < entry] if short else lv[lv["price"] > entry]
    if lv.empty:
        return "no_target"
    lv = lv.assign(dist=(lv["price"] - entry).abs()).sort_values("dist")
    risk = abs(entry - stop)
    lv = lv.assign(rr=lv["dist"] / risk)
    pick = lv.iloc[0] if cfg.target_mode == "nearest" else (
        lv[lv["rr"] >= cfg.min_rr].iloc[0] if (lv["rr"] >= cfg.min_rr).any() else lv.iloc[0])
    row.update(target=float(pick["price"]), target_kind=pick["kind"], rr=float(pick["rr"]))
    if pick["rr"] < cfg.min_rr:
        return "rr_below_min"
    return "valid"
