"""Phase 3 tests: sizing, prop guard, 1m fill simulation, news handling, no-lookahead, metrics, Monte Carlo."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import metrics
from src.backtest.engine import _deadlines, next_blackout_start, run_guarded, run_raw
from src.backtest.fills import M1, Costs, simulate
from src.backtest.montecarlo import day_samples, simulate_phase
from src.news.calendar import BlackoutStatus, NewsCalendar
from src.risk.guard import AccountState, Limits, SignalCandidate, evaluate
from src.risk.sizing import position_size
from src.timeutils import session_end_utc

NY = "America/New_York"
OK_NEWS = BlackoutStatus(available=True, blackout=False)


def ny(s):
    return pd.Timestamp(s, tz=NY).tz_convert("UTC")


def m1_frame(rows, start="2024-06-11 10:00", unit="ms"):
    """rows: (o, h, l, c) per minute, start in NY wall time. Index stored in ms like parquet."""
    idx = pd.date_range(ny(start), periods=len(rows), freq="1min").as_unit(unit)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


NO_COST = Costs(spread=0.0, stop_slippage=0.0)


# ------------------------------------------------------------------------------------- sizing

def test_sizing_rounds_down(settings):
    # 2500 x 0.5% = 12.50; stop 34 pts x $1 -> 0.3676 lots -> 0.36
    r = position_size(34.0, settings)
    assert r.lots == pytest.approx(0.36) and r.risk_usd == pytest.approx(12.24)
    assert r.risk_usd <= r.max_risk_usd


def test_sizing_exact_multiple_not_lost_to_float(settings):
    settings.broker.lot_value_per_point = 1.0
    r = position_size(12.5 / 0.3, settings)       # exactly 0.30 lots
    assert r.lots == pytest.approx(0.30)


def test_sizing_blocks_below_min_lot_never_rounds_up(settings):
    settings.broker.lot_value_per_point = 20.0
    r = position_size(100.0, settings)             # 12.5 / 2000 = 0.00625 < 0.01
    assert r.lots == 0 and r.blocked_reason == "below_min_lot"


def test_sizing_invalid_stop(settings):
    assert position_size(0.0, settings).blocked_reason == "invalid_stop_distance"


# -------------------------------------------------------------------------------------- guard

def sig(t="2024-06-11 10:00", session="ny_am", stop=20.0):
    return SignalCandidate(time=ny(t), session=session, stop_pts=stop, direction="long")


def fresh(settings):
    st = AccountState.fresh(Limits.from_settings(settings))
    st.roll_day(pd.Timestamp("2024-06-11").date())
    return st


def test_guard_allows_clean_signal(settings):
    d = evaluate(sig(), fresh(settings), OK_NEWS, settings)
    assert d.allowed and d.reasons == () and d.size.lots > 0


@pytest.mark.parametrize("news,reason", [
    (None, "news_unavailable"),
    (BlackoutStatus(available=False, blackout=False), "news_unavailable"),
    (BlackoutStatus(available=True, blackout=True), "news_blackout"),
])
def test_guard_news_fail_safe(settings, news, reason):
    d = evaluate(sig(), fresh(settings), news, settings)
    assert not d.allowed and reason in d.reasons


def test_guard_session_rules(settings):
    assert "outside_session" in evaluate(sig(session=None), fresh(settings), OK_NEWS, settings).reasons
    late = evaluate(sig("2024-06-11 11:20"), fresh(settings), OK_NEWS, settings)   # 10 min left < 15
    assert "too_close_to_session_end" in late.reasons


def test_guard_daily_buffer_and_firm_limit(settings):
    st = fresh(settings)
    lim = st.limits
    assert lim.daily_limit_usd == pytest.approx(100) and lim.daily_buffer_usd == pytest.approx(50)
    st.record_close(-50.0)
    d = evaluate(sig(), st, OK_NEWS, settings)
    assert "daily_loss_buffer_reached" in d.reasons
    st2 = fresh(settings)
    st2.day_start_balance = st2.balance + 95          # pretend 95 already lost today (buffer ignored)
    assert "could_breach_daily_limit" in evaluate(sig(), st2, OK_NEWS, settings).reasons


def test_guard_consecutive_losses_and_caps(settings):
    st = fresh(settings)
    st.record_close(-5)
    st.record_close(-5)
    assert "consecutive_losses_today" in evaluate(sig(), st, OK_NEWS, settings).reasons
    st = fresh(settings)
    st.record_signal("ny_am")
    st.record_signal("ny_am")
    d = evaluate(sig(), st, OK_NEWS, settings)
    assert "max_signals_per_session" in d.reasons and "max_signals_per_day" not in d.reasons
    st.record_signal("ny_pm")
    assert "max_signals_per_day" in evaluate(sig("2024-06-11 14:00", "ny_pm"), st, OK_NEWS, settings).reasons


def test_guard_new_day_resets_daily_counters_not_drawdown(settings):
    st = fresh(settings)
    st.record_close(-30)
    st.record_close(-30)
    st.roll_day(pd.Timestamp("2024-06-12").date())
    assert st.consecutive_losses == 0 and st.day_loss == 0 and st.drawdown == pytest.approx(60)


def test_guard_drawdown_halt_needs_manual_reset(settings):
    st = fresh(settings)
    assert st.limits.dd_buffer_usd == pytest.approx(140)     # 70% of 8% of 2500
    st.record_close(-140)
    assert st.halted
    st.roll_day(pd.Timestamp("2024-06-12").date())
    assert "halted_max_drawdown_buffer" in evaluate(sig(), st, OK_NEWS, settings).reasons
    st.manual_reset()
    assert "halted_max_drawdown_buffer" not in evaluate(sig(), st, OK_NEWS, settings).reasons


def test_guard_confidence_threshold(settings):
    st = fresh(settings)
    assert "confidence_below_threshold" in evaluate(sig(), st, OK_NEWS, settings, 0.4, 0.55).reasons
    assert evaluate(sig(), st, OK_NEWS, settings, 0.6, 0.55).allowed


def test_guard_reports_all_reasons(settings):
    st = fresh(settings)
    st.record_close(-60)
    st.record_close(-1)
    d = evaluate(sig(session=None), st, BlackoutStatus(available=True, blackout=True), settings)
    assert {"news_blackout", "outside_session", "daily_loss_buffer_reached",
            "consecutive_losses_today"} <= set(d.reasons)


# ------------------------------------------------------------------------------------ fills

def run_sim(rows, direction, entry, stop, target, costs=NO_COST, order_min=30, flat_min=None):
    m = M1.from_frame(m1_frame(rows))
    armed = ny("2024-06-11 10:00")
    od = armed + pd.Timedelta(minutes=order_min)
    fd = armed + pd.Timedelta(minutes=flat_min or len(rows))
    return simulate(m, direction, entry, stop, target, armed, od, fd, costs)


def test_long_target():
    rows = [(105, 106, 104, 105), (104, 104, 99.5, 101), (101, 108, 100.5, 107), (107, 111, 106, 110)]
    f = run_sim(rows, "long", 100, 95, 110)
    assert f.outcome == "tp" and f.fill_time == ny("2024-06-11 10:01") and f.pnl_pts == 10
    assert f.exit_time == ny("2024-06-11 10:04")


def test_short_stop_with_spread_and_slippage():
    rows = [(98, 100.5, 97, 99), (99, 103, 98, 102)]       # ask high on bar 2 = 104 >= 103.5 stop
    f = run_sim(rows, "short", 100, 103.5, 90, Costs(spread=1.0, stop_slippage=0.5))
    assert f.outcome == "sl" and f.exit_price == pytest.approx(104.0) and f.pnl_pts == pytest.approx(-4.0)


def test_long_entry_needs_ask_to_reach_limit():
    rows = [(101, 102, 99.5, 101)] * 3                      # bid low 99.5, ask low 101.5 > 100
    assert run_sim(rows, "long", 100, 95, 110, Costs(2.0, 0)).outcome == "expired"
    assert run_sim(rows, "long", 100, 95, 110, Costs(0.4, 0)).filled


def test_same_bar_stop_and_target_is_a_loss():
    rows = [(101, 101, 99.8, 100), (100, 111, 94, 100)]
    assert run_sim(rows, "long", 100, 95, 110).outcome == "sl"


def test_fill_bar_target_is_ignored_but_stop_counts():
    tgt_on_fill = [(111, 112, 99, 105), (105, 106, 104, 105)]   # touches 110 and 100 on one bar
    f = run_sim(tgt_on_fill, "long", 100, 95, 110, flat_min=2)
    assert f.outcome == "session_end"
    stop_on_fill = [(101, 101, 94, 96)]
    assert run_sim(stop_on_fill, "long", 100, 95, 110).outcome == "sl"


def test_gap_through_stop_fills_at_open():
    rows = [(101, 101, 99.9, 100), (92, 93, 91, 92)]
    f = run_sim(rows, "long", 100, 95, 110, Costs(0, 0.5))
    assert f.outcome == "sl" and f.exit_price == pytest.approx(91.5)


def test_flatten_at_deadline_uses_last_bar_close_and_pays_spread():
    rows = [(101, 101, 99, 100), (100, 102, 99, 101), (101, 103, 100, 102), (102, 104, 101, 103)]
    f = run_sim(rows, "short", 100.5, 110, 80, Costs(1.0, 0), flat_min=3)
    assert f.outcome == "session_end" and f.exit_price == pytest.approx(103.0)   # close 102 + spread
    assert f.exit_time == ny("2024-06-11 10:03")


def test_order_expires_without_touch():
    rows = [(105, 106, 104, 105)] * 10
    assert run_sim(rows, "long", 100, 95, 110, order_min=5).outcome == "expired"


def test_fill_ignores_bars_before_armed_and_after_exit():
    """No lookahead: changing data before the signal or after the exit changes nothing."""
    pre = [(100, 100, 90, 95)] * 5                                      # would fill+stop if seen
    rows = [(105, 106, 104, 105), (104, 104, 99.5, 101), (101, 111, 100.5, 110), (110, 111, 109, 110)]
    m_a = M1.from_frame(m1_frame(pre + rows, start="2024-06-11 09:55"))
    rows_b = rows[:3] + [(110, 300, 1, 110)]                            # wild bar after the exit
    m_b = M1.from_frame(m1_frame(pre + rows_b, start="2024-06-11 09:55"))
    armed = ny("2024-06-11 10:00")
    args = ("long", 100, 95, 110, armed, armed + pd.Timedelta("30min"), armed + pd.Timedelta("60min"), NO_COST)
    fa, fb = simulate(m_a, *args), simulate(m_b, *args)
    assert fa.outcome == fb.outcome == "tp" and fa.exit_time == fb.exit_time and fa.mfe_pts == fb.mfe_pts


def test_ns_and_ms_indexes_give_same_result():
    rows = [(105, 106, 104, 105), (104, 104, 99.5, 101), (101, 111, 100.5, 110)]
    armed = ny("2024-06-11 10:00")
    args = ("long", 100, 95, 110, armed, armed + pd.Timedelta("30min"), armed + pd.Timedelta("60min"), NO_COST)
    a = simulate(M1.from_frame(m1_frame(rows, unit="ms")), *args)
    b = simulate(M1.from_frame(m1_frame(rows, unit="ns")), *args)
    assert a.outcome == b.outcome == "tp" and a.exit_time == b.exit_time


# ------------------------------------------------------------------------- news + deadlines

def calendar(settings, *ny_times):
    ev = pd.DataFrame({"ts": [ny(t) for t in ny_times], "title": ["CPI"] * len(ny_times)})
    return NewsCalendar.from_settings(ev, settings)


def setup_row(armed="2024-06-11 10:00", expires="2024-06-11 10:30", direction="long",
              entry=100.0, stop=95.0, target=110.0, session="ny_am", setup_type="sweep_mss_fvg"):
    return {"status": "valid", "setup_type": setup_type, "direction": direction, "session": session,
            "armed_at": ny(armed), "expires_at": ny(expires), "entry": entry, "stop": stop,
            "target": target, "rr": (target - entry) / (entry - stop), "risk_pts": abs(entry - stop)}


def test_next_blackout_start(settings):
    cal = calendar(settings, "2024-06-11 10:00", "2024-06-11 14:00")
    assert next_blackout_start(cal, ny("2024-06-11 09:00")) == ny("2024-06-11 09:52")
    assert next_blackout_start(cal, ny("2024-06-11 09:55")) == ny("2024-06-11 13:52")   # already inside


def test_news_cuts_order_and_position(settings):
    cal = calendar(settings, "2024-06-11 10:20")                       # blackout 10:12-10:28
    s = pd.Series(setup_row())
    armed, od, fd, orr, frr = _deadlines(s, settings, cal)
    assert od == fd == ny("2024-06-11 10:11") and orr == "news_cancel" and frr == "news_flat"
    rows = [(101, 101, 99.9, 100)] + [(101, 102, 100.5, 101)] * 20
    m = M1.from_frame(m1_frame(rows))
    f = simulate(m, "long", 100, 95, 110, armed, od, fd, NO_COST, orr, frr)
    assert f.outcome == "news_flat" and f.exit_time == ny("2024-06-11 10:11")


def test_session_end_deadline_dst(settings):
    assert session_end_utc(ny("2024-01-10 10:00"), "11:30") == pd.Timestamp("2024-01-10 16:30", tz="UTC")
    assert session_end_utc(ny("2024-07-10 10:00"), "11:30") == pd.Timestamp("2024-07-10 15:30", tz="UTC")
    # day of the March switch (Sunday 2024-03-10) and the Monday after
    assert session_end_utc(ny("2024-03-11 10:00"), "16:00") == pd.Timestamp("2024-03-11 20:00", tz="UTC")
    assert session_end_utc(ny("2024-11-04 10:00"), "16:00") == pd.Timestamp("2024-11-04 21:00", tz="UTC")


# ----------------------------------------------------------------------- engine (account)

def losing_day_m1():
    """10:00-11:30 NY: price dips to 99 (fills a 100 long), then falls to 90 (stops 95)."""
    rows = []
    for k in range(90):
        rows.append((100.5, 101, 99, 100) if k % 30 == 1 else (96, 97, 90, 92) if k % 30 == 3 else
                    (101, 101.5, 100.5, 101))
    return M1.from_frame(m1_frame(rows))


def test_guarded_stops_after_two_consecutive_losses(settings):
    setups = pd.DataFrame([setup_row("2024-06-11 10:00", "2024-06-11 10:02"),
                           setup_row("2024-06-11 10:30", "2024-06-11 10:32"),
                           setup_row("2024-06-11 11:00", "2024-06-11 11:02")])
    cal = calendar(settings, "2024-06-20 08:30")
    res = run_guarded(setups, losing_day_m1(), settings, NO_COST, cal)
    assert list(res.trades["outcome"]) == ["sl", "sl"]
    assert "consecutive_losses_today" in res.blocks.iloc[0]["reasons"]
    assert res.trades["pnl_usd"].sum() == pytest.approx(-2 * 0.5 / 100 * 2500)


def test_guarded_blocks_inside_news_and_raw_ignores_news(settings):
    setups = pd.DataFrame([setup_row("2024-06-11 10:00", "2024-06-11 10:02")])
    cal = calendar(settings, "2024-06-11 10:05")
    res = run_guarded(setups, losing_day_m1(), settings, NO_COST, cal)
    assert res.trades.empty and "news_blackout" in res.blocks.iloc[0]["reasons"]
    raw = run_raw(setups, losing_day_m1(), settings, NO_COST)
    assert raw.iloc[0]["outcome"] == "sl" and raw.iloc[0]["r"] == pytest.approx(-1.0)


# --------------------------------------------------------------------------------- metrics

def tr(rs):
    return pd.DataFrame({"r": rs, "filled": True, "hit_1r": False, "hit_2r": False, "hit_3r": False})


def test_summary_known_values():
    s = metrics.summary(tr([2.0, -1.0, -1.0, 3.0, -1.0]), months=5)
    assert s["win_rate"] == pytest.approx(0.4) and s["expectancy_r"] == pytest.approx(0.4)
    assert s["profit_factor"] == pytest.approx(5 / 3) and s["max_drawdown_r"] == pytest.approx(2.0)
    assert s["longest_losing_streak"] == 2 and s["trades_per_month"] == pytest.approx(1.0)


def test_max_drawdown_from_start():
    assert metrics.max_drawdown(np.cumsum([-1, -1, 3, -2])) == pytest.approx(2.0)


# ------------------------------------------------------------------------------ Monte Carlo

def test_mc_always_winning_passes_and_always_losing_fails(settings):
    lim = Limits.from_settings(settings)
    rng = np.random.default_rng(0)
    win = pd.DataFrame({"pnl": [25.0], "worst": [0.0], "traded": [True]})
    p = simulate_phase(win, lim, 8.0, 2, 100, 500, rng)
    assert p.p_pass == 1.0 and p.median_days_to_pass == 8           # 200 / 25
    lose = pd.DataFrame({"pnl": [-25.0], "worst": [-25.0], "traded": [True]})
    p = simulate_phase(lose, lim, 8.0, 2, 100, 500, rng)
    assert p.p_pass == 0 and p.p_halt == 1.0                        # internal halt at -140 comes first
    blow = pd.DataFrame({"pnl": [-10.0], "worst": [-150.0], "traded": [True]})
    assert simulate_phase(blow, lim, 8.0, 2, 100, 500, rng).p_breach == 1.0


def test_mc_min_trading_days(settings):
    lim = Limits.from_settings(settings)
    big = pd.DataFrame({"pnl": [300.0], "worst": [0.0], "traded": [True]})
    p = simulate_phase(big, lim, 8.0, 2, 10, 100, np.random.default_rng(0))
    assert p.median_days_to_pass == 2                               # target hit day 1, 2 days needed


def test_day_samples_pad_no_trade_days():
    t = pd.DataFrame({"filled": [True, True], "fill_time": [ny("2024-06-11 10:00"), ny("2024-06-11 10:30")],
                      "trading_day": [pd.Timestamp("2024-06-11").date()] * 2, "pnl_usd": [-12.5, 25.0],
                      "mae_r": [1.0, 0.2], "risk_usd": [12.5, 12.5]})
    d = day_samples(t, 5)
    assert len(d) == 5 and d["traded"].sum() == 1
    row = d[d.traded].iloc[0]
    assert row.pnl == pytest.approx(12.5) and row.worst == pytest.approx(-15.0)   # -12.5 then -2.5 MAE


# ------------------------------------------------------------ review regressions (Phase 3)

def test_mae_on_stop_bar_is_the_realized_exit_not_the_spike():
    rows = [(101, 101, 99.9, 100), (100, 100, 50, 60)]       # stop bar spikes to 50
    f = run_sim(rows, "long", 100, 95, 110, Costs(0, 1.0))
    assert f.outcome == "sl" and f.pnl_pts == pytest.approx(-6) and f.mae_pts == pytest.approx(6)


def test_fill_bar_high_does_not_count_as_favourable_excursion():
    rows = [(125, 125, 100, 101), (101, 101, 94, 95)]        # high before the fill, then stopped
    f = run_sim(rows, "long", 100, 95, 110)
    assert f.outcome == "sl" and f.mfe_pts == 0


def test_breach_logged_once_and_account_stops(settings):
    """A gap through the stop loses more than the firm daily limit: one breach event, the account
    stops for the day, and the backtest starts a fresh account the next day."""
    setups = pd.DataFrame([setup_row("2024-06-11 10:00", "2024-06-11 10:02"),
                           setup_row("2024-06-11 10:30", "2024-06-11 10:32"),
                           setup_row("2024-06-12 10:00", "2024-06-12 10:02")])
    day1 = [(101, 101, 99, 100), (100, 100, 99.5, 100), (50, 52, 49, 51)] + [(101, 101.5, 100.5, 101)] * 50
    m = M1.from_frame(pd.concat([m1_frame(day1),
                                 m1_frame([(101, 101.5, 100.5, 101)] * 10, start="2024-06-12 10:00")]))
    res = run_guarded(setups, m, settings, NO_COST, calendar(settings, "2024-07-01 08:30"))
    ev = res.events["event"].tolist()
    assert res.trades.iloc[0]["pnl_usd"] == pytest.approx(-50 * 2.5)          # gapped to 50, 2.5 lots
    assert ev.count("FIRM_DAILY_LIMIT_BREACH") == 1 and "HALT" in ev and "RESET" in ev
    assert len(res.trades) == 2 and res.trades.iloc[-1]["account"] == 2
    assert "halted_max_drawdown_buffer" in res.blocks.iloc[0]["reasons"]


def test_mc_detects_intraday_halt(settings):
    lim = Limits.from_settings(settings)
    # every day dips to -90 realized and ends at -60: day 2 dips to -150 (buffer 140) but ends at -120
    d = pd.DataFrame({"pnl": [-60.0], "worst": [-90.0], "realized_low": [-90.0], "traded": [True]})
    p = simulate_phase(d, lim, 8.0, 2, 3, 50, np.random.default_rng(0))
    assert p.p_halt == 1.0


def test_guard_blocks_signal_just_before_blackout(settings):
    st = fresh(settings)
    news = BlackoutStatus(available=True, blackout=False, minutes_to_next=9.0)   # window starts in 1 min
    assert "news_imminent" in evaluate(sig(), st, news, settings).reasons
    news_far = BlackoutStatus(available=True, blackout=False, minutes_to_next=30.0)
    assert evaluate(sig(), st, news_far, settings).allowed


# ------------------------------------------------------------------- research (Phase 3b)

def test_min_rr_below_one_is_rejected_by_config(settings):
    from pydantic import ValidationError

    from src.backtest.research import apply_overrides
    with pytest.raises(ValidationError):
        apply_overrides(settings, {"setups.min_rr": 0.8})


def test_apply_overrides_unknown_key(settings):
    from src.backtest.research import apply_overrides
    with pytest.raises(KeyError):
        apply_overrides(settings, {"setups.not_a_setting": 1})


def test_research_variants_all_valid(settings):
    from src.backtest.research import apply_overrides
    for name, ov in settings.research.variants.items():
        assert apply_overrides(settings, ov).setups.min_rr >= 1.0, name
