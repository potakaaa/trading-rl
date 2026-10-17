"""RL trade filter: reward/env rules, no-lookahead features, and that it FINDS a planted edge."""
import numpy as np
import pandas as pd
import pytest

from src.ml.rl_filter import (
    ACTIONS,
    MARKET_FEATURES,
    _recent_strategy_r,
    fold_summary,
    reward,
    run_policy,
    train,
)


def test_reward_loss_aversion_and_day_stop():
    assert reward(1.0, 2.0, 0.0) == (2.0, 2.0)
    assert reward(1.0, -1.0, 0.0) == (pytest.approx(-1.25), -1.0)
    assert reward(0.5, -1.0, 0.0)[0] == pytest.approx(-0.625)
    rew, after = reward(1.0, -1.0, -1.5)        # crosses -2 -> extra -1
    assert after == -2.5 and rew == pytest.approx(-2.25)
    assert reward(1.0, -1.0, -2.5)[0] == pytest.approx(-1.25)   # penalty only once
    assert reward(0.0, -5.0, 0.0) == (0.0, 0.0)


def test_recent_strategy_r_uses_only_closed_trades():
    t = pd.to_datetime(["2024-01-01 15:00", "2024-01-01 15:30", "2024-01-01 16:00"], utc=True)
    c = pd.DataFrame({"setup_type": "x", "armed_at": t, "filled": True, "r": [1.0, -1.0, 3.0],
                      "exit_time": pd.to_datetime(["2024-01-01 15:40", "2024-01-01 15:50", "2024-01-01 17:00"],
                                                  utc=True)})
    v = _recent_strategy_r(c)
    assert np.isnan(v[0]) and np.isnan(v[1])        # nothing had closed yet at 15:00 / 15:30
    assert v[2] == pytest.approx(0.0)               # (1 - 1) / 2; the 3R trade is still open


def synthetic_candidates(n_days=500, seed=0):
    """3 candidates/day; candidates with feature `ret_1h_atr` > 0 win +1.5R 70%, others lose."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=d)
        for k in range(3):
            x = rng.normal()
            good = x > 0
            r = (1.5 if rng.random() < 0.7 else -1.0) if good else (1.0 if rng.random() < 0.25 else -1.0)
            row = {f: rng.normal() for f in MARKET_FEATURES}
            row.update(ret_1h_atr=x, setup_type="s", trading_day=day.date(), filled=True, r=r, r_2x=r - 0.05,
                       armed_at=day + pd.Timedelta(hours=14, minutes=30 * k))
            rows.append(row)
    return pd.DataFrame(rows)


def test_rl_finds_a_planted_edge_out_of_sample():
    c = synthetic_candidates()
    tr = c[c["armed_at"] < pd.Timestamp("2021-01-01", tz="UTC")]
    te = c[c["armed_at"] >= pd.Timestamp("2021-01-01", tz="UTC")]
    pol = train(tr)
    res = run_policy(pol, te)
    taken = res[res["action"] > 0]
    good_share = (te.loc[taken["idx"], "ret_1h_atr"] > 0).mean()
    s_rl, s_all = fold_summary(res), fold_summary(run_policy("all", te))
    assert good_share > 0.85, good_share                # it learned WHICH trades are good
    assert s_rl["r_per_unit"] > 0.4 and s_rl["total_r"] > s_all["total_r"]


def test_rl_learns_to_skip_pure_noise():
    rng = np.random.default_rng(1)
    c = synthetic_candidates(seed=2)
    c["r"] = np.where(rng.random(len(c)) < 0.4, 1.0, -1.0)    # negative expectancy, no signal
    pol = train(c[c["armed_at"] < pd.Timestamp("2021-01-01", tz="UTC")])
    res = run_policy(pol, c[c["armed_at"] >= pd.Timestamp("2021-01-01", tz="UTC")])
    assert (res["action"] > 0).mean() < 0.15                 # mostly skips
    assert set(res["action"].unique()) <= set(ACTIONS)


def test_state_only_counts_closed_trades():
    """Regression: a trade still open at the next decision must not show up in the state."""
    from src.ml.rl_filter import known_r
    taken = np.array([-3.0, 1.0])
    ns = lambda s: pd.Timestamp(s, tz="UTC").value  # noqa: E731
    exits = np.array([ns("2024-01-05 15:00"), ns("2024-01-02 16:00")])
    assert known_r(taken, exits, 2, ns("2024-01-03 14:30")) == pytest.approx(1.0)   # the -3R trade is still open
    assert known_r(taken, exits, 2, ns("2024-01-05 15:00")) == pytest.approx(-2.0)
    assert known_r(taken, exits, 0, ns("2024-01-05 15:00")) == 0.0


def test_policy_state_never_sees_open_trade_outcomes():
    """Two candidates; the first is a long-running loser. The policy's 2nd decision must be identical
    whether that loser ends -5R or +5R, because it is still open at the 2nd decision."""
    from src.ml.rl_filter import QPolicy
    seen = []

    class Spy(QPolicy):
        def act(self, x):
            seen.append(x[-2])
            return 1.0, 1.0

    def frame(r0):
        t0 = pd.Timestamp("2024-01-02 14:30", tz="UTC")
        rows = []
        for k, (r, hours) in enumerate([(r0, 72), (1.0, 2)]):
            row = {f: 0.0 for f in MARKET_FEATURES}
            row.update(armed_at=t0 + pd.Timedelta(hours=k), exit_time=t0 + pd.Timedelta(hours=k + hours),
                       r=r, setup_type="s", filled=True, trading_day="w1")
            rows.append(row)
        return pd.DataFrame(rows)

    for r0 in (-5.0, 5.0):
        run_policy(Spy(None, []), frame(r0))
    assert seen == [0.0, 0.0, 0.0, 0.0]
