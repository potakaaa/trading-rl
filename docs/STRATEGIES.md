# Strategy research round 2: pre-registration (written 2026-10-16, BEFORE any test was run)

This file is committed before any of these strategies is backtested. The rules, parameters and
pass/fail criteria below are frozen. A strategy that fails is reported as failed, never re-tuned
and re-tested on the same data.

## Common rules (same as the live system)
- US100 (Dukascopy BID, 1m execution), NY sessions AM 09:30-11:30 and PM 13:30-16:00 (NY time).
- Flat by the end of the session the trade was opened in. 0.5% risk per trade. Equity Edge limits,
  internal buffers and news blackout apply in the guarded run. Target is never below 1R (Rald's rule).
- Costs: spread 2.0 pts (and a 2x stress run), stop slippage 1 pt. Pessimistic intrabar rules.
- Entries are **market orders at the open of the next 1m bar** after the 5m signal bar closes
  (long pays the spread). R is measured from the actual fill.
- ATR = Wilder ATR(14) on 5m bars, as in the detectors. A trade whose stop is wider than 3 x ATR
  is skipped (avoids huge stops). One trade per strategy per session unless stated.
- Data: research on 2019-01-01 .. 2024-12-31 only. 2025-01-01 onward stays untouched until the
  single final test.

## Strategies
| id | idea | signal (decided at a 5m bar close) | stop | target | exit |
|---|---|---|---|---|---|
| `orb15` | Opening-range breakout | OR = 09:30-09:45 high/low. 09:45-11:00: first 5m close above ORH -> long, below ORL -> short. Max one per day | other side of the OR | 1.5R | 11:30 |
| `orb30` | Same, 30-min OR | OR = 09:30-10:00; signals 10:00-11:00 | other side of the OR | 1.5R | 11:30 |
| `vwap_revert` | VWAP band fade | session VWAP from 09:30 (tick-volume weighted), band = ±2 sd of (close - VWAP) since 09:30 (>= 6 bars). In 10:00-11:15 or 13:45-15:30: the previous bar closed outside the band and this bar closes back inside -> fade toward VWAP | extreme of the last 3 bars ± 0.1 ATR | VWAP (skip if < 1R) | session end |
| `momo_open` | Opening momentum | at the 10:00 close: move = close(10:00) - open(09:30). If abs(move) >= 0.3 x previous day's range (18:00-17:00 day) -> trade in its direction | opposite extreme of 09:30-10:00 | 2R | 11:30 |
| `momo_lasthalf` | Intraday momentum (first half hour predicts last half hour) | at the 15:30 close: direction = sign(close(10:00) - previous 16:00 close); trade if non-zero | 1.5 x ATR from entry | 1R | 16:00 |
| `gap_fill` | Overnight gap fade | gap = open(09:30) - previous 16:00 close. If 0.25 <= abs(gap) / daily ATR(14) <= 1.0: at the 09:35 close fade toward the previous close | beyond the 09:30-09:35 extreme + 0.1 ATR | previous 16:00 close (skip if < 1R) | 11:30 |
| `pd_break` | Prior-day high/low breakout | in either session: first 5m close above PDH (long) / below PDL (short), PDH/PDL of the previous 18:00-17:00 trading day, once per level per day | breakout bar's low (long) / high (short) ± 0.1 ATR | 2R | session end |

## Pass criteria (raw run, 2019-2024, all must hold)
1. at least 60 filled trades
2. expectancy > 0 R **with 2x spread**
3. t-stat of the mean R >= 2.0 at base costs
4. positive total R in at least 4 of the 6 years
Win rate is reported for every strategy but is NOT a pass criterion (a high win rate with small wins
and big losses still loses money).

## RL trade filter (offline reinforcement learning, fitted Q-iteration)
- Candidates: every signal from all 7 strategies plus the ICT setups, each with its raw outcome
  (R) simulated independently.
- Episode = one trading day; candidates in time order. State = market features known at the signal
  time (strategy, direction, time of day, weekday, ATR percentile, gap, distance to VWAP in sd,
  position vs PDH/PDL, first-half-hour move, stop size in ATR, minutes to/since high-impact news)
  plus the account state within the day (R so far today, trades taken today).
- Actions: skip, half size, full size. Reward = size x R - 0.25 x size x max(0, -R) (loss aversion)
  and -1 extra if the day's R falls to -2 or below (daily-limit pressure).
- Q-function: LightGBM regressor on (state, action), 3 fitted-Q iterations, discount 1 within a day.
  The policy takes the best action and only trades if its Q value > 0.
- Walk-forward: expanding training window starting 2019-01, test 6 months, roll 6 months:
  test folds 2021H1 .. 2024H2 (8 folds). Nothing from a test fold is used to train for it.
- Compared, per fold: take-all-candidates, each passing strategy alone, and the RL policy.
- RL passes if its out-of-sample mean R per taken trade > 0 with 2x spread, it beats take-all in
  at least 5 of the 8 folds, and its total out-of-sample R is positive.

## Final test (once)
Every strategy that passes, and the RL filter if it passes, is trained/fixed on 2019-2024 and run
**once** on 2025-01-01 .. latest data. Those numbers go to Rald, who picks what the live paper
engine trades. If nothing passes, that is the result.
