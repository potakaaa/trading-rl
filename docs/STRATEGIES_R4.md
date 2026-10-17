# Research round 4: intraday ML (volatility gate + gradient-boosting direction) - pre-registration
Written 2026-10-17, before any round-4 model was trained or any round-4 label was looked at.

Rald's choices: intraday only, NY AM + PM sessions, 30-120 minute horizon, volatility gate +
LightGBM direction model, neural-network challenger only if the GBM shows signal, final test on the
last 6 months, then paper trading.

## Data and periods
- US100 Dukascopy BID 1m bars (same feed as live). Research period: 2019-01-01 .. 2026-04-15.
  **Final test: 2026-04-16 .. 2026-10-15**, used once at the very end. Neither the 2025-26 holdout
  rule nor round 1-3 results are used to choose anything here.
- Costs as everywhere: spread 2.0 pts, stop slippage 1.0 pt (2x stress = 4.0 / 2.0). Market entry at
  the next 1m open after the decision time (long pays the spread); pessimistic intrabar rules.

## The trade being predicted (fixed, identical for every model)
- **Decision times (NY):** 09:45, 10:00, 10:15, 10:30 and 13:45, 14:00, 14:15, 14:30, 14:45, 15:00.
  That is 10 per day, each with 60 minutes to the session end or more.
- **Direction:** long or short, chosen by the model or skipped.
- **Stop:** 1.0 x ATR(14) of 15-minute bars at the decision time. **Target:** 1.5R (Rald's rule: >= 1R).
- **Time exit:** 60 minutes after entry, at session end at the latest.
- **Label:** the net R of that exact trade, simulated on 1m bars with costs (base and 2x), for both
  directions. A second label, `range60`, is (high - low over the next 60 minutes) / ATR15, the realized activity.

## Features (computed only from data complete at the decision time)
- **Price moves:** returns over the last 5/15/30/60/120 minutes in ATR15 units.
- **Location:** position vs the session open, the session VWAP (tick-volume weighted), previous-day high/low,
  and the overnight (18:00-09:30) high/low, all in ATR15 units.
- **Volatility and range:** realized volatility over the last 30/60 minutes; ATR15 / daily ATR(14);
  today's range so far / daily ATR; tick-volume ratio (last 30 min vs the 20-day average at that time).
- **Calendar and news:** minute of day, weekday, overnight gap / daily ATR, previous day's return / daily ATR,
  minutes to and since the nearest high-impact USD event.

## Models
- **Stage 1, volatility gate:** a LightGBM regressor predicts log(range60). A decision is tradable only if
  its predicted range60 is above the **median of predictions on its training window** (only the more
  active half of the moments).
- **Stage 2, direction:** two LightGBM regressors predict the net R of the long and of the short trade.
  Among gated decisions, trade the side with the higher prediction if it exceeds threshold tau.
  tau is chosen **on the training window only**: its last 12 months are held out inside the window, and
  tau is picked from {0.00, 0.05, 0.10, 0.15, 0.20, 0.30} to maximise mean R with at least 100 trades
  there. The models are then refit on the full window.
- Conservative LightGBM settings: 300 trees, learning rate 0.03, 15 leaves, min 100 samples per leaf,
  0.8 row/column subsampling, L2 = 1.
- **One position at a time:** while a trade is open, later decision times are skipped. The usual
  guard caps still apply: max 2 per session and 3 per day.

## Validation
Expanding-window walk-forward with 6-month test folds: 2021H1, 2021H2, ..., 2025H2, plus
2026-01-01 .. 2026-04-15 (11 folds). The training data for a fold ends before it starts (labels only reach
60 min ahead, and everything is intraday, so nothing overlaps). These are reported side by side:
(a) the gated GBM, (b) the GBM without the gate, (c) gate only with a random direction.

## Pass criteria (pooled out-of-sample folds; all must hold)
1. at least 150 trades
2. mean R > 0 with 2x costs
3. t-stat of mean R >= 3.0 (raised from 2.0 because many ideas have now been tried)
4. positive total R in at least 7 of the 11 folds

## Conditional next steps (pre-committed)
- If the GBM passes: train a PyTorch 1D-CNN challenger (last 120 1m returns plus the same features)
  under the same protocol, then run the better model ONCE on the final 6 months, then paper trading.
- If the GBM fails: no neural network on this setup, and the final 6 months stay untouched.
