# Round 10: meta-labelled stretch breakout on US100. Pre-registered 2026-10-18, before any run

**Goal (set by Rald):** an average of $5 per trade at 0.5% risk (mean R >= 0.40) and at 1% risk (mean R >= 0.20).
**Measured only out of sample** (walk-forward). In-sample numbers never count.

## Candidates (primary signal, fixed)
- Crabel opening-stretch breakout on **every** day (no NR filter).
- Stretch = mean over the last 10 days of min(open - low, high - open).
- Buy stop at the 09:30 open + stretch, sell stop at open - stretch; the first trigger wins.
- Stop at the opposite level, target 3R, otherwise exit at 16:00. Round-5 execution and costs (2.0 / 1.0, 2x stress).
- This is round 9's grid variant `crabel|filt=none,mult=1.0,exit=3.0`.

## Features (known at the trigger time; daily values from complete prior days only)
- **Yesterday's range:** its rank among the last 7 days' ranges (1 = narrowest), inside-day flag, and range / 20-day ATR.
- **Recent history:** the 5-day return in ATR, and where yesterday's close sat within its range (0..1).
- **Today so far:** gap (09:30 open - previous close) / ATR, overnight (18:00-09:30) range / ATR, trigger minute after
  09:30, direction, and stretch / ATR.
- **Context:** the trade direction times the 5-day return sign (with or against the recent trend), the ATR percentile
  over 1 year, weekday, and whether a high-impact USD event falls on the day.

## Model
- **Learner:** LightGBM regressor of the trade's net R (conservative: 200 trees, learning rate 0.03, 7 leaves,
  min 50 per leaf, 0.8 subsampling, L2 1).
- **Walk-forward:** the same 11 test folds (2021H1 .. 2025H2, 2026-01-01 .. 04-15), expanding window. tau is picked on
  the last 12 months of the training window from {0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40}, maximising mean R with
  at least 30 trades there. Trades with prediction > tau are taken.
- **Baselines reported alongside:** all candidates, and the NR4 filter alone.

## Verdict
- **Goal reached at 1% risk** if pooled out-of-sample mean R >= 0.20. **At 0.5% risk** if >= 0.40.
- Also required for trust: at least 100 out-of-sample trades, mean R > 0 with 2x costs, t >= 2.5, and positive in at
  least 7 of 11 folds.
- **Also reported:** the max drawdown in $ at each risk level against Equity Edge's $200 static limit.
- If the goal is reached and trusted, the final test (2026-04-16 .. 2026-10-15, still untouched) is run once.
