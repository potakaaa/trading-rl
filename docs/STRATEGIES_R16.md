# Round 16: equity-curve filter on all walk-forward families + stricter bar. Pre-registered 2026-10-18, before any run

## New success bar (set by Rald after round 15)
A candidate for "$5/trade at risk X" needs ALL of the following, measured on pooled walk-forward out-of-sample trades,
2021-01 .. 2026-04-15:
1. mean R >= 0.20 (X = 1%) or >= 0.40 (X = 0.5%)
2. >= 100 trades, mean R > 0 with 2x costs, t >= 2.5, positive in >= 7 of 11 folds
3. **worst drawdown < $200 at X**, i.e. < 8 R at 1% or < 16 R at 0.5%
4. **positive mean R over the last 2 years** (2024-04-16 .. 2026-04-15)

## Method
- **Families:** crabel, williams, noise (round 9) and orb, gap_go, drive, inside, pm_trend (round 15). In each fold the
  variant is chosen exactly as before (best mean R over the previous 3 years, >= 50 trades).
- **Equity-curve filter (fixed, not tuned):** a chosen trade is taken only if the sum of R of that variant's last 20
  **shadow** trades that had closed before its entry is > 0. Shadow trades = every signal of the variant, taken or not.
- Each family is scored on its own. The pre-registered portfolio is all families whose individual result is positive
  out of sample after the filter, max 3 trades per day.
