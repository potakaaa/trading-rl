# Round 7: cross-market information. Pre-registered 2026-10-17, before any cross-market data was analysed

Question: does information from other markets (which often lead or confirm Nasdaq moves) let the round-4
model pick US100 trades well enough to clear Rald's bar? This is a new trial: t >= 3 and the $5 rule apply.

## Identical to round 4
- **Decisions and trade:** the same decision times. The trade is stop 1.0 x ATR15, target 1.5R, exit after 60 minutes
  or at session end, market entry.
- **Same everything else:** costs (2.0 / 1.0, 2x stress), research period 2019-01-01 .. 2026-04-15, and 11 walk-forward
  folds. The tau grid and tau selection on the last 12 months of each training window are unchanged, as are the
  LightGBM settings and the sequencing (one position at a time, max 2 per session and 3 per day).
- **No volatility gate:** round 4 showed the gate hurt, and both arms below are compared without it.
- **Final test:** 2026-04-16 .. 2026-10-15, still untouched, run once, only if this passes.

## Cross markets (Dukascopy 1m, features only, never traded)
US500, US30 (Dow), US2000 (Russell 2000), DXY (dollar index), USTBOND (US T-bond), XAUUSD, EURUSD, USDJPY,
OIL (WTI).

## New features (36), computed from bars complete at the decision time t
For each cross market X, with ATR_X = ATR(14) of X's 15-minute bars at t:
- `X_ret5`, `X_ret15`, `X_ret60`: X's return over the last 5/15/60 minutes, divided by ATR_X
- `X_div60`: US100's ret60 (in US100 ATR15 units) minus `X_ret60`, a relative-strength divergence
If X has no bar within 5 minutes before t, its features are missing (LightGBM handles missing values).

## Arms (both evaluated, the same folds and seeds)
- `base`: the round-4 direction model with its 22 features (the reference, without the gate)
- `cross`: the same model with the 22 + 36 features

## Pass criteria (arm `cross`, pooled out-of-sample, all must hold)
1. at least 150 trades
2. mean R > 0.4 (more than $5 per trade) at base costs, and mean R > 0 with 2x costs
3. t-stat >= 3.0
4. positive in at least 7 of 11 folds
For information: `cross` minus `base` in R per trade (does cross-market data add value at all?), and the top
cross-market features by LightGBM gain.
