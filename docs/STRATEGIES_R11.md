# Round 11: best-trade-of-the-day selection across three setups. Pre-registered 2026-10-18, before any run
Same goal and verdict rules as round 10 (out-of-sample mean R >= 0.20 for $5 at 1%, >= 0.40 for $5 at 0.5%,
trusted only with >= 100 trades, mean R > 0 with 2x costs, t >= 2.5, >= 7/11 positive folds; drawdown in $ reported).

- **Candidates:** the round-9 trades of three fixed variants, `crabel|filt=none,mult=1.0,exit=3.0`,
  `williams|k=0.5,stop=half,exit=3.0` and `noise|band=1.25,every=30,trail=band`.
- **Features:** the round-10 features plus `family` (0/1/2) and the candidate's stop distance / daily ATR.
- **Model and folds:** the same LightGBM and the same 11 walk-forward folds, with tau picked on the last 12 months of
  training from {0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40} (at least 30 trades).
- **Selection:** each day, take only the candidate with the highest prediction, and only if it is > tau.
  Candidates are scored before entry with pre-entry information only. The chosen trade is the highest-scoring one
  among the day's candidates (all known by their trigger times). To avoid lookahead within the day, a candidate can
  only be chosen if no earlier candidate that day also cleared tau. In other words, the FIRST candidate of the day
  whose score clears tau is taken, and the rest are skipped.
