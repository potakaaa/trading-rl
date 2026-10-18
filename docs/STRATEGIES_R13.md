# Round 13: pullback entry after the stretch breakout. Pre-registered 2026-10-18, before any run
Same goal and verdict rules as round 10 (out-of-sample mean R >= 0.20 = $5 at 1%; >= 0.40 = $5 at 0.5%;
trusted only with >= 100 trades, mean R > 0 at 2x costs, t >= 2.5, >= 7/11 positive folds).

- **Trigger:** the Crabel stretch breakout (stretch = 10-day mean of min(open - low, high - open), levels =
  09:30 open +/- stretch). It only decides the DIRECTION; it does not enter.
- **Entry:** after the trigger bar, a limit order at the halfway level (open + 0.5 x stretch for longs,
  open - 0.5 x stretch for shorts), valid until 11:30 NY. Longs fill when the ask reaches it. No fill = no trade.
- **Stop and target:** the stop is the opposite stretch level (as in Crabel), the target 3R from the fill, otherwise
  exit at 16:00. Costs 2.0 / 1.0 (2x stress), pessimistic intrabar rules.
- **Arms:** (a) all pullback trades, and (b) the round-10 meta-model (same features, LightGBM, tau rule, folds)
  trained on these trades.
