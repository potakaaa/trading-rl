# Round 6: neural network (1D-CNN). Pre-registered 2026-10-17, before any network was trained

Rald asked to explore neural networks after round 5. The round-4 plan said "no NN unless the GBM
passes". This is therefore a **new, separately pre-registered round**. It counts as another trial, so
the bar stays at t >= 3 plus Rald's $5 rule.

## Trade, data, costs: identical to round 4
- **Decisions:** 10 per NY session day (09:45-10:30, 13:45-15:00 every 15 min). **Trade:** stop 1.0 x ATR15,
  target 1.5R, time exit 60 min / session end. Market entry at the next 1m open. Spread 2.0, slippage 1.0 (2x stress).
- **Data:** research 2019-01-01 .. 2026-04-15. **Final test 2026-04-16 .. 2026-10-15** (still untouched), run once,
  only if the network passes.

## Inputs (all known at the decision time)
- **Sequence:** the last 120 complete 1m bars before the decision, 4 channels, each scaled by ATR15 at the
  decision time: (close_k - close_t), (high_k - low_k), (close_k - open_k), and log tick volume relative
  to its 120-bar mean.
- **Features:** the same 22 round-4 features, standardised with the training window's mean and std, with
  missing values set to 0.

## Targets
- y_long = 1 if the long trade hits its 1.5R target (net R >= 1.49), else 0. y_short likewise.
  A target hit nets about +$18, comfortably over the $5 bar.

## Network (PyTorch, CPU/Apple GPU)
- **Layers:** Conv1d(4->16, k=5) -> ReLU -> Conv1d(16->32, k=5, stride 2) -> ReLU -> global average pool ->
  concat(22 features) -> Linear(54->64) -> ReLU -> Dropout(0.3) -> Linear(64->2) -> sigmoid.
- **Training:** Adam lr 1e-3, weight decay 1e-3, batch 256, binary cross-entropy, at most 30 epochs, early
  stopping (patience 5) on an inner validation set = the last 12 months of the training window.
- **Ensemble:** the average of 3 networks with seeds 0, 1, 2.

## Decision rule
- Take the side with the higher predicted probability p, and trade if p >= theta.
- theta is chosen **on the inner validation set only** from {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60},
  maximising mean R with at least 50 trades there. The networks are then retrained on the full window
  (with the same number of epochs as the early-stopping point).
- One position at a time; max 2 per session and 3 per day (same sequencing as round 4).

## Validation and pass criteria
- **Walk-forward:** expanding window, the same 11 test folds as round 4 (2021H1 .. 2025H2, 2026-01..04-15).
- **Pass (pooled out-of-sample trades, all must hold):** >= 150 trades; **mean R > 0.4 (> $5 per trade)
  at base costs**; mean R > 0 with 2x costs; t-stat >= 3.0; positive in >= 7 of 11 folds.
- **Reported:** win rate (> $5 = win), $ per trade, the comparison with the round-4 GBM and with random direction.
