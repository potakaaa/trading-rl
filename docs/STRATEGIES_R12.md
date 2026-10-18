# Round 12: exit selection for the round-10 meta-model. Pre-registered 2026-10-18, before any run
Same goal and verdict rules as round 10.
- **Candidates:** the round-10 stretch-breakout entries (every day, mult 1.0, stop at the opposite level), each simulated
  with 4 exits: 2R, 3R, 5R target, or hold to 16:00 (`eod`).
- **Exit choice:** in each walk-forward fold, the exit with the highest mean R over **all** training-window candidates
  (expanding window) is chosen, then the round-10 meta-model (same features, LightGBM, tau rule) is trained on that
  exit's labels and applied to the test fold. Only past data chooses the exit.
