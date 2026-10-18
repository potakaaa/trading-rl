# Round 14: opening drive + order-flow confirmation at 10:00. Pre-registered 2026-10-18, before any run
Same goal and verdict rules as round 10.
- **Signal at 10:00 NY:** move = close(09:59 bar) - open(09:30). Trade in the direction of the move if |move| >= 0.5 x stretch
  (the stretch as in round 13, from complete prior days).
- **Entry:** market at the 10:00 open. **Stop:** the opposite extreme of 09:30-10:00; skip if the risk > 1.5 x stretch.
  **Target:** 3R, otherwise exit at 16:00.
- **Arms:**
  - (a) all signals
  - (b) only when NQ order flow agrees: sign(of_delta3 for 09:57-10:00) == the move direction
  - (c) the round-10-style meta-model on (a), with the round-10 features plus the 6 round-8 order-flow features at 10:00

## Final test (one time), started 2026-10-18 at Rald's request
Variant 14(b) exactly as above, unchanged, on 2026-04-16 .. 2026-10-15. Order flow for 09:57-10:00 is bought from Databento for those days only (quoted first, budget cap respected).

**Result:** FAIL. 15 trades, -0.320 R per trade (2x costs -0.333 R), t -0.94. 14(a): 33 trades, -0.145 R.
