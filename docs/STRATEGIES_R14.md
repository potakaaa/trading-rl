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
