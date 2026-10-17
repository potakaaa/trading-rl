# Round 8: Nasdaq futures order flow (Databento). Pre-registered 2026-10-17, before any order-flow data was downloaded

Rald provided a Databento key with $125 free credit and asked to protect it. The design is shaped by the budget:
- **Only the 4 morning decision times** (09:45, 10:00, 10:15, 10:30 NY), the round-4 decisions inside the AM session.
- **Only the 10 minutes of NQ trades before each decision** (09:35-09:45, 09:50-10:00, 10:05-10:15, 10:20-10:30).
- **Estimated cost:** about $86 for 2019-01-01 .. 2026-04-15 (sampled with Databento's free cost API). A **hard stop at
  $100 total spend** is enforced in code, metered from the bytes received ($26.1/GB).

## Data
- Databento GLBX.MDP3, schema `trades`, symbol `NQ.v.0` (front-month by volume, continuous). Each trade has a
  price, a size and the aggressor `side` (A = sell aggressor / hit the bid, B = buy aggressor / lifted the ask).
- **Trades and costs:** identical to round 4 (US100 CFD, stop 1 x ATR15, target 1.5R, 60 min, spread 2.0 / slippage
  1.0, 2x stress). Same research period. **Final test 2026-04-16 .. 2026-10-15** (untouched), run once, only if this passes.

## Order-flow features (only from the window ending at the decision time)
For the last 10 minutes and for the last 3 minutes:
- `of_delta10`, `of_delta3`: (buy-aggressor volume - sell-aggressor volume) / total volume
- `of_vol10_rel`: total volume / its median at the same time of day over the previous 20 days
- `of_trades10_rel`: trade count, the same relative measure
- `of_big_delta10`: delta of large trades only (size >= 10 contracts) / total large-trade volume
- `of_avg_size10`: mean trade size / its 20-day median at the same time of day
- `of_vwap_dist10`: (last price - VWAP of the 10-minute window) / ATR15 of US100
Missing (no trades in the window) -> missing value.

## Arms (the same 11 walk-forward folds, LightGBM settings, tau grid and selection, sequencing; no gate)
- `base_am`: the round-4 direction model with its 22 features, AM decisions only
- `flow`: the same model with the 22 + 7 order-flow features

## Pass criteria (arm `flow`, pooled out-of-sample, all must hold)
1. at least 150 trades
2. mean R > 0.4 (more than $5 per trade) at base costs, and mean R > 0 with 2x costs
3. t-stat >= 3.0
4. positive in at least 7 of 11 folds
For information: `flow` minus `base_am`, and the gain share of the order-flow features.
