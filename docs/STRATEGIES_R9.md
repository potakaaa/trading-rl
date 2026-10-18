# Round 9: walk-forward optimisation of the small-edge US100 setups. Pre-registered 2026-10-18, before any run

Rald asked to keep improving on US100/NAS100. The three round-5 setups with a small positive edge (Williams
volatility breakout, Crabel NR7 stretch breakout, Zarattini noise-area momentum) are given parameter grids.
**Parameters are only ever chosen on past data**, and only the unseen next period counts. That makes the
result honest even though many variants are tried.

## Common
- **Data:** US100 Dukascopy BID 1m, 2019-01-01 .. 2026-04-15 (final test 2026-04-16 .. 2026-10-15 untouched,
  run once only if something passes).
- **Execution and risk:** round-5 execution, costs (2.0 / 1.0 and 2x stress) and pessimistic rules. 0.5% risk, flat
  by 16:00.
- **Hard stops:** every variant has one (prop safety), and R = net P&L / initial stop distance.

## Grids
**`williams`** (30 variants):
- k in {0.25, 0.40, 0.50, 0.65, 0.80} x yesterday's range from the 09:30 open (buy stop above, sell stop below)
- stop in {open, half} (half = halfway between the open and the entry level)
- exit in {16:00, 2R target, 3R target}

**`crabel`** (27 variants):
- day filter in {NR7, NR4, none}
- stretch multiplier in {1.0, 1.5, 2.0}
- exit in {16:00, 2R target, 3R target}
- the stop is at the opposite stretch level

**`noise`** (12 variants):
- band multiplier in {1.0, 1.25, 1.5} on sigma
- check interval in {30, 15} minutes
- trailing exit in {max/min(band, VWAP), band only}
- hard stop at entry -/+ (band multiplier x sigma x price), and R measured against that stop

## Walk-forward selection
- **Folds:** the 11 test folds used before (2021H1 .. 2025H2, 2026-01-01 .. 04-15).
- **Choosing a variant:** for each setup and fold, the variant with the highest mean R on the **previous 3 years**
  (at least 50 trades there) is chosen and traded unchanged in that fold.
- **Portfolio:** the three chosen setups run side by side. Each opens at most one position at a time, and at most
  3 trades per day are taken overall (the first 3 by entry time).

## Pass criteria (pooled out-of-sample, per setup AND for the portfolio; a setup or the portfolio passes if all hold)
1. at least 150 trades
2. mean R > 0.4 (more than $5 per trade) at base costs, and mean R > 0 with 2x costs
3. t-stat >= 3.0
4. positive in at least 7 of 11 folds

**Reported:** PBO, the share of folds where the chosen variant's out-of-sample R is below the median out-of-sample R of
all variants of that setup in the same fold (above 0.5 = selection is no better than chance). Also the variants
chosen over time, and $ per month.
