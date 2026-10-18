# Round 15: new intraday US100 families, walk-forward selected. Pre-registered 2026-10-18, before any run
Rald: keep searching history for $5/trade. Hits are **candidates** for paper trading, never confirmed (no holdout left).

## Families and grids (round-5 execution/costs; stop orders and market entries as defined there; flat 16:00)
- `orb`: opening range of N minutes from 09:30 (N in {15, 30, 60}). Bracket of stop orders at OR high/low, valid until
  11:30; stop at the opposite side; target in {2R, 3R, 5R}; filter in {none, NR4}. (18 variants)
- `inside`: if yesterday was an inside day (high < prior high and low > prior low), a bracket at yesterday's high/low from
  09:30 to 16:00, stop at the other side, target in {2R, 3R, eod}. (3 variants)
- `pm_trend`: at 14:00, move = close(13:59) - open(09:30). If |move| >= m x stretch (m in {0.5, 1.0}), trade its direction
  at market, stop at the opposite 13:00-14:00 extreme, target in {2R, 3R}. (4 variants)
- `gap_go`: if |gap| / daily ATR >= g (g in {0.3, 0.6}) and the 09:30-09:35 candle closes in the gap's direction, trade
  it at market at 09:35, stop at that candle's opposite extreme, target in {2R, 3R}. (4 variants)
- `drive`: the round-14 opening drive with min_move in {0.3, 0.5, 0.75} x stretch, target in {2R, 3R, 5R} and
  max_risk in {1.0, 1.5} x stretch. (18 variants)

## Selection and verdict
- **Selection:** as round 9. For each family and each of the 11 folds, the variant with the best mean R over the previous
  3 years (at least 50 trades) is chosen and traded on the fold.
- **Candidate for $5 at 1% risk** if pooled out-of-sample mean R >= 0.20, with >= 100 trades, mean R > 0 at 2x costs,
  t >= 2.5 and >= 7/11 positive folds. **Candidate for $5 at 0.5%** if mean R >= 0.40 with the same conditions.
- **Also reported:** PBO, the trades per month, and the worst drawdown in $ at both risk levels.
