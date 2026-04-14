# Round-0 Analysis Summary

## Method
The trader file was parsed from the mixed-format `Timestamp IST` field at the daily level and joined to the fear/greed index on date. The raw files had no duplicate rows and no missing values; the main data-cleaning work was timestamp normalization and sentiment alignment.
The supplied trader file does not contain a leverage column, so trade size is used as the sizing proxy in the segment comparisons.

## Key Findings
- Fear days had higher average daily PnL than Greed days (39,012 vs 15,848), but also higher daily volatility.
- Directional long bias was stronger on Fear days (64.7%) than on Greed days (44.7%).
- Frequent traders outperformed infrequent traders in both sentiment regimes, with the gap more visible in Fear: 3,431,836 vs 664,430 total PnL.
- High-size traders generated more absolute PnL, but win rates were weaker than lower-size traders in both Fear (38.6% vs 42.1%) and Greed (32.3% vs 45.6%).
- Consistent traders were more resilient than inconsistent traders, especially in Neutral days (41.6% vs 25.3%).

The date merge matched 211,218 of 211,224 trade rows (99.9972% coverage).

## Actionable Rules
- During Fear days, favor frequent, directionally long-biased traders and reduce allocation to the highest-size cohort unless their recent win rate stays above the peer median.
- During Greed days, cap position size more aggressively and concentrate risk in consistent accounts; large tickets still make money, but the win-rate penalty is too large to ignore.

## Files
- Charts: `outputs/charts/`
- Tables: `outputs/tables/`