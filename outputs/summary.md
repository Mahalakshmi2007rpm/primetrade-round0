# Round-0 Analysis Summary

## Objective
Assess how Bitcoin sentiment (Fear/Greed) relates to Hyperliquid trader behavior and outcomes, then derive practical risk rules.

## Methodology
- Parsed trader timestamps from mixed-format Timestamp IST and aligned both datasets on daily date.
- Verified data quality: no missing values and no duplicate rows in source files.
- Merge quality: 211,218 of 211,224 trade rows matched sentiment (99.9972% coverage).
- Engineered metrics: total/mean PnL, win rate, trade-size proxy, directional long/short mix, and daily PnL volatility (drawdown proxy).
- Segmented traders into frequent vs infrequent, higher-size vs lower-size, and consistent vs inconsistent groups.

Note: the trader file does not include a leverage column, so trade size is used as the practical sizing proxy.

## Key Findings
- Fear days had higher average daily PnL than Greed days (39,012 vs 15,848), but also higher daily volatility.
- Directional long bias was stronger on Fear days (64.7%) than on Greed days (44.7%).
- Frequent traders outperformed infrequent traders in both sentiment regimes, with the gap more visible in Fear: 3,431,836 vs 664,430 total PnL.
- High-size traders generated more absolute PnL, but win rates were weaker than lower-size traders in both Fear (38.6% vs 42.1%) and Greed (32.3% vs 45.6%).
- Consistent traders were more resilient than inconsistent traders, especially in Neutral days (41.6% vs 25.3%).

## Actionable Strategy Rules
1. Fear regime rule:
Favor frequent traders with stable directional execution, but constrain size escalation in the highest-size cohort unless recent win rate remains above peer median.
2. Greed regime rule:
Apply tighter size caps and prioritize allocation to consistent accounts; absolute upside remains, but hit-rate deterioration is material for larger tickets.

## Deliverables
- Notebook: notebooks/round0_analysis.ipynb
- Reproducible script: src/analyze_sentiment.py
- Charts: outputs/charts/
- Tables: outputs/tables/
