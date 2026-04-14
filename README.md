# Primetrade Round-0 Assignment

This repo contains a reproducible analysis of how Bitcoin market sentiment relates to Hyperliquid trader behavior and performance.

## Contents

- `notebooks/round0_analysis.ipynb` - notebook walkthrough of the analysis
- `src/analyze_sentiment.py` - reusable pipeline that builds the tables and charts
- `data/raw/` - downloaded source data
- `outputs/charts/` - exported figures
- `outputs/tables/` - exported summary tables
- `outputs/summary.md` - short write-up with findings and strategy rules

## Setup

The analysis was built with Python 3.10 in the repo virtual environment.

```bash
pip install -r requirements.txt
```

## Run

Regenerate tables, charts, and the summary write-up:

```bash
python src/analyze_sentiment.py
```

Open the notebook for the narrative version of the analysis:

```bash
jupyter notebook notebooks/round0_analysis.ipynb
```

## Main Findings

- Fear days had higher average daily PnL than Greed days, but also higher daily volatility.
- Directional long bias was stronger on Fear days than on Greed days.
- Frequent traders outperformed infrequent traders in both sentiment regimes.
- High-size traders generated more absolute PnL, but win rates were lower than lower-size traders.
- Consistent traders were more resilient than inconsistent traders, especially in Neutral days.
- The supplied trader file does not include a leverage column, so trade size is used as the practical sizing proxy.

## Strategy Rules

- During Fear days, favor frequent, directionally long-biased traders and reduce allocation to the highest-size cohort unless recent win rate remains strong.
- During Greed days, cap position size more aggressively and concentrate risk in consistent accounts.
