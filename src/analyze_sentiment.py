from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
OUTPUT_DIR = ROOT / 'outputs'
CHART_DIR = OUTPUT_DIR / 'charts'
TABLE_DIR = OUTPUT_DIR / 'tables'


def ensure_dirs() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(RAW_DIR / 'hyperliquid_trader_data.csv')
    sentiment = pd.read_csv(RAW_DIR / 'fear_greed_index.csv')
    return trades, sentiment


def prepare_data(trades: pd.DataFrame, sentiment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = trades.copy()
    sentiment = sentiment.copy()

    trades['date'] = pd.to_datetime(trades['Timestamp IST'], dayfirst=True, errors='coerce').dt.normalize()
    trades['is_win'] = trades['Closed PnL'] > 0
    trades['is_directional'] = trades['Direction'].str.contains('Long|Short', na=False)
    trades['trade_side'] = np.where(
        trades['Direction'].str.contains('Long', na=False),
        'Long',
        np.where(trades['Direction'].str.contains('Short', na=False), 'Short', pd.NA),
    )

    sentiment['date'] = pd.to_datetime(sentiment['date'])
    sentiment['sentiment_group'] = np.select(
        [
            sentiment['classification'].isin(['Fear', 'Extreme Fear']),
            sentiment['classification'].isin(['Greed', 'Extreme Greed']),
        ],
        ['Fear', 'Greed'],
        default='Neutral',
    )

    merged = trades.merge(
        sentiment[['date', 'value', 'classification', 'sentiment_group']],
        on='date',
        how='left',
        validate='many_to_one',
    )
    return trades, sentiment, merged


def build_tables(
    raw_trades: pd.DataFrame,
    raw_sentiment: pd.DataFrame,
    trades: pd.DataFrame,
    sentiment: pd.DataFrame,
    merged: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    raw_quality = pd.DataFrame(
        {
            'dataset': ['fear_greed_index', 'hyperliquid_trader_data'],
            'rows': [len(raw_sentiment), len(raw_trades)],
            'columns': [raw_sentiment.shape[1], raw_trades.shape[1]],
            'missing_values': [int(raw_sentiment.isna().sum().sum()), int(raw_trades.isna().sum().sum())],
            'duplicate_rows': [int(raw_sentiment.duplicated().sum()), int(raw_trades.duplicated().sum())],
        }
    )

    merge_coverage = pd.DataFrame(
        {
            'metric': ['trade_rows', 'matched_rows', 'unmatched_rows', 'match_rate'],
            'value': [
                len(merged),
                int(merged['classification'].notna().sum()),
                int(merged['classification'].isna().sum()),
                float(merged['classification'].notna().mean()),
            ],
        }
    )

    analysis = merged[merged['classification'].notna()].copy()

    sentiment_counts = analysis['classification'].value_counts(dropna=False).rename_axis('classification').reset_index(name='trade_rows')

    sentiment_summary = analysis.groupby('sentiment_group', dropna=False).agg(
        trades=('Closed PnL', 'size'),
        total_pnl=('Closed PnL', 'sum'),
        mean_pnl=('Closed PnL', 'mean'),
        median_pnl=('Closed PnL', 'median'),
        win_rate=('is_win', 'mean'),
        avg_trade_size=('Size USD', 'mean'),
        avg_abs_pnl=('Closed PnL', lambda s: s.abs().mean()),
        directional_trades=('is_directional', 'sum'),
        daily_dates=('date', 'nunique'),
    ).reset_index()
    directional = merged[merged['is_directional']].groupby('sentiment_group')['trade_side'].apply(lambda s: (s == 'Long').mean())
    sentiment_summary['directional_long_share'] = sentiment_summary['sentiment_group'].map(directional)
    sentiment_summary['directional_short_share'] = 1 - sentiment_summary['directional_long_share']
    sentiment_summary['pnl_per_trade'] = sentiment_summary['total_pnl'] / sentiment_summary['trades']

    daily_summary = analysis.groupby(['date', 'classification', 'sentiment_group'], dropna=False).agg(
        trades=('Closed PnL', 'size'),
        total_pnl=('Closed PnL', 'sum'),
        mean_pnl=('Closed PnL', 'mean'),
        win_rate=('is_win', 'mean'),
        avg_trade_size=('Size USD', 'mean'),
        directional_trades=('is_directional', 'sum'),
    ).reset_index()

    daily_sentiment = daily_summary.groupby('sentiment_group', dropna=False).agg(
        days=('date', 'nunique'),
        avg_daily_pnl=('total_pnl', 'mean'),
        median_daily_pnl=('total_pnl', 'median'),
        daily_pnl_std=('total_pnl', 'std'),
        daily_pnl_min=('total_pnl', 'min'),
        daily_pnl_max=('total_pnl', 'max'),
        loss_days=('total_pnl', lambda s: int((s < 0).sum())),
        avg_win_rate=('win_rate', 'mean'),
        avg_trades=('trades', 'mean'),
    ).reset_index()

    account_summary = analysis.groupby('Account').agg(
        trades=('Closed PnL', 'size'),
        total_pnl=('Closed PnL', 'sum'),
        mean_pnl=('Closed PnL', 'mean'),
        win_rate=('is_win', 'mean'),
        pnl_std=('Closed PnL', 'std'),
        avg_size_usd=('Size USD', 'mean'),
        median_size_usd=('Size USD', 'median'),
        directional_trades=('is_directional', 'sum'),
        directional_long_share=('trade_side', lambda s: (s == 'Long').mean()),
    ).reset_index()
    account_summary['consistency'] = account_summary['total_pnl'] / account_summary['pnl_std'].replace(0, np.nan)
    account_summary['consistency'] = account_summary['consistency'].fillna(0)

    account_summary['freq_segment'] = pd.qcut(account_summary['trades'].rank(method='first'), 2, labels=['Infrequent', 'Frequent'])
    account_summary['size_segment'] = pd.qcut(account_summary['avg_size_usd'].rank(method='first'), 2, labels=['Lower size', 'Higher size'])
    account_summary['consistency_segment'] = pd.qcut(account_summary['consistency'].rank(method='first'), 2, labels=['Inconsistent', 'Consistent'])

    account_behavior = analysis.merge(
        account_summary[['Account', 'freq_segment', 'size_segment', 'consistency_segment']],
        on='Account',
        how='left',
    )

    segment_tables: dict[str, pd.DataFrame] = {}
    for segment_col in ['freq_segment', 'size_segment', 'consistency_segment']:
        segment_tables[segment_col] = account_behavior.groupby(['sentiment_group', segment_col], dropna=False, observed=True).agg(
            trades=('Closed PnL', 'size'),
            total_pnl=('Closed PnL', 'sum'),
            win_rate=('is_win', 'mean'),
            avg_trade_size=('Size USD', 'mean'),
            directional_long_share=('trade_side', lambda s: (s == 'Long').mean()),
        ).reset_index()

    return {
        'raw_quality': raw_quality,
        'merge_coverage': merge_coverage,
        'sentiment_counts': sentiment_counts,
        'sentiment_summary': sentiment_summary,
        'daily_summary': daily_summary,
        'daily_sentiment': daily_sentiment,
        'account_summary': account_summary,
        'freq_segment': segment_tables['freq_segment'],
        'size_segment': segment_tables['size_segment'],
        'consistency_segment': segment_tables['consistency_segment'],
    }


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, frame in tables.items():
        frame.to_csv(TABLE_DIR / f'{name}.csv', index=False)


def save_charts(merged: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    sns.set_theme(style='whitegrid', context='talk')

    daily = tables['daily_summary'].copy()
    daily['pnl_bucket'] = pd.cut(
        daily['total_pnl'],
        bins=20,
        labels=False,
        duplicates='drop',
    )

    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    sentiment_palette = {
        'Extreme Fear': '#8b1e3f',
        'Fear': '#c94f4f',
        'Neutral': '#9aa0a6',
        'Greed': '#3c8d2f',
        'Extreme Greed': '#1b5e20',
    }
    analysis = merged[merged['classification'].notna()].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=analysis,
        x='classification',
        y='Closed PnL',
        order=sentiment_order,
        hue='classification',
        dodge=False,
        legend=False,
        palette=sentiment_palette,
        ax=ax,
        showfliers=False,
    )
    ax.set_title('Trade PnL Distribution by Sentiment')
    ax.set_xlabel('Sentiment classification')
    ax.set_ylabel('Closed PnL')
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    fig.savefig(CHART_DIR / 'pnl_boxplot_by_sentiment.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        data=daily,
        x='date',
        y='total_pnl',
        hue='sentiment_group',
        palette={'Fear': '#c94f4f', 'Greed': '#3c8d2f', 'Neutral': '#9aa0a6'},
        alpha=0.8,
        ax=ax,
    )
    sns.lineplot(data=daily, x='date', y='total_pnl', color='#1f1f1f', alpha=0.2, ax=ax)
    ax.axhline(0, color='#333333', linewidth=1)
    ax.set_title('Daily Total PnL by Sentiment Group')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total PnL')
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    fig.savefig(CHART_DIR / 'daily_total_pnl_by_sentiment.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_df = tables['sentiment_summary'].copy()
    plot_df['sentiment_group'] = pd.Categorical(plot_df['sentiment_group'], categories=['Fear', 'Greed', 'Neutral'], ordered=True)
    plot_df = plot_df.sort_values('sentiment_group')

    sns.barplot(data=plot_df, x='sentiment_group', y='win_rate', hue='sentiment_group', dodge=False, legend=False, palette={'Fear': '#c94f4f', 'Greed': '#3c8d2f', 'Neutral': '#9aa0a6'}, ax=axes[0])
    axes[0].set_title('Win Rate by Sentiment Group')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Win rate')
    axes[0].set_ylim(0, 1)

    sns.barplot(data=plot_df, x='sentiment_group', y='directional_long_share', hue='sentiment_group', dodge=False, legend=False, palette={'Fear': '#c94f4f', 'Greed': '#3c8d2f', 'Neutral': '#9aa0a6'}, ax=axes[1])
    axes[1].set_title('Directional Long Share by Sentiment Group')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Long share')
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(CHART_DIR / 'behavior_by_sentiment.png', dpi=180)
    plt.close(fig)

    pivot = tables['freq_segment'].pivot(index='sentiment_group', columns='freq_segment', values='total_pnl').reindex(index=['Fear', 'Greed', 'Neutral'])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=',.0f', cmap='RdYlGn', center=0, ax=ax)
    ax.set_title('Total PnL by Frequency Segment and Sentiment')
    ax.set_xlabel('Trader segment')
    ax.set_ylabel('Sentiment group')
    fig.tight_layout()
    fig.savefig(CHART_DIR / 'segment_pnl_heatmap.png', dpi=180)
    plt.close(fig)


def write_summary_markdown(tables: dict[str, pd.DataFrame]) -> None:
    sentiment = tables['sentiment_summary']
    coverage = tables['merge_coverage']
    daily = tables['daily_sentiment']
    accounts = tables['account_summary']
    freq = tables['freq_segment']
    size = tables['size_segment']
    consistency = tables['consistency_segment']

    fear = sentiment[sentiment['sentiment_group'] == 'Fear'].iloc[0]
    greed = sentiment[sentiment['sentiment_group'] == 'Greed'].iloc[0]
    frequent_fear = freq[(freq['sentiment_group'] == 'Fear') & (freq['freq_segment'] == 'Frequent')].iloc[0]
    frequent_greed = freq[(freq['sentiment_group'] == 'Greed') & (freq['freq_segment'] == 'Frequent')].iloc[0]
    higher_size_fear = size[(size['sentiment_group'] == 'Fear') & (size['size_segment'] == 'Higher size')].iloc[0]
    higher_size_greed = size[(size['sentiment_group'] == 'Greed') & (size['size_segment'] == 'Higher size')].iloc[0]
    consistent_neutral = consistency[(consistency['sentiment_group'] == 'Neutral') & (consistency['consistency_segment'] == 'Consistent')].iloc[0]
    inconsistent_neutral = consistency[(consistency['sentiment_group'] == 'Neutral') & (consistency['consistency_segment'] == 'Inconsistent')].iloc[0]

    lines = [
        '# Round-0 Analysis Summary',
        '',
        '## Method',
        'The trader file was parsed from the mixed-format `Timestamp IST` field at the daily level and joined to the fear/greed index on date. The raw files had no duplicate rows and no missing values; the main data-cleaning work was timestamp normalization and sentiment alignment.',
        '',
        '## Key Findings',
        f"- Fear days had higher average daily PnL than Greed days ({daily.loc[daily['sentiment_group'] == 'Fear', 'avg_daily_pnl'].iloc[0]:,.0f} vs {daily.loc[daily['sentiment_group'] == 'Greed', 'avg_daily_pnl'].iloc[0]:,.0f}), but also higher daily volatility.",
        f"- Directional long bias was stronger on Fear days ({fear['directional_long_share']:.1%}) than on Greed days ({greed['directional_long_share']:.1%}).",
        f"- Frequent traders outperformed infrequent traders in both sentiment regimes, with the gap more visible in Fear: {frequent_fear['total_pnl']:,.0f} vs {freq[(freq['sentiment_group'] == 'Fear') & (freq['freq_segment'] == 'Infrequent')]['total_pnl'].iloc[0]:,.0f} total PnL.",
        f"- High-size traders generated more absolute PnL, but win rates were weaker than lower-size traders in both Fear ({higher_size_fear['win_rate']:.1%} vs {size[(size['sentiment_group'] == 'Fear') & (size['size_segment'] == 'Lower size')]['win_rate'].iloc[0]:.1%}) and Greed ({higher_size_greed['win_rate']:.1%} vs {size[(size['sentiment_group'] == 'Greed') & (size['size_segment'] == 'Lower size')]['win_rate'].iloc[0]:.1%}).",
        f"- Consistent traders were more resilient than inconsistent traders, especially in Neutral days ({consistent_neutral['win_rate']:.1%} vs {inconsistent_neutral['win_rate']:.1%}).",
        '',
        f"The date merge matched {int(coverage.loc[coverage['metric'] == 'matched_rows', 'value'].iloc[0]):,} of {int(coverage.loc[coverage['metric'] == 'trade_rows', 'value'].iloc[0]):,} trade rows ({coverage.loc[coverage['metric'] == 'match_rate', 'value'].iloc[0]:.4%} coverage).",
        '',
        '## Actionable Rules',
        '- During Fear days, favor frequent, directionally long-biased traders and reduce allocation to the highest-size cohort unless their recent win rate stays above the peer median.',
        '- During Greed days, cap position size more aggressively and concentrate risk in consistent accounts; large tickets still make money, but the win-rate penalty is too large to ignore.',
        '',
        '## Files',
        '- Charts: `outputs/charts/`',
        '- Tables: `outputs/tables/`',
    ]

    (OUTPUT_DIR / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


def print_console_summary(tables: dict[str, pd.DataFrame]) -> None:
    quality = tables['raw_quality']
    coverage = tables['merge_coverage']
    sentiment = tables['sentiment_summary']
    daily = tables['daily_sentiment']
    accounts = tables['account_summary']

    print('Raw quality')
    print(quality.to_string(index=False))
    print('\nMerge coverage')
    print(coverage.to_string(index=False))
    print('\nSentiment summary')
    print(sentiment.to_string(index=False, float_format=lambda x: f'{x:,.3f}'))
    print('\nDaily sentiment summary')
    print(daily.to_string(index=False, float_format=lambda x: f'{x:,.3f}'))
    print('\nAccount headline')
    top = accounts.sort_values('total_pnl', ascending=False).head(5)
    bottom = accounts.sort_values('total_pnl').head(5)
    print('Top 5 accounts by PnL')
    print(top.to_string(index=False, float_format=lambda x: f'{x:,.3f}'))
    print('Bottom 5 accounts by PnL')
    print(bottom.to_string(index=False, float_format=lambda x: f'{x:,.3f}'))


def main() -> None:
    ensure_dirs()
    raw_trades, raw_sentiment = load_data()
    trades, sentiment, merged = prepare_data(raw_trades, raw_sentiment)
    tables = build_tables(raw_trades, raw_sentiment, trades, sentiment, merged)
    save_tables(tables)
    save_charts(merged, tables)
    write_summary_markdown(tables)
    print_console_summary(tables)


if __name__ == '__main__':
    main()
