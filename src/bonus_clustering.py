from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyze_sentiment import load_data, prepare_data, build_tables

OUTPUT_DIR = ROOT / 'outputs'
TABLE_DIR = OUTPUT_DIR / 'tables'
CHART_DIR = OUTPUT_DIR / 'charts'


def _choose_k(features_scaled: np.ndarray, min_k: int = 2, max_k: int = 6) -> int:
    best_k = min_k
    best_score = -1.0
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(features_scaled)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(features_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _cluster_name(row: pd.Series) -> str:
    pnl = row['total_pnl']
    win_rate = row['win_rate']
    trade_size = row['avg_size_usd']
    activity = row['trades']

    if pnl > 0 and win_rate >= 0.45 and activity >= 4000:
        return 'Systematic Winners'
    if pnl > 0 and trade_size >= 8000:
        return 'Conviction Size Traders'
    if pnl <= 0 and activity >= 3000:
        return 'Overactive Underperformers'
    if win_rate < 0.35:
        return 'Low Hit-Rate Traders'
    return 'Balanced Opportunists'


def run_clustering() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_trades, raw_sentiment = load_data()
    trades, sentiment, merged = prepare_data(raw_trades, raw_sentiment)
    tables = build_tables(raw_trades, raw_sentiment, trades, sentiment, merged)

    account = tables['account_summary'].copy()
    feature_cols = [
        'trades',
        'total_pnl',
        'win_rate',
        'avg_size_usd',
        'median_size_usd',
        'directional_long_share',
        'consistency',
    ]

    model_df = account[['Account'] + feature_cols].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_df[feature_cols])

    best_k = _choose_k(X_scaled, min_k=2, max_k=min(6, len(model_df) - 1))
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=30)
    model_df['cluster_id'] = kmeans.fit_predict(X_scaled)

    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feature_cols,
    )
    cluster_centers['cluster_id'] = np.arange(best_k)
    cluster_centers['archetype'] = cluster_centers.apply(_cluster_name, axis=1)

    labeled = model_df.merge(cluster_centers[['cluster_id', 'archetype']], on='cluster_id', how='left')

    summary = labeled.groupby(['cluster_id', 'archetype']).agg(
        accounts=('Account', 'size'),
        avg_trades=('trades', 'mean'),
        avg_total_pnl=('total_pnl', 'mean'),
        median_total_pnl=('total_pnl', 'median'),
        avg_win_rate=('win_rate', 'mean'),
        avg_trade_size=('avg_size_usd', 'mean'),
        avg_consistency=('consistency', 'mean'),
    ).reset_index().sort_values('avg_total_pnl', ascending=False)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    plot_df = labeled.copy()
    plot_df['pc1'] = components[:, 0]
    plot_df['pc2'] = components[:, 1]

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    labeled.to_csv(TABLE_DIR / 'trader_archetypes.csv', index=False)
    summary.to_csv(TABLE_DIR / 'trader_archetype_summary.csv', index=False)
    cluster_centers.to_csv(TABLE_DIR / 'trader_cluster_centers.csv', index=False)

    plt.figure(figsize=(10, 6))
    for archetype, group in plot_df.groupby('archetype'):
        plt.scatter(group['pc1'], group['pc2'], alpha=0.75, label=archetype)
    plt.title('Trader Archetypes (PCA projection of clustered features)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'trader_archetypes_scatter.png', dpi=180)
    plt.close()

    return labeled, summary, cluster_centers


def main() -> None:
    labeled, summary, centers = run_clustering()
    print('Generated clustering outputs:')
    print(f'- trader_archetypes rows: {len(labeled)}')
    print(f'- archetype summary rows: {len(summary)}')
    print(f'- cluster centers rows: {len(centers)}')


if __name__ == '__main__':
    main()
