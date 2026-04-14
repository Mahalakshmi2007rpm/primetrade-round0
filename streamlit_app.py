from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
TABLE_DIR = ROOT / 'outputs' / 'tables'
CHART_DIR = ROOT / 'outputs' / 'charts'

st.set_page_config(page_title='Trader vs Sentiment Dashboard', layout='wide')
st.title('Trader Performance vs Market Sentiment')
st.caption('Interactive dashboard for Primetrade Round-0 analysis outputs')


def load_table(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


sentiment_summary = load_table('sentiment_summary.csv')
daily_summary = load_table('daily_summary.csv')
freq_segment = load_table('freq_segment.csv')
size_segment = load_table('size_segment.csv')
consistency_segment = load_table('consistency_segment.csv')
archetypes = load_table('trader_archetypes.csv')
archetype_summary = load_table('trader_archetype_summary.csv')
bonus_report = load_table('bonus_classification_report.csv')
bonus_cm = load_table('bonus_confusion_matrix.csv')

if sentiment_summary.empty:
    st.error('No analysis tables found. Run src/analyze_sentiment.py first.')
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric('Fear PnL / trade', f"{sentiment_summary.loc[sentiment_summary['sentiment_group']=='Fear', 'pnl_per_trade'].iloc[0]:.2f}")
col2.metric('Greed PnL / trade', f"{sentiment_summary.loc[sentiment_summary['sentiment_group']=='Greed', 'pnl_per_trade'].iloc[0]:.2f}")
col3.metric('Neutral PnL / trade', f"{sentiment_summary.loc[sentiment_summary['sentiment_group']=='Neutral', 'pnl_per_trade'].iloc[0]:.2f}")

st.subheader('Sentiment Summary')
st.dataframe(sentiment_summary, use_container_width=True)

if not daily_summary.empty:
    st.subheader('Daily PnL Explorer')
    sentiment_choices = sorted(daily_summary['sentiment_group'].dropna().unique().tolist())
    selected = st.multiselect('Sentiment groups', sentiment_choices, default=sentiment_choices)
    plot_df = daily_summary[daily_summary['sentiment_group'].isin(selected)].copy()
    plot_df['date'] = pd.to_datetime(plot_df['date'])

    fig = px.scatter(
        plot_df,
        x='date',
        y='total_pnl',
        color='sentiment_group',
        hover_data=['classification', 'trades', 'win_rate', 'avg_trade_size'],
        title='Daily Total PnL by Sentiment Group',
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader('Segment Comparison')
segment_choice = st.selectbox('Choose segment table', ['freq_segment', 'size_segment', 'consistency_segment'])
segment_map = {
    'freq_segment': freq_segment,
    'size_segment': size_segment,
    'consistency_segment': consistency_segment,
}
selected_df = segment_map[segment_choice]
if not selected_df.empty:
    st.dataframe(selected_df, use_container_width=True)

    segment_col = [c for c in selected_df.columns if c.endswith('_segment')][0]
    bar = px.bar(
        selected_df,
        x='sentiment_group',
        y='total_pnl',
        color=segment_col,
        barmode='group',
        title=f'Total PnL by {segment_col} and Sentiment',
    )
    st.plotly_chart(bar, use_container_width=True)

st.subheader('Trader Archetypes (Clustering Bonus)')
if archetypes.empty or archetype_summary.empty:
    st.info('No archetype outputs found. Run src/bonus_clustering.py to generate them.')
else:
    st.dataframe(archetype_summary, use_container_width=True)

    archetype_filter = st.multiselect(
        'Filter archetypes',
        sorted(archetypes['archetype'].dropna().unique().tolist()),
        default=sorted(archetypes['archetype'].dropna().unique().tolist()),
    )
    filtered = archetypes[archetypes['archetype'].isin(archetype_filter)].copy()

    scatter = px.scatter(
        filtered,
        x='trades',
        y='total_pnl',
        color='archetype',
        size='avg_size_usd',
        hover_data=['Account', 'win_rate', 'consistency'],
        title='Trader Archetypes: Activity vs Total PnL',
    )
    st.plotly_chart(scatter, use_container_width=True)

    st.caption('Top accounts by total PnL within filtered archetypes')
    st.dataframe(
        filtered.sort_values('total_pnl', ascending=False).head(20),
        use_container_width=True,
    )

st.subheader('Prediction Bonus: Next-Day Profitability Buckets')
if bonus_report.empty or bonus_cm.empty:
    st.info('No prediction outputs found. Run notebook section 11 to generate bonus prediction tables.')
else:
    report_cols = ['precision', 'recall', 'f1-score', 'support']
    available_cols = [c for c in report_cols if c in bonus_report.columns]
    st.caption('Model quality by class (loss / flat / profit)')
    st.dataframe(bonus_report, use_container_width=True)

    if available_cols:
        score_view = bonus_report.set_index(bonus_report.columns[0])[available_cols]
        fig_report = px.bar(
            score_view.reset_index(),
            x=score_view.index.name,
            y=[c for c in ['precision', 'recall', 'f1-score'] if c in score_view.columns],
            barmode='group',
            title='Prediction Metrics by Class',
        )
        st.plotly_chart(fig_report, use_container_width=True)

    cm_name_col = bonus_cm.columns[0]
    cm_df = bonus_cm.copy().set_index(cm_name_col)
    cm_plot = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale='Blues',
        title='Confusion Matrix (Actual vs Predicted)',
    )
    cm_plot.update_xaxes(title='Predicted class')
    cm_plot.update_yaxes(title='Actual class')
    st.plotly_chart(cm_plot, use_container_width=True)

st.subheader('Static Charts')
for chart_name in [
    'pnl_boxplot_by_sentiment.png',
    'daily_total_pnl_by_sentiment.png',
    'behavior_by_sentiment.png',
    'segment_pnl_heatmap.png',
    'trader_archetypes_scatter.png',
]:
    chart_path = CHART_DIR / chart_name
    if chart_path.exists():
        st.image(str(chart_path), caption=chart_name)
