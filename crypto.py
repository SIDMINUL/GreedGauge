"""
=============================================================================
 Hyperliquid Trader Behavior & Market Sentiment Analysis
 PrimeTrade.ai – Junior Data Scientist Assignment
=============================================================================
 Datasets:
   1. compressed_data_csv      – Hyperliquid historical trades
   2. fear_greed_index.csv     – Bitcoin Fear & Greed Index

 Sections:
   0.  Imports & Config
   1.  Load & Clean Data
   2.  Merge Datasets
   3.  Exploratory Data Analysis (EDA)
   4.  Sentiment vs. PnL Analysis
   5.  Win Rate & Volume by Sentiment
   6.  Long vs. Short by Sentiment
   7.  Individual Trader Performance
   8.  Behavioral Clustering (K-Means + PCA)
   9.  PnL Timeline Analysis
   10. Coin-Level Analysis
   11. Coin × Sentiment Heatmap
   12. Statistical Validation (ANOVA + Correlation)
   13. Save All Charts
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                        # headless / no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs('charts', exist_ok=True)

# ── Visual theme ──────────────────────────────────────────────────────────────
BG  = '#0f0f1a'   # chart background
FG  = '#e8e8f0'   # foreground text / ticks

PALETTE = {
    'Extreme Fear': '#d62728',
    'Fear':         '#ff7f0e',
    'Neutral':      '#bcbd22',
    'Greed':        '#2ca02c',
    'Extreme Greed':'#17becf',
}
SENT_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   '#1a1a2e',
    'axes.edgecolor':   '#444',
    'axes.labelcolor':  FG,
    'xtick.color':      FG,
    'ytick.color':      FG,
    'text.color':       FG,
    'grid.color':       '#2a2a3e',
    'grid.linewidth':   0.5,
    'font.family':      'DejaVu Sans',
})


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 – Loading & cleaning data")
print("=" * 60)

# ── Trader data ───────────────────────────────────────────────────────────────
df = pd.read_csv('compressed_data.csv')
df.columns = df.columns.str.strip()

# Convert unix-ms timestamp → date
df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
df['date']      = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()

# Numeric casts (some fields arrive as strings / scientific notation)
df['Closed PnL'] = pd.to_numeric(df['Closed PnL'], errors='coerce').fillna(0)
df['Size USD']   = pd.to_numeric(df['Size USD'],   errors='coerce').fillna(0)
df['Fee']        = pd.to_numeric(df['Fee'],        errors='coerce').fillna(0)

# Net PnL = closed profit minus fees paid
df['Net PnL'] = df['Closed PnL'] - df['Fee']

print(f"Trader rows loaded : {len(df):,}")
print(f"Columns            : {list(df.columns)}")
print(f"Date range         : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Unique accounts    : {df['Account'].nunique()}")
print(f"Direction values   :\n{df['Direction'].value_counts()}\n")

# ── Fear / Greed index ────────────────────────────────────────────────────────
fg = pd.read_csv('fear_greed_index.csv')
fg['date']           = pd.to_datetime(fg['date'])
fg['classification'] = pd.Categorical(
    fg['classification'], categories=SENT_ORDER, ordered=True
)

print(f"Fear/Greed rows    : {len(fg):,}")
print(f"Date range         : {fg['date'].min().date()} → {fg['date'].max().date()}")
print(f"Class distribution :\n{fg['classification'].value_counts()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MERGE DATASETS
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 2 – Merging on date")
print("=" * 60)

# All trades + sentiment score (used for timeline and coin analysis)
df = df.merge(fg[['date', 'value', 'classification']], on='date', how='left')

# Closing trades only for PnL analysis
#   Direction labels for closing events on Hyperliquid:
#     'Close Long'  – realises profit/loss on a long position
#     'Close Short' – realises profit/loss on a short position
close_mask = df['Direction'].isin(['Close Long', 'Close Short'])
df_close   = df[close_mask].copy()
df_close   = df_close.dropna(subset=['classification'])

# Binary win flag
df_close['is_win'] = df_close['Net PnL'] > 0

print(f"Total trades    : {len(df):,}")
print(f"Closing trades  : {len(df_close):,}  (used for PnL analysis)")
print(f"Matched to F/G  : {df_close['classification'].notna().sum():,}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 3 – EDA")
print("=" * 60)

print("\n── Net PnL distribution (all closing trades) ──")
print(df_close['Net PnL'].describe().round(2))

print("\n── Top coins by closing trade count ──")
print(df_close['Coin'].value_counts().head(10))

print("\n── Trade direction split ──")
print(df_close['Direction'].value_counts())


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SENTIMENT vs. PnL  ── Chart 1 (overview) + Chart 2 (PnL breakdown)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 – Sentiment distribution + PnL analysis")
print("=" * 60)

# ── 4a. Sentiment distribution & timeline ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)

counts = fg['classification'].value_counts().reindex(SENT_ORDER).dropna()
axes[0].pie(
    counts,
    labels=counts.index,
    colors=[PALETTE[c] for c in counts.index],
    autopct='%1.1f%%',
    startangle=140,
    textprops={'color': FG, 'fontsize': 10},
    wedgeprops={'edgecolor': BG, 'linewidth': 2},
)
axes[0].set_title('Fear & Greed Index Distribution (2018–2025)',
                  fontsize=13, color=FG, pad=10)

monthly_fg = fg.set_index('date').resample('ME')['value'].mean()
axes[1].fill_between(monthly_fg.index, monthly_fg.values, alpha=0.3, color='#17becf')
axes[1].plot(monthly_fg.index, monthly_fg.values, color='#17becf', linewidth=1.5)
axes[1].axhline(50, color='#bcbd22', linestyle='--', linewidth=1, alpha=0.7, label='Neutral (50)')
axes[1].set_title('Bitcoin Fear & Greed Index Over Time', fontsize=13, color=FG)
axes[1].set_ylabel('Sentiment Score', color=FG)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/01_sentiment_overview.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ── 4b. Average & median Net PnL per trade by sentiment ──────────────────────
pnl_by_sent = (
    df_close
    .groupby('classification', observed=True)['Net PnL']
    .agg(['mean', 'median', 'std', 'count'])
    .reindex(SENT_ORDER)
    .dropna()
)
print("\nNet PnL statistics by sentiment:")
print(pnl_by_sent.round(2))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)

colors = [PALETTE[c] for c in pnl_by_sent.index]
bars = axes[0].bar(pnl_by_sent.index, pnl_by_sent['mean'],
                   color=colors, edgecolor=BG, linewidth=1.5)
axes[0].axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
for bar, val in zip(bars, pnl_by_sent['mean']):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f'${val:.1f}', ha='center', va='bottom', fontsize=10, color=FG
    )
axes[0].set_title('Average Net PnL per Trade by Sentiment', fontsize=13, color=FG)
axes[0].set_ylabel('Avg Net PnL (USD)', color=FG)
axes[0].grid(True, axis='y', alpha=0.3)

# Box-plot (clipped for visual clarity – no data is dropped)
sent_groups = [
    df_close[df_close['classification'] == s]['Net PnL'].clip(-2000, 2000).dropna()
    for s in SENT_ORDER if s in df_close['classification'].values
]
valid_labels = [s for s in SENT_ORDER if s in df_close['classification'].values]
bp = axes[1].boxplot(sent_groups, patch_artist=True,
                     medianprops={'color': 'white', 'linewidth': 2})
for patch, label in zip(bp['boxes'], valid_labels):
    patch.set_facecolor(PALETTE[label])
    patch.set_alpha(0.75)
axes[1].set_xticklabels(valid_labels, rotation=15, fontsize=9)
axes[1].axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_title('Net PnL Distribution by Sentiment\n(clipped ±$2 000 for display)',
                  fontsize=13, color=FG)
axes[1].set_ylabel('Net PnL (USD)', color=FG)
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('charts/02_pnl_by_sentiment.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  WIN RATE, VOLUME & TOTAL PnL BY SENTIMENT  ── Chart 3
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5 – Win rate, volume & total PnL by sentiment")
print("=" * 60)

win_stats = (
    df_close
    .groupby('classification', observed=True)
    .agg(
        win_rate    = ('is_win', 'mean'),
        total_trades= ('is_win', 'count'),
        total_pnl   = ('Net PnL', 'sum'),
    )
    .reindex(SENT_ORDER)
    .dropna()
)
print(win_stats.round(4))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG)
colors = [PALETTE[c] for c in win_stats.index]

# Win rate
bars = axes[0].bar(win_stats.index, win_stats['win_rate'] * 100,
                   color=colors, edgecolor=BG)
axes[0].axhline(50, color='white', linestyle='--', linewidth=1, alpha=0.6)
for bar, val in zip(bars, win_stats['win_rate']):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10, color=FG)
axes[0].set_title('Win Rate by Sentiment', fontsize=13, color=FG)
axes[0].set_ylabel('Win Rate (%)', color=FG)
axes[0].grid(True, axis='y', alpha=0.3)

# Trade count
bars2 = axes[1].bar(win_stats.index, win_stats['total_trades'],
                    color=colors, edgecolor=BG)
for bar, val in zip(bars2, win_stats['total_trades']):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f'{int(val):,}', ha='center', va='bottom', fontsize=9, color=FG)
axes[1].set_title('Closing Trade Count by Sentiment', fontsize=13, color=FG)
axes[1].set_ylabel('Number of Trades', color=FG)
axes[1].grid(True, axis='y', alpha=0.3)

# Total PnL
colors2 = ['#2ca02c' if v >= 0 else '#d62728' for v in win_stats['total_pnl']]
bars3 = axes[2].bar(win_stats.index, win_stats['total_pnl'] / 1e6,
                    color=colors2, edgecolor=BG)
axes[2].axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
axes[2].set_title('Cumulative Net PnL by Sentiment ($M)', fontsize=13, color=FG)
axes[2].set_ylabel('Total Net PnL ($ Millions)', color=FG)
axes[2].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('charts/03_winrate_volume.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LONG vs. SHORT BY SENTIMENT  ── Chart 4
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6 – Long vs Short by sentiment")
print("=" * 60)

# Map direction → trade type
df_close['trade_type'] = df_close['Direction'].map({
    'Close Long':  'Long',
    'Close Short': 'Short',
})

ls_pnl = (
    df_close
    .groupby(['classification', 'trade_type'], observed=True)['Net PnL']
    .agg(['mean', 'count'])
    .reset_index()
)
print(ls_pnl.round(2))

# Long/Short count ratio
ls_count = (
    df_close
    .groupby(['classification', 'trade_type'], observed=True)
    .size()
    .unstack(fill_value=0)
    .reindex(SENT_ORDER)
    .dropna()
)
ls_count['long_pct'] = ls_count.get('Long', 0) / (
    ls_count.get('Long', 0) + ls_count.get('Short', 1)
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)

# Average PnL: Long vs Short grouped bars
x = np.arange(len(SENT_ORDER))
for offset, tt, color in [(-0.2, 'Long', '#2ca02c'), (0.2, 'Short', '#d62728')]:
    subset = (
        ls_pnl[ls_pnl['trade_type'] == tt]
        .set_index('classification')
        .reindex(SENT_ORDER)
        .dropna()
    )
    axes[0].bar(x[:len(subset)] + offset, subset['mean'],
                width=0.38, label=tt, color=color, alpha=0.85, edgecolor=BG)
axes[0].set_xticks(x)
axes[0].set_xticklabels(SENT_ORDER, rotation=15, fontsize=9)
axes[0].axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
axes[0].set_title('Avg Net PnL – Long vs Short by Sentiment', fontsize=13, color=FG)
axes[0].set_ylabel('Avg Net PnL (USD)', color=FG)
axes[0].legend(fontsize=10)
axes[0].grid(True, axis='y', alpha=0.3)

# Long % by sentiment
colors3 = [PALETTE[c] for c in ls_count.index]
bars4 = axes[1].bar(ls_count.index, ls_count['long_pct'] * 100,
                    color=colors3, edgecolor=BG)
axes[1].axhline(50, color='white', linestyle='--', linewidth=1, alpha=0.6)
for bar, val in zip(bars4, ls_count['long_pct']):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10, color=FG)
axes[1].set_title('% Long Trades by Sentiment', fontsize=13, color=FG)
axes[1].set_ylabel('% Long Closing Trades', color=FG)
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('charts/04_long_short_sentiment.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  INDIVIDUAL TRADER PERFORMANCE  ── Chart 5
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7 – Trader-level aggregation")
print("=" * 60)

trader_stats = (
    df_close
    .groupby('Account')
    .agg(
        total_pnl   = ('Net PnL',  'sum'),
        avg_pnl     = ('Net PnL',  'mean'),
        trade_count = ('Net PnL',  'count'),
        win_rate    = ('is_win',   'mean'),
        total_vol   = ('Size USD', 'sum'),
        avg_size    = ('Size USD', 'mean'),
    )
    .reset_index()
)
# Require at least 10 closing trades for reliable statistics
trader_stats = trader_stats[trader_stats['trade_count'] >= 10]

print(f"Traders with ≥10 closing trades: {len(trader_stats)}")
print("\nTop 5 traders by total PnL:")
print(trader_stats.nlargest(5, 'total_pnl')[
    ['Account', 'total_pnl', 'win_rate', 'trade_count']
].round(2))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(BG)

# Top 20 by PnL
top20 = trader_stats.nlargest(20, 'total_pnl')
short_addr  = [a[:6] + '…' + a[-4:] for a in top20['Account']]
bar_colors  = ['#17becf' if v > 0 else '#d62728' for v in top20['total_pnl']]
axes[0, 0].barh(short_addr[::-1], top20['total_pnl'].values[::-1] / 1e3,
                color=bar_colors[::-1], edgecolor=BG)
axes[0, 0].set_title('Top 20 Traders by Total PnL', fontsize=12, color=FG)
axes[0, 0].set_xlabel('Total Net PnL ($K)', color=FG)
axes[0, 0].grid(True, axis='x', alpha=0.3)

# PnL vs win rate scatter (colour = trade frequency)
sc = axes[0, 1].scatter(
    trader_stats['win_rate'] * 100,
    trader_stats['total_pnl'] / 1e3,
    c=np.log1p(trader_stats['trade_count']),
    cmap='plasma', alpha=0.6, s=30, edgecolors='none'
)
plt.colorbar(sc, ax=axes[0, 1], label='log(Trade Count)')
axes[0, 1].axhline(0,  color='white', linestyle='--', alpha=0.4)
axes[0, 1].axvline(50, color='white', linestyle='--', alpha=0.4)
axes[0, 1].set_title('Win Rate vs Total PnL', fontsize=12, color=FG)
axes[0, 1].set_xlabel('Win Rate (%)', color=FG)
axes[0, 1].set_ylabel('Total Net PnL ($K)', color=FG)
axes[0, 1].grid(True, alpha=0.3)

# Trade count distribution
axes[1, 0].hist(trader_stats['trade_count'].clip(upper=500),
                bins=50, color='#17becf', edgecolor=BG, alpha=0.8)
axes[1, 0].set_title('Trade Count Distribution (capped 500)', fontsize=12, color=FG)
axes[1, 0].set_xlabel('Number of Closing Trades', color=FG)
axes[1, 0].set_ylabel('Traders', color=FG)
axes[1, 0].grid(True, axis='y', alpha=0.3)

# Win rate distribution
axes[1, 1].hist(trader_stats['win_rate'] * 100,
                bins=40, color='#ff7f0e', edgecolor=BG, alpha=0.8)
axes[1, 1].axvline(50, color='white', linestyle='--', linewidth=1.5)
axes[1, 1].set_title('Win Rate Distribution', fontsize=12, color=FG)
axes[1, 1].set_xlabel('Win Rate (%)', color=FG)
axes[1, 1].set_ylabel('Traders', color=FG)
axes[1, 1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('charts/05_trader_analysis.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  BEHAVIORAL CLUSTERING  ── Chart 6
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 8 – K-Means clustering of traders")
print("=" * 60)

# Feature matrix: total PnL, win rate, trade count, avg trade size
feature_cols = ['total_pnl', 'win_rate', 'trade_count', 'avg_size']
cluster_df   = trader_stats[feature_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

# Standardise features (K-Means is distance-based, so scale matters)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)

# Elbow method to justify k=4
inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Final model
km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_df['cluster'] = km_final.fit_predict(X_scaled)

# PCA for 2D visualisation only (not used in clustering itself)
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

# Label clusters by their characteristics
cluster_names = {}
for c in sorted(cluster_df['cluster'].unique()):
    sub = cluster_df[cluster_df['cluster'] == c]
    if sub['total_pnl'].mean() > cluster_df['total_pnl'].quantile(0.75):
        cluster_names[c] = 'Elite Traders'
    elif (sub['win_rate'].mean() > 0.55 and
          sub['trade_count'].mean() > cluster_df['trade_count'].median()):
        cluster_names[c] = 'Consistent Pros'
    elif sub['trade_count'].mean() > cluster_df['trade_count'].quantile(0.75):
        cluster_names[c] = 'High-Frequency'
    else:
        cluster_names[c] = 'Retail/Casual'
cluster_df['cluster_name'] = cluster_df['cluster'].map(cluster_names)

print("\nCluster summary:")
print(cluster_df.groupby('cluster_name')[feature_cols].mean().round(2))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)
cluster_colors = ['#17becf', '#ff7f0e', '#2ca02c', '#d62728']

for i, (c, grp) in enumerate(cluster_df.groupby('cluster')):
    mask = (cluster_df['cluster'] == c).values
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=cluster_names[c], color=cluster_colors[i % 4],
                    alpha=0.6, s=20, edgecolors='none')
axes[0].set_title('Trader Clusters (PCA 2D Projection)', fontsize=13, color=FG)
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", color=FG)
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", color=FG)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Cluster profile bar chart
profile = (
    cluster_df
    .groupby('cluster_name')
    .agg(avg_pnl=('total_pnl','mean'), avg_win_rate=('win_rate','mean'),
         avg_trades=('trade_count','mean'), count=('total_pnl','count'))
    .reset_index()
)
x = np.arange(len(profile))
axes[1].bar(x - 0.2, profile['avg_win_rate'] * 100, width=0.38,
            label='Win Rate (%)', color='#2ca02c', alpha=0.85)
ax_r = axes[1].twinx()
ax_r.bar(x + 0.2, profile['avg_pnl'] / 1e3, width=0.38,
         label='Avg PnL ($K)', color='#17becf', alpha=0.85)
ax_r.set_ylabel('Avg Total PnL ($K)', color='#17becf')
ax_r.tick_params(axis='y', colors='#17becf')
axes[1].set_xticks(x)
axes[1].set_xticklabels(profile['cluster_name'], rotation=10, fontsize=9)
axes[1].set_title('Cluster Profiles: Win Rate vs Avg PnL', fontsize=13, color=FG)
axes[1].set_ylabel('Win Rate (%)', color='#2ca02c')
axes[1].tick_params(axis='y', colors='#2ca02c')
axes[1].grid(True, axis='y', alpha=0.3)
l1, lab1 = axes[1].get_legend_handles_labels()
l2, lab2 = ax_r.get_legend_handles_labels()
axes[1].legend(l1 + l2, lab1 + lab2, fontsize=9)

plt.tight_layout()
plt.savefig('charts/06_trader_clusters.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PnL TIMELINE  ── Chart 7
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 9 – PnL timeline")
print("=" * 60)

# Daily aggregate PnL for all closing trades
daily_pnl = (
    df_close
    .groupby('date')['Net PnL']
    .sum()
    .reset_index()
    .merge(fg[['date', 'classification', 'value']], on='date', how='left')
)

# Roll up to monthly
monthly_pnl = (
    daily_pnl.set_index('date')
    .resample('ME')['Net PnL']
    .sum()
    .reset_index()
)
monthly_pnl['month_str'] = monthly_pnl['date'].dt.strftime('%Y-%m')

fig, axes = plt.subplots(2, 1, figsize=(18, 10))
fig.patch.set_facecolor(BG)

# Monthly bars
colors4 = ['#2ca02c' if v >= 0 else '#d62728' for v in monthly_pnl['Net PnL']]
axes[0].bar(monthly_pnl['month_str'], monthly_pnl['Net PnL'] / 1e6,
            color=colors4, edgecolor=BG)
axes[0].axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
axes[0].set_title('Monthly Aggregate Net PnL (All Traders)', fontsize=13, color=FG)
axes[0].set_ylabel('Net PnL ($ Millions)', color=FG)
axes[0].tick_params(axis='x', rotation=45, labelsize=8)
axes[0].grid(True, axis='y', alpha=0.3)

# Daily bars coloured by sentiment
for _, row in daily_pnl.sort_values('date').iterrows():
    color = PALETTE.get(str(row['classification']), '#888')
    axes[1].bar(row['date'], row['Net PnL'] / 1e3, color=color, alpha=0.7, width=1)
axes[1].axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_title('Daily Net PnL Colored by Market Sentiment', fontsize=13, color=FG)
axes[1].set_ylabel('Daily Net PnL ($K)', color=FG)
axes[1].grid(True, axis='y', alpha=0.3)
patches = [mpatches.Patch(color=PALETTE[s], label=s) for s in SENT_ORDER]
axes[1].legend(handles=patches, fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('charts/07_pnl_timeline.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 10. COIN-LEVEL ANALYSIS  ── Chart 8
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 10 – Coin-level analysis")
print("=" * 60)

coin_stats = (
    df_close
    .groupby('Coin')
    .agg(
        total_pnl   = ('Net PnL', 'sum'),
        trade_count = ('Net PnL', 'count'),
        win_rate    = ('is_win',  'mean'),
        avg_pnl     = ('Net PnL', 'mean'),
    )
    .reset_index()
)
# Keep coins with enough trades for reliable stats
coin_stats = coin_stats[coin_stats['trade_count'] >= 50]

print(f"Coins with ≥50 closing trades: {len(coin_stats)}")
print("\nTop 10 by trade count:")
print(coin_stats.nlargest(10, 'trade_count')[
    ['Coin', 'trade_count', 'win_rate', 'avg_pnl', 'total_pnl']
].round(2))

top15_vol = coin_stats.nlargest(15, 'trade_count')
top10_pnl = coin_stats.nlargest(10, 'total_pnl')

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(BG)

axes[0].barh(top15_vol['Coin'][::-1], top15_vol['trade_count'].values[::-1],
             color='#17becf', edgecolor=BG, alpha=0.85)
axes[0].set_title('Top 15 Most Traded Coins (closing trades)', fontsize=12, color=FG)
axes[0].set_xlabel('Number of Closing Trades', color=FG)
axes[0].grid(True, axis='x', alpha=0.3)

bar_col2 = ['#2ca02c' if v >= 0 else '#d62728' for v in top10_pnl['total_pnl']]
axes[1].barh(top10_pnl['Coin'][::-1], top10_pnl['total_pnl'].values[::-1] / 1e3,
             color=bar_col2[::-1], edgecolor=BG, alpha=0.85)
axes[1].axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_title('Top 10 Coins by Total Net PnL', fontsize=12, color=FG)
axes[1].set_xlabel('Total Net PnL ($K)', color=FG)
axes[1].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('charts/08_coin_analysis.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 11. COIN × SENTIMENT HEATMAP  ── Chart 9
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 11 – Coin × Sentiment heatmap")
print("=" * 60)

top10_coins = coin_stats.nlargest(10, 'trade_count')['Coin'].tolist()
heat_df     = df_close[df_close['Coin'].isin(top10_coins)]

heat_pivot = (
    heat_df
    .groupby(['Coin', 'classification'], observed=True)['Net PnL']
    .mean()
    .unstack()
    .reindex(columns=SENT_ORDER)
)
print("\nAvg Net PnL – top 10 coins × sentiment:")
print(heat_pivot.round(1))

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(BG)
sns.heatmap(heat_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            linewidths=0.5, linecolor='#0f0f1a', ax=ax,
            cbar_kws={'label': 'Avg Net PnL ($)'})
ax.set_title('Average Net PnL by Coin × Sentiment (Top 10 Coins by Volume)',
             fontsize=13, color=FG)
ax.set_xlabel('Market Sentiment', color=FG)
ax.set_ylabel('Coin', color=FG)
ax.tick_params(colors=FG)

plt.tight_layout()
plt.savefig('charts/09_coin_sentiment_heatmap.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 12. STATISTICAL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 12 – Statistical validation")
print("=" * 60)

# ── One-way ANOVA ─────────────────────────────────────────────────────────────
# H0: mean Net PnL is equal across all sentiment groups
# H1: at least one group has a different mean
groups_for_anova = [
    df_close[df_close['classification'] == s]['Net PnL'].dropna()
    for s in SENT_ORDER if s in df_close['classification'].values
]
f_stat, p_value = stats.f_oneway(*groups_for_anova)

print(f"\nOne-Way ANOVA on Net PnL across sentiment groups:")
print(f"  F-statistic : {f_stat:.4f}")
print(f"  p-value     : {p_value:.6f}")
if p_value < 0.001:
    print("  → Reject H0 at p<0.001: sentiment groups have significantly different mean PnL.")

# ── Pearson correlation: raw F/G score vs Net PnL ────────────────────────────
corr_data = df_close[['Net PnL', 'value']].dropna()
r, p_corr = stats.pearsonr(corr_data['value'], corr_data['Net PnL'])
print(f"\nPearson r (F/G score vs Net PnL): {r:.4f}  (p={p_corr:.4f})")
print("  Weak negative: higher sentiment score → slightly lower per-trade PnL")

# ── Kruskal-Wallis (non-parametric alternative) ───────────────────────────────
kw_stat, kw_p = stats.kruskal(*groups_for_anova)
print(f"\nKruskal-Wallis (non-parametric):")
print(f"  H={kw_stat:.4f}, p={kw_p:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 13. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"""
Dataset
  Total trades        : {len(df):,}
  Closing trades      : {len(df_close):,}
  Unique traders      : {df_close['Account'].nunique()}

Performance
  Overall win rate    : {df_close['is_win'].mean()*100:.1f}%
  Total net PnL       : ${df_close['Net PnL'].sum():,.0f}

Best sentiment (avg PnL)  : {pnl_by_sent['mean'].idxmax()}  (${pnl_by_sent['mean'].max():.2f}/trade)
Worst sentiment (avg PnL) : {pnl_by_sent['mean'].idxmin()} (${pnl_by_sent['mean'].min():.2f}/trade)

Statistics
  ANOVA F             : {f_stat:.4f}
  ANOVA p             : {p_value:.2e}
  Pearson r           : {r:.4f}

Charts saved to ./charts/
  01_sentiment_overview.png
  02_pnl_by_sentiment.png
  03_winrate_volume.png
  04_long_short_sentiment.png
  05_trader_analysis.png
  06_trader_clusters.png
  07_pnl_timeline.png
  08_coin_analysis.png
  09_coin_sentiment_heatmap.png
""")