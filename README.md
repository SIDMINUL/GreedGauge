# 📊 GreedGauge

> **Gauging greed to understand trader behavior.**

GreedGauge is an interactive **crypto trading analytics dashboard** that explores the relationship between **Bitcoin market sentiment** and **Hyperliquid trader performance**. It combines historical trading activity with the Bitcoin Fear & Greed Index to study profitability, win rate, trading behavior, trader performance, coin-level patterns, activity, and statistical relationships.

🌐 **Live Dashboard:** https://greedgauge0808.streamlit.app/

## ✨ What the Project Does

GreedGauge has two complementary parts:

1. **Interactive Streamlit dashboard** — explore the data using filters and dynamic analytics.
2. **Full Python analysis pipeline** — reproduce the deeper statistical and machine-learning analysis and generate nine research charts.

## 📊 Interactive Dashboard

The Streamlit application provides:

- 📅 Interactive **date-range filtering**
- 🪙 **Coin-level filtering**
- 📈 Cumulative **Net PnL over time**
- 💰 Total Net PnL
- 🎯 Overall **win rate**
- 👥 Trader and trade-count KPIs
- 😨😐🤑 **PnL by Fear & Greed sentiment**
- ↕️ Long vs. Short performance
- 🪙 Top coins ranked by Net PnL
- 🏆 Trader performance leaderboard
- 🕐 Trading activity by day and hour
- 📐 Pearson correlation between sentiment value and Net PnL
- 🔍 Data diagnostics and methodology information

The dashboard matches trading activity with the nearest available daily Fear & Greed observation within a **1-day tolerance**. When live sentiment data is available, the dashboard uses the Alternative.me Fear & Greed API; otherwise it falls back to the bundled CSV dataset.

## 🧠 Full Analysis Pipeline

`crypto.py` performs the deeper exploratory and statistical analysis:

1. Load and clean Hyperliquid trade data
2. Load Bitcoin Fear & Greed Index data
3. Align trading activity with daily sentiment
4. Perform exploratory data analysis
5. Analyze PnL across sentiment regimes
6. Compare win rate, trade volume, and profitability
7. Analyze Long vs. Short behavior
8. Evaluate individual trader performance
9. Perform K-Means behavioral clustering
10. Reduce trader features with PCA
11. Analyze daily and monthly PnL trends
12. Analyze coin-level performance
13. Build coin × sentiment comparisons and statistical tests

## 🖼️ Generated Analysis Charts

The `charts/` directory contains nine visual outputs produced by the full analysis pipeline:

| Chart | Purpose |
|---|---|
| `01_sentiment_overview.png` | Fear & Greed distribution and timeline |
| `02_pnl_by_sentiment.png` | PnL statistics across sentiment regimes |
| `03_winrate_volume.png` | Win rate, trading volume, and profitability |
| `04_long_short_sentiment.png` | Long vs. Short performance by sentiment |
| `05_trader_analysis.png` | Trader-level performance analysis |
| `06_trader_clusters.png` | K-Means behavioral clusters with PCA |
| `07_pnl_timeline.png` | Daily and monthly PnL trends |
| `08_coin_analysis.png` | Coin-level performance |
| `09_coin_sentiment_heatmap.png` | Coin × sentiment PnL comparison |

These images are **static research outputs**. The Streamlit dashboard independently recreates the most useful analyses interactively from the underlying datasets.

## 📁 Project Structure

```text
GreedGauge/
├── app.py                  # Interactive Streamlit dashboard
├── crypto.py               # Full analysis and chart-generation pipeline
├── run_analysis.py         # Runs crypto.py using compressed trade data
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
├── compressed_data.csv.gz  # Compressed Hyperliquid trade dataset
├── fear_greed_index.csv    # Bundled Fear & Greed dataset
└── charts/                 # Nine generated analysis charts
    ├── 01_sentiment_overview.png
    ├── 02_pnl_by_sentiment.png
    ├── 03_winrate_volume.png
    ├── 04_long_short_sentiment.png
    ├── 05_trader_analysis.png
    ├── 06_trader_clusters.png
    ├── 07_pnl_timeline.png
    ├── 08_coin_analysis.png
    └── 09_coin_sentiment_heatmap.png
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SIDMINUL/GreedGauge.git
cd GreedGauge
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full analysis

The trade dataset is stored as `compressed_data.csv.gz`. `run_analysis.py` temporarily decompresses the file, runs `crypto.py`, and cleans up the temporary uncompressed dataset.

```bash
python run_analysis.py
```

The nine charts will be generated in `charts/`.

### 4. Run the interactive dashboard

```bash
streamlit run app.py
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Data analysis and application logic |
| Streamlit | Interactive analytics dashboard |
| Pandas | Data cleaning, transformation, aggregation, and merging |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Seaborn | Statistical visualizations and heatmaps |
| Scikit-learn | K-Means clustering, PCA, and feature processing |
| SciPy | Statistical testing and analysis |
| Requests | Live Fear & Greed API integration |

## 🗂️ Datasets

### Hyperliquid Historical Trades

`compressed_data.csv.gz` contains historical trading activity, including trader accounts, coins, execution prices, trade sizes, directions, closed PnL, fees, and timestamps.

### Bitcoin Fear & Greed Index

`fear_greed_index.csv` contains daily Bitcoin market-sentiment observations ranging from **Extreme Fear** to **Extreme Greed**. The dashboard can supplement this bundled dataset with current data from the Alternative.me API when available.

## 🔬 Methodology

### Net PnL

```text
Net PnL = Closed PnL − Fees
```

Trading timestamps are converted into calendar dates using automatic Unix timestamp-unit detection. Sentiment observations are normalized to daily dates and matched to trading activity using a nearest-date join with a maximum 1-day tolerance.

### Statistical Analysis

The full analysis includes descriptive statistics, Pearson correlation, ANOVA, and Kruskal-Wallis testing where applicable. These analyses identify relationships and differences in the historical data but **do not establish causation**.

### Trader Segmentation

Trader behavior is analyzed using aggregated trading features and K-Means clustering, followed by PCA for lower-dimensional visualization.

## ⚠️ Limitations

- The trader sample represents activity from a specific exchange and is not representative of the entire cryptocurrency market.
- Historical correlation does not imply causation.
- Clustering results depend on feature selection and model configuration.
- Results depend on the quality and time coverage of the underlying datasets.
- Live sentiment availability depends on the external API.
- This project is intended for analytics and research, not trading execution.

## 📌 Disclaimer

GreedGauge is an **exploratory data-science project**. Historical patterns and statistical relationships should not be interpreted as financial advice, trading recommendations, or guarantees of future performance.

## 👨‍💻 Author

**Abdul Momin Siddiqui**

GitHub: https://github.com/SIDMINUL
