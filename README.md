# 📊 GreedGauge

> **Gauging greed to understand trader behavior.**

GreedGauge is a data-science project that studies the relationship between **Bitcoin market sentiment** and **Hyperliquid trader performance**. It combines historical trade data with the Bitcoin Fear & Greed Index to analyze profitability, win rates, trading direction, trader behavior, coin-level performance, and sentiment-driven patterns.

## ✨ Features

- 📈 Net PnL analysis across market-sentiment regimes
- 😨 Fear & Greed Index integration
- 📊 Win-rate, trade-volume, and cumulative-PnL analysis
- ↔️ Long vs. Short performance comparison
- 👤 Trader-level performance analysis
- 🧬 K-Means behavioral clustering with PCA visualization
- 🪙 Coin-level performance analysis
- 🔥 Coin × sentiment heatmap
- 📐 Statistical validation using ANOVA, Pearson correlation, and Kruskal-Wallis
- 🌐 Streamlit dashboard for quick interactive exploration
- 🖼️ Nine generated analysis charts

## 🧠 Analysis Pipeline

The full analysis in `crypto.py` covers:

1. Load and clean trade and sentiment datasets
2. Merge trades with the Fear & Greed Index by date
3. Exploratory data analysis
4. PnL by sentiment
5. Win rate, volume, and total PnL by sentiment
6. Long vs. Short performance
7. Individual trader performance
8. K-Means behavioral clustering + PCA
9. Daily and monthly PnL timelines
10. Coin-level performance
11. Coin × sentiment heatmap
12. Statistical validation
13. Final summary and chart generation

## 📊 Streamlit Dashboard

`app.py` provides a lightweight interactive dashboard using the same project datasets.

It currently displays:

- Total trades
- Unique traders
- Unique coins
- Cumulative net PnL over time
- Average PnL by market sentiment
- Trade counts by sentiment

Run it with:

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
GreedGauge/
├── app.py
├── crypto.py
├── run_analysis.py
├── requirements.txt
├── README.md
├── .gitignore
├── compressed_data.csv.gz
├── fear_greed_index.csv
└── charts/
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

The trade dataset is stored as `compressed_data.csv.gz`. The wrapper temporarily decompresses it, runs `crypto.py`, and removes the temporary uncompressed file when finished.

```bash
python run_analysis.py
```

The nine charts are generated inside `charts/`.

### 4. Run the dashboard

```bash
streamlit run app.py
```

## 📦 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Data analysis and application logic |
| Pandas | Data loading, cleaning, aggregation, and merging |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualizations and heatmaps |
| Scikit-learn | K-Means, PCA, and feature scaling |
| SciPy | Statistical testing and correlation analysis |
| Streamlit | Interactive dashboard |

## 🗂️ Datasets

### Hyperliquid Historical Trades

Stored as `compressed_data.csv.gz`. The dataset contains historical trader activity including account, coin, execution price, trade size, direction, closed PnL, fees, and timestamps.

### Bitcoin Fear & Greed Index

Stored as `fear_greed_index.csv`. It contains daily sentiment observations ranging from **Extreme Fear** to **Extreme Greed**.

## 📈 Generated Charts

| Chart | Analysis |
|---|---|
| `01_sentiment_overview.png` | Fear & Greed distribution and timeline |
| `02_pnl_by_sentiment.png` | PnL statistics by sentiment |
| `03_winrate_volume.png` | Win rate, volume, and total PnL |
| `04_long_short_sentiment.png` | Long vs. Short performance |
| `05_trader_analysis.png` | Trader performance and distributions |
| `06_trader_clusters.png` | K-Means clusters and PCA projection |
| `07_pnl_timeline.png` | Monthly and daily PnL timeline |
| `08_coin_analysis.png` | Coin-level performance |
| `09_coin_sentiment_heatmap.png` | Coin × sentiment PnL comparison |

## ⚠️ Limitations

- The trader sample represents activity from a single exchange and should not be treated as representative of the entire crypto market.
- Historical relationships between sentiment and PnL do not establish causation.
- Trader clustering depends on the selected features and K-Means configuration.
- Results are sensitive to the available time period and dataset quality.

## 📌 Note

The analysis is intended for **data-science and exploratory research purposes**. Historical patterns should not be interpreted as financial advice or guaranteed future trading signals.
