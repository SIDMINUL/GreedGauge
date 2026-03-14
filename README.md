# 📊 GreedGauge

> **Gauging greed to trade smarter.**
> A data science project exploring the relationship between Bitcoin market sentiment and trader performance on the Hyperliquid decentralised exchange.

---

## 🧠 Overview

GreedGauge merges two powerful datasets — **Hyperliquid historical trade data** and the **Bitcoin Fear & Greed Index** — to uncover how market psychology drives trader profitability. The project surfaces hidden behavioral patterns, segments traders into archetypes, and delivers actionable insights for smarter crypto trading strategies.

This analysis was built as part of the **PrimeTrade.ai Junior Data Scientist Assignment**.

---

## 🔍 Key Findings

- 📈 Traders earn **3.2× higher average PnL** during Fear vs Greed markets ($112.6 vs $56.0 per trade)
- 🏆 Win rates peak during **Fear (86.6%)** and drop to their lowest during **Extreme Greed (75.8%)**
- 📉 **Extreme Greed = worst returns** — consistent across coins, directions, and trader types
- 🔁 Long trades outperform shorts in **every single sentiment regime**
- 🧬 4 distinct trader archetypes identified: Elite, Consistent Pros, High-Frequency, Retail/Casual
- ✅ Differences are **statistically significant** — ANOVA F=10.13, p<0.001

---

## 📁 Project Structure

```
GreedGauge/
│
├── analysis.py                  # Main analysis script (all 13 steps)
├── requirements.txt             # Python dependencies
├── README.md                    # You are here
│
├── data/
│   ├── compressed_data_csv      # Hyperliquid historical trades
│   └── fear_greed_index.csv     # Bitcoin Fear & Greed Index
│
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

---

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

### 3. Add the datasets
Place both data files inside a `data/` folder:
```
data/compressed_data_csv
data/fear_greed_index.csv
```

### 4. Run the analysis
```bash
python analysis.py
```

All 9 charts will be saved automatically to the `charts/` directory.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, merging |
| `numpy` | Numerical operations |
| `matplotlib` | Chart generation |
| `seaborn` | Heatmap visualisation |
| `scikit-learn` | K-Means clustering, PCA, StandardScaler |
| `scipy` | ANOVA, Pearson correlation, Kruskal-Wallis |

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📊 Analysis Pipeline

| Step | Description |
|---|---|
| 1 | Load & clean both datasets |
| 2 | Merge on date — join sentiment to every trade |
| 3 | Exploratory Data Analysis (EDA) |
| 4 | Average & median PnL by sentiment + box-plots |
| 5 | Win rate, volume & cumulative PnL by sentiment |
| 6 | Long vs Short performance by sentiment |
| 7 | Individual trader aggregation & ranking |
| 8 | K-Means behavioral clustering + PCA visualisation |
| 9 | Monthly & daily PnL timeline coloured by sentiment |
| 10 | Coin-level performance analysis |
| 11 | Coin × Sentiment cross-tab heatmap |
| 12 | Statistical validation (ANOVA, Pearson r, Kruskal-Wallis) |
| 13 | Final summary printout |

---

## 🗂️ Datasets

### Hyperliquid Historical Trades
- **211,224 rows** across 32 unique trader accounts
- Covers **April 2023 – June 2025**
- Columns: `Account`, `Coin`, `Execution Price`, `Size USD`, `Side`, `Direction`, `Closed PnL`, `Fee`, `Timestamp`, and more

### Bitcoin Fear & Greed Index
- **2,644 daily observations** from February 2018 – May 2025
- Score: 0 (Extreme Fear) → 100 (Extreme Greed)
- Categories: `Extreme Fear`, `Fear`, `Neutral`, `Greed`, `Extreme Greed`

---

## 📈 Sample Charts

| Chart | Insight |
|---|---|
| Sentiment Overview | Distribution of Fear/Greed days across 7 years |
| PnL by Sentiment | Fear-period trades earn 3× more than Greed |
| Win Rate by Sentiment | Win rate drops sharply during Extreme Greed |
| Long vs Short | Long bias profitable in every sentiment regime |
| Trader Clusters | 4 behavioural archetypes via K-Means + PCA |
| Coin × Sentiment Heatmap | BTC most stable; altcoins thrive in Fear only |

---

## 💡 Strategic Insights

1. **Scale up during Fear** — increase position size when the index reads 0–40
2. **Reduce exposure during Greed** — both win rate and PnL per trade deteriorate
3. **Stick to BTC for all-weather trading** — most consistent across all sentiment regimes
4. **Reserve high-beta altcoins for Fear dips** — their edge disappears during Greed
5. **Long bias was justified in 2023–2025** — shorts underperformed in every sentiment regime

---

## ⚠️ Limitations

- Small trader cohort (32 accounts) — findings may not generalise broadly
- No leverage-normalised returns — risk-adjusted comparison is limited
- Single exchange (Hyperliquid) — venue-specific dynamics may apply
- Timeframe bias — dataset spans a predominantly bullish market cycle

---

## 🤝 Acknowledgements

- **Hyperliquid** for the on-chain trade data
- **Alternative.me** for the Bitcoin Fear & Greed Index
- **PrimeTrade.ai** for the assignment brief and dataset

---

*GreedGauge — because knowing when the market is greedy is half the edge.*
