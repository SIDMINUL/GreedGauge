import gzip
import streamlit as st
import pandas as pd

st.set_page_config(page_title="GreedGauge", page_icon="📊", layout="wide")
st.title("📊 GreedGauge")
st.caption("Bitcoin market sentiment vs. Hyperliquid trader performance")


@st.cache_data
def load_data():
    with gzip.open("compressed_data.csv.gz", "rb") as f:
        trades = pd.read_csv(f)
    sentiment = pd.read_csv("fear_greed_index.csv")

    trades.columns = trades.columns.str.strip()
    sentiment.columns = sentiment.columns.str.strip()

    trades["Timestamp"] = pd.to_numeric(trades["Timestamp"], errors="coerce")
    trades["date"] = pd.to_datetime(trades["Timestamp"], unit="ms", errors="coerce").dt.normalize()
    trades["Closed PnL"] = pd.to_numeric(trades["Closed PnL"], errors="coerce").fillna(0)
    trades["Fee"] = pd.to_numeric(trades["Fee"], errors="coerce").fillna(0)
    trades["Net PnL"] = trades["Closed PnL"] - trades["Fee"]

    return trades, sentiment


try:
    trades, sentiment = load_data()
except Exception as exc:
    st.error(f"Could not load the project datasets: {exc}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Trades", f"{len(trades):,}")
col2.metric("Traders", f"{trades['Account'].nunique():,}")
col3.metric("Coins", f"{trades['Coin'].nunique():,}")

st.subheader("Net PnL")
net_pnl = trades.groupby("date", dropna=True)["Net PnL"].sum().cumsum()
net_pnl.index.name = "Date"
st.line_chart(net_pnl)

st.subheader("PnL by Sentiment")
possible_sentiment = [
    c for c in sentiment.columns
    if c.lower() in {"classification", "sentiment", "value_classification"}
]

if possible_sentiment:
    sentiment_col = possible_sentiment[0]
    date_candidates = [
        c for c in sentiment.columns
        if c.lower() in {"date", "timestamp", "datetime"}
    ]

    if date_candidates:
        date_col = date_candidates[0]
        sentiment[date_col] = pd.to_datetime(
            sentiment[date_col], errors="coerce"
        ).dt.normalize()
        sentiment[sentiment_col] = sentiment[sentiment_col].astype("string")

        merged = trades.merge(
            sentiment[[date_col, sentiment_col]],
            left_on="date",
            right_on=date_col,
            how="left",
        )
        grouped = (
            merged.dropna(subset=[sentiment_col])
            .groupby(sentiment_col, observed=True)["Net PnL"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )

        grouped.index.name = "Sentiment"
        st.dataframe(grouped, width="stretch")
        st.bar_chart(grouped["mean"])
    else:
        st.info("A date column was not found in the sentiment dataset.")
else:
    st.info(
        "Sentiment column format was not recognized. "
        "The raw dataset is still available in the repository."
    )
