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
    trades["date"] = pd.to_datetime(trades["Timestamp"], unit="ms").dt.normalize()
    trades["Closed PnL"] = pd.to_numeric(trades["Closed PnL"], errors="coerce").fillna(0)
    trades["Fee"] = pd.to_numeric(trades["Fee"], errors="coerce").fillna(0)
    trades["Net PnL"] = trades["Closed PnL"] - trades["Fee"]
    return trades, sentiment

try:
    trades, sentiment = load_data()
except Exception as exc:
    st.error(f"Could not load the project datasets: {exc}")
    st.stop()

st.metric("Trades", f"{len(trades):,}")
st.metric("Traders", f"{trades['Account'].nunique():,}")
st.metric("Coins", f"{trades['Coin'].nunique():,}")

st.subheader("Net PnL")
st.line_chart(trades.groupby("date")["Net PnL"].sum().cumsum())

st.subheader("PnL by Sentiment")
possible_sentiment = [c for c in sentiment.columns if c.lower() in {"classification", "sentiment", "value_classification"}]
if possible_sentiment:
    sentiment_col = possible_sentiment[0]
    date_candidates = [c for c in sentiment.columns if c.lower() in {"date", "timestamp", "datetime"}]
    if date_candidates:
        sentiment[date_candidates[0]] = pd.to_datetime(sentiment[date_candidates[0]], errors="coerce").dt.normalize()
        merged = trades.merge(sentiment[[date_candidates[0], sentiment_col]], left_on="date", right_on=date_candidates[0], how="left")
        grouped = merged.groupby(sentiment_col)["Net PnL"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        st.dataframe(grouped, use_container_width=True)
        st.bar_chart(grouped["mean"])
else:
    st.info("Sentiment column format was not recognized. The raw dataset is still available in the repository.")
