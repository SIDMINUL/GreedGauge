import gzip

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="GreedGauge", page_icon="📊", layout="wide")
st.title("📊 GreedGauge")
st.caption("Bitcoin market sentiment vs. Hyperliquid trader performance")


@st.cache_data(ttl=3600)
def load_sentiment_data():
    """Load a full historical Fear & Greed series, preferring live data."""
    local = pd.read_csv("fear_greed_index.csv")
    local.columns = local.columns.str.strip()

    try:
        response = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": 3650, "format": "json"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", [])
        live = pd.DataFrame(rows)

        if not live.empty and "timestamp" in live.columns:
            live["date"] = pd.to_datetime(
                pd.to_numeric(live["timestamp"], errors="coerce"),
                unit="s",
                errors="coerce",
                utc=True,
            ).dt.tz_localize(None).dt.normalize()
            live["classification"] = live.get(
                "value_classification", live.get("classification")
            )
            live = live[["date", "classification"]].dropna(subset=["date", "classification"])
            live["classification"] = live["classification"].astype("string").str.strip()

            local["date"] = pd.to_datetime(local["date"], errors="coerce")
            local["classification"] = local["classification"].astype("string").str.strip()
            local = local[["date", "classification"]].dropna()

            combined = pd.concat([local, live], ignore_index=True)
            return combined.drop_duplicates("date", keep="last").sort_values("date")
    except Exception:
        pass

    local["date"] = pd.to_datetime(local["date"], errors="coerce")
    local["classification"] = local["classification"].astype("string").str.strip()
    return local.dropna(subset=["date", "classification"]).drop_duplicates("date").sort_values("date")


@st.cache_data
def load_data():
    with gzip.open("compressed_data.csv.gz", "rb") as f:
        trades = pd.read_csv(f)

    trades.columns = trades.columns.str.strip()

    timestamp = pd.to_numeric(trades["Timestamp"], errors="coerce")
    median_timestamp = timestamp.dropna().median()
    if median_timestamp >= 1e17:
        timestamp_unit = "ns"
    elif median_timestamp >= 1e14:
        timestamp_unit = "us"
    elif median_timestamp >= 1e11:
        timestamp_unit = "ms"
    else:
        timestamp_unit = "s"

    trades["date"] = (
        pd.to_datetime(timestamp, unit=timestamp_unit, errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    trades["Closed PnL"] = pd.to_numeric(trades["Closed PnL"], errors="coerce")
    trades["Fee"] = pd.to_numeric(trades["Fee"], errors="coerce").fillna(0)
    trades["Net PnL"] = trades["Closed PnL"].fillna(0) - trades["Fee"]

    sentiment = load_sentiment_data()
    return trades, sentiment, timestamp_unit


try:
    trades, sentiment, timestamp_unit = load_data()
except Exception as exc:
    st.error(f"Could not load the project datasets: {exc}")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Trades", f"{len(trades):,}")
m2.metric("Traders", f"{trades['Account'].nunique():,}")
m3.metric("Coins", f"{trades['Coin'].nunique():,}")

st.subheader("Net PnL")
pnl_timeline = (
    trades.dropna(subset=["date"])
    .groupby("date", as_index=True)["Net PnL"]
    .sum()
    .cumsum()
)
st.line_chart(pnl_timeline, width="stretch")

st.subheader("PnL by Sentiment")
pnl_trades = trades.dropna(subset=["date", "Closed PnL"]).copy()

if "Direction" in pnl_trades.columns:
    directions = pnl_trades["Direction"].astype("string").str.strip()
    closing_trades = pnl_trades[directions.isin({"Close Long", "Close Short"})].copy()
    if not closing_trades.empty:
        pnl_trades = closing_trades
        analysis_source = "standard closing-direction rows"
    else:
        analysis_source = "valid Closed PnL rows (close labels not recognized)"
else:
    analysis_source = "valid Closed PnL rows"

left = pnl_trades.sort_values("date").copy()
right = sentiment.rename(columns={"date": "sentiment_date", "classification": "sentiment"})
right = right.sort_values("sentiment_date")

merged = pd.merge_asof(
    left,
    right,
    left_on="date",
    right_on="sentiment_date",
    direction="nearest",
    tolerance=pd.Timedelta(days=1),
)

matched = merged["sentiment"].notna().sum()

if matched > 0:
    grouped = (
        merged.dropna(subset=["sentiment"])
        .groupby("sentiment", observed=True)["Net PnL"]
        .agg(mean="mean", count="count", total="sum")
        .sort_values("mean", ascending=False)
    )
    st.dataframe(grouped, width="stretch")
    st.bar_chart(grouped["mean"], width="stretch")
    st.caption(
        f"Matched {matched:,} trades using {analysis_source}. "
        f"Trade dates: {left['date'].min().date()} → {left['date'].max().date()} | "
        f"Fear & Greed dates: {right['sentiment_date'].min().date()} → "
        f"{right['sentiment_date'].max().date()} | timestamp unit: {timestamp_unit}"
    )
else:
    st.warning("No trade dates overlap with the available Fear & Greed history.")
    st.info(
        f"Trade dates: {left['date'].min().date()} → {left['date'].max().date()} | "
        f"Fear & Greed dates: {right['sentiment_date'].min().date()} → "
        f"{right['sentiment_date'].max().date()}"
    )

with st.expander("Data diagnostics"):
    st.write(f"Parsed trade timestamp unit: **{timestamp_unit}**")
    st.write(f"Trade rows with valid dates: **{trades['date'].notna().sum():,}**")
    st.write(f"Trade date range: **{trades['date'].min().date()} → {trades['date'].max().date()}**")
    st.write(f"Fear & Greed date range: **{sentiment['date'].min().date()} → {sentiment['date'].max().date()}**")
    st.write(f"Rows used for sentiment analysis: **{len(pnl_trades):,}**")
