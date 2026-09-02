import gzip

import pandas as pd
import streamlit as st

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

    # Hyperliquid timestamps are normally Unix milliseconds. Detect other
    # common Unix timestamp units defensively so the merge remains reliable.
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

    trades["date"] = pd.to_datetime(
        timestamp, unit=timestamp_unit, errors="coerce", utc=True
    ).dt.tz_localize(None).dt.normalize()

    trades["Closed PnL"] = pd.to_numeric(
        trades["Closed PnL"], errors="coerce"
    ).fillna(0)
    trades["Fee"] = pd.to_numeric(trades["Fee"], errors="coerce").fillna(0)
    trades["Net PnL"] = trades["Closed PnL"] - trades["Fee"]

    sentiment_date = next(
        (c for c in sentiment.columns if c.lower() in {"date", "timestamp", "datetime"}),
        None,
    )
    sentiment_col = next(
        (
            c
            for c in sentiment.columns
            if c.lower() in {"classification", "sentiment", "value_classification"}
        ),
        None,
    )

    if sentiment_date is None or sentiment_col is None:
        raise ValueError(
            "Fear & Greed dataset must contain a date and classification column."
        )

    sentiment[sentiment_date] = pd.to_datetime(
        sentiment[sentiment_date], errors="coerce", utc=True
    ).dt.tz_localize(None).dt.normalize()
    sentiment[sentiment_col] = sentiment[sentiment_col].astype("string").str.strip()
    sentiment = sentiment.dropna(subset=[sentiment_date, sentiment_col])
    sentiment = sentiment.drop_duplicates(subset=[sentiment_date], keep="last")

    return trades, sentiment, sentiment_date, sentiment_col


try:
    trades, sentiment, sentiment_date, sentiment_col = load_data()
except Exception as exc:
    st.error(f"Could not load the project datasets: {exc}")
    st.stop()

# Dashboard KPIs
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

# PnL analysis should use realized/closing trades, matching the analysis script.
if "Direction" in trades.columns:
    closing_mask = trades["Direction"].astype("string").str.strip().isin(
        ["Close Long", "Close Short"]
    )
    pnl_trades = trades.loc[closing_mask].copy()
else:
    pnl_trades = trades.copy()

# Exact daily merge: both datasets are normalized to midnight first.
merged = pnl_trades.merge(
    sentiment[[sentiment_date, sentiment_col]],
    left_on="date",
    right_on=sentiment_date,
    how="left",
)

matched = merged[sentiment_col].notna().sum()

if matched == 0:
    # A one-day nearest-date fallback handles timezone/data-source boundary
    # differences without inventing sentiment values beyond one day.
    left = pnl_trades.dropna(subset=["date"]).sort_values("date").copy()
    right = sentiment[[sentiment_date, sentiment_col]].sort_values(sentiment_date).copy()
    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on=sentiment_date,
        direction="nearest",
        tolerance=pd.Timedelta(days=1),
    )
    matched = merged[sentiment_col].notna().sum()

if matched > 0:
    grouped = (
        merged.dropna(subset=[sentiment_col])
        .groupby(sentiment_col, observed=True)["Net PnL"]
        .agg(mean="mean", count="count", total="sum")
        .sort_values("mean", ascending=False)
    )

    st.dataframe(grouped, width="stretch")
    st.bar_chart(grouped["mean"], width="stretch")
    st.caption(f"Matched {matched:,} closing trades to a Fear & Greed sentiment day.")
else:
    st.warning(
        "No closing trades could be matched to the Fear & Greed dates. "
        "Check the dataset date ranges."
    )
