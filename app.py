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

    # Hyperliquid timestamps are normally Unix milliseconds. Detect common
    # Unix timestamp units defensively.
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

    trades["Closed PnL"] = pd.to_numeric(
        trades["Closed PnL"], errors="coerce"
    )
    trades["Fee"] = pd.to_numeric(trades["Fee"], errors="coerce").fillna(0)
    trades["Net PnL"] = trades["Closed PnL"].fillna(0) - trades["Fee"]

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

    sentiment[sentiment_date] = (
        pd.to_datetime(sentiment[sentiment_date], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
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

# Prefer realized/closing trades when the dataset exposes standard
# Hyperliquid close-direction labels. If those labels are different or absent,
# use rows with an actual Closed PnL value instead of showing an empty chart.
pnl_trades = trades.dropna(subset=["date", "Closed PnL"]).copy()

if "Direction" in pnl_trades.columns:
    directions = pnl_trades["Direction"].astype("string").str.strip()
    standard_close_labels = {"Close Long", "Close Short"}
    closing_trades = pnl_trades[directions.isin(standard_close_labels)].copy()

    if not closing_trades.empty:
        pnl_trades = closing_trades
        analysis_source = "standard closing-direction rows"
    else:
        analysis_source = "rows with a valid Closed PnL value (close labels not recognized)"
else:
    analysis_source = "rows with a valid Closed PnL value"

# Match on the calendar date. If exact dates do not overlap, use a one-day
# nearest-date tolerance to handle timezone/data-source boundary differences.
left = pnl_trades.sort_values("date").copy()
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
    st.caption(
        f"Matched {matched:,} trades using {analysis_source}. "
        f"Trade dates: {left['date'].min().date()} → {left['date'].max().date()} | "
        f"Fear & Greed dates: {right[sentiment_date].min().date()} → "
        f"{right[sentiment_date].max().date()}"
    )
else:
    st.warning(
        "The trade and Fear & Greed datasets have no overlapping dates within "
        "the 1-day matching tolerance."
    )
    st.info(
        f"Trade dates: {left['date'].min().date()} → {left['date'].max().date()} | "
        f"Fear & Greed dates: {right[sentiment_date].min().date()} → "
        f"{right[sentiment_date].max().date()}"
    )
