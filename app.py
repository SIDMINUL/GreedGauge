import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GreedGauge", page_icon="📊", layout="wide")

DATA_FILE = Path("compressed_data.csv.gz")
LOCAL_SENTIMENT_FILE = Path("fear_greed_index.csv")


@st.cache_data(show_spinner="Loading trading data…")
def load_trades():
    with gzip.open(DATA_FILE, "rb") as f:
        df = pd.read_csv(f)
    df.columns = df.columns.str.strip()

    timestamp = pd.to_numeric(df["Timestamp"], errors="coerce")
    median_timestamp = timestamp.dropna().median()
    if median_timestamp >= 1e17:
        unit = "ns"
    elif median_timestamp >= 1e14:
        unit = "us"
    elif median_timestamp >= 1e11:
        unit = "ms"
    else:
        unit = "s"

    df["date"] = pd.to_datetime(timestamp, unit=unit, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    df["Closed PnL"] = pd.to_numeric(df.get("Closed PnL"), errors="coerce")
    df["Fee"] = pd.to_numeric(df.get("Fee", 0), errors="coerce").fillna(0)
    df["Net PnL"] = df["Closed PnL"].fillna(0) - df["Fee"]
    df["Size USD"] = pd.to_numeric(df.get("Size USD", 0), errors="coerce").fillna(0)
    df["Execution Price"] = pd.to_numeric(df.get("Execution Price", 0), errors="coerce")
    return df, unit


@st.cache_data(show_spinner="Loading Fear & Greed data…")
def load_sentiment():
    sentiment = pd.read_csv(LOCAL_SENTIMENT_FILE)
    sentiment.columns = sentiment.columns.str.strip()
    date_col = next(c for c in sentiment.columns if c.lower() in {"date", "timestamp", "datetime"})
    class_col = next(c for c in sentiment.columns if c.lower() in {"classification", "sentiment", "value_classification"})
    sentiment[date_col] = pd.to_datetime(sentiment[date_col], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    sentiment[class_col] = sentiment[class_col].astype("string").str.strip()
    sentiment["value"] = pd.to_numeric(sentiment.get("value"), errors="coerce")
    return sentiment.dropna(subset=[date_col, class_col]), date_col, class_col


@st.cache_data(ttl=3600, show_spinner="Checking latest Fear & Greed data…")
def load_live_sentiment():
    try:
        import requests
        response = requests.get("https://api.alternative.me/fng/?limit=0", timeout=10)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", [])
        if not rows:
            return None
        live = pd.DataFrame(rows)
        live["date"] = pd.to_datetime(pd.to_numeric(live["timestamp"], errors="coerce"), unit="s", errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        live["classification"] = live["value_classification"].astype("string").str.strip()
        live["value"] = pd.to_numeric(live["value"], errors="coerce")
        return live[["date", "value", "classification"]].dropna(subset=["date", "classification"])
    except Exception:
        return None


trades, timestamp_unit = load_trades()
local_sentiment, sentiment_date, sentiment_class = load_sentiment()
live_sentiment = load_live_sentiment()

if live_sentiment is not None and not live_sentiment.empty:
    sentiment = live_sentiment.copy()
    sentiment_source = "live Alternative.me API"
else:
    sentiment = local_sentiment[[sentiment_date, "value", sentiment_class]].rename(columns={sentiment_date: "date", sentiment_class: "classification"})
    sentiment_source = "bundled fear_greed_index.csv"

# Keep only rows that can be meaningfully analyzed.
valid_trades = trades.dropna(subset=["date"]).copy()

# Sidebar filters
st.sidebar.header("Dashboard filters")
min_trade_date = valid_trades["date"].min().date()
max_trade_date = valid_trades["date"].max().date()
start_date, end_date = st.sidebar.date_input("Date range", value=(min_trade_date, max_trade_date), min_value=min_trade_date, max_value=max_trade_date)
if start_date > end_date:
    start_date, end_date = end_date, start_date

coin_options = sorted(valid_trades["Coin"].dropna().astype(str).unique()) if "Coin" in valid_trades else []
selected_coins = st.sidebar.multiselect("Coins", coin_options, default=coin_options[:20])

if selected_coins:
    filtered = valid_trades[valid_trades["Coin"].astype(str).isin(selected_coins)].copy()
else:
    filtered = valid_trades.copy()
filtered = filtered[(filtered["date"].dt.date >= start_date) & (filtered["date"].dt.date <= end_date)]

# Header
st.title("📊 GreedGauge")
st.caption("Hyperliquid trader performance through the lens of Bitcoin market sentiment")

# KPIs
closed = filtered.dropna(subset=["Closed PnL"])
traders = filtered["Account"].nunique() if "Account" in filtered else 0
coins = filtered["Coin"].nunique() if "Coin" in filtered else 0
wins = (closed["Closed PnL"] > 0).sum()
losses = (closed["Closed PnL"] < 0).sum()
win_rate = wins / (wins + losses) * 100 if wins + losses else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Trades", f"{len(filtered):,}")
k2.metric("Traders", f"{traders:,}")
k3.metric("Coins", f"{coins:,}")
k4.metric("Net PnL", f"${filtered['Net PnL'].sum():,.2f}")
k5.metric("Win Rate", f"{win_rate:.1f}%")

# PnL over time
st.header("Performance")
pnl_daily = filtered.groupby("date")["Net PnL"].sum().sort_index()
cumulative = pnl_daily.cumsum()
st.line_chart(cumulative, width="stretch")

# Sentiment join
sentiment = sentiment.drop_duplicates("date", keep="last").sort_values("date")
merged = pd.merge_asof(filtered.sort_values("date"), sentiment, on="date", direction="nearest", tolerance=pd.Timedelta(days=1))
matched = merged.dropna(subset=["classification"]).copy()

st.subheader("PnL by Market Sentiment")
if matched.empty:
    st.warning("No overlapping trade and Fear & Greed dates were found for the selected filters.")
else:
    sentiment_summary = matched.groupby("classification", observed=True)["Net PnL"].agg(Mean="mean", Total="sum", Trades="count").sort_values("Mean", ascending=False)
    st.dataframe(sentiment_summary, width="stretch")
    st.bar_chart(sentiment_summary["Mean"], width="stretch")
    st.caption(f"Matched {len(matched):,} trades using {sentiment_source}.")

# Two analysis sections
st.header("Trading behavior")
col1, col2 = st.columns(2)
with col1:
    if "Direction" in filtered:
        direction_summary = filtered.groupby(filtered["Direction"].astype(str))["Net PnL"].agg(Total="sum", Trades="count").sort_values("Total", ascending=False)
        st.subheader("Long / Short performance")
        st.dataframe(direction_summary, width="stretch")
        st.bar_chart(direction_summary["Total"], width="stretch")
with col2:
    if "Coin" in filtered:
        coin_summary = filtered.groupby("Coin")["Net PnL"].agg(Total="sum", Trades="count").sort_values("Total", ascending=False).head(15)
        st.subheader("Top coins by Net PnL")
        st.dataframe(coin_summary, width="stretch")
        st.bar_chart(coin_summary["Total"], width="stretch")

# Trader leaderboard
if "Account" in filtered:
    st.header("Trader leaderboard")
    trader_summary = filtered.groupby("Account")["Net PnL"].agg(Total="sum", Trades="count", Avg="mean").sort_values("Total", ascending=False)
    st.dataframe(trader_summary.head(20), width="stretch")
    st.bar_chart(trader_summary.head(15)["Total"], width="stretch")

# Activity heatmap-style table
st.header("Activity")
activity = filtered.assign(day=filtered["date"].dt.day_name(), hour=filtered["date"].dt.hour).pivot_table(index="day", columns="hour", values="Net PnL", aggfunc="sum", fill_value=0)
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
activity = activity.reindex(weekday_order).dropna(how="all")
st.dataframe(activity, width="stretch")

# Statistical relationship
st.header("Statistical analysis")
if len(matched) >= 3 and "value" in matched.columns:
    corr = matched[["value", "Net PnL"]].corr(method="pearson").iloc[0, 1]
    st.metric("Pearson correlation: Fear & Greed value vs Net PnL", f"{corr:.3f}")
    st.caption("Correlation describes association, not causation.")

# Methodology / diagnostics
with st.expander("Data diagnostics & methodology"):
    st.write(f"Trade timestamp unit detected: `{timestamp_unit}`")
    st.write(f"Trade dates: `{min_trade_date}` → `{max_trade_date}`")
    st.write(f"Sentiment dates: `{sentiment['date'].min().date()}` → `{sentiment['date'].max().date()}`")
    st.write(f"Sentiment source: `{sentiment_source}`")
    st.write(f"Rows in current filter: `{len(filtered):,}`")
    st.write("Sentiment is matched to the nearest available daily value within a 1-day tolerance; unmatched rows are excluded from sentiment analysis.")

st.caption("GreedGauge is an analytics project, not financial advice.")
