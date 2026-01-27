import streamlit as st
import pandas as pd
import os
from supabase import create_client

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Market News Intelligence",
    layout="wide"
)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# =============================
# LOAD DATA
# =============================
assets = pd.DataFrame(
    supabase.table("assets").select("*").execute().data
)

news = pd.DataFrame(
    supabase.table("news").select("*").execute().data
)

metrics = pd.DataFrame(
    supabase.table("daily_metrics").select("*").execute().data
)

briefs = pd.DataFrame(
    supabase.table("market_briefs").select("*").execute().data
)


# =============================
# UI
# =============================
st.title("📊 Market News Intelligence")

asset_name = st.selectbox(
    "Select an asset",
    assets["name"]
)

asset = assets[assets["name"] == asset_name].iloc[0]
asset_id = asset["asset_id"]

st.subheader(f"{asset_name} ({asset['ticker']})")


# =============================
# METRICS
# =============================
asset_metrics = metrics[metrics["asset_id"] == asset_id]

if not asset_metrics.empty:
    latest = asset_metrics.sort_values("metric_date").iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average sentiment", f"{latest['avg_sentiment']:.2f}")
    col2.metric("News volume", int(latest["news_volume"]))
    col3.metric("Sentiment volatility", f"{latest['sentiment_std']:.2f}")
    col4.metric("Signal", latest["signal"].replace("_", " ").title())
else:
    st.info("No metrics available.")


# =============================
# MARKET BRIEF
# =============================
st.markdown("### 🧠 Market Brief")

asset_brief = briefs[briefs["scope"] == asset["ticker"]]

if not asset_brief.empty:
    st.write(asset_brief.sort_values("generated_at").iloc[-1]["content"])
else:
    st.info("No market brief available.")


# =============================
# NEWS LIST (CLEAN LINKS)
# =============================
st.markdown("### 📰 Recent News")

asset_news = news[news["asset_id"] == asset_id] \
    .sort_values("published_at", ascending=False) \
    .head(10)

for _, row in asset_news.iterrows():
    st.markdown(
        f"- [{row['title']}]({row['url']}) "
        f"<span style='color:gray;font-size:0.8em'>({row['source']})</span>",
        unsafe_allow_html=True
    )
