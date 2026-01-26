import streamlit as st
from supabase import create_client
from datetime import date

# =============================
# Supabase client
# =============================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================
# Load assets
# =============================
assets = supabase.table("assets").select("*").execute().data
asset_options = [f"{a['ticker']} - {a['name']}" for a in assets]
selected_asset = st.selectbox("Select an asset", asset_options)
asset = assets[asset_options.index(selected_asset)]

# =============================
# Show latest metrics
# =============================
st.header(f"Metrics for {asset['name']} ({asset['ticker']})")

metrics = supabase.table("daily_metrics") \
    .select("*") \
    .eq("asset_id", asset["asset_id"]) \
    .order("metric_date", desc=True) \
    .execute().data

if metrics:
    latest = metrics[0]
    st.metric("Average sentiment", f"{latest['avg_sentiment']:.2f}")
    st.metric("News volume", latest["news_volume"])
    st.metric("Sentiment volatility", f"{latest['sentiment_std']:.2f}")
    st.metric("Signal", latest["signal"])
else:
    st.info("No metrics yet for this asset.")

# =============================
# Show market brief
# =============================
st.header("Market Brief")
briefs = supabase.table("market_briefs") \
    .select("*") \
    .eq("scope", asset["ticker"]) \
    .order("generated_at", desc=True) \
    .execute().data

if briefs:
    st.write(briefs[0]["content"])
else:
    st.info("No market brief yet.")

# =============================
# Show recent news
# =============================
st.header("Recent News")
news = supabase.table("news") \
    .select("*") \
    .eq("asset_id", asset["asset_id"]) \
    .order("published_at", desc=True) \
    .limit(10) \
    .execute().data

if news:
    for n in news:
        st.markdown(f"**{n['title']}** ({n['source']}, {n['published_at'][:10]})")
        st.markdown(n["content"])
        st.markdown(f"[Link]({n['url']})")
        st.write("---")
else:
    st.info("No news yet.")
