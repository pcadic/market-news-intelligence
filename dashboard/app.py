import streamlit as st
import pandas as pd
from supabase import create_client
import os
from dotenv import load_dotenv

# -------------------------------------------------
# Config
# -------------------------------------------------
st.set_page_config(
    page_title="Market News Intelligence",
    layout="wide"
)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------------------------
# Load assets (FIXED COLUMN NAME)
# -------------------------------------------------
assets_res = supabase.table("assets").select(
    "asset_id, name, ticker"
).order("name").execute()

assets_df = pd.DataFrame(assets_res.data)

if assets_df.empty:
    st.error("No assets found.")
    st.stop()

# -------------------------------------------------
# Asset selector (robust)
# -------------------------------------------------
selected_asset_id = st.selectbox(
    "Select an asset",
    options=assets_df["asset_id"].tolist(),
    format_func=lambda x: (
        assets_df.loc[assets_df["asset_id"] == x, "name"].values[0]
        + " ("
        + assets_df.loc[assets_df["asset_id"] == x, "ticker"].values[0]
        + ")"
    )
)

asset_row = assets_df[assets_df["asset_id"] == selected_asset_id].iloc[0]
asset_name = asset_row["name"]
ticker = asset_row["ticker"]

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    f"## {asset_name} <span style='color:#6f6f6f; font-weight:400'>({ticker})</span>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# Daily metrics
# -------------------------------------------------
metrics_res = supabase.table("daily_metrics") \
    .select("*") \
    .eq("asset_id", selected_asset_id) \
    .order("date", desc=True) \
    .limit(1) \
    .execute()

if metrics_res.data:
    m = metrics_res.data[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average sentiment", f"{m['avg_sentiment']:.2f}")
    col2.metric("News volume", m["news_count"])
    col3.metric("Sentiment volatility", f"{m['sentiment_volatility']:.2f}")
    col4.metric("Signal", m["signal"])
else:
    st.warning("No metrics available.")

# -------------------------------------------------
# News
# -------------------------------------------------
st.markdown("### Latest news")

news_res = supabase.table("news") \
    .select("title, url, source, published_at") \
    .eq("asset_id", selected_asset_id) \
    .order("published_at", desc=True) \
    .execute()

news_df = pd.DataFrame(news_res.data)

if news_df.empty:
    st.info("No news available.")
else:
    news_df["published_at"] = pd.to_datetime(news_df["published_at"]).dt.date

    MAX_VISIBLE = 5
    visible_news = news_df.head(MAX_VISIBLE)
    extra_news = news_df.iloc[MAX_VISIBLE:]

    for _, row in visible_news.iterrows():
        st.markdown(
            f"- **[{row['title']}]({row['url']})**  \n"
            f"<span style='color:#6f6f6f; font-size:0.85em'>"
            f"{row['source']} · {row['published_at']}"
            f"</span>",
            unsafe_allow_html=True
        )

    if not extra_news.empty:
        with st.expander(f"Show {len(extra_news)} more articles"):
            for _, row in extra_news.iterrows():
                st.markdown(
                    f"- **[{row['title']}]({row['url']})**  \n"
                    f"<span style='color:#6f6f6f; font-size:0.85em'>"
                    f"{row['source']} · {row['published_at']}"
                    f"</span>",
                    unsafe_allow_html=True
                )
