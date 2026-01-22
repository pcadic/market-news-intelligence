import pandas as pd
import yfinance as yf
from supabase import create_client, Client
from datetime import datetime
from tqdm import tqdm

# ============================================================
# 1️⃣ Supabase client (variables d'environnement)
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()  # charge .env

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# 2️⃣ Helpers
# ============================================================

def compute_signal(avg_sentiment, sentiment_std, news_volume):
    """
    Simple heuristic signal:
    - Buy if sentiment > 0.5
    - Sell if sentiment < -0.5
    - Hold otherwise
    """
    if avg_sentiment > 0.5:
        return "buy"
    elif avg_sentiment < -0.5:
        return "sell"
    else:
        return "hold"

# ============================================================
# 3️⃣ Fetch assets from Supabase
# ============================================================

def fetch_assets():
    response = supabase.table("assets").select("*").execute()
    return response.data if response.data else []

# ============================================================
# 4️⃣ Fetch news from yfinance
# ============================================================

def fetch_news_for_asset(ticker):
    try:
        yf_ticker = yf.Ticker(ticker)
        news_items = yf_ticker.news  # list of dicts
        return news_items
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

def fetch_news():
    assets = fetch_assets()
    print("DEBUG – Assets:", assets)
    all_articles = []
    for asset in tqdm(assets, desc="Fetching news for assets"):
        ticker = asset["ticker"]
        news_items = fetch_news_for_asset(ticker)
        for n in news_items:
            article = {
                "news_id": n.get("uuid", f"{ticker}_{n.get('providerPublishTime', datetime.now().timestamp())}"),
                "asset_id": asset["id"],
                "published_at": datetime.fromtimestamp(n.get("providerPublishTime", datetime.now().timestamp())).isoformat(),
                "title": n.get("title", ""),
                "content": n.get("summary", ""),
                "url": n.get("link", ""),
                "source": n.get("publisher", ""),
            }
            all_articles.append(article)
    return all_articles

# ============================================================
# 5️⃣ Store raw news in Supabase
# ============================================================

def store_raw_news(article):
    supabase.table("news").upsert(article).execute()

# ============================================================
# 6️⃣ NLP placeholder (à remplacer par ton modèle Hugging Face)
# ============================================================

def process_nlp_for_unprocessed_news():
    # Récupérer toutes les news sans NLP
    news_rows = supabase.table("news").select("*").execute().data
    for n in news_rows:
        # Fake sentiment for demo (remplacer par ton modèle HF)
        sentiment_score = 0.2  # placeholder
        supabase.table("news").update({
            "news_nlp": [{"sentiment_score": sentiment_score}]
        }).eq("news_id", n["news_id"]).execute()

# ============================================================
# 7️⃣ Compute daily metrics in Python
# ============================================================

def compute_daily_metrics():
    news_rows = supabase.table("news").select(
        "news_id, asset_id, published_at, news_nlp(sentiment_score)"
    ).execute().data

    if not news_rows:
        print("No news found for metrics.")
        return

    records = []
    for n in news_rows:
        if "news_nlp" in n and n["news_nlp"]:
            for nlp in n["news_nlp"]:
                records.append({
                    "news_id": n["news_id"],
                    "asset_id": n["asset_id"],
                    "published_at": n["published_at"],
                    "sentiment_score": nlp.get("sentiment_score", 0)
                })

    if not records:
        print("No NLP sentiment scores found.")
        return

    df = pd.DataFrame(records)
    df["metric_date"] = pd.to_datetime(df["published_at"]).dt.date

    daily_metrics = df.groupby(["asset_id", "metric_date"]).agg(
        avg_sentiment=("sentiment_score", "mean"),
        news_volume=("sentiment_score", "count"),
        sentiment_std=("sentiment_score", "std")
    ).reset_index()

    for _, row in daily_metrics.iterrows():
        supabase.table("daily_metrics").upsert({
            "asset_id": row["asset_id"],
            "metric_date": row["metric_date"].isoformat(),
            "avg_sentiment": row["avg_sentiment"],
            "news_volume": row["news_volume"],
            "sentiment_std": row["sentiment_std"],
            "signal": compute_signal(
                row["avg_sentiment"], 
                row["sentiment_std"], 
                row["news_volume"]
            )
        }).execute()

    print("Daily metrics computed and upserted successfully.")

# ============================================================
# 8️⃣ Main pipeline
# ============================================================

def run_pipeline():
    print("Fetching news...")
    articles = fetch_news()
    print(f"Fetched {len(articles)} articles.")

    for article in articles:
        store_raw_news(article)

    print("Running NLP...")
    process_nlp_for_unprocessed_news()

    print("Computing daily metrics...")
    compute_daily_metrics()

    print("Pipeline completed successfully.")

# ============================================================
if __name__ == "__main__":
    run_pipeline()
