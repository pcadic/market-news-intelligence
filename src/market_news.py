import os
import feedparser
import requests
from datetime import datetime, timezone
from tqdm import tqdm
from supabase import create_client, Client
from transformers import pipeline

# =====================
# ENV & SUPABASE
# =====================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =====================
# NLP MODEL (HF)
# =====================
sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

# =====================
# RSS SOURCES (STABLE)
# =====================
REUTERS_RSS = "https://feeds.reuters.com/reuters/businessNews"
YAHOO_RSS = "https://finance.yahoo.com/rss/topstories"

# =====================
# HELPERS
# =====================
def get_assets():
    response = supabase.table("assets").select("*").execute()
    return response.data or []

def fetch_rss(url):
    return feedparser.parse(url).entries

def asset_match(article_title, asset):
    keywords = [
        asset["ticker"].replace(".TO", ""),
        asset["name"].lower()
    ]
    title = article_title.lower()
    return any(k.lower() in title for k in keywords)

def analyze_sentiment(text):
    result = sentiment_model(text[:512])[0]
    return result["label"], float(result["score"])

# =====================
# PIPELINE
# =====================
def run_pipeline():
    print("Fetching assets...")
    assets = get_assets()
    print(f"{len(assets)} assets found")

    print("Fetching RSS feeds...")
    articles = fetch_rss(REUTERS_RSS) + fetch_rss(YAHOO_RSS)

    print(f"{len(articles)} raw articles fetched")

    rows_to_insert = []

    for asset in tqdm(assets, desc="Processing assets"):
        for article in articles:
            title = article.get("title", "")
            link = article.get("link", "")

            if not title or not asset_match(title, asset):
                continue

            sentiment, score = analyze_sentiment(title)

            rows_to_insert.append({
                "asset_id": asset["asset_id"],
                "title": title,
                "url": link,
                "published_at": datetime.now(timezone.utc).isoformat(),
                "sentiment": sentiment,
                "sentiment_score": score,
                "source": "RSS"
            })

    print(f"{len(rows_to_insert)} articles to ingest")

    if rows_to_insert:
        supabase.table("market_news").insert(rows_to_insert).execute()
        print("News ingested successfully")
    else:
        print("No matching news found")

    compute_daily_metrics()

# =====================
# METRICS
# =====================
def compute_daily_metrics():
    print("Computing daily metrics...")

    today = datetime.now(timezone.utc).date().isoformat()

    news = supabase.table("market_news") \
        .select("asset_id,sentiment_score") \
        .gte("published_at", f"{today}T00:00:00") \
        .execute().data

    if not news:
        print("No news found for metrics")
        return

    metrics = {}

    for n in news:
        aid = n["asset_id"]
        metrics.setdefault(aid, []).append(n["sentiment_score"])

    rows = []
    for aid, scores in metrics.items():
        avg = sum(scores) / len(scores)

        recommendation = (
            "BUY" if avg > 0.4 else
            "SELL" if avg < -0.4 else
            "HOLD"
        )

        rows.append({
            "asset_id": aid,
            "date": today,
            "avg_sentiment": avg,
            "recommendation": recommendation
        })

    supabase.table("daily_metrics").upsert(
        rows,
        on_conflict="asset_id,date"
    ).execute()

    print("Daily metrics updated")

# =====================
# ENTRY POINT
# =====================
if __name__ == "__main__":
    run_pipeline()
