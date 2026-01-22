import os
import requests
import feedparser
from datetime import datetime
from typing import List, Dict

import pandas as pd
from supabase import create_client, Client

from transformers import pipeline

# ============================================================
# CONFIG
# ============================================================

RSS_SOURCES = [
    {
        "name": "Yahoo Finance",
        "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSX&region=US&lang=en-US"
    },
    {
        "name": "Reuters Business",
        "url": "https://feeds.reuters.com/reuters/businessNews"
    }
]

MIN_ARTICLE_LENGTH = 500
POS_THRESHOLD = 0.2
NEG_THRESHOLD = -0.2
HIGH_STD = 0.6

# ============================================================
# SUPABASE CLIENT
# ============================================================

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

# ============================================================
# NLP MODELS (HUGGING FACE)
# ============================================================

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

# ============================================================
# UTILITIES
# ============================================================

def fetch_assets() -> List[Dict]:
    response = supabase.table("assets").select("*").execute()
    return response.data


def is_duplicate(url: str) -> bool:
    if not url:
        return False
    response = (
        supabase
        .table("news")
        .select("news_id")
        .eq("url", url)
        .execute()
    )
    return len(response.data) > 0


# ============================================================
# 1️⃣ FETCH NEWS
# ============================================================

def fetch_news() -> List[Dict]:
    articles = []

    for source in RSS_SOURCES:
        feed = feedparser.parse(source["url"])

        for entry in feed.entries:
            content = entry.get("summary", "")
            published = entry.get("published_parsed")

            if not published:
                continue

            articles.append({
                "source": source["name"],
                "title": entry.title,
                "content": content,
                "url": entry.get("link"),
                "published_at": datetime(*published[:6])
            })

    return articles


# ============================================================
# 2️⃣ CLEAN & FILTER
# ============================================================

def clean_news(articles: List[Dict]) -> List[Dict]:
    cleaned = []

    for a in articles:
        if len(a["content"]) < MIN_ARTICLE_LENGTH:
            continue
        if is_duplicate(a["url"]):
            continue

        cleaned.append(a)

    return cleaned


# ============================================================
# 3️⃣ MAP NEWS TO ASSETS
# ============================================================

def map_news_to_asset(article: Dict, assets: List[Dict]):
    title = article["title"].upper()
    content = article["content"].upper()

    for asset in assets:
        if asset["ticker"].upper() in title:
            return asset["asset_id"]
        if asset["name"].upper() in content:
            return asset["asset_id"]

    return None


# ============================================================
# 4️⃣ STORE RAW NEWS
# ============================================================

def store_raw_news(article: Dict, asset_id):
    supabase.table("news").insert({
        "asset_id": asset_id,
        "source": article["source"],
        "title": article["title"],
        "content": article["content"],
        "url": article["url"],
        "published_at": article["published_at"].isoformat()
    }).execute()


# ============================================================
# 5️⃣ NLP PROCESSING
# ============================================================

def summarize_text(text: str) -> str:
    result = summarizer(
        text,
        max_length=120,
        min_length=50,
        do_sample=False
    )
    return result[0]["summary_text"]


def analyze_sentiment(text: str):
    result = sentiment_model(text[:512])[0]
    label = result["label"].lower()

    score = result["score"]
    if label == "negative":
        score = -score
    elif label == "neutral":
        score = 0.0

    return score, label


def process_nlp_for_unprocessed_news():
    news_rows = (
        supabase
        .table("news")
        .select("news_id, content")
        .execute()
        .data
    )

    processed_ids = (
        supabase
        .table("news_nlp")
        .select("news_id")
        .execute()
        .data
    )

    processed_ids = {row["news_id"] for row in processed_ids}

    for row in news_rows:
        if row["news_id"] in processed_ids:
            continue

        summary = summarize_text(row["content"])
        score, label = analyze_sentiment(row["content"])

        supabase.table("news_nlp").insert({
            "news_id": row["news_id"],
            "summary": summary,
            "sentiment_score": score,
            "sentiment_label": label,
            "model_name": "finbert + bart-large-cnn"
        }).execute()


# ============================================================
# 6️⃣ DAILY METRICS (Python uniquement)
# ============================================================

import pandas as pd

def compute_daily_metrics():
    # Récupérer toutes les news avec leur sentiment
    news_rows = supabase.table("news").select(
        "news_id, asset_id, published_at, news_nlp(sentiment_score)"
    ).execute().data

    # Vérifier qu'on a des données
    if not news_rows:
        print("No news with NLP found yet.")
        return

    # Convertir en DataFrame
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

    # Calcul des métriques journalières
    daily_metrics = df.groupby(["asset_id", "metric_date"]).agg(
        avg_sentiment=("sentiment_score", "mean"),
        news_volume=("sentiment_score", "count"),
        sentiment_std=("sentiment_score", "std")
    ).reset_index()

    # Calculer le signal et upsert dans Supabase
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
# 7️⃣ MAIN PIPELINE
# ============================================================

def run_pipeline():
    print("Fetching assets...")
    assets = fetch_assets()

    print("Fetching news...")
    articles = fetch_news()

    print("Cleaning news...")
    articles = clean_news(articles)

    print(f"{len(articles)} articles to ingest")

    for article in articles:
        asset_id = map_news_to_asset(article, assets)
        store_raw_news(article, asset_id)

    print("Running NLP...")
    process_nlp_for_unprocessed_news()

    print("Computing daily metrics...")
    compute_daily_metrics()

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
