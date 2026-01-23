import os
import feedparser
from datetime import datetime, timezone
from supabase import create_client
from tqdm import tqdm
from transformers import pipeline

# ======================
# CONFIG
# ======================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

# ======================
# UTILS
# ======================

def to_iso_datetime(entry):
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(
            *entry.published_parsed[:6],
            tzinfo=timezone.utc
        ).isoformat()
    return None


def google_news_rss(query: str) -> str:
    query = query.replace(" ", "%20")
    return (
        "https://news.google.com/rss/search"
        f"?q={query}"
        "&hl=en-CA&gl=CA&ceid=CA:en"
    )

# ======================
# LOAD ASSETS
# ======================

assets = supabase.table("assets").select("asset_id, ticker").execute().data
print(f"{len(assets)} assets found")

news_rows = []

# ======================
# FETCH NEWS
# ======================

for asset in tqdm(assets, desc="Fetching news"):
    ticker = asset["ticker"]
    rss_url = google_news_rss(f"{ticker} stock")

    feed = feedparser.parse(rss_url)

    for entry in feed.entries:
        news_rows.append({
            "asset_id": asset["asset_id"],
            "title": entry.get("title"),
            "source": entry.get("source", {}).get("title"),
            "url": entry.get("link"),
            "published_at": to_iso_datetime(entry),
            "content": entry.get("summary", "")
        })

print(f"{len(news_rows)} articles fetched")

# ======================
# INSERT NEWS
# ======================

inserted_news = []

if news_rows:
    inserted_news = (
        supabase
        .table("news")
        .insert(news_rows)
        .execute()
        .data
    )

# ======================
# NLP (FinBERT)
# ======================

nlp_rows = []

for news in tqdm(inserted_news, desc="Running sentiment"):
    text = news["title"] or ""
    if not text:
        continue

    result = sentiment_model(text)[0]

    nlp_rows.append({
        "news_id": news["news_id"],
        "sentiment_label": result["label"].lower(),
        "sentiment_score": float(result["score"]),
        "model_name": "ProsusAI/finbert"
    })

if nlp_rows:
    supabase.table("news_nlp").insert(nlp_rows).execute()

# ======================
# DAILY METRICS
# ======================

import statistics
from collections import defaultdict
from datetime import date

def compute_daily_metrics():
    rows = (
        supabase
        .table("news")
        .select("news_id, asset_id, published_at, news_nlp(sentiment_score)")
        .execute()
        .data
    )

    grouped = defaultdict(list)

    for r in rows:
        if not r["news_nlp"]:
            continue

        d = r["published_at"][:10]  # YYYY-MM-DD
        grouped[(r["asset_id"], d)].append(
            r["news_nlp"]["sentiment_score"]
        )

    metrics = []

    for (asset_id, d), scores in grouped.items():
        avg = statistics.mean(scores)
        std = statistics.pstdev(scores) if len(scores) > 1 else 0
        volume = len(scores)

        if avg > 0.6 and volume >= 3:
            signal = "positive_momentum"
        elif avg < 0.4:
            signal = "caution"
        elif std > 0.25:
            signal = "high_uncertainty"
        else:
            signal = "neutral"

        metrics.append({
            "asset_id": asset_id,
            "metric_date": d,
            "avg_sentiment": round(avg, 3),
            "sentiment_std": round(std, 3),
            "news_volume": volume,
            "signal": signal
        })

    if metrics:
        supabase.table("daily_metrics").upsert(
            metrics,
            on_conflict="asset_id,metric_date"
        ).execute()

    print("Daily metrics computed")



compute_daily_metrics()

print("Pipeline completed successfully")

