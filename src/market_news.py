# ======================================================
# MARKET NEWS INTELLIGENCE — STABLE PIPELINE
# No HF API • No upsert hell • One script • Deterministic
# ======================================================

import os
import feedparser
from datetime import datetime, date
from collections import defaultdict
from tqdm import tqdm

from supabase import create_client, Client
from transformers import pipeline

# ======================================================
# ENV
# ======================================================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# ======================================================
# CLIENTS
# ======================================================

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    device=-1  # CPU
)

# ======================================================
# 1. LOAD ASSETS
# ======================================================

assets = supabase.table("assets").select("*").execute().data
print(f"{len(assets)} assets found")

# ======================================================
# 2. FETCH NEWS (NO UPSERT, NO URL CONSTRAINT)
# ======================================================

news_rows = []

print("Fetching news...")

for asset in tqdm(assets, desc="Fetching news"):
    query = f"{asset['ticker']} stock".replace(" ", "%20")
    rss_url = (
        "https://news.google.com/rss/search"
        f"?q={query}&hl=en-CA&gl=CA&ceid=CA:en"
    )

    feed = feedparser.parse(rss_url)

    for entry in feed.entries:
        if not hasattr(entry, "published_parsed"):
            continue

        published_at = datetime(
            *entry.published_parsed[:6]
        ).isoformat()

        news_rows.append({
            "asset_id": asset["asset_id"],
            "source": "Google News",
            "title": entry.title,
            "content": entry.get("summary", entry.title),
            "url": entry.get("link"),
            "published_at": published_at
        })

print(f"{len(news_rows)} articles fetched")

if news_rows:
    supabase.table("news").insert(news_rows).execute()

# ======================================================
# 3. SENTIMENT — FINBERT LOCAL
# ======================================================

news_items = supabase.table("news").select("*").execute().data
nlp_rows = []

print("Running sentiment analysis...")

for item in tqdm(news_items, desc="Sentiment"):
    try:
        result = sentiment_pipeline(item["content"][:512])[0]
        label = result["label"].lower()
        score = result["score"]

        sentiment_score = (
            score if label == "positive"
            else -score if label == "negative"
            else 0.0
        )

        nlp_rows.append({
            "news_id": item["news_id"],
            "sentiment_score": sentiment_score,
            "sentiment_label": label,
            "summary": item["content"][:300],
            "model_name": "ProsusAI/finbert"
        })

    except Exception as e:
        print(f"Sentiment error on news_id={item['news_id']}: {e}")

if nlp_rows:
    supabase.table("news_nlp").insert(nlp_rows).execute()

# ======================================================
# 4. DAILY METRICS (RULE-BASED, STABLE)
# ======================================================

print("Computing daily metrics...")

rows = supabase.table("news_nlp") \
    .select("sentiment_score, news:news_id(asset_id, published_at)") \
    .execute().data

metrics = defaultdict(list)

for row in rows:
    asset_id = row["news"]["asset_id"]
    d = row["news"]["published_at"][:10]
    metrics[(asset_id, d)].append(row["sentiment_score"])

metric_rows = []

for (asset_id, d), scores in metrics.items():
    avg = sum(scores) / len(scores)
    std = (sum((s - avg) ** 2 for s in scores) / len(scores)) ** 0.5

    if avg > 0.15:
        signal = "positive_momentum"
    elif avg < -0.15:
        signal = "caution"
    elif std > 0.4:
        signal = "high_uncertainty"
    else:
        signal = "neutral"

    metric_rows.append({
        "asset_id": asset_id,
        "metric_date": d,
        "avg_sentiment": avg,
        "news_volume": len(scores),
        "sentiment_std": std,
        "signal": signal
    })

if metric_rows:
    supabase.table("daily_metrics").insert(metric_rows).execute()

# ======================================================
# 5. MARKET BRIEFS (NO LLM — ALWAYS WORKS)
# ======================================================

print("Generating market briefs...")

today = date.today()

for asset in assets:
    rows = [m for m in metric_rows if m["asset_id"] == asset["asset_id"]]
    if not rows:
        continue

    avg_sent = sum(r["avg_sentiment"] for r in rows) / len(rows)
    total_news = sum(r["news_volume"] for r in rows)
    dominant_signal = max(
        set(r["signal"] for r in rows),
        key=lambda s: sum(x["signal"] == s for x in rows)
    )

    brief = (
        f"Over the latest period, {asset['name']} ({asset['ticker']}) "
        f"received {total_news} news mentions. "
        f"The average sentiment was {avg_sent:.2f}, "
        f"indicating a {dominant_signal.replace('_', ' ')} environment. "
        f"Media coverage suggests sustained market attention."
    )

    supabase.table("market_briefs").insert({
        "period_start": today.isoformat(),
        "period_end": today.isoformat(),
        "scope": asset["ticker"],
        "content": brief,
        "model_name": "rule_based_v1"
    }).execute()

print("✅ PIPELINE COMPLETED SUCCESSFULLY")
