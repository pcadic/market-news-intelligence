import os
import feedparser
from datetime import datetime, timedelta, date
from urllib.parse import quote_plus
from tqdm import tqdm

from supabase import create_client, Client
from transformers import pipeline

# =============================
# CONFIG
# =============================
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

LOOKBACK_DAYS = 1  # <-- PARAMETRABLE
SENTIMENT_MODEL = "ProsusAI/finbert"
BRIEF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# =============================
# CLIENTS
# =============================
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
    tokenizer=SENTIMENT_MODEL
)

brief_pipeline = pipeline(
    "text-generation",
    model=BRIEF_MODEL,
    tokenizer=BRIEF_MODEL,
    max_new_tokens=250,
    do_sample=False
)

# =============================
# 1. FETCH ASSETS
# =============================
assets = supabase.table("assets").select("*").execute().data
print(f"{len(assets)} assets found")

# =============================
# 2. FETCH NEWS (RSS)
# =============================
news_rows = []

print("Fetching news...")
for asset in tqdm(assets):
    query = quote_plus(f"{asset['ticker']} stock")
    rss_url = (
        "https://news.google.com/rss/search"
        f"?q={query}&hl=en-CA&gl=CA&ceid=CA:en"
    )

    feed = feedparser.parse(rss_url)

    for entry in feed.entries:
        if not hasattr(entry, "published_parsed"):
            continue

        published = datetime(*entry.published_parsed[:6]).isoformat()

        news_rows.append({
            "asset_id": asset["asset_id"],
            "source": "Google News",
            "title": entry.title,
            "content": entry.get("summary", entry.title),
            "url": entry.link,
            "published_at": published
        })

print(f"{len(news_rows)} articles fetched")

if news_rows:
    supabase.table("news").insert(news_rows).execute()

# =============================
# 3. NLP (FinBERT)
# =============================
news = supabase.table("news").select("*").execute().data

nlp_rows = []

print("Running sentiment...")
for item in tqdm(news):
    result = sentiment_pipeline(item["content"][:512])[0]

    label = result["label"].lower()
    score = result["score"]

    sentiment_score = score if label == "positive" else -score if label == "negative" else 0.0

    nlp_rows.append({
        "news_id": item["news_id"],
        "summary": item["content"][:300],
        "sentiment_score": sentiment_score,
        "sentiment_label": label,
        "model_name": SENTIMENT_MODEL
    })

if nlp_rows:
    supabase.table("news_nlp").upsert(nlp_rows, on_conflict="news_id").execute()

# =============================
# 4. DAILY METRICS
# =============================
today = date.today()

metrics = {}

for row in supabase.table("news_nlp") \
    .select("sentiment_score, news:news_id(asset_id, published_at)") \
    .execute().data:

    asset_id = row["news"]["asset_id"]
    d = row["news"]["published_at"][:10]

    key = (asset_id, d)

    metrics.setdefault(key, []).append(row["sentiment_score"])

metric_rows = []

for (asset_id, d), scores in metrics.items():
    avg = sum(scores) / len(scores)
    std = (sum((s - avg) ** 2 for s in scores) / len(scores)) ** 0.5

    if avg > 0.15:
        signal = "positive_momentum"
    elif avg < -0.15:
        signal = "caution"
    elif std > 0.5:
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
    supabase.table("daily_metrics").upsert(
        metric_rows,
        on_conflict="asset_id,metric_date"
    ).execute()

# =============================
# 5. MARKET BRIEFS (PER ASSET)
# =============================
start_date = today - timedelta(days=LOOKBACK_DAYS)

metrics = supabase.table("daily_metrics") \
    .select("*") \
    .gte("metric_date", start_date.isoformat()) \
    .execute().data

print("Generating market briefs...")

for asset in assets:
    rows = [m for m in metrics if m["asset_id"] == asset["asset_id"]]

    if not rows:
        continue

    avg_sent = sum(r["avg_sentiment"] for r in rows) / len(rows)
    total_news = sum(r["news_volume"] for r in rows)
    avg_std = sum(r["sentiment_std"] for r in rows) / len(rows)

    signal = max(set(r["signal"] for r in rows), key=lambda s: sum(x["signal"] == s for x in rows))

    prompt = f"""
Asset: {asset['name']} ({asset['ticker']})
Period: {start_date} to {today}

Metrics:
- Average sentiment: {avg_sent:.2f}
- News volume: {total_news}
- Sentiment volatility: {avg_std:.2f}
- Dominant signal: {signal}

Write a concise professional market brief.
"""

    output = brief_pipeline(prompt)[0]["generated_text"]

    supabase.table("market_briefs").insert({
        "period_start": start_date.isoformat(),
        "period_end": today.isoformat(),
        "scope": asset["ticker"],
        "content": output,
        "model_name": BRIEF_MODEL
    }).execute()

print("Pipeline completed successfully.")
