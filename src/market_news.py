import os
import requests
import feedparser
from datetime import datetime, timedelta, date
from urllib.parse import quote_plus
from tqdm import tqdm
from supabase import create_client

# =============================
# ENV
# =============================
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# =============================
# HF MODELS (STABLE)
# =============================
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
GEN_MODEL = "google/flan-t5-small"

SENTIMENT_API = f"https://api-inference.huggingface.co/models/{SENTIMENT_MODEL}"
GEN_API = f"https://api-inference.huggingface.co/models/{GEN_MODEL}"

# =============================
# LOAD ASSETS
# =============================
assets = supabase.table("assets").select("*").execute().data
print(f"{len(assets)} assets found")

# =============================
# FETCH NEWS (NO UPSERT, NO URL CONSTRAINT)
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

# =============================
# SENTIMENT ANALYSIS (HF API)
# =============================
news_items = supabase.table("news").select("*").execute().data
nlp_rows = []

print("Running sentiment analysis...")

for item in tqdm(news_items):
    try:
        r = requests.post(
            SENTIMENT_API,
            headers=HF_HEADERS,
            json={"inputs": item["content"][:512]},
            timeout=30
        )
        result = r.json()[0]

        label = result["label"].lower()
        score = result["score"]

        sentiment_score = (
            score if label == "positive"
            else -score if label == "negative"
            else 0.0
        )

        nlp_rows.append({
            "news_id": item["news_id"],
            "summary": item["content"][:300],
            "sentiment_score": sentiment_score,
            "sentiment_label": label,
            "model_name": SENTIMENT_MODEL
        })

    except Exception as e:
        print(f"Sentiment error: {e}")

if nlp_rows:
    supabase.table("news_nlp").upsert(
        nlp_rows, on_conflict="news_id"
    ).execute()

# =============================
# DAILY METRICS
# =============================
metrics = {}

rows = supabase.table("news_nlp") \
    .select("sentiment_score, news:news_id(asset_id, published_at)") \
    .execute().data

for row in rows:
    asset_id = row["news"]["asset_id"]
    d = row["news"]["published_at"][:10]
    metrics.setdefault((asset_id, d), []).append(row["sentiment_score"])

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
# MARKET BRIEFS (ROBUST)
# =============================
today = date.today()
start_date = today - timedelta(days=1)

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

    prompt = (
        f"Write a short market brief.\n"
        f"Asset: {asset['name']} ({asset['ticker']})\n"
        f"Average sentiment: {avg_sent:.2f}\n"
        f"News volume: {total_news}"
    )

    try:
        r = requests.post(
            GEN_API,
            headers=HF_HEADERS,
            json={"inputs": prompt},
            timeout=30
        )
        text = r.json()[0]["generated_text"]

        supabase.table("market_briefs").insert({
            "period_start": start_date.isoformat(),
            "period_end": today.isoformat(),
            "scope": asset["ticker"],
            "content": text,
            "model_name": GEN_MODEL
        }).execute()

    except Exception as e:
        print(f"Brief error for {asset['ticker']}: {e}")

print("Pipeline completed successfully.")
