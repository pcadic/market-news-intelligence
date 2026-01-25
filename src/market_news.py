import os
import requests
import feedparser
from datetime import datetime, date
from tqdm import tqdm
from supabase import create_client

# ==============================
# ENV
# ==============================
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==============================
# HF CONFIG
# ==============================
FINBERT_MODEL = "ProsusAI/finbert"
GEN_MODEL = "tiiuae/falcon-7b-instruct"

FINBERT_API = f"https://api-inference.huggingface.co/models/{FINBERT_MODEL}"
GEN_API = f"https://api-inference.huggingface.co/models/{GEN_MODEL}"

HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ==============================
# LOAD ASSETS
# ==============================
assets = supabase.table("assets").select("*").execute().data
print(f"{len(assets)} assets found")

# ==============================
# FETCH NEWS
# ==============================
news_rows = []

print("Fetching news...")
for asset in tqdm(assets, desc="Fetching news"):
    query = f"{asset['ticker']} stock"
    rss_url = (
        "https://news.google.com/rss/search"
        f"?q={query.replace(' ', '%20')}&hl=en-CA&gl=CA&ceid=CA:en"
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

# ==============================
# NLP – FINBERT SENTIMENT
# ==============================
news_items = supabase.table("news").select("*").execute().data
nlp_rows = []

print("Running sentiment analysis...")

for item in tqdm(news_items, desc="Running sentiment"):
    payload = {
        "inputs": item["content"][:512]
    }

    response = requests.post(
        FINBERT_API,
        headers=HF_HEADERS,
        json=payload,
        timeout=30
    )

    result = response.json()

    if isinstance(result, list) and len(result) > 0:
        label = result[0]["label"].lower()
        score = resu
import os
import feedparser
import requests
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
HF_TOKEN = os.environ["HF_TOKEN"]

LOOKBACK_DAYS = 1  # 1=daily, 7=weekly, 30=monthly

SENTIMENT_MODEL = "ProsusAI/finbert"
BRIEF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

HF_API_URL = f"https://api-inference.huggingface.co/models/{BRIEF_MODEL}"
HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# =============================
# CLIENTS
# =============================
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
    tokenizer=SENTIMENT_MODEL
)

# =============================
# 1. FETCH ASSETS
# =============================
assets = supabase.table("assets").select("*").execute().data
print(f"{len(assets)} assets found")

# =============================
# 2. FETCH NEWS
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
# 3. NLP — FinBERT
# =============================
news = supabase.table("news").select("*").execute().data
nlp_rows = []

print("Running sentiment...")
for item in tqdm(news):
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
        "summary": item["content"][:300],
        "sentiment_score": sentiment_score,
        "sentiment_label": label,
        "model_name": SENTIMENT_MODEL
    })

if nlp_rows:
    supabase.table("news_nlp").upsert(
        nlp_rows, on_conflict="news_id"
    ).execute()

# =============================
# 4. DAILY METRICS
# =============================
today = date.today()
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
# 5. MARKET BRIEFS — HF API
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
    signal = max(set(r["signal"] for r in rows),
                 key=lambda s: sum(x["signal"] == s for x in rows))

    prompt = (
        f"Asset: {asset['name']} ({asset['ticker']})\n"
        f"Period: {start_date} to {today}\n\n"
        f"Metrics:\n"
        f"- Average sentiment: {avg_sent:.2f}\n"
        f"- News volume: {total_news}\n"
        f"- Sentiment volatility: {avg_std:.2f}\n"
        f"- Dominant signal: {signal}\n\n"
        f"Write a concise professional market brief."
    )

    response = requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 250,
                "temperature": 0.2,
                "return_full_text": False
            }
        },
        timeout=60
    )

    output = response.json()[0]["generated_text"]

    supabase.table("market_briefs").insert({
        "period_start": start_date.isoformat(),
        "period_end": today.isoformat(),
        "scope": asset["ticker"],
        "content": output,
        "model_name": BRIEF_MODEL
    }).execute()

print("Pipeline completed successfully.")
