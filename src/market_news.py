import os
import requests
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
HF_TOKEN = os.environ["HF_TOKEN"]

LOOKBACK_DAYS = 1  # Nombre de jours pour les briefs (1 = daily)

# Modèles HF
SENTIMENT_MODEL = "ProsusAI/finbert"
BRIEF_MODEL = "tiiuae/falcon-7b-instruct"  # Modèle actif pour générer les briefs

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
# 2. FETCH NEWS (Google News RSS)
# =============================
news_rows = []

print("Fetching news...")
for asset in tqdm(assets):
    query = quote_plus(f"{asset['ticker']} stock")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-CA&gl=CA&ceid=CA:en"
    feed = feedparser.parse(rss_url)

    for entry in feed.entries:
        if not hasattr(entry, "published_parsed"):
            continue
        published_at = datetime(*entry.published_parsed[:6]).isoformat()

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
    supabase.table("news").upsert(news_rows, on_conflict="url").execute()

# =============================
# 3. NLP – FinBERT Sentiment
# =============================
news_items = supabase.table("news").select("*").execute().data
nlp_rows = []

print("Running sentiment analysis...")
for item in tqdm(news_items):
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
rows = supabase.table("news_nlp") \
    .select("sentiment_score, news:news_id(asset_id, published_at)") \
    .execute().data

metrics = {}
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
    supabase.table("daily_metrics").upsert(metric_rows, on_conflict="asset_id,metric_date").execute()

# =============================
# 5. MARKET BRIEFS – HF API
# =============================
start_date = today - timedelta(days=LOOKBACK_DAYS)
metrics_rows = supabase.table("daily_metrics").select("*").gte("metric_date", start_date.isoformat()).execute().data

print("Generating market briefs...")
for asset in assets:
    asset_metrics = [m for m in metrics_rows if m["asset_id"] == asset["asset_id"]]
    if not asset_metrics:
        continue

    avg_sent = sum(m["avg_sentiment"] for m in asset_metrics) / len(asset_metrics)
    total_news = sum(m["news_volume"] for m in asset_metrics)
    avg_std = sum(m["sentiment_std"] for m in asset_metrics) / len(asset_metrics)
    signal_counts = {}
    for m in asset_metrics:
        signal_counts[m["signal"]] = signal_counts.get(m["signal"], 0) + 1
    dominant_signal = max(signal_counts, key=signal_counts.get)

    prompt = (
        f"Asset: {a
