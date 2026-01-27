import os
import feedparser
from datetime import datetime, date, timedelta
from urllib.parse import quote_plus
from collections import defaultdict

from tqdm import tqdm
from supabase import create_client
from transformers import pipeline


# =============================
# ENV
# =============================
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# =============================
# CONFIG
# =============================
LOOKBACK_DAYS = 7
SENTIMENT_MODEL = "ProsusAI/finbert"


# =============================
# LOAD FINBERT (LOCAL)
# =============================
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
    tokenizer=SENTIMENT_MODEL
)


# =============================
# 1. LOAD ASSETS
# =============================
assets = supabase.table("assets").select("*").execute().data
print(f"{len(assets)} assets found")


# =============================
# 2. FETCH NEWS (GOOGLE NEWS)
# =============================
news_rows = []
seen_urls = set()

print("Fetching news...")

for asset in tqdm(assets, desc="Fetching news"):
    query = quote_plus(f"{asset['ticker']} stock")
    rss_url = (
        "https://news.google.com/rss/search"
        f"?q={query}&hl=en-CA&gl=CA&ceid=CA:en"
    )

    feed = feedparser.parse(rss_url)

    for entry in feed.entries:
        if not hasattr(entry, "published_parsed"):
            continue

        url = entry.get("link")
        if not url or url in seen_urls:
            continue

        seen_urls.add(url)

        published_at = datetime(
            *entry.published_parsed[:6]
        ).isoformat()

        news_rows.append({
            "asset_id": asset["asset_id"],
            "source": "Google News",
            "title": entry.title,
            "content": entry.get("summary", entry.title),
            "url": url,
            "published_at": published_at
        })

print(f"{len(news_rows)} articles fetched")

if news_rows:
    supabase.table("news").insert(news_rows).execute()


# =============================
# 3. NLP — FINBERT SENTIMENT
# =============================
news_items = supabase.table("news").select("*").execute().data
nlp_rows = []

print("Running sentiment analysis...")

for item in tqdm(news_items, desc="Running sentiment"):
    try:
        result = sentiment_pipeline(item["content"][:512])[0]
    except Exception as e:
        print("Sentiment error:", e)
        continue

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
# 4. DAILY METRICS (MEILLEUR SIGNAL)
# =============================
metrics = defaultdict(list)

rows = supabase.table("news_nlp") \
    .select("sentiment_score, news:news_id(asset_id, published_at)") \
    .execute().data

for row in rows:
    asset_id = row["news"]["asset_id"]
    d = row["news"]["published_at"][:10]
    metrics[(asset_id, d)].append(row["sentiment_score"])

metric_rows = []

for (asset_id, d), scores in metrics.items():
    avg = sum(scores) / len(scores)
    std = (sum((s - avg) ** 2 for s in scores) / len(scores)) ** 0.5
    volume = len(scores)

    if volume < 3:
        signal = "low_coverage"
    elif std > 0.35:
        signal = "high_uncertainty"
    elif avg > 0.15:
        signal = "positive_momentum"
    elif avg < -0.15:
        signal = "negative_pressure"
    else:
        signal = "neutral"

    metric_rows.append({
        "asset_id": asset_id,
        "metric_date": d,
        "avg_sentiment": avg,
        "news_volume": volume,
        "sentiment_std": std,
        "signal": signal
    })

if metric_rows:
    supabase.table("daily_metrics").upsert(
        metric_rows,
        on_conflict="asset_id,metric_date"
    ).execute()


# =============================
# 5. MARKET BRIEFS (RULE-BASED, EXPLICABLE)
# =============================
today = date.today()
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
    dominant_signal = max(
        set(r["signal"] for r in rows),
        key=lambda s: sum(x["signal"] == s for x in rows)
    )

    content = (
        f"Over the past {LOOKBACK_DAYS} days, {asset['name']} ({asset['ticker']}) "
        f"was mentioned in {total_news} news articles. "
        f"The average sentiment score was {avg_sent:.2f}. "
        f"The dominant market signal is **{dominant_signal.replace('_', ' ')}**. "
        f"This signal reflects the balance between sentiment strength, volatility, "
        f"and news coverage."
    )

    supabase.table("market_briefs").insert({
        "period_start": start_date.isoformat(),
        "period_end": today.isoformat(),
        "scope": asset["ticker"],
        "content": content,
        "model_name": "rule_based_v1"
    }).execute()

print("Pipeline completed successfully.")
