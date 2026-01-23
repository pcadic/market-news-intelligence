import os
import feedparser
import datetime
import statistics
from supabase import create_client
from transformers import pipeline
from tqdm import tqdm
from urllib.parse import quote_plus

# =============================
# CONFIG
# =============================
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# FinBERT sentiment model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="yiyanghkust/finbert-tone"
)

TODAY = datetime.date.today()

# =============================
# STEP 1 — FETCH ASSETS
# =============================
assets = supabase.table("assets").select("*").execute().data
print(f"{len(assets)} assets found")

# =============================
# STEP 2 — FETCH NEWS (REAL, WORKING)
# =============================
news_rows = []

for asset in tqdm(assets, desc="Fetching news"):
    query = quote_plus(f"{asset['ticker']} stock")
    rss_url = (
        "https://news.google.com/rss/search"
        f"?q={query}&hl=en-CA&gl=CA&ceid=CA:en"
    )

    feed = feedparser.parse(rss_url)

    for entry in feed.entries:
        news_rows.append({
            "asset_id": asset["asset_id"],
            "source": "Google News",
            "title": entry.title,
            "content": entry.get("summary", entry.title),
            "url": entry.link,
            "published_at": datetime.datetime(*entry.published_parsed[:6])
        })

print(f"{len(news_rows)} articles fetched")

if news_rows:
    supabase.table("news").insert(news_rows).execute()

# =============================
# STEP 3 — NLP WITH FINBERT
# =============================
news = supabase.table("news").select("*").execute().data
existing_nlp = supabase.table("news_nlp").select("news_id").execute().data
processed_ids = {n["news_id"] for n in existing_nlp}

nlp_rows = []

for n in tqdm(news, desc="Running FinBERT"):
    if n["news_id"] in processed_ids:
        continue

    text = (n["title"] + ". " + n["content"])[:512]
    result = sentiment_model(text)[0]

    nlp_rows.append({
        "news_id": n["news_id"],
        "sentiment_score": result["score"],
        "sentiment_label": result["label"].lower(),
        "model_name": "yiyanghkust/finbert-tone"
    })

if nlp_rows:
    supabase.table("news_nlp").insert(nlp_rows).execute()

# =============================
# STEP 4 — DAILY METRICS
# =============================
metrics = {}

for nlp in nlp_rows:
    news_item = next(n for n in news if n["news_id"] == nlp["news_id"])
    asset_id = news_item["asset_id"]

    metrics.setdefault(asset_id, []).append(nlp["sentiment_score"])

metric_rows = []

for asset_id, scores in metrics.items():
    avg = statistics.mean(scores)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0

    if avg > 0.6:
        signal = "positive_momentum"
    elif avg < 0.4:
        signal = "caution"
    elif std > 0.2:
        signal = "high_uncertainty"
    else:
        signal = "neutral"

    metric_rows.append({
        "asset_id": asset_id,
        "metric_date": TODAY,
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
# STEP 5 — MARKET BRIEF (LLM TEXT)
# =============================
metrics_today = supabase.table("daily_metrics") \
    .select("*") \
    .eq("metric_date", TODAY.isoformat()) \
    .execute().data

summary = f"Market overview for {TODAY}:\n\n"

for m in metrics_today:
    asset = next(a for a in assets if a["asset_id"] == m["asset_id"])
    summary += (
        f"- {asset['ticker']} ({asset['name']}): "
        f"{m['signal']} | Avg sentiment: {round(m['avg_sentiment'],2)} "
        f"({m['news_volume']} articles)\n"
    )

supabase.table("market_briefs").insert({
    "period_start": TODAY,
    "period_end": TODAY,
    "scope": "TSX",
    "content": summary,
    "model_name": "finbert-aggregation"
}).execute()

print("Pipeline completed successfully.")
