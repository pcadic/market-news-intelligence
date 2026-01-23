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

# FinBERT (standard du marché)
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

all_news = []

# ======================
# FETCH NEWS
# ======================

for asset in tqdm(assets, desc="Fetching news"):
    ticker = asset["ticker"]
    rss_url = google_news_rss(f"{ticker} stock")

    feed = feedparser.parse(rss_url)

    for entry in feed.entries:
        all_news.append({
            "asset_id": asset["asset_id"],
            "title": entry.get("title"),
            "source": entry.get("source", {}).get("title"),
            "url": entry.get("link"),
            "published_at": to_iso_datetime(entry),
            "content": entry.get("summary", "")
        })

print(f"{len(all_news)} articles fetched")

# ======================
# INSERT NEWS
# ======================

if all_news:
    supabase.table("news").insert(all_news).execute()

# ======================
# NLP (FinBERT)
# ======================

nlp_rows = []

for row in tqdm(all_news, desc="Running sentiment"):
    text = row["title"] or ""
    if not text:
        continue

    result = sentiment_model(text)[0]

    nlp_rows.append({
        "asset_id": row["asset_id"],
        "url": row["url"],
        "sentiment": result["label"],
        "confidence": float(result["score"])
    })

if nlp_rows:
    supabase.table("news_nlp").insert(nlp_rows).execute()

print("Pipeline completed successfully")
