def analyze_sentiment(text: str) -> tuple[str, float]:
    try:
        response = hf_sentiment(text)

        # Cas 1 : réponse vide
        if not response or not isinstance(response, list):
            return "neutral", 0.0

        # Cas 2 : format inattendu
        item = response[0]
        if "label" not in item or "score" not in item:
            return "neutral", 0.0

        return item["label"].lower(), float(item["score"])

    except Exception as e:
        print(f"Sentiment fallback used: {e}")
        return "neutral", 0.0
