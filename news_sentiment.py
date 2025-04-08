# news_sentiment.py
import requests
from textblob import TextBlob


API_KEY = "aba9e0cbd7984ffb8f83bea761fdbd59"  # Replace with your key

def fetch_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    articles = data.get("articles", [])[:10]  # Get latest 10 articles
    return articles

def analyze_sentiment(articles):
    sentiments = []
    for article in articles:
        title = article['title']
        blob = TextBlob(title)
        polarity = blob.sentiment.polarity
        sentiments.append({
            "title": title,
            "polarity": polarity
        })
    return sentiments

def summarize_sentiment(sentiments):
    if not sentiments:
        return "No news available.", 0

    avg_polarity = sum([s['polarity'] for s in sentiments]) / len(sentiments)
    summary = "Neutral"
    if avg_polarity > 0.1:
        summary = "Positive"
    elif avg_polarity < -0.1:
        summary = "Negative"

    return summary, avg_polarity
