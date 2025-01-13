import yfinance as yf
from transformers import pipeline
import numpy as np
import pandas as pd
import finnhub
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Load sentiment analysis model
try:
    pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    print("[INFO] Loaded financial sentiment analysis model successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load the financial sentiment model: {e}")
    pipe = None

# Helper functions
def fetch_news(company, investing_horizon):
    load_dotenv()
    api_key = os.getenv("NEXT_PUBLIC_FINNHUB_API_KEY")
    finnhub_client = finnhub.Client(api_key=api_key)
    today = datetime.now()
    from_date = today - (timedelta(days=30) if investing_horizon == "short-term" else timedelta(days=365*5))
    news = finnhub_client.company_news(symbol=company, _from=from_date.strftime('%Y-%m-%d'), to=today.strftime('%Y-%m-%d'))
    return news if news else []

def preprocess_articles(articles):
    return [article["summary"] for article in articles if article.get("summary")]

def analyze_sentiment(articles):
    if not pipe or not articles:
        return 0
    sentiments = pipe(articles)
    total_confidence = sum(s["score"] for s in sentiments)
    positive_score = sum(s["score"] for s in sentiments if s["label"] == "positive")
    negative_score = sum(s["score"] for s in sentiments if s["label"] == "negative")
    return round(((positive_score - negative_score + total_confidence) / 2) / total_confidence * 100, 2)

def fetch_stock_data(company, investing_horizon):
    period = "1mo" if investing_horizon == "short-term" else "5y"
    stock_data = yf.download(company, period=period)
    if stock_data.empty:
        return None
    price_data = stock_data["Adj Close"] if "Adj Close" in stock_data.columns else stock_data["Close"]
    daily_returns = price_data.pct_change().dropna()
    return {
        "recent_growth": price_data.pct_change(periods=5).iloc[-1] * 100 if len(price_data) > 5 else np.nan,
        "historical_growth": (price_data.iloc[-1] / price_data.iloc[0] - 1) * 100 if len(price_data) > 1 else np.nan,
        "volatility": daily_returns.std() * np.sqrt(252) if not daily_returns.empty else np.nan
    }

def calculate_confidence(sentiment_score, stock_data, investing_horizon, company):
    if investing_horizon == "short-term":
        features = [[sentiment_score, stock_data.get("recent_growth", 0)]]
    else:
        features = [[sentiment_score, stock_data.get("historical_growth", 0), stock_data.get("volatility", 0)]]
    X_train = np.random.rand(100, len(features[0]))
    y_train = np.random.randint(60, 100, 100)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    features_scaled = scaler.transform(features)
    model = LinearRegression().fit(X_train_scaled, y_train)
    return round(min(max(model.predict(features_scaled)[0], 0), 100), 2)

def recommend_stocks(companies, preferences):
    recommendations = []
    for company in companies:
        news = fetch_news(company, preferences["investing_horizon"])
        sentiment_score = analyze_sentiment(preprocess_articles(news))
        stock_data = fetch_stock_data(company, preferences["investing_horizon"])
        if stock_data:
            confidence = calculate_confidence(sentiment_score, stock_data, preferences["investing_horizon"], company)
            if confidence > preferences["min_confidence"]:
                recommendations.append({"company": company, "confidence": confidence})
    return recommendations

# Flask application setup
app = Flask(__name__)
CORS(app)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        companies = data.get('companies', [])
        preferences = {
            'min_confidence': data.get('min_confidence', 50),
            'investing_horizon': data.get('investing_horizon', 'short-term')
        }
        recommendations = recommend_stocks(companies, preferences)
        return jsonify({"status": "success", "recommendations": recommendations})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
