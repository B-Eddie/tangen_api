import os
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
import finnhub
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
from dotenv import load_dotenv
from cachetools import TTLCache
import joblib 

app = Flask(__name__)
CORS(app)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "confidence_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    print("[INFO] Loaded financial sentiment analysis model successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load the financial sentiment model: {e}")
    pipe = None

# TTL Cache for sentiment analysis
cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes

def preprocess_articles(articles):
    """Extract and truncate summaries from articles."""
    return [article["summary"][:100] for article in articles if article.get("summary")]

def analyze_sentiment(articles):
    if not articles:
        return {"score": 0, "positive": 0, "negative": 0, "neutral": 0, "total": 0}
    
    cache_key = tuple(articles)  #  hashable cache keys
    if cache_key in cache:
        return cache[cache_key]

    try:
        MAX_CHUNK_LENGTH = 450
        chunks = []
        current_chunk = ""
        
        for article in articles:
            if len(current_chunk) + len(article) + 1 > MAX_CHUNK_LENGTH:
                chunks.append(current_chunk)
                current_chunk = article
            else:
                current_chunk += " " + article if current_chunk else article
        
        if current_chunk:
            chunks.append(current_chunk)
        
        all_sentiments = []
        for chunk in chunks:
            chunk_sentiment = pipe(chunk)
            all_sentiments.extend(chunk_sentiment)
        
        positive_count = sum(1 for s in all_sentiments if s["label"] == "positive")
        negative_count = sum(1 for s in all_sentiments if s["label"] == "negative")
        neutral_count = sum(1 for s in all_sentiments if s["label"] == "neutral")
        
        sentiment_result = {
            "score": sum(s["score"] for s in all_sentiments) / len(all_sentiments) if all_sentiments else 0,
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "total": len(all_sentiments)
        }
        
        cache[cache_key] = sentiment_result
        return sentiment_result
    except Exception as e:
        print(f"[ERROR] Failed to analyze sentiment: {e}")
        return {"score": 0, "positive": 0, "negative": 0, "neutral": 0, "total": 0}

def fetch_news(company, investing_horizon, start_date=None, end_date=None):
    try:
        load_dotenv()
        api_key = os.getenv("NEXT_PUBLIC_FINNHUB_API_KEY")
        finnhub_client = finnhub.Client(api_key=api_key)
        
        if start_date is None:
            today = datetime.now()
            start_date = today - (timedelta(days=30) if investing_horizon == "short-term" else timedelta(days=365 * 5))
        
        if end_date is None:
            end_date = datetime.now()
            
        news = finnhub_client.company_news(
            symbol=company,
            _from=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
        )
        return news or []
    except Exception as e:
        print(f"[ERROR] Failed to fetch news for {company}: {e}")
        return []

def fetch_stock_data(company, investing_horizon):
    try:
        period = "1mo" if investing_horizon == "short-term" else "5y"
        stock_data = yf.download(company, period=period, progress=False)

        if stock_data.empty:
            return None

        recent_growth = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-6]) - 1) * 100 if len(stock_data) > 5 else 0.0
        historical_growth = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100 if len(stock_data) > 1 else 0.0
        volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) if len(stock_data) > 1 else 0.0
                
        return {"recent_growth": recent_growth, "historical_growth": historical_growth, "volatility": volatility}
    except Exception as e:
        print(f"[ERROR] Failed to fetch stock data for {company}: {e}")
        return None

def collect_training_data(companies):
    training_data = []
    training_labels = []
    
    for company in companies:
        try:
            print(f"[INFO] Collecting data for {company}")
            stock_data = yf.download(company, period="1y", progress=False)
            if stock_data.empty:
                print(f"[WARNING] No stock data for {company}")
                continue
                
            window_size = 30
            price_data = stock_data['Close']
            data_points = 0
            
            for i in range(window_size, len(stock_data) - 30):
                try:
                    window_end = datetime.now() - timedelta(days=len(stock_data)-i)
                    historical_news = fetch_news(company, "short-term", 
                        start_date=window_end - timedelta(days=window_size), 
                        end_date=window_end)
                    articles = preprocess_articles(historical_news)
                    sentiment = analyze_sentiment(articles)
                    
                    window_returns = price_data[i-window_size:i].pct_change().dropna()
                    
                    historical_growth = ((price_data.iloc[i].iloc[0] / price_data.iloc[i-window_size].iloc[0]) - 1) * 100
                    volatility = window_returns.std().iloc[0] * np.sqrt(252)
                    future_return = ((price_data.iloc[i+30].iloc[0] / price_data.iloc[i].iloc[0]) - 1) * 100
                    
                    features = np.array([
                        float(sentiment['score']),
                        historical_growth,
                        volatility,
                        float(sentiment['positive']) / max(sentiment['total'], 1),
                        float(sentiment['negative']) / max(sentiment['total'], 1)
                    ], dtype=np.float64)
                    
                    training_data.append(features)
                    confidence_label = min(max((future_return + 20) * 2.5, 0), 100)
                    training_labels.append(confidence_label)
                    data_points += 1
                    print(f"[DEBUG] Features for {company}: {features}")
                    print(f"[DEBUG] Label: {confidence_label}")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process window for {company}: {e}")
                    continue
                    
            print(f"[INFO] Collected {data_points} data points for {company}")
                
        except Exception as e:
            print(f"[ERROR] Failed to collect training data for {company}: {e}")
            continue
    
    if not training_data:
        raise ValueError("No training data collected")
    
    X = np.array(training_data, dtype=np.float64)
    y = np.array(training_labels, dtype=np.float64)
    print(f"[DEBUG] Training data shape: {X.shape}")
    print(f"[DEBUG] Labels shape: {y.shape}")
    return X, y

def train_confidence_model(training_companies, force_retrain=False):
    if not force_retrain and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("[INFO] Loading existing model...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    
    print("[INFO] Training new model...")
    X_train, y_train = collect_training_data(training_companies)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("[INFO] Model saved successfully")
    
    return model, scaler

def calculate_confidence(company, sentiment_score, stock_data, num_articles=0, model=None, scaler=None):
    try:
        if model is None or scaler is None:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                model = joblib.load(MODEL_PATH)
                scaler = joblib.load(SCALER_PATH)
            else:
                print("[WARNING] No saved model found, training new model...")
                model, scaler = train_confidence_model(['AAPL', 'GOOGL', 'MSFT', 'AMZN'])
        
        features = np.array([[
            float(sentiment_score.get('score', 0)),
            float(stock_data.get("historical_growth", 0)),
            float(stock_data.get("volatility", 0)),
            float(sentiment_score.get('positive', 0)) / max(sentiment_score.get('total', 1), 1),
            float(sentiment_score.get('negative', 0)) / max(sentiment_score.get('total', 1), 1)
        ]], dtype=np.float64)
        
        features_scaled = scaler.transform(features)
        confidence = float(model.predict(features_scaled)[0])
        
        return round(min(max(confidence, 0), 100), 2)
        
    except Exception as e:
        print(f"[ERROR] Failed to calculate confidence for {company}: {e}")
        return 0

def recommend_stocks(companies, preferences):
    training_companies = ['AAPL', 'MSFT', 'GOOGL', 'META']
    try:
        model, scaler = train_confidence_model(training_companies)
    except ValueError as e:
        print(f"[ERROR] Failed to train model: {e}")
        model, scaler = None, None
    
    recommendations = []
    for company in companies:
        news = fetch_news(company, preferences["investing_horizon"])
        articles = preprocess_articles(news)
        sentiment = analyze_sentiment(articles)
        stock_data = fetch_stock_data(company, preferences["investing_horizon"])
        
        if stock_data:
            confidence = calculate_confidence(
                company,
                sentiment_score=sentiment,
                stock_data=stock_data,
                num_articles=sentiment["total"],
                model=model,
                scaler=scaler
            )
            if confidence > 0:
                recommendations.append({
                    "company": company,
                    "confidence": confidence,
                    "sentiment": sentiment,
                })
    return recommendations


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    try:
        if request.method == "GET":
            companies = request.args.getlist('companies')
            investing_horizon = request.args.get('investing_horizon', 'short-term')
        else:  # POST
            data = request.json
            companies = data.get('companies', [])
            investing_horizon = data.get('investing_horizon', 'short-term')

        recommendations = recommend_stocks(companies, {"investing_horizon": investing_horizon})
        return jsonify({"status": "success", "recommendations": recommendations})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
