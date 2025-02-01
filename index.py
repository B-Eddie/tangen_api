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
CORS(app, resources={
    r"/recommend": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": "*",
        "expose_headers": "*",
        "supports_credentials": True,
        "max_age": 86400
    }
})

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "confidence_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    print("Loaded financial sentiment analysis model successfully.")
except Exception as e:
    print(f"Failed to load the financial sentiment model: {e}")
    pipe = None

# TTL Cache for sentiment analysis
cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes

def preprocess_articles(articles):
    """Extract and truncate summaries from articles."""
    return [article["summary"][:100] for article in articles if article.get("summary")]

def analyze_sentiment(articles):
    if not articles:
        return {"score": 0, "positive": 0, "negative": 0, "neutral": 0, "total": 0}
    
    cache_key = tuple(articles)
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
        print(f"Failed to analyze sentiment: {e}")
        return {"score": 0, "positive": 0, "negative": 0, "neutral": 0, "total": 0}

def recommend_stocks(companies, context, horizon="short-term"):
    """
    Generate stock recommendations based on data analysis and sentiment.

    Args:
        companies (list): List of company ticker symbols.
        context (dict): Additional context for recommendations.
        horizon (str): Investing horizon ("short-term" or "long-term").

    Returns:
        list: Recommended stocks with analysis metadata.
    """
    recommendations = []

    for company in companies:
        try:
            print(f"Analyzing {company} for {horizon} investment.")

            # Fetch stock data and news
            stock_data = fetch_stock_data(company, horizon)
            news = fetch_news(company, horizon)
            
            if not stock_data or not news:
                print(f"Skipping {company} due to insufficient data.")
                continue

            # Analyze sentiment from news
            articles = preprocess_articles(news)
            sentiment = analyze_sentiment(articles)

            # Prepare the recommendation entry
            recommendations.append({
                "company": company,
                "stock_data": stock_data,
                "sentiment": sentiment
            })
        except Exception as e:
            print(f"Failed to analyze {company}: {e}")

    # Sort recommendations by positive sentiment score
    recommendations.sort(key=lambda x: x['sentiment']['positive'], reverse=True)

    return recommendations

def fetch_news(company, investing_horizon, start_date=None, end_date=None):
    try:
        load_dotenv()
        api_key = os.getenv("NEXT_PUBLIC_FINNHUB_API_KEY")
        finnhub_client = finnhub.Client(api_key=api_key)
        
        if start_date is None:
            today = datetime.now()
            # Adjust news fetch period based on horizon
            lookback_days = 30 if investing_horizon == "short-term" else 180
            start_date = today - timedelta(days=lookback_days)
        
        if end_date is None:
            end_date = datetime.now()
            
        news = finnhub_client.company_news(
            symbol=company,
            _from=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
        )
        print(f"Fetched {len(news)} news articles for {company}")
        return news or []
    except Exception as e:
        print(f"Failed to fetch news for {company}: {e}")
        return []

def get_appropriate_period(company, investing_horizon="long-term"):
    """
    Determine the appropriate period for stock data based on available history and investing horizon.
    
    Args:
        company (str): Stock ticker symbol
        investing_horizon (str): Either "short-term" or "long-term"
        
    Returns:
        str: Appropriate period string for yfinance
    """
    if investing_horizon == "short-term":
        return "1mo"
        
    try:
        # Try to get max historical data first to check availability
        test_data = yf.download(company, period="max", progress=False)
        days_available = len(test_data)
        
        print(f"Found {days_available} days of historical data for {company}")
        
        # Determine appropriate period based on available history
        if days_available >= 1250:  # About 5 years of trading days
            return "5y"
        elif days_available >= 500:  # About 2 years
            return "2y"
        elif days_available >= 250:  # About 1 year
            return "1y"
        elif days_available >= 125:  # About 6 months
            return "6mo"
        else:
            return "3mo"  # Minimum period for long-term analysis
            
    except Exception as e:
        print(f"Error checking historical data for {company}: {e}")
        # Fall back to 1 year if there's an error
        return "1y"

def fetch_stock_data(company, investing_horizon):
    try:
        # Determine appropriate period based on stock's history
        period = "1mo" if investing_horizon == "short-term" else get_appropriate_period(company)
        print(f"Fetching {period} of data for {company}")
        
        stock_data = yf.download(company, period=period, progress=False)
        print(f"Retrieved {len(stock_data)} data points for {company}")

        if stock_data.empty:
            print(f"No data found for {company}")
            return None

        # Calculate metrics based on available data
        min_lookback = 5  # Minimum number of days needed for calculations
        if len(stock_data) < min_lookback:
            print(f"Insufficient data points for {company}")
            return None

        recent_lookback = min(5, len(stock_data) - 1)
        recent_growth = ((stock_data['Close'].iloc[-1].item() / stock_data['Close'].iloc[-recent_lookback-1].item()) - 1) * 100
        historical_growth = ((stock_data['Close'].iloc[-1].item() / stock_data['Close'].iloc[0].item()) - 1) * 100
        volatility = stock_data['Close'].pct_change().std().item() * np.sqrt(252)
                
        return {
            "recent_growth": recent_growth,
            "historical_growth": historical_growth,
            "volatility": volatility,
            "data_points": len(stock_data)
        }
    except Exception as e:
        print(f"Failed to fetch stock data for {company}: {e}")
        return None

# ADDITIONAL LONG-TERM LOGIC CAN BE PLACED HERE

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

        print(f"Processing recommendation request for {companies} with {investing_horizon} horizon")
        recommendations = recommend_stocks(companies, {"investing_horizon": investing_horizon}, horizon=investing_horizon)
        
        if not recommendations:
            print(f"No recommendations generated for {companies}")
            return jsonify({"status": "success", "recommendations": []})

        return jsonify({
            "status": "success",
            "recommendations": recommendations,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "horizon": investing_horizon
            }
        })
    except Exception as e:
        print(f"Error processing recommendation request: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 32771))
    app.run(host='0.0.0.0', port=port)
