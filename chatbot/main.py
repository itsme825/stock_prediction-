import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression




import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def classify_sentiment(title):
    inputs = tokenizer(title, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    label = model.config.id2label[predicted_class_id].upper()

    return 'POSITIVE' if 'POSITIVE' in label else 'NEGATIVE'


import requests

def get_news_data(ticker, start_date, end_date):
    api_key = "5e55cd905f1b4c97b3deab676e335974"
    base_url = "https://newsapi.org/v2/everything"

    query = f"{ticker} stock"
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')

    all_articles = []

    page = 1
    while True:
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 100,
            "page": page,
            "apiKey": api_key,
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        if "articles" not in data:
            print("Error fetching news:", data)
            break

        articles = data["articles"]
        if not articles:
            break

        all_articles.extend(articles)
        if len(articles) < 100:
            break
        page += 1

    news_df = pd.DataFrame([
        {"Date": pd.to_datetime(article["publishedAt"]),
         "Title": article["title"]}
        for article in all_articles
    ])

    if news_df.empty:
        return pd.DataFrame(columns=["Date", "Title", "sentiment", "DateOnly"])

    news_df['DateOnly'] = news_df['Date'].dt.date
    news_df['Title'] = news_df['Title'].str.lower()
    news_df['sentiment'] = news_df['Title'].apply(classify_sentiment)
    news_df = news_df[news_df['sentiment'] != 'NEUTRAL']

    news_df = news_df.groupby('DateOnly').head(5).reset_index(drop=True)
    news_df['Date'] = pd.to_datetime(news_df['Date']).dt.date


    return news_df





def process_sentiment_data(news):
    news['DateOnly'] = pd.to_datetime(news['Date']).dt.date

    daily_counts = news.groupby(['DateOnly', 'sentiment']).size().unstack(fill_value=0)
    daily_counts = daily_counts.reindex(columns=['POSITIVE', 'NEGATIVE'], fill_value=0)

    daily_counts['7day_avg_positive'] = daily_counts['POSITIVE'].rolling(window=7, min_periods=1).sum()
    daily_counts['7day_avg_negative'] = daily_counts['NEGATIVE'].rolling(window=7, min_periods=1).sum()

    total = daily_counts['POSITIVE'] + daily_counts['NEGATIVE']
    daily_counts['7day_pct_positive'] = daily_counts['POSITIVE'] / total.replace(0, pd.NA)

    result_df = daily_counts.reset_index()
    return result_df


def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.droplevel(1)
    stock_data['Pct_Change'] = stock_data['Close'].pct_change() * 100
    stock_data['Price'] = stock_data['Close']
    stock_data = stock_data.reset_index()  
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    return stock_data



def get_stock_deviation_range(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    df = pd.DataFrame()
    df[ticker] = data["Adj Close"]
    return(round(df[ticker].std(),2))


def combine_data(result_df, stock_data):
    stock_data = stock_data.copy()
    stock_data['DateOnly'] = pd.to_datetime(stock_data['Date']).dt.date
    stock_data = stock_data.set_index('DateOnly')
    
    result_df['DateOnly'] = pd.to_datetime(result_df['DateOnly']).dt.date
    combined_df = result_df.set_index('DateOnly').join(stock_data[['Pct_Change']], how='inner')
    
    combined_df['lagged_7day_pct_positive'] = combined_df['7day_pct_positive'].shift(1)
    combined_df = combined_df.fillna(0)
    
    return combined_df




def get_stock_plot(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    
    if data.empty:
        st.error("No data returned. Check ticker or date range.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Adj Close"], label=ticker)
    plt.title(f"{ticker} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    st.pyplot(plt)



def fit_linear_model(df):
    df = df.dropna()
    X = df[['lagged_7day_pct_positive']]
    y = df['Pct_Change']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = model.score(X, y)
    st.write("R-squared:", r2)

    if r2 < 0.4:
        st.warning("⚠️ TOO LOW OF A SCORE – Consider relying more on market price trends.")

    st.write("### Predictions vs Actual:")
    for actual, pred in zip(y.values, y_pred):
        st.write(f"Actual: {actual:.2f}%, Predicted: {pred:.2f}%")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='blue', label='Actual')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted')
    plt.xlabel('Lagged 7-day % Positive Sentiment')
    plt.ylabel('Stock % Change')
    plt.title('Linear Regression: Sentiment vs Stock Change')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)  



    
    
    
    


    
    

