import streamlit as st
from transformers import pipeline
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime

from main import (
    get_news_data,
    process_sentiment_data,
    get_stock_data,
    get_stock_deviation_range,
    get_stock_plot,
    fit_linear_model,
    process_sentiment_data,
    combine_data
)

ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

SUPPORTED_TASKS = {
    "sentiment": "process_sentiment_data",
    "price": "get_stock_data",
    "deviation": "get_stock_deviation_range",
    "plot": "get_stock_plot",
    "regression": "fit_linear_model"
}

def extract_ticker(text):
    entities = ner(text)
    for ent in entities:
        if ent['entity_group'] in ['ORG', 'MISC']:
            return ent['word'].upper()
    return None

def detect_intent(text):
    labels = list(SUPPORTED_TASKS.keys()) + ["unknown"]
    result = classifier(text, labels)
    return result['labels'][0]

def handle_intent(intent, ticker, start_date, end_date):
    if intent == "sentiment":
        news = get_news_data(ticker, start_date, end_date)
        st.write("### Sentiment Analysis Result")
        st.dataframe(news)

    elif intent == "price":
        df = get_stock_data(ticker, start_date, end_date)
        st.write("### Stock Price Data")
        st.dataframe(df)

    elif intent == "deviation":
        dev = get_stock_deviation_range(ticker, start_date, end_date)
        st.write(f"### Deviation of {ticker} between {start_date.date()} and {end_date.date()} is **{dev}%**")

    elif intent == "plot":
        get_stock_plot(ticker,start_date,end_date)

    elif intent == "regression":
        news = get_news_data(ticker, start_date, end_date)
        if news.empty:
            st.warning("No news available for regression analysis.")
            return
        sentiment_df = process_sentiment_data(news)
        stock_df = get_stock_data(ticker, start_date, end_date)
        combined_df = combine_data(sentiment_df, stock_df)
        st.write("### Running Regression Model...")
        fit_linear_model(combined_df)
    else:
        st.error("❌ Sorry, I can't perform that action.")

# --- Streamlit UI ---



st.set_page_config(page_title="Stock Analyzer Chatbot")
st.title("🤖 Stock Analyzer Chatbot")

st.markdown("Hello! I'm a stock analysis assistant. Tell me what you'd like to know.")

if "user_input" not in st.session_state:
    st.session_state.user_input = None
if "intent" not in st.session_state:
    st.session_state.intent = None
if "ticker" not in st.session_state:
    st.session_state.ticker = None

user_input = st.chat_input("Ask anything about stock data...")

if user_input:
    st.session_state.user_input = user_input
    st.session_state.intent = detect_intent(user_input)
    st.session_state.ticker = extract_ticker(user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    if not st.session_state.ticker:
        st.chat_message("assistant").markdown("❓ Sorry, I couldn't find any stock ticker in your request.")
    elif st.session_state.intent not in SUPPORTED_TASKS:
        st.chat_message("assistant").markdown("❌ Sorry, I don't support that action.")
    else:
        st.chat_message("assistant").markdown(
            f"🔍 Processing **{st.session_state.intent}** for **{st.session_state.ticker}**..."
        )

if st.session_state.intent and st.session_state.ticker:
    if st.session_state.intent != "regression":
        with st.expander("🔧 Select Date Range"):
            start_date = st.date_input("Start Date", value=datetime.date(2025, 6, 1), key="start")
            end_date = st.date_input("End Date", value=datetime.date(2025, 6, 14), key="end")

        if st.button("🔍 Run"):
            handle_intent(
                st.session_state.intent,
                st.session_state.ticker,
                pd.to_datetime(start_date),
                pd.to_datetime(end_date)
            )
    else:
        fixed_start = pd.to_datetime("2025-06-04")
        fixed_end = pd.to_datetime("2025-06-17")
        handle_intent(st.session_state.intent, st.session_state.ticker, fixed_start, fixed_end)

