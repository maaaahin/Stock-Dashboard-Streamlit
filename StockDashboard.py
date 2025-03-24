import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews


# Setting page configuration
st.set_page_config(page_title="InvestInsight", page_icon="📈", layout="wide")

# Title and Ticker Input
st.title("📈 InvestInsight")
st.write("Analyze stock data, track financials, and get the latest financial news insights!")

st.sidebar.header("Enter Stock Information")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL", help="Enter a stock ticker symbol, e.g., 'AAPL' for Apple.")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# Fetch stock data and info
ticker_obj = yf.Ticker(ticker)
info = ticker_obj.info
stock_name = info.get("shortName", "Stock Information Not Available")
st.header(f"{stock_name} - {ticker}")

# Stock Price Chart
st.subheader("📉 Stock Price Over Time")
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
if not data.empty:
    fig = px.line(data, x=data.index, y=data['Adj Close'].squeeze(), title=f"{stock_name} Adjusted Closing Price")
    fig.update_traces(line=dict(color="#2E86C1"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No data available for this ticker and date range.")

# Tabbed Sections for Pricing Data, Fundamentals, News, and OpenAI
pricing_data, fundamental_data, news, stock_suggestions = st.tabs(
    ["📊 Pricing Data", "📋 Fundamental Data", "📰 Top Financial News", "🌟 Stock Suggestions"]
)

# Pricing Data Section
with pricing_data:
    st.subheader("💹 Pricing Movements")
    if not data.empty:
        data["% Change"] = data["Adj Close"].pct_change()
        data.dropna(inplace=True)
        st.write(data)

        # Calculate and display annual return, standard deviation, and risk-adjusted return
        annual_return = data["% Change"].mean() * 252 * 100
        stdev = np.std(data["% Change"]) * np.sqrt(252)
        risk_adj_return = annual_return / (stdev * 100)

        st.metric("Annual Return", f"{annual_return:.2f}%")
        st.metric("Standard Deviation", f"{stdev * 100:.2f}%")
        st.metric("Risk-Adjusted Return", f"{risk_adj_return:.2f}")
    else:
        st.write("No pricing data to display.")
# fundamental data section 
with fundamental_data:
    key = 'R1YZE1WFIKIW7KRE'
    fd = FundamentalData(key, output_format="pandas")
    st.subheader("📑 Financial Statements")

    with st.spinner("Fetching balance sheet..."):
        try:
            bs_data = fd.get_balance_sheet_annual(ticker)[0].T
            balance_sheet = bs_data[2:]
            balance_sheet.columns = list(bs_data.iloc[0])
            st.write("### Balance Sheet")
            st.write(balance_sheet)
        except:
            st.write("Balance Sheet data unavailable.")

    with st.spinner("Fetching income statement..."):
        try:
            is_data = fd.get_income_statement_annual(ticker)[0].T
            income_statement = is_data[2:]
            income_statement.columns = list(is_data.iloc[0])
            st.write("### Income Statement")
            st.write(income_statement)
        except:
            st.write("Income Statement data unavailable.")

    with st.spinner("Fetching cash flow statement..."):
        try:
            cf_data = fd.get_cash_flow_annual(ticker)[0].T
            cash_flow = cf_data[2:]
            cash_flow.columns = list(cf_data.iloc[0])
            st.write("### Cash Flow Statement")
            st.write(cash_flow)
        except:
            st.write("Cash Flow Statement data unavailable.")


import time
from concurrent.futures import ThreadPoolExecutor

# News Section
with news:
    st.subheader(f"📰 Financial News for {ticker}")

    with st.spinner("Fetching latest news..."):
        sn = StockNews(ticker, save_news=False)
        df_news = sn.read_rss()

    if not df_news.empty:
        def render_news(i):
            st.write(f"**News {i+1}:** {df_news['title'][i]}")
            st.caption(df_news["published"][i])
            st.write(df_news["summary"][i])
            st.write(f"**Sentiment - Title:** {df_news['sentiment_title'][i]} | **News:** {df_news['sentiment_summary'][i]}")
            st.write("---")

        with ThreadPoolExecutor() as executor:
            executor.map(render_news, range(min(10, len(df_news))))
    else:
        st.write("No news available for this ticker.")

# A* Search Function
def a_star_search(stocks, start_date, end_date, target_return, max_risk):
    suggestions = []
    for stock in stocks:
        try:
            # Fetch stock data
            data = yf.download(stock, start=start_date, end=end_date)
            if not data.empty:
                data["% Change"] = data["Adj Close"].pct_change()
                data.dropna(inplace=True)

                # Calculate metrics
                annual_return = data["% Change"].mean() * 252 * 100
                stdev = np.std(data["% Change"]) * np.sqrt(252)

                # Heuristic: Minimize risk and maximize return
                heuristic = abs(annual_return - target_return) + max(0, stdev - max_risk)

                suggestions.append((stock, annual_return, stdev, heuristic))
        except Exception as e:
            print(f"Error processing stock {stock}: {e}")
    
    # Sort by heuristic (lowest is better)
    suggestions.sort(key=lambda x: x[3])
    return suggestions

# Stock Suggestions Section
import random

with stock_suggestions:
    st.subheader("🌟 Stock Suggestions")

    selection = st.selectbox("Select Stock Suggestion Method:", ["A* Search on Historic Data", "News Sentiment"])

    if selection == "A* Search on Historic Data":
        st.write("Set your preferences for stock suggestions using A* Search:")
        target_return = st.slider("Target Annual Return (%)", 0, 50, 15)
        max_risk = st.slider("Maximum Risk (Standard Deviation %)", 0, 50, 20)

        stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
                  "V", "XOM", "WMT", "PG", "JPM", "MA", "LLY", "HD", "CVX", "MRK",
                  "PEP", "KO", "ABBV", "BAC", "AVGO", "COST", "MCD", "PFE", "TMO"]

        if st.button("Find Stocks Using A* Search"):
            with st.spinner("Running A* search on stock data..."):
                suggestions = a_star_search(stocks, start_date, end_date, target_return, max_risk)

                if suggestions:
                    st.write("### Top Stock Suggestions Based on A* Search:")
                    for stock, annual_return, risk, heuristic in suggestions[:5]:
                        st.write(f"**{stock}** - Annual Return: {annual_return:.2f}%, Risk: {risk:.2f}%, Score: {heuristic:.2f}")
                else:
                    st.write("No suitable stocks found based on your criteria.")

    elif selection == "News Sentiment":
        st.subheader("📈 Stock Suggestions Based on Latest Financial News Sentiment")

        tickers_to_analyze = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
                              "V", "XOM", "WMT", "PG", "JPM", "MA", "LLY", "HD", "CVX", "MRK",
                              "PEP", "KO", "ABBV", "BAC", "AVGO", "COST", "MCD", "PFE", "TMO"]

        stock_sentiments = {}

        with st.spinner("Analyzing news sentiment across multiple tickers..."):
            for ticker in tickers_to_analyze:
                try:
                    sn = StockNews(ticker, save_news=False)
                    df_news = sn.read_rss()
                    sentiment_score = df_news['sentiment_title'].mean() if not df_news.empty else 0
                    stock_sentiments[ticker] = sentiment_score
                    time.sleep(random.uniform(0.5, 1.5))  # Adds delay to avoid rate limits
                except:
                    stock_sentiments[ticker] = 0  # fallback on error

        sorted_sentiments = sorted(stock_sentiments.items(), key=lambda x: x[1], reverse=True)
        st.write("### Stocks with Positive News Sentiment:")
        for ticker, sentiment in sorted_sentiments[:5]:
            st.write(f"**{ticker}** - Sentiment Score: {sentiment:.2f}")

