import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from data_loader import load_stock_data
from data_preprocessing import preprocess_data
from model import train_model, evaluate_model
from news_sentiment import fetch_news, analyze_sentiment, summarize_sentiment

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Price Prediction App")

# Search bar
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", "AAPL")

if ticker:
    df = load_stock_data(ticker)

    # Stock information
    st.subheader(f"📄 Stock Information: {ticker.upper()}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        st.markdown(f"**Company Name:** {info.get('longName', 'N/A')}")
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
        st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
        st.markdown(f"**Market Cap:** {info.get('marketCap', 'N/A'):,}")
        st.markdown(f"**Country:** {info.get('country', 'N/A')}")
    except:
        st.warning("Could not retrieve full stock information.")

    # 🔍 News Sentiment Analysis
    st.subheader("📰 Sentiment Analysis Based on News")
    with st.spinner("Analyzing latest news..."):
        articles = fetch_news(ticker)
        sentiments = analyze_sentiment(articles)
        summary, avg_polarity = summarize_sentiment(sentiments)

        # Emoji and color summary
        sentiment_emoji = "😐"
        sentiment_color = "gray"
        if avg_polarity > 0.1:
            sentiment_emoji = "😄"
            sentiment_color = "green"
        elif avg_polarity < -0.1:
            sentiment_emoji = "😞"
            sentiment_color = "red"

        st.markdown(
            f"<span style='color:{sentiment_color}; font-size:18px;'>"
            f"<b>Sentiment Summary</b>: {summary} {sentiment_emoji} (Polarity Score: {avg_polarity:.2f})"
            f"</span>",
            unsafe_allow_html=True
        )

        # Bar chart for sentiment breakdown
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        for item in sentiments:
            if item["polarity"] > 0.1:
                sentiment_counts["Positive"] += 1
            elif item["polarity"] < -0.1:
                sentiment_counts["Negative"] += 1
            else:
                sentiment_counts["Neutral"] += 1

        sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=["Sentiment", "Count"])
        bar_fig = px.bar(sentiment_df, x="Sentiment", y="Count", color="Sentiment", color_discrete_map={
            "Positive": "green",
            "Neutral": "gray",
            "Negative": "red"
        })
        st.plotly_chart(bar_fig, use_container_width=True)

        with st.expander("View News Headlines with Sentiment Scores"):
            for item in sentiments:
                st.markdown(f"- {item['title']} (`{item['polarity']:.2f}`)")

    # Preprocess and train model
    X_train, X_test, y_train, y_test, dates_train, dates_test = preprocess_data(df, return_dates=True)
    model = train_model(X_train, y_train)
    predictions, mse = evaluate_model(model, X_test, y_test)

    # Align predictions and actuals with dates
    actual = y_test.values.flatten()
    predicted = predictions.flatten()
    dates = dates_test[-len(predicted):]

    results = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Actual Price": actual,
        "Predicted Price": predicted
    })
    results["Difference"] = results["Actual Price"] - results["Predicted Price"]
    results["Difference"] = results["Difference"].round(2)

    # 📊 Actual vs Predicted Chart
    st.subheader("📊 Actual vs Predicted Stock Price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results["Date"], y=results["Actual Price"], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=results["Date"], y=results["Predicted Price"], mode='lines', name='Predicted', line=dict(dash='dash')))
    fig.update_layout(xaxis_title='Date', yaxis_title='Price', legend_title='Legend')
    st.plotly_chart(fig, use_container_width=True)

    # 📌 Current vs Predicted - Key Metrics
    st.subheader("📌 Current vs Predicted Price")
    latest = results.iloc[-1]
    diff = latest['Difference']
    diff_color = "green" if diff >= 0 else "red"
    diff_icon = "⬆️" if diff >= 0 else "⬇️"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Date", str(latest['Date'].date()))
    with col2:
        st.metric("Actual", f"${latest['Actual Price']:.2f}")
    with col3:
        st.metric("Predicted", f"${latest['Predicted Price']:.2f}")
    with col4:
        st.markdown(
            f"<div style='color:{diff_color}; font-weight:bold; font-size:20px;'>"
            f"Difference: {diff:+.2f} {diff_icon}"
            f"</div>",
            unsafe_allow_html=True
        )

    # 📉 MSE
    st.subheader("📉 Model Evaluation")
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
