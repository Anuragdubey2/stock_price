# 📈 Stock Price Prediction App with Sentiment Analysis

This is a Streamlit-based web application that predicts future stock prices using machine learning and analyzes recent news headlines for sentiment. The app provides stock-specific insights including actual vs predicted prices, model evaluation, and sentiment-based news analysis.

## 🚀 Features

- 🔎 **Search any stock** by ticker symbol (e.g., AAPL, TSLA)
- 📄 **Company Info** including sector, industry, and market cap
- 🧠 **ML-based Price Prediction** using historical stock data
- 📰 **News Sentiment Analysis** using real-time headlines
- 📊 **Visuals** for sentiment breakdown and actual vs predicted prices
- 🧮 **Model Evaluation** using Mean Squared Error (MSE)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stock-predictor-app.git
   cd stock-predictor-app
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download TextBlob corpora (for sentiment analysis)**:
   ```bash
   python -m textblob.download_corpora
   ```

5. **Set up your NewsAPI key**:
   - Open `news_sentiment.py`
   - Replace `YOUR_NEWSAPI_KEY` with your actual API key from [https://newsapi.org](https://newsapi.org)

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 📂 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── data_loader.py          # Fetches historical stock data using yfinance
├── data_preprocessing.py   # Cleans and prepares data for ML model
├── model.py                # Trains and evaluates the regression model
├── news_sentiment.py       # Fetches news and performs sentiment analysis
├── requirements.txt        # Python dependencies
├── README.md               # This file
```

## 🧠 Model Details

- Features used: Open, High, Low, Close, Volume, and lag features
- Model: Scikit-learn's `LinearRegression`
- Evaluation metric: Mean Squared Error (MSE)

## 📸 Screenshots

*(Add screenshots here if desired)*

## 📜 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

Made with ❤️ using Streamlit and Python.
