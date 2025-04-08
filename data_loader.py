import yfinance as yf
import pandas as pd

def load_stock_data(ticker):
    df = yf.download(ticker, start="2015-01-01")
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df
