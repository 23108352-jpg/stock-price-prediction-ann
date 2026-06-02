import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def get_stock_data(ticker="AAPL", period="5y"):
    """Download historical stock data from Yahoo Finance."""
    df = yf.download(ticker, period=period, auto_adjust=True)
    df.dropna(inplace=True)
    return df


def add_features(df):
    """Add technical indicators as features."""
    df = df.copy()

    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = -delta.where(delta < 0, 0).rolling(14).mean()
    rs    = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12      = df['Close'].ewm(span=12, adjust=False).mean()
    ema26      = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    # Volatility (rolling std)
    df['Volatility'] = df['Close'].rolling(window=10).std()

    df.dropna(inplace=True)
    return df


def prepare_data(df, window=60):
    """Scale features and create sliding-window sequences."""
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'SMA_50', 'EMA_10', 'RSI', 'MACD', 'Volatility'
    ]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    X, y = [], []
    close_col = features.index('Close')

    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i])
        # Label: 1 = price went Up, 0 = Down
        y.append(1 if scaled[i][close_col] > scaled[i - 1][close_col] else 0)

    X, y   = np.array(X), np.array(y)
    split  = int(0.8 * len(X))

    return X[:split], X[split:], y[:split], y[split:], scaler
