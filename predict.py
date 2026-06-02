"""
predict.py
----------
Predict tomorrow's direction for any stock using the saved model.

Usage:
    python predict.py
    python predict.py --ticker GOOGL
"""

import argparse
import numpy as np

from preprocess import get_stock_data, add_features, prepare_data
from model import load_model, model_exists


def parse_args():
    parser = argparse.ArgumentParser(description="Predict next-day stock direction")
    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Stock ticker (default: AAPL)")
    parser.add_argument("--window", type=int, default=60,
                        help="Sliding window size (must match training)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("  Stock Market ANN — Single Prediction")
    print("  Muhammad Mubeen Khan | 23108352 | SZABIST")
    print("=" * 50)

    if not model_exists():
        print("\n❌  No saved model found. Run  python train.py  first.")
        return

    # Download recent data
    print(f"\nFetching recent data for {args.ticker}…")
    df = get_stock_data(args.ticker, period="1y")
    df = add_features(df)

    if len(df) < args.window + 1:
        print(f"❌  Not enough data. Need at least {args.window + 1} rows.")
        return

    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'SMA_50', 'EMA_10', 'RSI', 'MACD', 'Volatility'
    ]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    # Use the last `window` rows as input
    X_latest = scaled[-args.window:].reshape(1, args.window, len(features))

    model      = load_model()
    prob       = model.predict(X_latest, verbose=0)[0][0]
    direction  = "📈 UP   (BUY signal)" if prob >= 0.5 else "📉 DOWN (SELL signal)"
    confidence = prob if prob >= 0.5 else 1 - prob

    last_close = df['Close'].iloc[-1]

    print(f"\n{'─'*40}")
    print(f"  Ticker       : {args.ticker}")
    print(f"  Last Close   : ${float(last_close):.2f}")
    print(f"  Prediction   : {direction}")
    print(f"  Confidence   : {confidence*100:.1f}%")
    print(f"  Raw Prob     : {prob:.4f}")
    print(f"{'─'*40}")
    print("\n⚠️  Disclaimer: For educational purposes only.")
    print("   Do NOT use this for real investment decisions.\n")


if __name__ == "__main__":
    main()
