"""
train.py
--------
Run this file directly to download data, train the ANN, and save the model.

Usage:
    python train.py
    python train.py --ticker TSLA --period 5y
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from preprocess import get_stock_data, add_features, prepare_data
from model import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train ANN for Stock Prediction")
    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--period", type=str, default="5y",
                        help="Data period: 1y 2y 3y 5y (default: 5y)")
    parser.add_argument("--window", type=int, default=60,
                        help="Sliding window size (default: 60)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("  Stock Market ANN — Training Script")
    print("  Muhammad Mubeen Khan | 23108352 | SZABIST")
    print("=" * 50)

    # ── 1. Download data ───────────────────────────────
    print(f"\n[1/4] Downloading {args.ticker} data ({args.period})…")
    df = get_stock_data(args.ticker, args.period)
    print(f"      Rows downloaded: {len(df):,}")

    # ── 2. Feature engineering ─────────────────────────
    print("\n[2/4] Engineering features (SMA, EMA, RSI, MACD, Volatility)…")
    df = add_features(df)
    print(f"      Final dataset shape: {df.shape}")

    # ── 3. Prepare sequences ───────────────────────────
    print(f"\n[3/4] Creating sliding-window sequences (window={args.window})…")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, args.window)
    print(f"      X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"      y_train up%: {y_train.mean()*100:.1f}%  |  "
          f"y_test up%: {y_test.mean()*100:.1f}%")

    # ── 4. Train model ─────────────────────────────────
    print("\n[4/4] Training ANN model…")
    model, history = train_model(X_train, y_train)
    print("\n✅  Model saved to saved_model/ann_stock_model.h5")

    # ── Plot training curves ───────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training Results — {args.ticker}", fontsize=13)

    ax1.plot(history.history['accuracy'],     label='Train Acc')
    ax1.plot(history.history['val_accuracy'], label='Val Acc')
    ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history.history['loss'],     label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Loss'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("📊  Training curves saved to training_curves.png")

    print("\nDone! Run  streamlit run app.py  to launch the web app.")


if __name__ == "__main__":
    main()
