"""
evaluate.py
-----------
Load the saved model and print full evaluation metrics + save plots.

Usage:
    python evaluate.py
    python evaluate.py --ticker MSFT --period 3y
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

from preprocess import get_stock_data, add_features, prepare_data
from model import load_model, model_exists


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved ANN model")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--period", type=str, default="5y")
    parser.add_argument("--window", type=int, default=60)
    return parser.parse_args()


def print_metrics(y_test, y_pred):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print("\n── Evaluation Metrics ──────────────────────────")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("────────────────────────────────────────────────")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Down (0)', 'Up (1)']))


def plot_confusion_matrix(y_test, y_pred, ticker):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {ticker}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("📊  Confusion matrix saved to confusion_matrix.png")


def plot_predictions(y_test, y_pred, y_prob, ticker, n=150):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(f"Predictions — {ticker} (first {n} test samples)")

    ax1.plot(y_test[:n],  label='Actual',    linewidth=1.5, alpha=0.8)
    ax1.plot(y_pred[:n],  label='Predicted', linewidth=1.2, linestyle='--', alpha=0.8)
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(['Down', 'Up'])
    ax1.set_title('Actual vs Predicted Direction')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(y_prob[:n], color='darkorange', linewidth=1)
    ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Threshold')
    ax2.fill_between(range(n), 0.5, y_prob[:n],
                     where=y_prob[:n] >= 0.5, alpha=0.15, color='green', label='Up')
    ax2.fill_between(range(n), y_prob[:n], 0.5,
                     where=y_prob[:n] <  0.5, alpha=0.15, color='red',   label='Down')
    ax2.set_title('Prediction Confidence (Probability)')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    print("📊  Prediction chart saved to predictions.png")


def main():
    args = parse_args()

    print("=" * 50)
    print("  Stock Market ANN — Evaluation Script")
    print("  Muhammad Mubeen Khan | 23108352 | SZABIST")
    print("=" * 50)

    if not model_exists():
        print("\n❌  No saved model found. Run  python train.py  first.")
        return

    print(f"\n[1/3] Loading data for {args.ticker}…")
    df = get_stock_data(args.ticker, args.period)
    df = add_features(df)

    print(f"[2/3] Preparing test sequences (window={args.window})…")
    _, X_test, _, y_test, _ = prepare_data(df, args.window)

    print("[3/3] Running inference…")
    model     = load_model()
    y_prob    = model.predict(X_test, verbose=0).flatten()
    y_pred    = (y_prob > 0.5).astype(int)

    print_metrics(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, args.ticker)
    plot_predictions(y_test, y_pred, y_prob, args.ticker)

    print("\n✅  Evaluation complete!")


if __name__ == "__main__":
    main()
