"""
data_exploration.py
-------------------
Exploratory Data Analysis (EDA) — visualize raw stock data and indicators.

Usage:
    python data_exploration.py
    python data_exploration.py --ticker TSLA --period 3y
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from preprocess import get_stock_data, add_features


def parse_args():
    parser = argparse.ArgumentParser(description="EDA for stock data")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--period", type=str, default="3y")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Running EDA for {args.ticker} ({args.period})…")
    df = get_stock_data(args.ticker, args.period)
    df = add_features(df)

    print(f"\nDataset shape : {df.shape}")
    print(f"Date range    : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"\nBasic stats:\n{df[['Open','High','Low','Close','Volume']].describe().round(2)}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"EDA — {args.ticker}", fontsize=15, fontweight='bold')
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

    # 1. Close + MAs
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['Close'],   label='Close',  linewidth=1.5)
    ax1.plot(df.index, df['SMA_10'],  label='SMA 10', alpha=0.8, linestyle='--')
    ax1.plot(df.index, df['SMA_50'],  label='SMA 50', alpha=0.8, linestyle='--')
    ax1.set_title('Close Price & Moving Averages')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # 2. Volume
    ax2 = fig.add_subplot(gs[1, :])
    ax2.bar(df.index, df['Volume'], color='steelblue', alpha=0.6, width=1)
    ax2.set_title('Volume'); ax2.grid(alpha=0.3)

    # 3. RSI
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(df.index, df['RSI'], color='purple', linewidth=1)
    ax3.axhline(70, color='red',   linestyle='--', alpha=0.6)
    ax3.axhline(30, color='green', linestyle='--', alpha=0.6)
    ax3.set_title('RSI (14)'); ax3.grid(alpha=0.3)

    # 4. MACD
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(df.index, df['MACD'], color='orange', linewidth=1)
    ax4.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax4.set_title('MACD'); ax4.grid(alpha=0.3)

    # 5. Volatility
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.plot(df.index, df['Volatility'], color='red', linewidth=1)
    ax5.set_title('Volatility (10-day std)'); ax5.grid(alpha=0.3)

    # 6. Correlation heatmap
    ax6 = fig.add_subplot(gs[3, 1])
    cols = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'MACD', 'Volatility']
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax6, annot_kws={"size": 7}, linewidths=0.5)
    ax6.set_title('Feature Correlation')

    plt.savefig("eda_report.png", dpi=150, bbox_inches='tight')
    print("\n📊  EDA report saved to eda_report.png")


if __name__ == "__main__":
    main()
