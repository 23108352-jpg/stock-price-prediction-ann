"""
config.py
---------
Central configuration for the Stock ANN project.
Edit values here to change behaviour across all scripts.
"""

# ── Data settings ──────────────────────────────────────────────────────────────
DEFAULT_TICKER  = "AAPL"        # Default stock ticker
DEFAULT_PERIOD  = "5y"          # How much history to download
WINDOW_SIZE     = 60            # Sliding window (days) for sequences
TRAIN_SPLIT     = 0.80          # 80 % train, 20 % test

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_50", "EMA_10", "RSI", "MACD", "Volatility",
]

# ── Technical indicator settings ───────────────────────────────────────────────
SMA_SHORT   = 10
SMA_LONG    = 50
EMA_SPAN    = 10
RSI_PERIOD  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
VOL_WINDOW  = 10

# ── Model hyperparameters ──────────────────────────────────────────────────────
LEARNING_RATE   = 0.001
EPOCHS          = 50
BATCH_SIZE      = 32
DROPOUT_RATE    = 0.3
VALIDATION_SPLIT = 0.10

LAYER_SIZES = [256, 128, 64, 32]   # Hidden layer sizes

# ── Callbacks ──────────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE = 7
REDUCE_LR_PATIENCE  = 3
REDUCE_LR_FACTOR    = 0.5

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "saved_model/ann_stock_model.h5"

# ── Prediction threshold ───────────────────────────────────────────────────────
PRED_THRESHOLD = 0.5     # Probability above this = UP
