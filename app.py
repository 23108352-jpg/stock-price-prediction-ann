import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

from preprocess import get_stock_data, add_features, prepare_data
from model import train_model, load_model, model_exists

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock ANN Predictor",
    page_icon="📈",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#1f77b4; }
    .sub-title   { font-size:1rem;   color:#555; margin-bottom:1rem; }
    .metric-card { background:#f0f4ff; border-radius:10px; padding:1rem;
                   text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📈 Stock Market Trend Prediction Using ANN</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Muhammad Mubeen Khan &nbsp;|&nbsp; Reg: 23108352 &nbsp;|&nbsp; '
    'SZABIST Islamabad &nbsp;|&nbsp; BS(AI)-5B</p>',
    unsafe_allow_html=True
)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker  = st.text_input("Stock Ticker Symbol", value="AAPL",
                             help="e.g. AAPL, GOOGL, MSFT, TSLA")
    period  = st.selectbox("Historical Data Period",
                            ["1y", "2y", "3y", "5y"], index=3)
    window  = st.slider("Sliding Window Size", 30, 120, 60,
                         help="Number of past days used per prediction")
    retrain = st.checkbox("🔁 Force Retrain Model", value=False)

    st.divider()
    run_btn = st.button("🚀 Run Prediction", use_container_width=True)

    st.divider()
    st.markdown("**Popular Tickers**")
    col1, col2 = st.columns(2)
    with col1:
        st.code("AAPL\nGOOGL\nMSFT")
    with col2:
        st.code("TSLA\nAMZN\nMETA")

# ── Main content ───────────────────────────────────────────────────────────────
if not run_btn:
    st.info("👈  Configure settings in the sidebar and click **Run Prediction** to start.")
    st.markdown("""
    ### How it works
    1. **Data Collection** — Downloads historical OHLCV data via Yahoo Finance
    2. **Feature Engineering** — Adds SMA, EMA, RSI, MACD, Volatility indicators
    3. **ANN Model** — Multi-layer dense network trained on 80 % of data
    4. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix
    5. **Visualization** — Interactive charts of predicted vs actual trends
    """)
    st.stop()

# ── Step 1: Data ───────────────────────────────────────────────────────────────
with st.spinner(f"📥 Downloading {ticker} data…"):
    try:
        df = get_stock_data(ticker, period)
        df = add_features(df)
    except Exception as e:
        st.error(f"❌ Could not fetch data: {e}")
        st.stop()

st.success(f"✅ Loaded **{len(df):,}** rows for **{ticker}**")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data & Charts", "🧠 Training", "📋 Evaluation", "🔮 Predictions"]
)

# ── Tab 1: Data & Charts ───────────────────────────────────────────────────────
with tab1:
    st.subheader("Recent Stock Data")
    st.dataframe(df.tail(10), use_container_width=True)

    st.subheader("📉 Close Price with Moving Averages")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df['Close'],   label='Close',  linewidth=1.5)
    ax.plot(df.index, df['SMA_10'],  label='SMA 10', alpha=0.8, linestyle='--')
    ax.plot(df.index, df['SMA_50'],  label='SMA 50', alpha=0.8, linestyle='--')
    ax.set_title(f"{ticker} — Close Price & Moving Averages")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RSI")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(df.index, df['RSI'], color='purple')
        ax2.axhline(70, color='red',   linestyle='--', alpha=0.6, label='Overbought 70')
        ax2.axhline(30, color='green', linestyle='--', alpha=0.6, label='Oversold 30')
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        st.pyplot(fig2)

    with col2:
        st.subheader("MACD")
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.plot(df.index, df['MACD'], color='orange')
        ax3.axhline(0, color='black', linestyle='--', alpha=0.4)
        ax3.grid(alpha=0.3)
        st.pyplot(fig3)

# ── Prepare data ───────────────────────────────────────────────────────────────
with st.spinner("⚙️ Preparing sequences…"):
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, window)

# ── Tab 2: Training ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🧠 Model Architecture")
    st.code("""
Input  →  Flatten
        →  Dense(256, ReLU) → BatchNorm → Dropout(0.3)
        →  Dense(128, ReLU) → BatchNorm → Dropout(0.3)
        →  Dense(64,  ReLU) → Dropout(0.2)
        →  Dense(32,  ReLU)
        →  Dense(1, Sigmoid)   ← Binary Output: Up / Down

Optimizer : Adam (lr=0.001)
Loss      : Binary Crossentropy
Callbacks : EarlyStopping + ReduceLROnPlateau
    """)

    st.info(f"Training samples: **{len(X_train):,}** | Test samples: **{len(X_test):,}**")

    if retrain or not model_exists():
        progress = st.progress(0, text="Training in progress…")
        with st.spinner("Training ANN model (this may take a minute)…"):
            model, history = train_model(X_train, y_train)
        progress.progress(100, text="Training complete!")
        st.success("✅ Model trained and saved!")

        # Training curves
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 4))
        ax4a.plot(history.history['accuracy'],     label='Train')
        ax4a.plot(history.history['val_accuracy'], label='Val')
        ax4a.set_title('Accuracy'); ax4a.legend(); ax4a.grid(alpha=0.3)

        ax4b.plot(history.history['loss'],     label='Train')
        ax4b.plot(history.history['val_loss'], label='Val')
        ax4b.set_title('Loss'); ax4b.legend(); ax4b.grid(alpha=0.3)
        st.pyplot(fig4)
    else:
        model = load_model()
        st.info("ℹ️ Loaded existing saved model. Check **Force Retrain** to retrain.")

# ── Predictions ────────────────────────────────────────────────────────────────
with st.spinner("🔮 Running predictions…"):
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred      = (y_pred_prob > 0.5).astype(int)

acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

# ── Tab 3: Evaluation ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("📋 Performance Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc*100:.1f}%")
    c2.metric("F1 Score",  f"{f1:.3f}")
    c3.metric("Test Samples", f"{len(y_test):,}")
    c4.metric("Up Predictions", f"{int(y_pred.sum()):,}")

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred,
                                   target_names=['Down (0)', 'Up (1)'],
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3),
                 use_container_width=True)

    st.subheader("Confusion Matrix")
    fig5, ax5 = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'], ax=ax5)
    ax5.set_xlabel("Predicted"); ax5.set_ylabel("Actual")
    ax5.set_title("Confusion Matrix")
    st.pyplot(fig5)

# ── Tab 4: Predictions ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("🔮 Predicted vs Actual Trend (first 150 test samples)")
    n = min(150, len(y_test))
    fig6, ax6 = plt.subplots(figsize=(12, 4))
    ax6.plot(range(n), y_test[:n],      label='Actual',    alpha=0.8, linewidth=1.5)
    ax6.plot(range(n), y_pred[:n],      label='Predicted', alpha=0.7, linewidth=1.2,
             linestyle='--')
    ax6.set_yticks([0, 1]); ax6.set_yticklabels(['Down', 'Up'])
    ax6.set_title("Actual vs Predicted Direction"); ax6.legend(); ax6.grid(alpha=0.3)
    st.pyplot(fig6)

    st.subheader("Prediction Confidence (Probability)")
    fig7, ax7 = plt.subplots(figsize=(12, 3))
    ax7.plot(range(n), y_pred_prob[:n], color='darkorange', linewidth=1)
    ax7.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Threshold 0.5')
    ax7.fill_between(range(n), 0.5, y_pred_prob[:n],
                     where=y_pred_prob[:n] >= 0.5, alpha=0.2, color='green', label='Up')
    ax7.fill_between(range(n), y_pred_prob[:n], 0.5,
                     where=y_pred_prob[:n] < 0.5,  alpha=0.2, color='red',   label='Down')
    ax7.set_title("Model Confidence"); ax7.legend(); ax7.grid(alpha=0.3)
    st.pyplot(fig7)

st.divider()
st.caption("ANN Lab Project | SZABIST Islamabad | Department of Robotics & AI")
