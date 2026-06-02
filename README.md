# 📈 Stock Market Trend Prediction Using ANN

[![CI](https://github.com/YOUR_USERNAME/stock-ann-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/stock-ann-prediction/actions)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io)

> **ANN Lab Project — SZABIST Islamabad | BS(AI)-5B**  
> **Student:** Muhammad Mubeen Khan | **Reg:** 23108352  
> **Submitted to:** Mr. Hassan Mujtaba

A deep learning project that uses an **Artificial Neural Network (ANN)** to predict
whether a stock price will move **Up ↑ or Down ↓** the next trading day.

---

## 📁 Project Structure

```
stock-ann-prediction/
│
├── app.py                  ← Streamlit web dashboard
├── model.py                ← ANN architecture & training
├── preprocess.py           ← Data collection & feature engineering
├── train.py                ← CLI training script
├── evaluate.py             ← Model evaluation & plots
├── predict.py              ← Single stock prediction
├── data_exploration.py     ← EDA charts & correlation heatmap
├── config.py               ← Central configuration
├── setup.py                ← Package setup
├── requirements.txt        ← Dependencies
├── CONTRIBUTING.md         ← How to contribute
├── CHANGELOG.md            ← Version history
├── LICENSE                 ← MIT License
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml          ← GitHub Actions CI
├── tests/
│   ├── test_preprocess.py  ← Unit tests for preprocessing
│   └── test_model.py       ← Unit tests for model
└── saved_model/            ← Auto-created after training
```

---

## 🧠 Features

| Feature | Detail |
|---------|--------|
| Live Data | Downloads via `yfinance` (Yahoo Finance) |
| Indicators | SMA, EMA, RSI, MACD, Volatility |
| ANN Model | 4 hidden layers, BatchNorm, Dropout |
| Output | Binary: Up (1) / Down (0) |
| Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Dashboard | Interactive Streamlit app with 4 tabs |

---

## ⚙️ Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/stock-ann-prediction.git
cd stock-ann-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run in order
```bash
python data_exploration.py   # EDA charts
python train.py              # Train the model
python evaluate.py           # Evaluate & save plots
python predict.py            # Predict tomorrow's direction
streamlit run app.py         # Launch web dashboard
```

### CLI options
```bash
python train.py    --ticker TSLA --period 3y
python evaluate.py --ticker TSLA --period 3y
python predict.py  --ticker GOOGL
```

---

## 🏗️ ANN Architecture

```
Input (60 days × 11 features)
    ↓  Flatten
    ↓  Dense(256, ReLU) → BatchNorm → Dropout(0.3)
    ↓  Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    ↓  Dense(64,  ReLU) → Dropout(0.2)
    ↓  Dense(32,  ReLU)
    ↓  Dense(1, Sigmoid)   →  Up (≥0.5) / Down (<0.5)

Optimizer : Adam (lr=0.001)
Loss      : Binary Crossentropy
Callbacks : EarlyStopping + ReduceLROnPlateau
```

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New App** → select your repo
4. Set main file to `app.py`
5. Click **Deploy** ✅

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.  
Do **NOT** use it for real investment decisions.  
Stock markets are unpredictable and this model does not guarantee accuracy.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
