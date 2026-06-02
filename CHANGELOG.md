# Changelog

All notable changes to this project will be documented here.

---

## [1.0.0] - 2024

### Added
- Initial release of Stock Market Trend Prediction Using ANN
- `preprocess.py` — data download, feature engineering (SMA, EMA, RSI, MACD, Volatility)
- `model.py` — ANN architecture with BatchNorm & Dropout, train/load utilities
- `train.py` — standalone CLI training script with `--ticker` and `--period` flags
- `evaluate.py` — evaluation script saving confusion matrix & prediction plots
- `predict.py` — single-stock next-day direction predictor
- `data_exploration.py` — full EDA with correlation heatmap and indicator charts
- `app.py` — interactive 4-tab Streamlit dashboard
- `requirements.txt` — pinned dependencies
- `README.md` — full project documentation
- `CONTRIBUTING.md` — contribution guidelines
- `LICENSE` — MIT license
- `.github/workflows/ci.yml` — GitHub Actions CI pipeline

---

## [Upcoming]
- LSTM / RNN model comparison
- Sentiment analysis integration
- Real-time prediction via live API
- Portfolio optimization module
