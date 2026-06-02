
import streamlit as st
import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Prediction App (ANN)")

model_path = "stock_ann_model.h5"

try:
    model = tf.keras.models.load_model(model_path)
except:
    st.error("Model file not found. Please place 'stock_ann_model.h5' in the same folder as app.py")
    st.stop()

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Predict"):
    stock = yf.download(symbol, start="2020-01-01", end="2025-01-01")
    stock = stock[['Open','High','Low','Volume','Close']].dropna()

    X = stock[['Open','High','Low','Volume']]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    preds = model.predict(X_scaled)

    stock['Predicted Close'] = preds

    st.subheader("Data")
    st.write(stock.tail())

    st.subheader("Graph")
    st.line_chart(stock[['Close','Predicted Close']])
