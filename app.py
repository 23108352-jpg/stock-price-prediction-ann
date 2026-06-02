import streamlit as st
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Prediction App (ANN)")

model = tf.keras.models.load_model("stock_ann_model.h5")

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Predict"):
    stock = yf.download(symbol, start="2020-01-01", end="2025-01-01")
    stock = stock[['Open','High','Low','Volume','Close']].dropna()

    X = stock[['Open','High','Low','Volume']]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    predictions = model.predict(X_scaled)
    stock["Predicted Close"] = predictions

    st.write(stock.tail())
    st.line_chart(stock[['Close', 'Predicted Close']])
