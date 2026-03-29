import yfinance as yf
import numpy as np
import os
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Stocks to train
stocks = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",

    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",

    "Gold": "GC=F",
    "DDR5 RAM (Micron)": "MU",
    "Meta": "META",
    "Netflix": "NFLX",
    "Intel": "INTC",
    "AMD": "AMD",

    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",

    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",

    "S&P 500": "^GSPC",
    "Nifty 50": "^NSEI"
}
os.makedirs("models", exist_ok=True)

def create_sequences(data, seq_length=100):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

for name, ticker in stocks.items():
    print(f"\nTraining {name} ({ticker})...")

    data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

    # check about data
    if data.empty:
        print(f"Skipping {ticker} (no data found)")
        continue

    data = data[['Open','High','Low','Close','Volume']]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = create_sequences(scaled_data)

    if len(X) == 0:
        print(f"Skipping {ticker} (not enough data)")
        continue

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    # Save using ticker (correct)
    model.save(f"models/{ticker}.h5")


print("\nAll models trained and saved.")
