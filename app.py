import tkinter as tk
from tkinter import messagebox
import yfinance as yf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
# stocks
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

def create_sequences(data, seq_length=60):
    X = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
    return np.array(X)

def predict_stock():
    selected = [stocks[name] for name, var in checkboxes.items() if var.get() == 1]

    if not selected:
        messagebox.showerror("Error", "Select at least one stock")
        return

    for ticker in selected:
        model_path = f"models/{ticker}.h5"

        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model for {ticker} not found. Run train.py first.")
            return

        print(f"\nPredicting {ticker}...")

        model = load_model(model_path, compile=False)

        data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
        data = data[['Open','High','Low','Close','Volume']]

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X = create_sequences(scaled_data)
        predictions = model.predict(X)

        dummy = np.zeros((predictions.shape[0], 5))
        dummy[:, 3] = predictions[:, 0] 

        predictions = scaler.inverse_transform(dummy)[:, 3].reshape(-1,1)

        real = data[['Close']].values[60:]
        mse = mean_squared_error(real, predictions)

        rmse = np.sqrt(mse)

# Accuracy (custom %)
        accuracy = 100 - (np.mean(np.abs((real - predictions) / real)) * 100)
        plt.figure(figsize=(8,4))
        plt.plot(real, label="Real")
        plt.plot(predictions, label="Predicted")

        plt.title(f"{ticker} | Accuracy: {accuracy:.2f}%")
        plt.legend()
        plt.show()
        
        print(f"{ticker} Accuracy: {accuracy:.2f}%")
        print(f"{ticker} RMSE: {rmse:.2f}")

# UI
root = tk.Tk()
root.title("Stock Predictor")

tk.Label(root, text="Select Stocks:", font=("Arial", 14)).pack()

checkboxes = {}

for name in stocks:
    var = tk.IntVar()
    cb = tk.Checkbutton(root, text=name, variable=var)
    cb.pack(anchor='w')
    checkboxes[name] = var

tk.Button(root, text="Predict", command=predict_stock).pack(pady=10)

root.mainloop()
