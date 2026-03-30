# Ai-stock-trend-analyzer

## 📌 Overview

This project is a multi-stock prediction system that uses **LSTM (Long Short-Term Memory)** neural networks to analyze historical stock data and predict future price trends. The system allows users to select multiple stocks, visualize predictions, and evaluate model accuracy.

---

## 🎯 Problem Statement

Investors and learners often lack simple tools to understand stock price behavior using data-driven methods. This project aims to provide an AI-based solution to analyze historical stock data and generate predictions for better decision-making.

---

## 🚀 Features

* 📊 Multi-stock selection (Apple, Tesla, NVIDIA, Gold, Crypto, etc.)
* 🤖 Deep Learning model (LSTM)
* 📈 Real vs Predicted graph visualization
* 📉 Accuracy metrics (RMSE & Accuracy %)
* 💾 Pre-trained models (no retraining required)
* 🖥️ Simple UI using Tkinter

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy & Pandas
* Matplotlib
* Scikit-learn
* yFinance API
* Tkinter (GUI)

---

## 📂 Project Structure

```
ai-stock-trend-analyzer/
│── train.py          # Train and save models
│── app.py            # UI and prediction
│── models/           # Saved trained models (.h5)
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/sunnyyadav54/ai-stock-trend-analyzer.git
cd ai-stock-trend-analyzer
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1: Train models (only once)

```
python train.py
```

### Step 2: Run application

```
python app.py
```
<img width="1920" height="1080" alt="Screenshot (207)" src="https://github.com/user-attachments/assets/66edcb18-6151-4b5b-8b15-03b3db12db20" />

---

## 📊 How it Works

1. Fetches historical stock data using yFinance
2. Preprocesses data (scaling + sequence creation)
3. Uses LSTM neural network for prediction
4. Compares predicted vs actual values
5. Displays results with graphs and accuracy

---

## 📈 Evaluation Metrics

* **RMSE (Root Mean Squared Error)**
* **Accuracy (%)** based on MAPE
<img width="1845" height="905" alt="Screenshot 2026-03-29 160929" src="https://github.com/user-attachments/assets/b233ca43-20d8-42d1-aa82-7793d1ec58aa" />

---

## ⚠️ Limitations

* Stock market is highly unpredictable
* Model depends only on historical data
* Cannot account for real-world events (news, policies, etc.)

---

## 📚 Future Improvements

* Add real-time prediction
* Improve UI (web-based using Streamlit)
* Include technical indicators (RSI, MACD)
* Save scaler for better consistency

---

## 📌 Conclusion

This project demonstrates how machine learning, specifically LSTM networks, can be used to identify patterns in time-series financial data and provide useful insights into stock price trends.

---

## 👨‍💻 Author

- SUNNY KUMAR YADAV
- REG. NO. - 25BCE11132
- YEAR - 1st year
- B.Tech CSE(core)

---
