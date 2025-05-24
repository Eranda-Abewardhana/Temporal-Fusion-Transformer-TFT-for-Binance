# 📊 Bitcoin Price Forecasting using Temporal Fusion Transformer (TFT) & LightGBM

This project builds a powerful hybrid forecasting pipeline to predict hourly Bitcoin prices using deep learning and ensemble techniques. It leverages the Temporal Fusion Transformer (TFT) for sequential prediction, combined with LightGBM to correct residual errors — delivering accurate and interpretable financial time series forecasts.

---

## 🚦 Project Pipeline

### 1. 🧱 Data Extraction
- Downloads **hourly BTC price data** over 90 days.
- Stored as `bitcoin_90days_hourly.csv` in the `Data Extraction` folder.
- Script: `get_data.py`

### 2. 🧹 Data Preprocessing
- Converts raw data into `train.csv`, `val.csv`, `test.csv` splits.
- Adds lag features, rolling averages, momentum indicators.
- Scales price using `StandardScaler` for deep learning compatibility.
- Script: `Load and Preprocess Data.ipynb` → outputs to `Preprocess_csvs/`.

### 3. 🔮 Temporal Fusion Transformer (TFT)
- Trains a deep learning model that uses attention and recurrent layers.
- Input includes: time signals (hour, day), technical indicators (RSI, MACD), and engineered features.
- Hyperparameters are tuned using **Optuna** for best validation performance.
- Script: `train_tft.py`

### 4. 🌳 LightGBM Residual Model
- Trained on the **residuals (errors)** between TFT predictions and actual prices.
- Learns to correct model bias, particularly during sudden price shifts.
- Script: `train_lightgbm.py`

### 5. 🔁 Hybrid Forecasting
- Final prediction = `TFT output` + `LightGBM correction`.
- Plots show actual vs predicted with Bollinger Bands and residual analysis.
- Script: `hybrid_predictions.py`

### 6. 🧠 N-BEATS (optional)
- A backup deep learning forecaster for ensembling.
- Script: `train_n-beats.py`

### 7. 📈 Next-Day Forecasting
- Predicts the **next 24 hours of Bitcoin price** using the trained TFT model.
- Visualized in a **TradingView-like candlestick chart** with interactive resolution switches (1H, 6H, 12H, 24H).
- Script: `forecast_next_day.py`

---

## 🛠 How to Run Everything

Use the `main.py` script to run all steps in order:

```bash
python main.py
