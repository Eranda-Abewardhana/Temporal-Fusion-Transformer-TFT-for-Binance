import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Paths ---
DATA_DIR = "../Preprocess Data/Preprocess_csvs"
ARTIFACT_DIR = "../Model_Training/Model Artifacts"
SCALER_PATH = os.path.join(ARTIFACT_DIR, "price_scaler.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "tft_model_final.ckpt")
TFT_PRED_PATH = os.path.join(ARTIFACT_DIR, "tft_predictions.csv")

# --- Load all datasets ---
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["datetime"])
df_val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"), parse_dates=["datetime"])
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), parse_dates=["datetime"])

# --- Combine datasets ---
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

# --- Feature engineering ---
df_all.sort_values("datetime", inplace=True)
df_all["group_id"] = "BTC"
df_all["time_idx"] = (df_all["datetime"] - df_all["datetime"].min()).dt.total_seconds() // 3600
df_all["time_idx"] = df_all["time_idx"].astype(int)
df_all["hour"] = df_all["datetime"].dt.hour
df_all["hour_sin"] = np.sin(2 * np.pi * df_all["hour"] / 24)
df_all["hour_cos"] = np.cos(2 * np.pi * df_all["hour"] / 24)
df_all["day_of_week"] = df_all["datetime"].dt.dayofweek
df_all["dow_sin"] = np.sin(2 * np.pi * df_all["day_of_week"] / 7)
df_all["dow_cos"] = np.cos(2 * np.pi * df_all["day_of_week"] / 7)
df_all["static_id"] = "btc"

# --- Load and apply scaler ---
with open(SCALER_PATH, 'rb') as f:
    scaler = joblib.load(f)
df_all["price_scaled"] = scaler.transform(df_all[["price"]])

# --- Add lag, rolling, and momentum features ---
lag_features = [1, 6, 12, 24, 48, 168]
for lag in lag_features:
    df_all[f"lag_{lag}"] = df_all["price_scaled"].shift(lag)
    df_all[f"rolling_mean_{lag}"] = df_all["price_scaled"].rolling(lag).mean()
    if lag > 1:
        df_all[f"rolling_std_{lag}"] = df_all["price_scaled"].rolling(lag).std()
df_all["momentum_24"] = df_all["price_scaled"] - df_all["lag_24"]
df_all.bfill(inplace=True)

# --- Dataset config ---
max_encoder_length = 336
max_prediction_length = 24

lag_cols = [f"lag_{l}" for l in lag_features]
roll_cols = [f"rolling_mean_{l}" for l in lag_features] + [f"rolling_std_{l}" for l in lag_features if l > 1]
known_reals = ["time_idx", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
unknown_reals = ["price_scaled"] + lag_cols + roll_cols + ["momentum_24"]

full_dataset = TimeSeriesDataSet(
    df_all,
    time_idx="time_idx",
    target="price_scaled",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=known_reals,
    time_varying_unknown_reals=unknown_reals,
    static_categoricals=["static_id"],
    allow_missing_timesteps=True
)

full_loader = full_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

# --- Load model ---
model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)

# --- Predict ---
pred_output = model.predict(full_loader, mode="raw", return_x=True)
raw_predictions = pred_output.output
x = pred_output.x

# --- Extract predictions ---
all_forecasts = []
for i in range(len(raw_predictions["prediction"])):
    decoder_time_idx = x["decoder_time_idx"][i].detach().cpu().numpy()
    decoder_dates = df_all[df_all["time_idx"].isin(decoder_time_idx)]["datetime"].values

    pred_scaled = raw_predictions["prediction"][i, :, 3].unsqueeze(-1).detach().numpy()
    pred_actual = scaler.inverse_transform(pred_scaled).flatten()

    min_len = min(len(decoder_dates), len(pred_actual))
    all_forecasts.append(pd.DataFrame({
        "datetime": decoder_dates[:min_len],
        "predicted_price": pred_actual[:min_len]
    }))

# --- Combine forecasts ---
all_predictions_df = pd.concat(all_forecasts, ignore_index=True)
all_predictions_df = all_predictions_df.drop_duplicates(subset="datetime")

# --- Save TFT predictions (for residual model) ---
tft_save_df = all_predictions_df.copy()
tft_save_df.rename(columns={"predicted_price": "tft_pred"}, inplace=True)
tft_save_df.to_csv(TFT_PRED_PATH, index=False)
print(f"📁 Saved TFT predictions to: {TFT_PRED_PATH}")

# --- Merge with actuals ---
actuals_df = df_all[["datetime", "price"]].drop_duplicates()
merged = pd.merge(actuals_df, all_predictions_df, on="datetime", how="inner")

# --- Accuracy Metrics ---
mae = mean_absolute_error(merged["price"], merged["predicted_price"])
mse = mean_squared_error(merged["price"], merged["predicted_price"])
rmse = np.sqrt(mse)
mape = np.mean(np.abs((merged["price"] - merged["predicted_price"]) / merged["price"])) * 100

print("📊 Forecast Accuracy on Entire Dataset:")
print(f"✅ MAE  : {mae:.4f}")
print(f"✅ MSE  : {mse:.4f}")
print(f"✅ RMSE : {rmse:.4f}")
print(f"✅ MAPE : {mape:.2f}%")

# --- Residuals ---
merged["residual"] = merged["price"] - merged["predicted_price"]

# --- Split train/test predictions ---
test_start = df_test["datetime"].min()
train_pred = merged[merged["datetime"] < test_start]
test_pred = merged[merged["datetime"] >= test_start]

# --- Plot: Forecast + Residual ---
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
    row_heights=[0.8, 0.2], subplot_titles=["Bitcoin Price Forecast", "Prediction Residuals"]
)

# Forecast plot
fig.add_trace(go.Scatter(x=train_pred["datetime"], y=train_pred["predicted_price"],
                         name="Predicted (Train)", line=dict(color="orange")), row=1, col=1)
fig.add_trace(go.Scatter(x=test_pred["datetime"], y=test_pred["predicted_price"],
                         name="Predicted (Test)", line=dict(color="red")), row=1, col=1)
fig.add_trace(go.Scatter(x=merged["datetime"], y=merged["price"],
                         name="Actual Price", line=dict(color="blue")), row=1, col=1)

# Residuals plot
fig.add_trace(go.Scatter(x=merged["datetime"], y=merged["residual"],
                         name="Residuals", line=dict(color="gray")), row=2, col=1)

# Highlight test region
fig.add_vrect(
    x0=test_start, x1=df_test["datetime"].max(),
    fillcolor="lightblue", opacity=0.3,
    layer="below", line_width=0,
    annotation_text="Test Data", annotation_position="top left", row=1, col=1
)

fig.update_layout(
    height=700,
    title="Bitcoin Price Forecast with Test Region and Residuals",
    xaxis_title="Time",
    yaxis_title="Price (USD)",
    xaxis2_title="Time",
    yaxis2_title="Residual"
)

fig.show()
