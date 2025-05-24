import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pytorch_forecasting import NBeats, TimeSeriesDataSet
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Paths ---
DATA_DIR = "../Preprocess Data/Preprocess_csvs"
ARTIFACT_DIR = "/Model Artifacts"
SCALER_PATH = os.path.join(ARTIFACT_DIR, "nbeats_price_scaler.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "nbeats_model_final.ckpt")
NBEATS_PRED_PATH = os.path.join(ARTIFACT_DIR, "nbeats_predictions.csv")

# --- Load datasets ---
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["datetime"])
df_val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"), parse_dates=["datetime"])
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), parse_dates=["datetime"])

# --- Combine datasets ---
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
df_all.sort_values("datetime", inplace=True)
df_all["group_id"] = "BTC"
df_all["time_idx"] = ((df_all["datetime"] - df_all["datetime"].min()) / pd.Timedelta(hours=1)).astype(int)

# --- Apply scaler ---
with open(SCALER_PATH, "rb") as f:
    scaler = joblib.load(f)
df_all["price_scaled"] = scaler.transform(df_all[["price"]])

# --- Dataset config ---
max_encoder_length = 336
max_prediction_length = 24

dataset = TimeSeriesDataSet(
    df_all,
    time_idx="time_idx",
    target="price_scaled",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["price_scaled"],
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=[],
    allow_missing_timesteps=True,
    target_normalizer=None
)

# --- Dataloader ---
# Create prediction dataset
predict_dataset = TimeSeriesDataSet.from_dataset(dataset, df_all, predict=True, stop_randomization=True)
dataloader = predict_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

# --- Load model ---
model = NBeats.load_from_checkpoint(MODEL_PATH)

# --- Predict ---
predictions = model.predict(dataloader, mode="raw", return_x=True)
raw_preds = predictions.output
x_vals = predictions.x

# --- Extract and inverse scale predictions ---
all_forecasts = []
for i in range(len(raw_preds["prediction"])):
    decoder_time_idx = x_vals["decoder_time_idx"][i].detach().cpu().numpy()
    decoder_dates = df_all[df_all["time_idx"].isin(decoder_time_idx)]["datetime"].values

    pred_scaled = raw_preds["prediction"][i].unsqueeze(-1).detach().numpy()
    pred_actual = scaler.inverse_transform(pred_scaled).flatten()

    min_len = min(len(decoder_dates), len(pred_actual))
    all_forecasts.append(pd.DataFrame({
        "datetime": decoder_dates[:min_len],
        "predicted_price": pred_actual[:min_len]
    }))

# --- Combine and save ---
all_predictions_df = pd.concat(all_forecasts, ignore_index=True).drop_duplicates(subset="datetime")
all_predictions_df.to_csv(NBEATS_PRED_PATH, index=False)
print(f"📁 Saved N-BEATS predictions to: {NBEATS_PRED_PATH}")

# --- Merge with actuals ---
actuals_df = df_all[["datetime", "price"]].drop_duplicates()
merged = pd.merge(actuals_df, all_predictions_df, on="datetime", how="inner")

# --- Accuracy metrics ---
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

# --- Plot ---
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
    row_heights=[0.8, 0.2], subplot_titles=["Bitcoin Price Forecast (N-BEATS)", "Prediction Residuals"]
)

fig.add_trace(go.Scatter(x=train_pred["datetime"], y=train_pred["predicted_price"],
                         name="Predicted (Train)", line=dict(color="orange")), row=1, col=1)
fig.add_trace(go.Scatter(x=test_pred["datetime"], y=test_pred["predicted_price"],
                         name="Predicted (Test)", line=dict(color="red")), row=1, col=1)
fig.add_trace(go.Scatter(x=merged["datetime"], y=merged["price"],
                         name="Actual Price", line=dict(color="blue")), row=1, col=1)

fig.add_trace(go.Scatter(x=merged["datetime"], y=merged["residual"],
                         name="Residuals", line=dict(color="gray")), row=2, col=1)

fig.add_vrect(
    x0=test_start, x1=df_test["datetime"].max(),
    fillcolor="lightblue", opacity=0.3,
    layer="below", line_width=0,
    annotation_text="Test Data", annotation_position="top left", row=1, col=1
)

fig.update_layout(
    height=700,
    title="N-BEATS Bitcoin Price Forecast with Residuals",
    xaxis_title="Time",
    yaxis_title="Price (USD)",
    xaxis2_title="Time",
    yaxis2_title="Residual"
)

fig.show()
