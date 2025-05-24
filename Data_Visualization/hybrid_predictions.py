import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

def hybrid_predictions():
    # --- Paths ---
    DATA_DIR = "Preprocess Data/Preprocess_csvs"
    ARTIFACT_DIR = "Model_Training/Model Artifacts"
    TFT_PRED_PATH = os.path.join(ARTIFACT_DIR, "tft_predictions.csv")
    RESIDUAL_MODEL_PATH = os.path.join(ARTIFACT_DIR, "lgb_residual_model.pkl")
    FINAL_PRED_PATH = os.path.join(ARTIFACT_DIR, "final_predictions_combined.csv")

    # --- Load datasets ---
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["datetime"])
    df_val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"), parse_dates=["datetime"])
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), parse_dates=["datetime"])
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # --- Feature Engineering (same as residual model) ---
    df_all.sort_values("datetime", inplace=True)
    df_all["time_idx"] = ((df_all["datetime"] - df_all["datetime"].min()).dt.total_seconds() // 3600).astype(int)
    df_all["hour"] = df_all["datetime"].dt.hour
    df_all["hour_sin"] = np.sin(2 * np.pi * df_all["hour"] / 24)
    df_all["hour_cos"] = np.cos(2 * np.pi * df_all["hour"] / 24)
    df_all["day_of_week"] = df_all["datetime"].dt.dayofweek
    df_all["dow_sin"] = np.sin(2 * np.pi * df_all["day_of_week"] / 7)
    df_all["dow_cos"] = np.cos(2 * np.pi * df_all["day_of_week"] / 7)

    df_all["price_scaled"] = StandardScaler().fit_transform(df_all[["price"]])

    # Lag and momentum
    for lag in [6, 24, 168]:
        df_all[f"lag_{lag}"] = df_all["price_scaled"].shift(lag)
        df_all[f"rolling_mean_{lag}"] = df_all["price_scaled"].rolling(lag).mean()
        df_all[f"rolling_std_{lag}"] = df_all["price_scaled"].rolling(lag).std()

    df_all["momentum_24"] = df_all["price_scaled"] - df_all["lag_24"]
    df_all.bfill(inplace=True)

    # --- Load TFT predictions ---
    tft_df = pd.read_csv(TFT_PRED_PATH, parse_dates=["datetime"])
    df_all = df_all.merge(tft_df, on="datetime", how="left")

    # --- Predict residuals ---
    model = joblib.load(RESIDUAL_MODEL_PATH)

    features = [col for col in df_all.columns if col not in ["datetime", "price", "price_scaled", "group_id", "residual", "tft_pred"]]
    df_all["residual_pred"] = model.predict(df_all[features])

    # --- Final prediction = TFT + Residual ---
    df_all["price_final"] = df_all["tft_pred"] + df_all["residual_pred"]

    # --- Save final predictions ---
    df_all[["datetime", "price", "tft_pred", "residual_pred", "price_final"]].to_csv(FINAL_PRED_PATH, index=False)
    print(f"✅ Saved final combined predictions to: {FINAL_PRED_PATH}")

    # --- Evaluation ---
    merged = df_all[["datetime", "price", "price_final"]].dropna()
    mae = mean_absolute_error(merged["price"], merged["price_final"])
    mse = mean_squared_error(merged["price"], merged["price_final"])
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((merged["price"] - merged["price_final"]) / merged["price"])) * 100

    print("\n📊 Combined Model Accuracy:")
    print(f"✅ MAE  : {mae:.4f}")
    print(f"✅ MSE  : {mse:.4f}")
    print(f"✅ RMSE : {rmse:.4f}")
    print(f"✅ MAPE : {mape:.2f}%")

    # --- Plot ---
    test_start = df_test["datetime"].min()
    train_pred = merged[merged["datetime"] < test_start]
    test_pred = merged[merged["datetime"] >= test_start]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[0.8, 0.2], subplot_titles=["Final Forecast: TFT + LightGBM Residual", "Residuals"]
    )

    fig.add_trace(go.Scatter(x=train_pred["datetime"], y=train_pred["price_final"],
                             name="Predicted (Train)", line=dict(color="green")), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_pred["datetime"], y=test_pred["price_final"],
                             name="Predicted (Test)", line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=merged["datetime"], y=merged["price"],
                             name="Actual Price", line=dict(color="blue")), row=1, col=1)

    fig.add_trace(go.Scatter(x=merged["datetime"],
                             y=merged["price"] - merged["price_final"],
                             name="Residuals", line=dict(color="gray")), row=2, col=1)

    fig.add_vrect(
        x0=test_start, x1=df_test["datetime"].max(),
        fillcolor="lightblue", opacity=0.3,
        layer="below", line_width=0,
        annotation_text="Test Region", annotation_position="top left", row=1, col=1
    )

    fig.update_layout(
        height=700,
        title="Bitcoin Price Forecast with Residual Correction (TFT + LightGBM)",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        xaxis2_title="Time",
        yaxis2_title="Residual"
    )

    fig.show()
