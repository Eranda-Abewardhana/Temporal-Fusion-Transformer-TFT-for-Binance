import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pytorch_forecasting import EncoderNormalizer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

def train_lightgbm_residual_model():
    DATA_DIR = "Preprocess Data/Preprocess_csvs"
    ARTIFACT_DIR = "Model_Training/Model Artifacts"
    TFT_PRED_PATH = os.path.join(ARTIFACT_DIR, "tft_predictions.csv")  # ⬅️ From previous TFT model
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["datetime"])
    df_val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"), parse_dates=["datetime"])

    # --- Combine and engineer ---
    df = pd.concat([df_train, df_val])
    df["time_idx"] = ((df["datetime"] - df["datetime"].min()).dt.total_seconds() // 3600).astype(int)
    df["hour"] = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # --- Lag & rolling ---
    df.sort_values("datetime", inplace=True)
    df["price_scaled"] = StandardScaler().fit_transform(df[["price"]])

    for lag in [6, 24, 168]:
        df[f"lag_{lag}"] = df["price_scaled"].shift(lag)
        df[f"rolling_mean_{lag}"] = df["price_scaled"].rolling(lag).mean()
        df[f"rolling_std_{lag}"] = df["price_scaled"].rolling(lag).std()

    df["momentum_24"] = df["price_scaled"] - df["lag_24"]
    df.bfill(inplace=True)

    # --- Load TFT predictions ---
    tft_df = pd.read_csv(TFT_PRED_PATH, parse_dates=["datetime"])
    df = df.merge(tft_df[["datetime", "tft_pred"]], on="datetime", how="left")

    # --- Compute residual ---
    df["residual"] = df["price"] - df["tft_pred"]

    # --- Train/val split ---
    split_point = int(len(df) * 0.8)
    df_train = df.iloc[:split_point].copy()
    df_val = df.iloc[split_point:].copy()

    features = [col for col in df.columns if col not in ["datetime", "price", "price_scaled", "group_id", "residual", "tft_pred"]]
    target = "residual"

    X_train = df_train[features]
    y_train = df_train[target]
    X_val = df_val[features]
    y_val = df_val[target]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.005,
        "verbosity": 1
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(period=100)
        ]
    )

    joblib.dump(model, os.path.join(ARTIFACT_DIR, "lgb_residual_model.pkl"))

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"✅ Residual LightGBM RMSE: {rmse:.4f}")


