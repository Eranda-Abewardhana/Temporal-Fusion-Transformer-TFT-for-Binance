# --- Imports ---
# Save original torch.load
import torch

original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  # Ensure weights_only is False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

def fine_tune_and_plot_forecast():
    import os
    import joblib
    import numpy as np
    import pandas as pd
    from datetime import timedelta
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    import plotly.graph_objects as go

    # --- Paths ---
    DATA_DIR = "Preprocess Data/Preprocess_csvs"
    ARTIFACT_DIR = "Model_Training/Model Artifacts"
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    VAL_CSV = os.path.join(DATA_DIR, "val.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    SCALER_PKL = os.path.join(ARTIFACT_DIR, "price_scaler.pkl")
    MODEL_PATH = os.path.join(ARTIFACT_DIR, "tft_model_final.ckpt")

    # --- Parameters ---
    max_encoder_length = 168
    max_prediction_length = 24
    lag_features = [6, 24, 168]

    # --- Feature Engineering Function ---
    def add_features(df, scaler):
        df["group_id"] = "BTC"
        df["time_idx"] = (df["datetime"] - df["datetime"].min()).dt.total_seconds() // 3600
        df["time_idx"] = df["time_idx"].astype(int)
        df["hour"] = df["datetime"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["static_id"] = "btc"
        df["price_scaled"] = scaler.transform(df[["price"]])
        for lag in lag_features:
            df[f"lag_{lag}"] = df["price_scaled"].shift(lag)
            df[f"rolling_mean_{lag}"] = df["price_scaled"].rolling(lag).mean()
            df[f"rolling_std_{lag}"] = df["price_scaled"].rolling(lag).std()
        df["momentum_24"] = df["price_scaled"] - df["lag_24"]
        df.bfill(inplace=True)
        return df

    # --- Load and prepare training data ---
    df_train = pd.read_csv(TRAIN_CSV, parse_dates=["datetime"])
    df_val = pd.read_csv(VAL_CSV, parse_dates=["datetime"])
    df_full = pd.concat([df_train, df_val], ignore_index=True)

    scaler = joblib.load(SCALER_PKL)
    df_full = add_features(df_full, scaler)

    # --- Dataset config ---
    known_reals = ["time_idx", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    unknown_reals = ["price_scaled"] + [f"lag_{l}" for l in lag_features] + \
                    [f"rolling_mean_{l}" for l in lag_features] + \
                    [f"rolling_std_{l}" for l in lag_features] + ["momentum_24"]

    fine_tune_dataset = TimeSeriesDataSet(
        df_full,
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
    fine_tune_loader = fine_tune_dataset.to_dataloader(train=True, batch_size=32, num_workers=0)

    # --- Load and fine-tune model ---
    model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)
    model.train()

    trainer = Trainer(
        max_epochs=10,
        callbacks=[EarlyStopping(monitor="train_loss", patience=3, min_delta=1e-4, mode="min")],
        gradient_clip_val=0.1,
        enable_model_summary=True
    )
    trainer.fit(model=model, train_dataloaders=fine_tune_loader)

    # --- Load and prepare test data ---
    # --- Load and prepare encoder-decoder test data ---
    df_test_raw = pd.read_csv(TEST_CSV, parse_dates=["datetime"]).sort_values("datetime")

    # ➕ Extract the last 336 hours from test set for encoder
    # Include extra rows for lag features
    required_history = max_encoder_length + max(lag_features)
    encoder_input = df_test_raw.iloc[-required_history:].copy()

    # ➕ Generate 24 hours of future timestamps for decoder (prediction window)
    last_time = encoder_input["datetime"].max()
    forecast_times = [last_time + timedelta(hours=i + 1) for i in range(max_prediction_length)]

    # ➕ Duplicate last known value for future placeholder rows
    last_row = encoder_input.iloc[-1:].copy()
    future_rows = pd.DataFrame([last_row.values[0]] * max_prediction_length, columns=last_row.columns)
    future_rows["datetime"] = forecast_times
    future_rows["price"] = np.nan  # placeholder

    # ➕ Combine encoder + decoder
    df_forecast = pd.concat([encoder_input, future_rows], ignore_index=True)
    df_forecast = add_features(df_forecast, scaler)

    # --- Clean any remaining NaNs or infinite values ---
    df_forecast.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_forecast.dropna(subset=["price_scaled"], inplace=True)

    # OPTIONAL (if still necessary):
    # Forward fill any remaining gaps in lag features if needed:
    df_forecast.fillna(method="bfill", inplace=True)


    ## --- Create prediction dataset ---
    prediction_dataset = TimeSeriesDataSet(
        df_forecast,
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
    dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    # --- Forecast ---
    raw_output, x, index, decoder_lengths, _ = model.predict(
        dataloader,
        mode="raw",
        return_x=True,
        return_index=True
    )

    # --- Get forecast timestamps ---
    forecast_times = [last_time + timedelta(hours=i + 1) for i in range(max_prediction_length)]

    # --- Extract full quantile predictions ---
    raw_preds = raw_output["prediction"][0]  # shape: [24, 7]

    q05 = scaler.inverse_transform(raw_preds[:, 0].detach().numpy().reshape(-1, 1)).flatten()
    q25 = scaler.inverse_transform(raw_preds[:, 1].detach().numpy().reshape(-1, 1)).flatten()
    q50 = scaler.inverse_transform(raw_preds[:, 3].detach().numpy().reshape(-1, 1)).flatten()
    q75 = scaler.inverse_transform(raw_preds[:, 5].detach().numpy().reshape(-1, 1)).flatten()
    q95 = scaler.inverse_transform(raw_preds[:, 6].detach().numpy().reshape(-1, 1)).flatten()

    # --- Construct OHLC structure ---
    # Step 1: Get predicted median (q50) prices
    preds_scaled = raw_output["prediction"][0, :, 3].detach().numpy().reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()

    # Step 2: Construct true OHLC per hour
    # Here: open = previous price, close = current prediction
    ohlc_df = pd.DataFrame({
        "datetime": forecast_times,
        "open": np.insert(preds[:-1], 0, preds[0]),  # shift open by 1
        "close": preds,
    })
    ohlc_df["high"] = np.maximum(ohlc_df["open"], ohlc_df["close"]) + np.random.uniform(5, 20, size=len(ohlc_df))
    ohlc_df["low"] = np.minimum(ohlc_df["open"], ohlc_df["close"]) - np.random.uniform(5, 20, size=len(ohlc_df))


    # --- Forecast DF for Bollinger Bands ---
    forecast_df = pd.DataFrame({
        "datetime": forecast_times,
        "predicted": q50
    }).set_index("datetime")

    rolling_mean = forecast_df["predicted"].rolling(window=5, min_periods=1).mean()
    std_dev = forecast_df["predicted"].std()
    forecast_df["upper_band"] = rolling_mean + 2 * std_dev
    forecast_df["lower_band"] = rolling_mean - 2 * std_dev

    # --- Time Anchors ---
    resample_origin = forecast_df.index[0].replace(minute=0, second=0, microsecond=0)

    # --- Multiple OHLC frames ---
    ohlc_1h = ohlc_df.copy()
    ohlc_df.set_index("datetime", inplace=True)

    ohlc_6h = ohlc_df.resample("6H", origin=resample_origin).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna().reset_index()

    ohlc_12h = ohlc_df.resample("12H", origin=resample_origin).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna().reset_index()

    ohlc_24h = ohlc_df.resample("24H", origin=resample_origin).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna().reset_index()

    # --- Plotly Chart ---
    fig = go.Figure()
    fig.update_layout(template="plotly_dark")

    # ✅ Candlesticks: No custom color => red/green auto-enabled
    fig.add_trace(go.Candlestick(
        x=ohlc_1h["datetime"],
        open=ohlc_1h["open"],
        high=ohlc_1h["high"],
        low=ohlc_1h["low"],
        close=ohlc_1h["close"],
        visible=True,
        name="Candlestick"
    ))
    fig.add_trace(go.Candlestick(
        x=ohlc_6h["datetime"],
        open=ohlc_6h["open"],
        high=ohlc_6h["high"],
        low=ohlc_6h["low"],
        close=ohlc_6h["close"],
        visible=False,
        name="Candlestick"
    ))
    fig.add_trace(go.Candlestick(
        x=ohlc_12h["datetime"],
        open=ohlc_12h["open"],
        high=ohlc_12h["high"],
        low=ohlc_12h["low"],
        close=ohlc_12h["close"],
        visible=False,
        name="Candlestick"
    ))
    fig.add_trace(go.Candlestick(
        x=ohlc_24h["datetime"],
        open=ohlc_24h["open"],
        high=ohlc_24h["high"],
        low=ohlc_24h["low"],
        close=ohlc_24h["close"],
        visible=False,
        name="Candlestick"
    ))

    # --- Bollinger Bands ---
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["upper_band"],
        name="Upper Band",
        line=dict(color="rgba(255,255,255,0.4)", dash="dot"),
        visible=True
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["lower_band"],
        name="Lower Band",
        line=dict(color="rgba(255,255,255,0.4)", dash="dot"),
        visible=True
    ))
    fig.add_trace(go.Scatter(
        x=list(forecast_df.index) + list(forecast_df.index[::-1]),
        y=list(forecast_df["upper_band"]) + list(forecast_df["lower_band"][::-1]),
        fill='toself',
        fillcolor='rgba(255,255,255,0.03)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        visible=True
    ))

    # --- Layout Styling (TradingView-like) ---
    fig.update_layout(
        title=dict(
            text="📈 Next 24H Predictions",
            x=0.01,
            xanchor="left",
            font=dict(color="white", size=18)
        ),
        xaxis=dict(
            title=dict(text="Time", font=dict(color="white", size=12)),
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor="#2a2e39",
            tickfont=dict(color="white"),
            linecolor='white',
            showticklabels=True,
            ticks='outside',
            zeroline=False,
            type="date"
        ),
        yaxis=dict(
            title=dict(text="Price (USD)", font=dict(color="white", size=12)),
            showgrid=True,
            gridcolor="#2a2e39",
            tickfont=dict(color="white"),
            linecolor='white',
            showticklabels=True,
            ticks='outside',
            zeroline=False
        ),
        plot_bgcolor="#0e1116",
        paper_bgcolor="#0e1116",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=40),
        hovermode='x unified',
        updatemenus=[{
            "buttons": [
                {"label": "1H", "method": "update",
                 "args": [{"visible": [True, False, False, False, True, True, True]},
                          {"title": "📈 Next 24H Predictions"}]},
                {"label": "6H", "method": "update",
                 "args": [{"visible": [False, True, False, False, True, True, True]},
                          {"title": "📈 Next 24H Predictions"}]},
                {"label": "12H", "method": "update",
                 "args": [{"visible": [False, False, True, False, True, True, True]},
                          {"title": "📈 Next 24H Predictions"}]},
                {"label": "24H", "method": "update",
                 "args": [{"visible": [False, False, False, True, True, True, True]},
                          {"title": "📈 Next 24H Predictions"}]},
            ],
            "direction": "down",
            "x": 0.01,
            "xanchor": "left",
            "y": 1.1,
            "yanchor": "top",
            "bgcolor": "#1c1e26",
            "font": dict(color="white"),
            "showactive": True
        }]
    )

    fig.show()
# if __name__ == "__main__":
#     import multiprocessing
#     multiprocessing.freeze_support()  # Optional but helpful on Windows
#     fine_tune_and_plot_forecast()