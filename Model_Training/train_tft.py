import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, EncoderNormalizer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from lightning.pytorch import Trainer
import torch
import multiprocessing
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import lightning.pytorch.tuner.lr_finder as lr_finder_module
import lightning.fabric.utilities.cloud_io as cloud_io


# Dynamically fetch NumPy's internal scalar and dtype classes
scalar = getattr(__import__("numpy.core.multiarray", fromlist=["scalar"]), "scalar")
_reconstruct = getattr(__import__("numpy.core.multiarray", fromlist=["_reconstruct"]), "_reconstruct")
dtype = np.dtype
float32_dtype_class = np.dtype("float32").__class__  # Fixes Float32DType issue
int32_dtype_class = np.dtype("int32").__class__
ndarray = getattr(__import__("numpy", fromlist=["ndarray"]), "ndarray")

# Register all required classes
torch.serialization.add_safe_globals([
    EncoderNormalizer,
    StandardScaler,
    np.generic,
    np.float32,
    np.float64,
    np.int64,
    np.int32,
    scalar,
    dtype,
    float32_dtype_class,
    int32_dtype_class,
    NaNLabelEncoder,
    _reconstruct,
    ndarray,
    np.dtype("str"),       # 👈 key fix: needed to allow np.str_ and np.dtypes.StrDType
    np.str_,
])

torch.set_float32_matmul_precision("medium")
torch.set_num_threads(min(4, multiprocessing.cpu_count() // 2))
os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())
os.environ["OPENBLAS_NUM_THREADS"] = str(torch.get_num_threads())
os.environ["MKL_NUM_THREADS"] = str(torch.get_num_threads())
os.environ["VECLIB_MAXIMUM_THREADS"] = str(torch.get_num_threads())
os.environ["NUMEXPR_NUM_THREADS"] = str(torch.get_num_threads())

def reduce_memory(df):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    return df

def train_tft_model():
    # Paths
    DATA_DIR = "Preprocess Data/Preprocess_csvs"
    ARTIFACT_DIR = "Model_Training/Model Artifacts"
    TUNING_DIR = "Model_Training/Optuna_Tuning"
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(TUNING_DIR, exist_ok=True)

    # Load data
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["datetime"])
    df_val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"), parse_dates=["datetime"])

    max_encoder_length = 336
    max_prediction_length = 24
    lag_features = [6, 24, 168]

    # Recombine just before dataset creation
    # Combine
    combined = pd.concat([df_train, df_val]).copy()

    # # Set parameters
    # required_val_rows = max_encoder_length + max(lag_features) + max_prediction_length  # 528
    #
    # # Validate combined length
    # if len(combined) < required_val_rows + 500:
    #     raise ValueError(f"Not enough data. Need at least {required_val_rows + 500} rows, got {len(combined)}.")
    #
    # # Split
    # df_val = combined.iloc[-(required_val_rows + 24):].copy()  # 24 extra buffer
    # df_train = combined.iloc[:-(required_val_rows + 24)].copy()
    #
    # # Fix time_idx cutoff for prediction mode
    # min_t_idx = df_val["time_idx"].max() - max_encoder_length - max_prediction_length + 1
    # df_val = df_val[df_val["time_idx"] >= min_t_idx].copy()
    #
    # print(f"✅ Training rows: {len(df_train)}, Validation rows: {len(df_val)}")
    # print("Filtered df_val time_idx range:", df_val["time_idx"].min(), "to", df_val["time_idx"].max())

    split_point = int(len(combined) * 0.8)
    df_train = combined.iloc[:split_point].copy()
    df_val = combined.iloc[split_point:].copy()

    # Feature engineering
    for df in [df_train, df_val]:
        df.sort_values("datetime", inplace=True)
        df["group_id"] = "BTC"
        df["time_idx"] = ((df["datetime"] - df["datetime"].min()).dt.total_seconds() // 3600).astype(int)
        df["hour"] = df["datetime"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["static_id"] = "btc"

    # Scale target
    scaler = StandardScaler()
    df_train["price_scaled"] = scaler.fit_transform(df_train[["price"]])
    df_val["price_scaled"] = scaler.transform(df_val[["price"]])
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "price_scaler.pkl"))

    # Lag, momentum, and rolling features (reduced)
    lag_features = [6, 24, 168]
    # lag_features = [6, 24]
    for df in [df_train, df_val]:
        for lag in lag_features:
            df[f"lag_{lag}"] = df["price_scaled"].shift(lag)
            df[f"rolling_mean_{lag}"] = df["price_scaled"].rolling(lag).mean()
            df[f"rolling_std_{lag}"] = df["price_scaled"].rolling(lag).std()
        df["momentum_24"] = df["price_scaled"] - df["lag_24"]
        df.bfill(inplace=True)

    # Downcast to float32
    df_train = reduce_memory(df_train)
    df_val = reduce_memory(df_val)

    # Dataset config
    max_encoder_length = 336
    # max_encoder_length = 168
    max_prediction_length = 24

    lag_cols = [f"lag_{l}" for l in lag_features]
    roll_cols = [f"rolling_mean_{l}" for l in lag_features] + [f"rolling_std_{l}" for l in lag_features]
    known_reals = ["time_idx", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    unknown_reals = ["price_scaled"] + lag_cols + roll_cols + ["momentum_24"]

    training = TimeSeriesDataSet(
        df_train,
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

    validation = TimeSeriesDataSet.from_dataset(training, df_val, predict=True, stop_randomization=True)

    batch_size = 64
    num_workers = min(4, multiprocessing.cpu_count() // 2)
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)

    def safe_load_patch(path, map_location=None):
        return torch.load(path, map_location=map_location, weights_only=False)

    lr_finder_module._load = safe_load_patch

    # ✅ Patch torch.load to force weights_only=False
    original_load_fn = torch.load

    def safe_load(path_or_file, map_location=None, *args, **kwargs):
        kwargs["weights_only"] = False
        return original_load_fn(path_or_file, map_location=map_location, **kwargs)

    # 🩹 Apply patch
    cloud_io._load = safe_load
    lr_finder_module._load = safe_load
    # Hyperparameter tuning (study object returned)
    study = optimize_hyperparameters(
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        model_path=TUNING_DIR,
        n_trials=10,
        max_epochs=20,
        timeout=1800,
        gradient_clip_val_range=(0.05, 0.5),
        hidden_size_range=(256, 384),
        attention_head_size_range=(2, 4),
        dropout_range=(0.1, 0.25),
        learning_rate_range=(1e-4, 5e-3),
        reduce_on_plateau_patience=3,
        use_learning_rate_finder=False,
        trainer_kwargs=dict(
            limit_train_batches=20,
            max_epochs=20,
            enable_progress_bar=False
        )
    )

    # Locate best checkpoint manually
    # best_trial_number = study.best_trial.number
    # trial_dir = os.path.join(TUNING_DIR, f"trial_{best_trial_number}", "checkpoints")
    # best_model_path = [os.path.join(trial_dir, f) for f in os.listdir(trial_dir) if f.endswith(".ckpt")][0]
    #
    # # Load best model from checkpoint
    # best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # Reconstruct best model from best trial params
    best_trial_params = study.best_trial.params

    best_model = TemporalFusionTransformer.from_dataset(
        training,
        loss=QuantileLoss(),
        learning_rate=best_trial_params["learning_rate"],
        hidden_size=best_trial_params["hidden_size"],
        hidden_continuous_size=best_trial_params["hidden_continuous_size"],
        attention_head_size=best_trial_params["attention_head_size"],
        dropout=best_trial_params["dropout"]
    )

    # Final training
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-4, mode="min"),
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            dirpath=ARTIFACT_DIR,
            filename="tft_best_model",
            monitor="val_loss",
            save_top_k=1,
            mode="min"
        )
    ]

    trainer = Trainer(
        max_epochs=100,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        limit_val_batches=1.0,
        enable_model_summary=True,
        log_every_n_steps=5,
        enable_progress_bar=False,
        accelerator="cpu"
    )

    trainer.fit(best_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.save_checkpoint(os.path.join(ARTIFACT_DIR, "tft_model_final.ckpt"))

    print("✅ Training complete with optimized performance.")

# if __name__ == "__main__":
#     import multiprocessing
#     multiprocessing.freeze_support()  # Optional but helpful on Windows
#     train_tft_model()


