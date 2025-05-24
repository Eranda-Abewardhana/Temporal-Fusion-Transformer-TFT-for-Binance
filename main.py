import os

from Data_Visualization.forecast_next_day import fine_tune_and_plot_forecast
from Data_Visualization.predict_tft import predict_tfts
from Model_Training.train_lightgbm import train_lightgbm_residual_model
from Model_Training.train_tft import train_tft_model
from Data_Visualization.hybrid_predictions import hybrid_predictions
import multiprocessing

def main():
    base_dir = os.path.dirname(__file__)

    print("🚀 Step 1: Downloading & saving Bitcoin price data...")
    exec(open(os.path.join(base_dir, "Data Extraction", "download_bitcoin_data.py"), encoding="utf-8").read())

    print("🧹 Step 2: Preprocessing data...")
    exec(open(os.path.join(base_dir, "Preprocess Data", "Load and Preprocess Data.py"), encoding="utf-8").read())

    print("📈 Step 3: Training TFT model...")
    train_tft_model()

    print("📈 Step 3.1: Predict TFT model...")
    predict_tfts()

    print("🌳 Step 4: Training LightGBM model...")
    train_lightgbm_residual_model()

    print("🔁 Step 6: Running hybrid predictions...")
    hybrid_predictions()

    print("📊 Step 7: Visualizing next-day forecast...")
    fine_tune_and_plot_forecast()

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Optional but good for Windows packaging
    main()
