#%% raw
# Load and Preprocess Data
#%%
import pandas as pd
import tft
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Data Extraction/bitcoin_90days_hourly.csv", parse_dates=["datetime"])
df["group_id"] = "BTC"  # group_id required by TFT even for univariate
df["time_idx"] = (df["datetime"] - df["datetime"].min()).dt.total_seconds() // 3600
df["time_idx"] = df["time_idx"].astype(int)

# Optional: scale the price
scaler = StandardScaler()
df["price_scaled"] = scaler.fit_transform(df[["price"]])

df.head()

#%% md
# Check for outliers using Plotly
#%%
import pandas as pd
import plotly.express as px
import os

# --- Sample data ---
# If you're not reading from file yet:
# df = pd.read_csv("your_file.csv", parse_dates=['datetime'])

# --- Plotly boxplot for outlier detection ---
fig = px.box(df, y="price", title="Outlier Detection in Price using Plotly")
fig.show()

# --- Sort and split ---
df = df.sort_values("time_idx")
total_length = len(df)
train_end = int(0.8 * total_length)
val_end = int(0.9 * total_length)

df_train = df.iloc[:train_end]
df_val = df.iloc[train_end:val_end]
df_test = df.iloc[val_end:]
#%%
def remove_outliers_iqr(df, column):
    """
    Removes outliers from the specified column using the IQR method.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers

    Returns:
        pd.DataFrame: Cleaned DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return cleaned_df
#%% raw
# Convert to PyTorch Forecasting Dataset
#%%
# ✅ Use relative path — creates directory in current working dir
DATASET_DIR = "Preprocess Data/Preprocess_csvs"
os.makedirs(DATASET_DIR, exist_ok=True)

# --- Remove outliers from 'price' ---
df = remove_outliers_iqr(df, 'price')
# --- Sort and split ---
df = df.sort_values("time_idx")
total_length = len(df)
train_end = int(0.8 * total_length)
val_end = int(0.9 * total_length)

df_train = df.iloc[:train_end]
df_val = df.iloc[train_end:val_end]
df_test = df.iloc[val_end:]

# --- Save to CSV ---
df_train.to_csv(os.path.join(DATASET_DIR, "train.csv"), index=False)
df_val.to_csv(os.path.join(DATASET_DIR, "val.csv"), index=False)
df_test.to_csv(os.path.join(DATASET_DIR, "test.csv"), index=False)

print("✅ Datasets saved in './Preprocess_csvs/'")
#%%
