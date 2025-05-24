import requests
import pandas as pd
import os

# --- Parameters ---
vs_currency = "usd"
days = "90"
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {"vs_currency": vs_currency, "days": days}

# --- Fetch Data ---
response = requests.get(url, params=params)
if response.status_code != 200:
    print(f"Error {response.status_code}: {response.text}")
    exit()

data = response.json()
if 'prices' not in data:
    print("No 'prices' key found in response.")
    exit()

# --- Process Data ---
prices = data['prices']
df = pd.DataFrame(prices, columns=["timestamp", "price"])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df[['datetime', 'price']]
df.set_index('datetime', inplace=True)

# ✅ Save to: Data Extraction/bitcoin_90days_hourly.csv
save_path = os.path.join(os.path.dirname(__file__), "Data Extraction", "bitcoin_90days_hourly.csv")
df.to_csv(save_path)
print(f"✅ Saved {len(df)} rows to {save_path}")
