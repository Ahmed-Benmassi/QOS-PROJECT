import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Step 1: Load the CSV and clean columns
df = pd.read_csv("bandwith_150.171.27.11.csv", skiprows=3)
df.columns = df.columns.str.strip()  # remove extra spaces

# Debugging: Show column names
print("✅ Cleaned Columns:", df.columns.tolist())

# Step 2: Check required columns
required_cols = ['_time', '_value', 'target']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Step 3: Filter for target IP
target_ip = '150.171.27.11'
df = df[df['target'] == target_ip]

# Step 4: Rename and convert columns
df = df.rename(columns={'_time': 'ds', '_value': 'y'})
df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
df['y'] = df['y'].astype(float)

# Optional: Preview the cleaned DataFrame
print("\n📊 Time Series Preview:")
print(df[['ds', 'y']].head())

# Step 5: Prophet modeling
model = Prophet()
model.fit(df[['ds', 'y']])

# Step 6: Make future forecast
future = model.make_future_dataframe(periods=1, freq="min")
forecast = model.predict(future)

# Step 7: Plot
print("\n📈 Plotting forecast...")
fig = model.plot(forecast)
plt.title("QoS Forecast (Bandwidth)")
plt.xlabel("Time")
plt.ylabel("bandwidth_mbps")
plt.tight_layout()
plt.show()

# Step 8: Alert on spikes
threshold = 0.15
print("\n🚨 Alerts for predicted spikes:")
for i, row in forecast.tail(6).iterrows():
    if row['yhat'] > threshold:
        print(f"[!] {row['ds']} → Predicted: {row['yhat']:.2f} Mbps")
    else :
       print(f"✅ No spike in prediction {i+1} → Predicted: {row['yhat']:.2f} Mbps")
     


