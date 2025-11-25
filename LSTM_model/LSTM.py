import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR

from datetime import datetime , timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from influxdb_client import InfluxDBClient ,Point, WritePrecision, WriteOptions
from sklearn.preprocessing import MinMaxScaler
from data_preprocessor import load_and_preprocess_data
from dotenv import load_dotenv
import os 
import time 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from influxdb_client.rest import ApiException
from influxdb_client import QueryApi

# Load environment variables
load_dotenv(dotenv_path=r"C:\Users\HP\Documents\qos project\LSTM_model\predicted.env")

INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api()

df= load_and_preprocess_data()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['bandwidth_mbps']])

def create_sequences(data, seq_length=10, target_feature=0):
    x, y = [], []
    for i in range(len(data)-seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])
    return np.array(x), np.array(y)

seq_length = 10
x, y = create_sequences(scaled_data, seq_length)

train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(Input(shape=(seq_length, x.shape[2])))
model.add(LSTM(50, activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")


early_stop = EarlyStopping(monitor='val_loss', patience=50,restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=80, batch_size=32, validation_split=0.2, callbacks=[early_stop],verbose=1)

# --- Predict ---
y_pred = model.predict(x_test)

# --- Inverse scale the predictions ---
y_test_scaled = y_test.reshape(-1, 1)  
y_pred_scaled = y_pred.reshape(-1, 1) 

# For single feature, inverse transform directly
bandwidth_pred = scaler.inverse_transform(y_pred_scaled).flatten()
bandwidth_actual = scaler.inverse_transform(y_test_scaled).flatten()

# --- Visualization ---
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

time_axis = range(seq_length, seq_length + len(bandwidth_pred))

plt.plot(bandwidth_actual, label="Actual Bandwidth", linewidth=2, color="#1f77b4")
plt.plot(bandwidth_pred, label="Predicted Bandwidth", linewidth=2, color="#ff7f0e", linestyle="--")
plt.title("Bandwidth Prediction (LSTM) - Raw Data", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Time Steps", fontsize=13)
plt.ylabel("Bandwidth (Mbps)", fontsize=13)
plt.legend(fontsize=12, loc="upper right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.text(len(bandwidth_pred) * 0.05, max(bandwidth_actual) * 0.9,
         f"Seq length: {seq_length}\nPatience: {early_stop.patience}\nEpochs run: {len(history.history['loss'])}",
         bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

# Loss plot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(history.history['loss'], label='Training Loss', color='royalblue')
ax.plot(history.history['val_loss'], label='Validation Loss', color='darkorange')
ax.set_title("LSTM Training vs Validation Loss", fontsize=14)
ax.set_xlabel("Epochs")
ax.set_ylabel("MSE Loss")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# --- Write predictions to InfluxDB with proper timestamps ---
def write_to_influx(bandwidth, timestamp):
    try:
        point = (
            Point("network_predicted")
            .field("bandwidth_mbps", float(bandwidth))
            .time(timestamp, WritePrecision.S)
        )
        write_api.write(bucket=INFLUXDB_BUCKET, record=point)
        print(f"✅ Data written: {timestamp}, bandwidth={bandwidth:.2f}Mbps")
        return True
    except Exception as e:
        print(f"❌ Error writing data: {e}")
        return False

try:
    health = client.health()
    print(f"🔌 InfluxDB Health: {health.status}")
except Exception as e:
    print(f"❌ Cannot connect to InfluxDB: {e}")

# Write predictions with proper timestamps
base_time = datetime.utcnow()
success_count = 0

for i, bandwidth in enumerate(bandwidth_pred):
    timestamp = base_time + timedelta(seconds=i)
    if write_to_influx(bandwidth, timestamp):  # Remove target parameter
        success_count += 1

print(f"📊 Successfully wrote {success_count}/{len(bandwidth_pred)} predictions")

# Force flush and close
write_api.flush()
time.sleep(2)  # Give time for writes to complete
# --- ADD THIS: Write actual bandwidth data to InfluxDB ---
print("📤 Writing ACTUAL bandwidth data to InfluxDB...")
actual_success_count = 0

for i, bandwidth in enumerate(bandwidth_actual):
    timestamp = base_time + timedelta(seconds=i)
   
    point = (
        Point("network_metrics")  # Different measurement for actual data
        .field("bandwidth_mbps", float(bandwidth))
        .time(timestamp, WritePrecision.S)
    )
    write_api.write(bucket=INFLUXDB_BUCKET, record=point)
    actual_success_count += 1

print(f"📊 Successfully wrote {actual_success_count}/{len(bandwidth_actual)} ACTUAL bandwidth points")

# Force flush again to ensure actual data is written
write_api.flush()
time.sleep(2)

# --- Verify the data was written ---
from influxdb_client import QueryApi

def verify_predictions():
    query_api = client.query_api()
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -1h)
      |> filter(fn: (r) => r._measurement == "network_predicted")
    '''
    
    try:
        tables = query_api.query(query)
        count = 0
        for table in tables:
            for record in table.records:
                print(f"📊 Found: {record.get_time()} | {record.get_field()} = {record.get_value()}")
                count += 1
        print(f"✅ Total records found in InfluxDB: {count}")
        return count
    except Exception as e:
        print(f"❌ Query error: {e}")
        return 0

# Run verification
verify_predictions()

print(" Predictions written to InfluxDB successfully.")