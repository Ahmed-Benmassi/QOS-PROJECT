import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR

from datetime import datetime , timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from influxdb_client import InfluxDBClient ,Point, WritePrecision
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os 
import time 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

# Load environment variables
load_dotenv()

INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")


client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api()

# Step 1: Load the CSV and clean columns
df = pd.read_csv(r"C:\Users\HP\Documents\qos project\qos-data.csv", skiprows=3)
df.columns = df.columns.str.strip()  # remove extra spaces


# Pivot the data so each target has its own 3 metrics as columns
pivot_df = df.pivot_table(index=["_time", "target"], columns="_field", values="_value").reset_index()

# Ensure sorted by time
pivot_df = pivot_df.sort_values(by=["target", "_time"])

# --- Step 3: Convert _time to datetime and set as index ---
pivot_df["_time"] = pd.to_datetime(pivot_df["_time"])
pivot_df.set_index("_time", inplace=True)

# --- Step 4: Select only numeric columns for scaling ---
numeric_cols = pivot_df.select_dtypes(include=np.number).columns
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(pivot_df[numeric_cols])


def create_sequences(data, seq_length =10 ,target_col='bandwidth_mbps' ):
    x, y = [] ,[]
    for i in range(len(data)-seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])
    return np.array(x) ,np.array(y)

seq_length= 10
x,y =create_sequences(scaled_data,seq_length)

train_size= int(len(x)*0.8)
x_train, x_test=x[:train_size],x[:train_size:]
y_train, y_test=y[:train_size],y[:train_size:]

model =Sequential()
model.add(Input(shape=(seq_length, x.shape[2])))
model.add(LSTM(50, activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")


early_stop = EarlyStopping(monitor='val_loss', patience=50,restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=170, batch_size=32, validation_split=0.2, callbacks=[early_stop],verbose=1)

# --- Predict ---
y_pred = model.predict(x_test)

# --- Inverse scale the predictions ---
# Create dummy arrays to inverse transform correctly  
y_test_scaled=y_test.reshape(-1,1)  
y_pred_scaled=y_pred.reshape(-1,1) 

bandwidth_pred = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), 2)), y_pred_scaled), axis=1))[:,2]
bandwidth_actual = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), 2)), y_test_scaled), axis=1))[:,2]


# --- Visualization Style ---
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# --- Align Predicted and Actual Bandwidth ---
# Align time steps so predictions start after the sequence window
time_axis = range(seq_length, seq_length + len(bandwidth_pred))


plt.plot(bandwidth_actual,label="Actuel Bandwidth",linewidth=2,color ="#1f77b4")
plt.plot(bandwidth_pred,label ="Predicted Bandwidth", linewidth=2,color ="#ff7f0e", linestyle="--")
plt.title("Bandwidth Prediction (LSTM)", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Time Steps",fontsize=13)
plt.ylabel("Bandwidth (Mbps)",fontsize=13)
plt.legend(fontsize=12, loc="upper right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.text(len(bandwidth_pred) * 0.05, max(bandwidth_actual) * 0.9,
         f"Seq length: {seq_length}\nPatience: {early_stop.patience}\nEpochs run: {len(history.history['loss'])}",
         bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

# --- Optional: add loss plot below ---
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

# --- Write predictions to InfluxDB ---

for i , pred in enumerate(bandwidth_pred):
   time_stamp= df.index[train_size + seq_length +i ]
   point =Point("network_predicted ").field("bandwidth_mbps", pred).time(time_stamp,WritePrecision.S)
   write_api.write(bucket=INFLUXDB_BUCKET,record=point)

print("✅ Predictions written to InfluxDB successfully.")  

