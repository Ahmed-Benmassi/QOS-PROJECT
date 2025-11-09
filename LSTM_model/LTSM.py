from datetime import datetime , timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from influxdb_client import InfluxDBClient ,Point, WritePrecision
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os 
import time 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


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


def create_sequences(data, seq_length =10 ):
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
model.add(LSTM(50,activation="relu",input_shape=(seq_length,x.shape[2])))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")


early_stop = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop],verbose=1)

# --- Predict ---
y_pred = model.predict(x_test)

# --- Inverse scale the predictions ---
# Create dummy arrays to inverse transform correctly  
y_test_scaled=y_test.reshape(-1,1)  
y_pred_scaled=y_pred.reshape(-1,1) 

bandwidth_pred = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), 2)), y_pred_scaled), axis=1))[:,2]
bandwidth_actual = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), 2)), y_test_scaled), axis=1))[:,2]

plt.plot(bandwidth_actual,label="Actuel Bandwidth")
plt.plot(bandwidth_pred,label ="Predicted Bandwidth")
plt.title("Bandwidth Prediction (LSTM)")
plt.legend()
plt.show()

# --- Write predictions to InfluxDB ---

for i , pred in enumerate(bandwidth_pred):
   time_stamp= df.index[train_size + seq_length +i ]
   point =Point("network_predicted ").field("bandwidth_mbps", pred).time(time_stamp,WritePrecision.S)
   write_api.write(bucket=INFLUXDB_BUCKET,record=point)

print("✅ Predictions written to InfluxDB successfully.")  

