
from datetime import datetime , timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from influxdb_client import InfluxDBClient ,Point, WritePrecision
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import matplotlib.pyplot as plt 
 
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

query = f'''
from(bucket:  "{INFLUXDB_BUCKET} ")
  |> range(start: -7d)  // last 7 days
  |> filter(fn: (r) => r._measurement == "network_metrics")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time","target","latency_ms","packet_loss_percent","bandwidth_mbps"])
'''

tables = query_api.query_data_frame(query)
tables["_time"] =pd.to_datetime(tables["_time"])
tables.set_index("_time",inplace=True)

scaler =MinMaxScaler()
scaled_data =scaler.fit_transform(df)

def create_sequences(data, seq_length =10 ):
    x, y = [] ,[]
    for i in range(len(data)-seq_length):
        x.append(data[i:i+seq_length])
        x.append(data[i+seq_length, 2])
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

history = model.fit(
    x_train,y_train,
    epochs= 50
    batch_size =32
    validation_split=0.2
    verbose=1
  
)

early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])

 y_pred=model.predict(x_test)    
y_test_scaled=y_test.reshape(-1,1)  
y_pred_scaled=y_pred.reshape(-1,1) 

bandwidth_pred = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), 2)), y_pred_scaled), axis=1))[:,2]
bandwidth_actual = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), 2)), y_test_scaled), axis=1))[:,2]

plt.plot(bandwidth_actual,label="Actuel")
plt.plot(bandwidth_pred,label ="Predicted")
plt.legend()
plt.show()

for i , pred in enumerate(bandwidth_pred):
   time_stamp= df.index[train_size + seq_length +i ]
   point =Point("network_predicted ").field("bandwidth_mbps", pred).time(time_stamp,write_precision.s)
   write_api.write(bucket=INFLUXDB_BUCKET,record=point)

   

