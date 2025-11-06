from influxdb_client import InfluxDBClient
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
 
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
       

