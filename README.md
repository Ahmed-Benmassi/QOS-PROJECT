# 🚦QOS Network Supervision Tool (LITTLE UPDATE)

-This is a Python-based Quality of Service (QoS) monitoring and prediction tool that collects, analyzes, and forecasts network performance metrics including:

-Real-time monitoring of latency, packet loss, and bandwidth

-using PROPHET for bandwidth predictionfrom the collected data 

-LSTM Neural Network for bandwidth prediction ( using raw bandwith data i found i couldnt work with the data i detect for one small issue the dataset was small to use for ltsm)

-Historical data analysis and trend forecasting

-InfluxDB and Grafana integration for data storage and visualization

------------------------------------
# 📦Features
Continuous monitoring of a 3 target IP (default: 8.8.8.8, 1.1.1.1,  150.171.27.11)
Measures average latency and packet loss using ping
Tracks network bandwidth usage via psutil
Stores metrics in InfluxDB for easy dashboarding (e.g., Grafana)

## Facebook Prophet
- **Purpose**: Long-term trend analysis and seasonal forecasting
- **Features**:
  - Holiday effect modeling
  - Confidence intervals for predictions
- **Advantages**: Handles missing data, outliers, and seasonal patterns effectively


## 🤖Machine Learning

-LSTM Neural Network for time series forecasting

-Bandwidth prediction with sequence learning

-Multi-feature analysis (bandwidth, latency, packet loss)

-Early stopping and model optimization

## 📊Data Visualization

-Real-time prediction vs actual comparison plots

-Training history and loss curves

-Performance metrics display

## 💾Data Storage & Integration

-InfluxDB 2.x integration for time-series data

-Automated prediction logging

-Support for Grafana dashboards

-Historical data analysis

-------------------------------

## 📝Requirements
Python 3.x

InfluxDB 2.x running locally or remotely

InfluxDB CLI 2.x

Git to clone this repo


## 📚Python Packages


-subprocess

-prophet

-psutil

-time

-influxdb_client

-datetime

### Core dependencies

-tensorflow>=2.8.0 

-pandas>=1.3.0

-numpy>=1.21.0

-scikit-learn>=1.0.0 

-matplotlib>=3.5.0 

-seaborn>=0.11.0 

-python-dotenv>=0.19.0

-------------------------

## 🏗️Architecture Data Flow:

Data Collection → Network metrics from multiple targets Preprocessing → Cleaning, scaling, sequence creation Model Training → LSTM neural network Prediction → Future bandwidth forecasting Storage → InfluxDB time-series database(doing it later ) Visualization → Real-time plots and dashboards

----------------

-Model Configuration:

Sequence Length: 10 time steps LSTM Layers: 16-50 units with ReLU activation Training: Early stopping with 80 patience Validation: 20% split for model evaluation


## 📈Output & Metrics Performance Metrics

-Sequence Length: 10 time steps

-LSTM Layers: 16-50 units with ReLU activation

-Training: Early stopping with 80 patience

-Validation: 20% split for model evaluation

-MAE (Mean Absolute Error): Prediction accuracy in Mbps

-R² Score: Model fit quality (0-1 scale)

-Training Loss: Model convergence monitoring

-Prediction Horizon: 10-20 future time steps

### Visualizations

# prophet visualitaion of predicted data 

<img width="973" height="587" alt="image" src="https://github.com/user-attachments/assets/98310769-ba4c-4e23-a39f-ed2308c8c2fe" />

## 📊Sample Output (lstm)

<img width="1330" height="694" alt="image" src="https://github.com/user-attachments/assets/fdbd0b92-d2ea-469c-93dd-ec7bd9b81ffc" />

# predicted data from LSTM 
![WhatsApp Image 2025-11-25 at 18 36 41_aa53fef4](https://github.com/user-attachments/assets/d36c9456-b32d-4b07-a3da-70c5c87297e3)


## Actual vs Predicted bandwidth trends 
![WhatsApp Image 2025-11-25 at 23 03 29_c61d4b64](https://github.com/user-attachments/assets/c2fd9d85-a185-4862-8cd5-2b210d0f07cd)



-Training/validation loss curves

-Error distribution analysis -

Multi-target comparison charts

### 📊Data Summary:

Total data points: 1728 Bandwidth range: 0.37 - 975.47 Mbps Average bandwidth: 85.23 Mbps

### 🎉FINAL RESULTS: 

-MAE: 12.45 Mbps R² Score: 0.8347 Epochs trained: 156 Test samples: 345

✅Predictions written to InfluxDB successfully.

Next Steps:i am Considering about adding real-time dashboard integration with Grafana for live monitoring and alerting capabilities.












