🚦 QOS Network Supervision Tool
A Python-based Quality of Service (QoS) Monitoring & Prediction System for real-time network performance tracking and machine learning forecasting. Continuously measures latency, packet loss, bandwidth, and predicts future trends using LSTM neural networks.

⚡ Features
📡 Real-Time Monitoring
24/7 QoS tracking of 3 target IPs (8.8.8.8, 1.1.1.1, 150.171.27.11)

Latency & packet loss measurement via ping

Bandwidth usage tracking with psutil

Lightweight continuous monitoring loop

🤖 Machine Learning Engine
LSTM Neural Network for bandwidth prediction

Multi-feature analysis: bandwidth, latency, packet loss

Sequence learning with 10-step windows

Early stopping with 80-epoch patience

Automatic data scaling and preprocessing

📊 Visualization & Analytics
Real-time prediction vs actual plots

Training history and loss curves

Error distribution analysis

Multi-target comparison charts

Grafana-ready for dashboard integration

💾 Data Storage
InfluxDB 2.x integration for time-series data

Automated prediction logging


⚙️ Requirements
Python 3.x
InfluxDB 2.x running locally or remotely
InfluxDB CLI 2.x
Git (to clone this repo)
📚 Python Packages
subprocess
psutil
time
influxdb_client
datetime
Core dependencies
-tensorflow>=2.8.0 -pandas>=1.3.0 -numpy>=1.21.0 -scikit-learn>=1.0.0 -matplotlib>=3.5.0 -seaborn>=0.11.0 -python-dotenv>=0.19.0

🏗️ Architecture Data Flow:

Data Collection → Network metrics from multiple targets Preprocessing → Cleaning, scaling, sequence creation Model Training → LSTM neural network Prediction → Future bandwidth forecasting Storage → InfluxDB time-series database(doing it later ) Visualization → Real-time plots and dashboards

Model Configuration:

Sequence Length: 10 time steps LSTM Layers: 16-50 units with ReLU activation Training: Early stopping with 80 patience Validation: 20% split for model evaluation
📈 Output & Metrics Performance Metrics

MAE (Mean Absolute Error): Prediction accuracy in Mbps R² Score: Model fit quality (0-1 scale) Training Loss: Model convergence monitoring Prediction Horizon: 10-20 future time steps

Visualizations

-Actual vs Predicted bandwidth trends -Training/validation loss curves -Error distribution analysis -Multi-target comparison charts

📊 Sample Output (lstm)

📊 Data Summary: Total data points: 1728 Bandwidth range: 0.37 - 975.47 Mbps Average bandwidth: 85.23 Mbps

🎉 FINAL RESULTS: MAE: 12.45 Mbps R² Score: 0.8347 Epochs trained: 156 Test samples: 345

✅ Predictions written to InfluxDB successfully.

Next Steps:i am Considering about adding real-time dashboard integration with Grafana for live monitoring and alerting capabilities.



