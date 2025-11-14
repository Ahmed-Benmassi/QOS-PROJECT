🚦 QOS Network Supervision Tool — Enhanced Overview

A Python-based Quality of Service (QoS) Monitoring & Prediction System designed for real-time network performance tracking and machine-learning-based forecasting.
This tool continuously measures latency, packet loss, bandwidth, and predicts future usage trends using an LSTM neural network — all with optional InfluxDB + Grafana integration.

⚡ Core Capabilities
📡 Real-Time Network Monitoring

Continuous QoS measurement for 3 target IPs
(default: 8.8.8.8, 1.1.1.1, 150.171.27.11)

Average latency & packet loss calculation via ping

Live bandwidth usage tracking with psutil

Lightweight and efficient loop for 24/7 monitoring

🤖 Machine Learning Engine
🔮 LSTM Time-Series Prediction

Predicts raw bandwidth usage (Mbps)

Multi-feature model:
→ bandwidth, latency, packet loss

Sequence learning (10-step windows)

Early stopping with 80-epoch patience

Automatic scaling and pre/post-processing

🧠 Model Configuration

Sequence Length: 10

LSTM Units: 16–50 (ReLU activation)

Validation Split: 20%

Prediction Horizon: 10–20 timesteps

📊 Visual Analytics
🚀 Built-in Plots

Actual vs. predicted bandwidth curves

Loss history for training & validation

Error distribution charts

Correlation heatmaps for multi-feature analysis

Multi-target comparison visualization

🎛️ Dashboard-Ready

Fully compatible with Grafana, for:

live bandwidth insights

QoS alerting

predictive trend dashboards

💾 Data Storage & Integration
🗄️ InfluxDB 2.x Support

Real-time metric insertion

Prediction logging

Works with local or remote instances

Ready for Grafana dashboards

📦 Requirements
🔧 System

Python 3.x

InfluxDB 2.x (optional, but recommended)

InfluxDB CLI

Git

🐍 Python Packages
subprocess
psutil
time
influxdb_client
datetime


Core ML / Data Stack:

tensorflow >= 2.8.0
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
python-dotenv >= 0.19.0

🧭 Data Flow Architecture
Data Collection
     ↓
Preprocessing  → cleaning, scaling, windowing
     ↓
Model Training (LSTM)
     ↓
Prediction Engine
     ↓
InfluxDB Storage (optional)
     ↓
Visualization (matplotlib / Grafana)

📈 Example Output (LSTM)
📊 Data Summary

Total samples: 1728

Bandwidth range: 0.37 – 975.47 Mbps

Average: 85.23 Mbps

🏁 Final Results

MAE: 12.45 Mbps

R² Score: 0.8347

Epochs trained: 156

Test samples: 345

✔️ Predictions successfully written to InfluxDB.

🚀 Next Steps
🔧 Planned Enhancements

Live Grafana dashboard for real-time:

bandwidth visualizations

latency & packet loss panels

ML prediction overlays

alerting (e.g., high latency, bandwidth drops)

