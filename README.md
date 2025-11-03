# QOS Network Supervision Tool (LITTLE UPDATE)

This is a Python-based Quality of Service (QoS) monitoring tool that collects and logs network performance metrics such as:

- Average latency (ms) via ping
- Packet loss (%)
- Network bandwidth usage (Mbps)

Metrics are written to an **InfluxDB** instance for analysis and visualization.

---

## 📦 Features

- Continuous monitoring of a 3 target IP (default: `8.8.8.8`, `1.1.1.1`, ` 150.171.27.11`)
- Measures average latency and packet loss using `ping`
- Tracks network bandwidth usage via `psutil`
- Stores metrics in InfluxDB for easy dashboarding (e.g., Grafana)

---

## ⚙️ Requirements

- Python 3.x
- InfluxDB 2.x running locally or remotely
- InfluxDB CLI 2.x 
- Git (to clone this repo)

### 📚 Python Packages

- subprocess
- psutil
- time
- influxdb_client
- datetime



