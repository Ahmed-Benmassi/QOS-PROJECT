import subprocess
import psutil
import time
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WritePrecision, WriteOptions


# --- CONFIGURATION ---
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "tFm0fX3cH6Nrhjgz75oUpTocP_sm2PEiZq4aRvNBZ9HuyVRMljKM5cLN3juv7-_BCqgIoD9B2xC98wk45K6p-A=="
INFLUXDB_ORG = "qos-supervision"
INFLUXDB_BUCKET = "network_metrics"

PING_TARGET = "8.8.8.8"
NETWORK_INTERFACE = "Wi-Fi"

# --- SETUP INFLUXDB CLIENT ---
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api()


# --- FUNCTIONS ---

def ping_test(target):
    try:
        ping_output = subprocess.check_output(
            ["ping", "-n", "5", target], universal_newlines=True
        )
        packet_loss_line = [line for line in ping_output.split('\n') if "Lost" in line][0]
        loss_percent_str = packet_loss_line.split('(')[1].split('%')[0]
        packet_loss_percent = int(loss_percent_str)

        latency_line = [line for line in ping_output.split('\n') if "Average" in line][0]
        latency_parts = latency_line.split(',')
        avg_part = [part for part in latency_parts if 'Average' in part][0]
        latency_avg = float(avg_part.split('=')[1].strip().replace("ms", ""))

        return latency_avg, packet_loss_percent
    except Exception as e:
        print(f"Ping test failed: {e}")
        return None, None


def get_bandwidth(interface):
    net1 = psutil.net_io_counters(pernic=True)[interface]
    time.sleep(4)
    net2 = psutil.net_io_counters(pernic=True)[interface]

    bytes_sent_per_sec = net2.bytes_sent - net1.bytes_sent
    bytes_recv_per_sec = net2.bytes_recv - net1.bytes_recv

    total_bytes_per_sec = bytes_sent_per_sec + bytes_recv_per_sec
    mbps = (total_bytes_per_sec * 8) / (1024 * 1024)
    return mbps

def write_to_influx(latency, packet_loss, bandwidth):
    point = (
        Point("network_metrics")
        .field("latency_ms", latency)
        .field("packet_loss_percent", packet_loss)
        .field("bandwidth_mbps", bandwidth)
        .time(datetime.utcnow(), WritePrecision.S)
    )
    write_api.write(bucket=INFLUXDB_BUCKET, record=point)
    print(f"Data written: latency={latency}ms, packet_loss={packet_loss}%, bandwidth={bandwidth:.2f}Mbps")

# --- MAIN LOOP ---

def main():
    while True:
        latency, packet_loss = ping_test(PING_TARGET)
        if latency is None:
            print("Skipping data write due to ping failure")
            time.sleep(60)
            continue

        bandwidth = get_bandwidth(NETWORK_INTERFACE)
        write_to_influx(latency, packet_loss, bandwidth)
        time.sleep(60)  # wait 1 min before next measurement

if __name__ == "__main__":
    main()
