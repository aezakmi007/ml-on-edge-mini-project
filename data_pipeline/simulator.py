"""
LogiEdge Sensor Simulator — Component C, Task C1
Simulates three sensor streams for a cold-chain refrigerated truck:
  - Temperature (1 Hz)
  - Vibration RMS (0.5 Hz)
  - Door events (discrete)
Publishes to local Mosquitto MQTT broker on localhost:1883
Usage:
  python simulator.py --anomaly none
  python simulator.py --anomaly temp_drift
  python simulator.py --anomaly vibration
  python simulator.py --anomaly combined
"""

import argparse
import json
import random
import time
import numpy as np
import paho.mqtt.client as mqtt

# ── MQTT Configuration ─────────────────────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TRUCK_ID    = "truck_001"

TOPIC_TEMP  = f"logibridge/trucks/{TRUCK_ID}/temperature"
TOPIC_VIB   = f"logibridge/trucks/{TRUCK_ID}/vibration"
TOPIC_DOOR  = f"logibridge/trucks/{TRUCK_ID}/door"

# ── Sensor Normal Parameters ───────────────────────────────────────────────────
TEMP_SETPOINT   = 4.0    # °C
TEMP_STD        = 0.3    # °C
TEMP_DRIFT_RATE = 0.08   # °C per reading (anomaly)

VIB_NORMAL_MEAN = 0.45   # g
VIB_NORMAL_STD  = 0.05   # g
VIB_FAULT_MEAN  = 1.20   # g (bearing wear anomaly)
VIB_FAULT_STD   = 0.15   # g

# Average time between door events (seconds)
DOOR_EVENT_INTERVAL = 45


def connect_mqtt() -> mqtt.Client:
    """Create and connect an MQTT client to the local broker."""
    client = mqtt.Client(client_id=f"simulator_{TRUCK_ID}")

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[MQTT] Connected to broker at {BROKER_HOST}:{BROKER_PORT}")
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    client.on_connect = on_connect
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()
    return client


def publish(client: mqtt.Client, topic: str, payload: dict):
    """Publish a JSON payload to an MQTT topic."""
    message = json.dumps(payload)
    client.publish(topic, message)
    print(f"[PUB] {topic} → {message}")


def run_simulator(anomaly_mode: str):
    """
    Main simulator loop.
    anomaly_mode: 'none' | 'temp_drift' | 'vibration' | 'combined'
    """
    client = connect_mqtt()
    time.sleep(1)  # Wait for broker connection to establish

    print(f"\n[SIM] Starting simulator | anomaly_mode={anomaly_mode}")
    print("[SIM] Press Ctrl+C to stop.\n")

    # State tracking
    temp_reading_count = 0   # counts temp readings for drift calculation
    door_state         = "CLOSED"
    last_door_time     = time.time()
    last_vib_time      = time.time()

    try:
        while True:
            now = time.time()

            # ── Temperature Stream (1 Hz) ──────────────────────────────────────
            if anomaly_mode in ("temp_drift", "combined"):
                # Linear drift: each reading adds 0.08°C
                temp = np.random.normal(
                    TEMP_SETPOINT + TEMP_DRIFT_RATE * temp_reading_count,
                    TEMP_STD
                )
            else:
                temp = np.random.normal(TEMP_SETPOINT, TEMP_STD)

            publish(client, TOPIC_TEMP, {
                "truck_id":  TRUCK_ID,
                "timestamp": now,
                "value":     round(float(temp), 4),
                "unit":      "celsius"
            })
            temp_reading_count += 1

            # ── Vibration Stream (0.5 Hz — every 2 seconds) ───────────────────
            if now - last_vib_time >= 2.0:
                if anomaly_mode in ("vibration", "combined"):
                    vib = np.random.normal(VIB_FAULT_MEAN, VIB_FAULT_STD)
                else:
                    vib = np.random.normal(VIB_NORMAL_MEAN, VIB_NORMAL_STD)

                vib = max(0.0, float(vib))   # vibration can't be negative
                publish(client, TOPIC_VIB, {
                    "truck_id":  TRUCK_ID,
                    "timestamp": now,
                    "value":     round(vib, 4),
                    "unit":      "g_rms"
                })
                last_vib_time = now

            # ── Door Events (discrete, random interval ~45 seconds) ───────────
            if now - last_door_time >= DOOR_EVENT_INTERVAL + random.uniform(-10, 10):
                door_state = "OPEN" if door_state == "CLOSED" else "CLOSED"
                publish(client, TOPIC_DOOR, {
                    "truck_id":  TRUCK_ID,
                    "timestamp": now,
                    "event":     door_state
                })
                last_door_time = now

            time.sleep(1)   # 1 Hz main loop

    except KeyboardInterrupt:
        print("\n[SIM] Simulator stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        print("[SIM] MQTT disconnected. Goodbye.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogiEdge Cold-Chain Sensor Simulator")
    parser.add_argument(
        "--anomaly",
        choices=["none", "temp_drift", "vibration", "combined"],
        default="none",
        help="Anomaly mode to simulate (default: none)"
    )
    args = parser.parse_args()
    run_simulator(args.anomaly)