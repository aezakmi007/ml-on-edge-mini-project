"""
LogiEdge Preprocessing Pipeline — Component C, Task C2
Subscribes to MQTT sensor streams, applies:
  1. 5-sample moving average filter (temperature + vibration)
  2. 30-second sliding window feature extraction (step = 10 seconds)
  3. Normalisation using saved training_stats.npy
Outputs a 6-value feature vector per window:
  [temp_mean, temp_std, temp_roc, vib_rms, vib_peak, vib_kurtosis]
"""

import json
import time
import threading
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque
from scipy import stats as scipy_stats

# ── MQTT Configuration ─────────────────────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT  = 1883
TRUCK_ID     = "truck_001"

TOPIC_TEMP = f"logibridge/trucks/{TRUCK_ID}/temperature"
TOPIC_VIB  = f"logibridge/trucks/{TRUCK_ID}/vibration"
TOPIC_DOOR = f"logibridge/trucks/{TRUCK_ID}/door"

# ── Window Configuration ───────────────────────────────────────────────────────
WINDOW_SECONDS = 30    # 30-second sliding window
STEP_SECONDS   = 10    # new window every 10 seconds
TEMP_HZ        = 1     # temperature samples per second
VIB_HZ         = 0.5  # vibration samples per 2 seconds

WINDOW_TEMP_SAMPLES = WINDOW_SECONDS * TEMP_HZ        # 30 samples
WINDOW_VIB_SAMPLES  = int(WINDOW_SECONDS * VIB_HZ)   # 15 samples
FILTER_SIZE         = 5    # moving average window size

# ── Raw buffers (store last WINDOW seconds of readings) ───────────────────────
temp_buffer = deque()   # (timestamp, value)
vib_buffer  = deque()   # (timestamp, value)

# ── Filter buffers (last 5 readings for moving average) ───────────────────────
temp_filter_buf = deque(maxlen=FILTER_SIZE)
vib_filter_buf  = deque(maxlen=FILTER_SIZE)

# ── Normalisation stats (loaded from training_stats.npy) ──────────────────────
norm_stats = None   # set by load_norm_stats()

# ── Feature window output list (used during dataset generation) ────────────────
feature_windows  = []   # list of (feature_vector, timestamp)
lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
# 1. MOVING AVERAGE FILTER
# ══════════════════════════════════════════════════════════════════════════════

def apply_moving_average(buffer: deque, new_value: float) -> float:
    """
    Add new_value to filter buffer, return 5-sample moving average.
    If fewer than 5 samples available, averages what we have.
    """
    buffer.append(new_value)
    return float(np.mean(buffer))


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION (30-second window)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(temp_values: np.ndarray, vib_values: np.ndarray) -> np.ndarray:
    """
    Extract 6 features from a 30-second window of sensor data.

    Features:
      0 — temp_mean    : mean temperature (°C)
      1 — temp_std     : temperature standard deviation (°C)
      2 — temp_roc     : temperature rate-of-change (°C/min)
      3 — vib_rms      : vibration RMS (g)
      4 — vib_peak     : vibration peak value (g)
      5 — vib_kurtosis : vibration kurtosis (dimensionless)
    """
    # Temperature features
    temp_mean = np.mean(temp_values)
    temp_std  = np.std(temp_values)

    # Rate of change: (last value - first value) / window duration in minutes
    if len(temp_values) >= 2:
        temp_roc = (temp_values[-1] - temp_values[0]) / (WINDOW_SECONDS / 60.0)
    else:
        temp_roc = 0.0

    # Vibration features
    vib_rms      = np.sqrt(np.mean(vib_values ** 2))
    vib_peak     = np.max(np.abs(vib_values))
    vib_kurtosis = float(scipy_stats.kurtosis(vib_values)) if len(vib_values) >= 4 else 0.0

    return np.array([temp_mean, temp_std, temp_roc, vib_rms, vib_peak, vib_kurtosis],
                    dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def load_norm_stats(stats_path: str = "data_pipeline/training_stats.npy") -> bool:
    """
    Load normalisation stats (mean, std per feature) from training_stats.npy.
    Returns True if loaded, False if file not found (pre-training mode).
    """
    global norm_stats
    try:
        norm_stats = np.load(stats_path, allow_pickle=True).item()
        print(f"[NORM] Loaded normalisation stats from {stats_path}")
        print(f"       mean = {np.round(norm_stats['mean'], 4)}")
        print(f"       std  = {np.round(norm_stats['std'],  4)}")
        return True
    except FileNotFoundError:
        print(f"[NORM] Warning: {stats_path} not found — running unnormalised (pre-training mode)")
        return False


def normalise(features: np.ndarray) -> np.ndarray:
    """
    Z-score normalise features using loaded training stats.
    If stats not loaded, return raw features.
    """
    if norm_stats is None:
        return features
    mean = norm_stats["mean"]
    std  = norm_stats["std"]
    std  = np.where(std == 0, 1e-8, std)   # avoid division by zero
    return (features - mean) / std


# ══════════════════════════════════════════════════════════════════════════════
# 4. WINDOW PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

last_window_time = time.time()

def process_window():
    """
    Called every STEP_SECONDS. Extracts features from current buffers
    if enough data is available.
    Returns normalised feature vector or None.
    """
    global last_window_time

    now = time.time()

    # Collect temperature readings from last 30 seconds
    temp_vals = np.array([v for (t, v) in temp_buffer
                          if now - t <= WINDOW_SECONDS])
    vib_vals  = np.array([v for (t, v) in vib_buffer
                          if now - t <= WINDOW_SECONDS])

    # Need at least half the expected samples to compute meaningful features
    if len(temp_vals) < WINDOW_TEMP_SAMPLES // 2:
        print(f"[WINDOW] Not enough temp data yet ({len(temp_vals)} samples), waiting...")
        return None
    if len(vib_vals) < WINDOW_VIB_SAMPLES // 2:
        print(f"[WINDOW] Not enough vib data yet ({len(vib_vals)} samples), waiting...")
        return None

    raw_features  = extract_features(temp_vals, vib_vals)
    norm_features = normalise(raw_features)

    print(f"\n[WINDOW] Feature vector extracted:")
    labels = ["temp_mean", "temp_std", "temp_roc", "vib_rms", "vib_peak", "vib_kurtosis"]
    for label, raw, norm in zip(labels, raw_features, norm_features):
        print(f"         {label:15s} raw={raw:8.4f}  normalised={norm:8.4f}")

    with lock:
        feature_windows.append({
            "timestamp": now,
            "raw":       raw_features.tolist(),
            "normalised": norm_features.tolist()
        })

    last_window_time = now
    return norm_features


# ══════════════════════════════════════════════════════════════════════════════
# 5. MQTT CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"[MQTT] Connected to broker at {BROKER_HOST}:{BROKER_PORT}")
        client.subscribe(TOPIC_TEMP)
        client.subscribe(TOPIC_VIB)
        client.subscribe(TOPIC_DOOR)
        print(f"[MQTT] Subscribed to:")
        print(f"       {TOPIC_TEMP}")
        print(f"       {TOPIC_VIB}")
        print(f"       {TOPIC_DOOR}")
    else:
        print(f"[MQTT] Connection failed with code {rc}")


def on_message(client, userdata, msg):
    global last_window_time

    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        now     = time.time()

        if msg.topic == TOPIC_TEMP:
            raw_val      = payload["value"]
            filtered_val = apply_moving_average(temp_filter_buf, raw_val)
            temp_buffer.append((now, filtered_val))
            # Keep only last 60 seconds in buffer
            while temp_buffer and now - temp_buffer[0][0] > 60:
                temp_buffer.popleft()

        elif msg.topic == TOPIC_VIB:
            raw_val      = payload["value"]
            filtered_val = apply_moving_average(vib_filter_buf, raw_val)
            vib_buffer.append((now, filtered_val))
            while vib_buffer and now - vib_buffer[0][0] > 60:
                vib_buffer.popleft()

        elif msg.topic == TOPIC_DOOR:
            print(f"[DOOR] Event: {payload['event']} at {payload['timestamp']:.0f}")

        # Trigger window processing every STEP_SECONDS
        if now - last_window_time >= STEP_SECONDS:
            process_window()

    except Exception as e:
        print(f"[ERROR] Failed to process message: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessor(stats_path: str = "data_pipeline/training_stats.npy"):
    """Start the MQTT subscriber and preprocessing loop."""
    load_norm_stats(stats_path)

    client = mqtt.Client(client_id="preprocessor_001")
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    print("\n[PREP] Preprocessing pipeline running. Waiting for sensor data...")
    print("[PREP] Press Ctrl+C to stop.\n")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[PREP] Preprocessor stopped.")
    finally:
        client.disconnect()


if __name__ == "__main__":
    run_preprocessor()