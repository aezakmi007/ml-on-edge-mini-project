"""
LogiEdge Inference Service — Component D, Task D2
Subscribes to sensor MQTT topics, runs preprocessing pipeline,
performs TFLite inference, and publishes results.

Environment variables:
  MODEL_PATH  : path to .tflite model file (default: model.tflite)
  TRUCK_ID    : truck identifier           (default: truck_001)
  BROKER_HOST : MQTT broker hostname       (default: localhost)

Publishes results to:
  logibridge/trucks/{truck_id}/inference
"""

import os
import json
import time
import threading
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque

# ── Configuration from environment variables ───────────────────────────────────
MODEL_PATH  = os.environ.get("MODEL_PATH",  "model.tflite")
TRUCK_ID    = os.environ.get("TRUCK_ID",    "truck_001")
BROKER_HOST = os.environ.get("BROKER_HOST", "localhost")
BROKER_PORT = int(os.environ.get("BROKER_PORT", "1883"))

# ── MQTT Topics ────────────────────────────────────────────────────────────────
TOPIC_TEMP      = f"logibridge/trucks/{TRUCK_ID}/temperature"
TOPIC_VIB       = f"logibridge/trucks/{TRUCK_ID}/vibration"
TOPIC_DOOR      = f"logibridge/trucks/{TRUCK_ID}/door"
TOPIC_INFERENCE = f"logibridge/trucks/{TRUCK_ID}/inference"

# ── Window Configuration ───────────────────────────────────────────────────────
WINDOW_SECONDS = 30
STEP_SECONDS   = 10
FILTER_SIZE    = 5

CLASS_NAMES    = {0: "Normal", 1: "Warning", 2: "Critical"}
CLASS_ACTIONS  = {
    0: "All systems normal.",
    1: "WARNING: Intervention needed within 2 hours.",
    2: "CRITICAL: Immediate escalation required!"
}

# ── Buffers ────────────────────────────────────────────────────────────────────
temp_buffer      = deque()
vib_buffer       = deque()
temp_filter_buf  = deque(maxlen=FILTER_SIZE)
vib_filter_buf   = deque(maxlen=FILTER_SIZE)
last_window_time = time.time()
lock             = threading.Lock()

# ── Normalisation stats ────────────────────────────────────────────────────────
norm_stats = None


# ══════════════════════════════════════════════════════════════════════════════
# LOAD TFLITE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_tflite_model(model_path: str):
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"[MODEL] Loaded: {model_path}")
    print(f"[MODEL] Input  dtype: {input_details[0]['dtype']}")
    print(f"[MODEL] Output dtype: {output_details[0]['dtype']}")
    return interpreter, input_details, output_details


# ══════════════════════════════════════════════════════════════════════════════
# LOAD NORMALISATION STATS
# ══════════════════════════════════════════════════════════════════════════════

def load_norm_stats():
    global norm_stats
    stats_candidates = [
        "training_stats.npy",
        "/opt/logibridge/training_stats.npy",
        "../data_pipeline/training_stats.npy"
    ]
    for path in stats_candidates:
        if os.path.exists(path):
            norm_stats = np.load(path, allow_pickle=True).item()
            print(f"[NORM] Loaded stats from {path}")
            return
    print("[NORM] Warning: training_stats.npy not found — running unnormalised")


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def moving_average(buf: deque, value: float) -> float:
    buf.append(value)
    return float(np.mean(buf))


def extract_features(temp_arr: np.ndarray, vib_arr: np.ndarray) -> np.ndarray:
    from scipy import stats as scipy_stats
    temp_mean    = np.mean(temp_arr)
    temp_std     = np.std(temp_arr)
    temp_roc     = (temp_arr[-1] - temp_arr[0]) / (WINDOW_SECONDS / 60.0)
    vib_rms      = np.sqrt(np.mean(vib_arr ** 2))
    vib_peak     = np.max(np.abs(vib_arr))
    vib_kurtosis = float(scipy_stats.kurtosis(vib_arr)) if len(vib_arr) >= 4 else 0.0
    return np.array([temp_mean, temp_std, temp_roc,
                     vib_rms,   vib_peak, vib_kurtosis], dtype=np.float32)


def normalise(features: np.ndarray) -> np.ndarray:
    if norm_stats is None:
        return features
    mean = norm_stats["mean"]
    std  = np.where(norm_stats["std"] == 0, 1e-8, norm_stats["std"])
    return (features - mean) / std


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(interpreter, input_details, output_details,
                  features: np.ndarray) -> tuple:
    """Run TFLite inference. Returns (class_id, confidence_scores)."""
    sample = features.reshape(1, -1).astype(np.float32)

    is_int8 = input_details[0]["dtype"] == np.int8
    if is_int8:
        scale, zero_point = input_details[0]["quantization"]
        if scale != 0:
            sample = sample / scale + zero_point
        sample = sample.astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    if output_details[0]["dtype"] == np.int8:
        out_scale, out_zp = output_details[0]["quantization"]
        scores = (output.astype(np.float32) - out_zp) * out_scale
    else:
        scores = output.astype(np.float32)

    scores     = scores[0]
    class_id   = int(np.argmax(scores))
    confidence = float(np.max(scores))
    return class_id, confidence, scores.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# WINDOW PROCESSING + PUBLISH
# ══════════════════════════════════════════════════════════════════════════════

def process_and_infer(mqtt_client, interpreter,
                      input_details, output_details):
    global last_window_time
    now = time.time()

    temp_vals = np.array([v for (t, v) in temp_buffer if now - t <= WINDOW_SECONDS])
    vib_vals  = np.array([v for (t, v) in vib_buffer  if now - t <= WINDOW_SECONDS])

    if len(temp_vals) < 10 or len(vib_vals) < 5:
        return

    raw_features  = extract_features(temp_vals, vib_vals)
    norm_features = normalise(raw_features)

    class_id, confidence, scores = run_inference(
        interpreter, input_details, output_details, norm_features
    )

    class_name = CLASS_NAMES[class_id]
    action     = CLASS_ACTIONS[class_id]

    result = {
        "truck_id":   TRUCK_ID,
        "timestamp":  now,
        "class_id":   class_id,
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "scores":     [round(s, 4) for s in scores],
        "action":     action,
        "model":      MODEL_PATH
    }

    mqtt_client.publish(TOPIC_INFERENCE, json.dumps(result))

    # Print alert levels with colour coding
    prefix = {0: "[  OK  ]", 1: "[  WARN ]", 2: "[ CRIT ]"}[class_id]
    print(f"{prefix} Class={class_id} ({class_name}) | "
          f"conf={confidence:.3f} | "
          f"temp_mean={raw_features[0]:.2f}°C | "
          f"vib_rms={raw_features[3]:.3f}g")

    if class_id == 2:
        print(f"         ⚠️  {action}")

    last_window_time = now


# ══════════════════════════════════════════════════════════════════════════════
# MQTT CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def make_on_message(mqtt_client, interpreter, input_details, output_details):
    def on_message(client, userdata, msg):
        global last_window_time
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            now     = time.time()

            if msg.topic == TOPIC_TEMP:
                val = moving_average(temp_filter_buf, payload["value"])
                temp_buffer.append((now, val))
                while temp_buffer and now - temp_buffer[0][0] > 60:
                    temp_buffer.popleft()

            elif msg.topic == TOPIC_VIB:
                val = moving_average(vib_filter_buf, payload["value"])
                vib_buffer.append((now, val))
                while vib_buffer and now - vib_buffer[0][0] > 60:
                    vib_buffer.popleft()

            elif msg.topic == TOPIC_DOOR:
                print(f"[DOOR] {payload['event']}")

            if now - last_window_time >= STEP_SECONDS:
                process_and_infer(mqtt_client, interpreter,
                                  input_details, output_details)

        except Exception as e:
            print(f"[ERROR] {e}")
    return on_message


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"[MQTT] Connected to {BROKER_HOST}:{BROKER_PORT}")
        client.subscribe(TOPIC_TEMP)
        client.subscribe(TOPIC_VIB)
        client.subscribe(TOPIC_DOOR)
        print(f"[MQTT] Subscribed to sensor topics for truck: {TRUCK_ID}")
        print(f"[MQTT] Publishing inference to: {TOPIC_INFERENCE}")
    else:
        print(f"[MQTT] Connection failed: rc={rc}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  LogiEdge Inference Service")
    print("=" * 60)
    print(f"  Model     : {MODEL_PATH}")
    print(f"  Truck ID  : {TRUCK_ID}")
    print(f"  Broker    : {BROKER_HOST}:{BROKER_PORT}")
    print("=" * 60)

    load_norm_stats()
    interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)

    client = mqtt.Client(client_id=f"inference_{TRUCK_ID}")
    client.on_connect = on_connect
    client.on_message = make_on_message(client, interpreter,
                                        input_details, output_details)
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    print("\n[INF] Inference service running. Waiting for sensor data...")
    print("[INF] Press Ctrl+C to stop.\n")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[INF] Inference service stopped.")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()