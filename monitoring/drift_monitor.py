"""
LogiEdge PSI Drift Monitor — Component E, Task E1

Population Stability Index (PSI) monitors model confidence score distribution.
When PSI > 0.25, the incoming data has drifted significantly from training data.

Workflow:
  Phase 1 — Build reference distribution (300 clean Normal-class windows)
  Phase 2 — Monitor live inference output every 60 seconds
  Phase 3 — Alert when PSI > 0.25

PSI formula:
  PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)

PSI thresholds:
  PSI < 0.10  : No drift (stable)
  0.10–0.25   : Moderate drift (monitor closely)
  PSI > 0.25  : Significant drift → ALERT

Usage:
  # Phase 1: generate reference distribution
  python3 monitoring/drift_monitor.py --mode reference

  # Phase 2: live monitoring (run simulator separately)
  python3 monitoring/drift_monitor.py --mode monitor
"""

import argparse
import json
import os
import time
import numpy as np
from collections import deque
import paho.mqtt.client as mqtt

# ── Paths ──────────────────────────────────────────────────────────────────────
REFERENCE_DIST_PATH = "monitoring/reference_dist.json"
MODEL_PATH          = "inference/model.tflite"
STATS_PATH          = "data_pipeline/training_stats.npy"

# ── MQTT Configuration ─────────────────────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TRUCK_ID    = "truck_001"
TOPIC_INFERENCE = f"logibridge/trucks/{TRUCK_ID}/inference"

# ── PSI Configuration ──────────────────────────────────────────────────────────
# 4 bins for confidence score distribution (assignment requirement)
BINS           = [0.0, 0.25, 0.50, 0.75, 1.01]  # 1.01 to include 1.0
BIN_LABELS     = ["[0.00–0.25)", "[0.25–0.50)", "[0.50–0.75)", "[0.75–1.00]"]
REFERENCE_SIZE = 300    # run 300 clean Normal-class windows for reference
MONITOR_WINDOW = 100    # PSI computed on last 100 inferences
MONITOR_INTERVAL = 60   # compute PSI every 60 seconds
PSI_ALERT_THRESHOLD    = 0.25
PSI_RECOVERY_THRESHOLD = 0.10

# ── Rolling inference buffer ───────────────────────────────────────────────────
confidence_buffer = deque(maxlen=MONITOR_WINDOW)
last_psi_time     = time.time()


# ══════════════════════════════════════════════════════════════════════════════
# PSI CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_bin_distribution(confidence_scores: list) -> np.ndarray:
    """
    Bin confidence scores into 4 bins and return percentage per bin.
    Adds small epsilon to avoid log(0) in PSI calculation.
    """
    counts, _ = np.histogram(confidence_scores, bins=BINS)
    total      = len(confidence_scores)
    pcts       = counts / total
    # Add epsilon to avoid division by zero or log(0)
    pcts       = np.where(pcts == 0, 1e-6, pcts)
    return pcts


def compute_psi(expected_pcts: np.ndarray,
                actual_pcts:   np.ndarray) -> float:
    """
    PSI = Σ (actual - expected) × ln(actual / expected)
    Higher PSI = more drift from reference distribution.
    """
    psi = np.sum((actual_pcts - expected_pcts) *
                 np.log(actual_pcts / expected_pcts))
    return float(psi)


def print_distribution_table(label: str,
                              pcts: np.ndarray,
                              scores: list = None):
    """Pretty-print the confidence score distribution."""
    print(f"\n  {'Bin':<18} {'Pct':>8}  {'Bar'}")
    print(f"  {'-'*50}")
    for i, (bin_label, pct) in enumerate(zip(BIN_LABELS, pcts)):
        bar = "█" * int(pct * 40)
        count = int(pct * len(scores)) if scores else ""
        print(f"  {bin_label:<18} {pct*100:>6.1f}%  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — BUILD REFERENCE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def build_reference_distribution():
    """
    Generate 300 clean Normal-class windows in batch,
    run TFLite inference on each, collect confidence scores,
    compute bin distribution, save as reference_dist.json.
    """
    print("=" * 60)
    print("  Phase 1: Building Reference Distribution")
    print(f"  Running {REFERENCE_SIZE} clean Normal-class windows")
    print("=" * 60)

    # ── Load TFLite model ──────────────────────────────────────────────────
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[MODEL] Loaded {MODEL_PATH}")

    # ── Load normalisation stats ───────────────────────────────────────────
    norm_stats = None
    if os.path.exists(STATS_PATH):
        norm_stats = np.load(STATS_PATH, allow_pickle=True).item()
        print(f"[NORM]  Loaded {STATS_PATH}")

    # ── Generate Normal-class feature vectors ──────────────────────────────
    from scipy import stats as scipy_stats

    TEMP_SETPOINT  = 4.0
    WINDOW_SECONDS = 30

    confidence_scores = []
    print(f"\n[REF]  Generating {REFERENCE_SIZE} Normal windows...")

    for i in range(REFERENCE_SIZE):
        # Simulate a 30-second Normal window
        temp = np.random.normal(TEMP_SETPOINT, 0.3, 30).astype(np.float32)
        vib  = np.abs(np.random.normal(0.45, 0.05, 15).astype(np.float32))

        # Extract features
        temp_mean    = np.mean(temp)
        temp_std     = np.std(temp)
        temp_roc     = (temp[-1] - temp[0]) / (WINDOW_SECONDS / 60.0)
        vib_rms      = np.sqrt(np.mean(vib ** 2))
        vib_peak     = np.max(np.abs(vib))
        vib_kurtosis = float(scipy_stats.kurtosis(vib))

        features = np.array([temp_mean, temp_std, temp_roc,
                              vib_rms, vib_peak, vib_kurtosis],
                             dtype=np.float32)

        # Normalise
        if norm_stats is not None:
            mean = norm_stats["mean"]
            std  = np.where(norm_stats["std"] == 0, 1e-8, norm_stats["std"])
            features = (features - mean) / std

        # Run inference
        sample = features.reshape(1, -1)
        is_int8 = input_details[0]["dtype"] == np.int8
        if is_int8:
            scale, zp = input_details[0]["quantization"]
            if scale != 0:
                sample = sample / scale + zp
            sample = sample.astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        if output_details[0]["dtype"] == np.int8:
            out_scale, out_zp = output_details[0]["quantization"]
            scores = (output.astype(np.float32) - out_zp) * out_scale
        else:
            scores = output.astype(np.float32)

        confidence = float(np.max(scores[0]))
        confidence_scores.append(confidence)

        if (i + 1) % 50 == 0:
            print(f"[REF]  {i+1}/{REFERENCE_SIZE} windows processed...")

    # ── Compute reference distribution ────────────────────────────────────
    reference_pcts = compute_bin_distribution(confidence_scores)
    mean_conf      = float(np.mean(confidence_scores))

    print(f"\n[REF]  Reference distribution computed:")
    print_distribution_table("Reference", reference_pcts, confidence_scores)
    print(f"\n[REF]  Mean confidence: {mean_conf:.4f}")

    # ── Save to JSON ──────────────────────────────────────────────────────
    os.makedirs("monitoring", exist_ok=True)
    reference_data = {
        "bin_labels":       BIN_LABELS,
        "bin_edges":        BINS,
        "reference_pcts":   reference_pcts.tolist(),
        "reference_size":   REFERENCE_SIZE,
        "mean_confidence":  mean_conf,
        "created_at":       time.time()
    }
    with open(REFERENCE_DIST_PATH, "w") as f:
        json.dump(reference_data, f, indent=2)

    print(f"\n[SAVE] Reference distribution saved to {REFERENCE_DIST_PATH}")
    print("[DONE] Phase 1 complete! ✅")
    print(f"       Now run: python3 monitoring/drift_monitor.py --mode monitor")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — LIVE MONITORING
# ══════════════════════════════════════════════════════════════════════════════

def load_reference():
    """Load saved reference distribution from JSON."""
    with open(REFERENCE_DIST_PATH, "r") as f:
        data = json.load(f)
    reference_pcts = np.array(data["reference_pcts"])
    print(f"[REF]  Loaded reference distribution from {REFERENCE_DIST_PATH}")
    print(f"[REF]  Reference size: {data['reference_size']} windows")
    print(f"[REF]  Mean confidence: {data['mean_confidence']:.4f}")
    return reference_pcts


def check_psi(reference_pcts: np.ndarray):
    """Compute PSI on current rolling buffer and print result."""
    global last_psi_time

    if len(confidence_buffer) < 10:
        print(f"[PSI]  Waiting for data... ({len(confidence_buffer)}/10 minimum)")
        return

    scores  = list(confidence_buffer)
    actual  = compute_bin_distribution(scores)
    psi     = compute_psi(reference_pcts, actual)
    n       = len(scores)

    # Status indicator
    if psi < 0.10:
        status = "✅ STABLE"
    elif psi < PSI_ALERT_THRESHOLD:
        status = "⚠️  MODERATE"
    else:
        status = "🚨 DRIFT DETECTED"

    print(f"\n[PSI]  {'─'*50}")
    print(f"[PSI]  PSI = {psi:.4f} | n={n} | {status}")
    print_distribution_table("Current", actual, scores)

    if psi > PSI_ALERT_THRESHOLD:
        print(f"\n[LOGIBRIDGE DRIFT ALERT] PSI={psi:.3f}")
        print("         Model confidence distribution has shifted significantly!")
        print("         Recommend: retrain model or investigate sensor data.")
    elif psi < PSI_RECOVERY_THRESHOLD and len(confidence_buffer) >= 20:
        print(f"[PSI]  ✅ PSI={psi:.3f} — Distribution stable (below 0.10)")

    last_psi_time = time.time()
    return psi


def run_monitor():
    """Subscribe to inference MQTT topic and monitor PSI continuously."""
    print("=" * 60)
    print("  Phase 2: Live PSI Drift Monitoring")
    print("=" * 60)

    if not os.path.exists(REFERENCE_DIST_PATH):
        print(f"[ERROR] {REFERENCE_DIST_PATH} not found!")
        print("        Run Phase 1 first:")
        print("        python3 monitoring/drift_monitor.py --mode reference")
        return

    reference_pcts = load_reference()

    print(f"\n[MON]  PSI alert threshold  : {PSI_ALERT_THRESHOLD}")
    print(f"[MON]  PSI recovery threshold: {PSI_RECOVERY_THRESHOLD}")
    print(f"[MON]  PSI check interval   : every {MONITOR_INTERVAL} seconds")
    print(f"[MON]  Rolling window size  : last {MONITOR_WINDOW} inferences")
    print(f"\n[MON]  Subscribing to: {TOPIC_INFERENCE}")
    print("[MON]  Waiting for inference results...\n")

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(TOPIC_INFERENCE)
            print(f"[MQTT] Connected. Subscribed to {TOPIC_INFERENCE}")
        else:
            print(f"[MQTT] Connection failed: rc={rc}")

    def on_message(client, userdata, msg):
        global last_psi_time
        try:
            payload    = json.loads(msg.payload.decode("utf-8"))
            confidence = payload.get("confidence", 0.0)
            class_id   = payload.get("class_id", 0)
            confidence_buffer.append(confidence)

            class_names = {0: "Normal", 1: "Warning", 2: "Critical"}
            print(f"[INF]  class={class_id} ({class_names.get(class_id,'?')}) | "
                  f"conf={confidence:.3f} | "
                  f"buffer={len(confidence_buffer)}/{MONITOR_WINDOW}")

            # Check PSI every MONITOR_INTERVAL seconds
            if time.time() - last_psi_time >= MONITOR_INTERVAL:
                check_psi(reference_pcts)

        except Exception as e:
            print(f"[ERROR] {e}")

    client = mqtt.Client(client_id="drift_monitor")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[MON]  Monitor stopped.")
        # Final PSI check on exit
        print("\n[MON]  Final PSI check:")
        check_psi(reference_pcts)
    finally:
        client.disconnect()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogiEdge PSI Drift Monitor")
    parser.add_argument(
        "--mode",
        choices=["reference", "monitor"],
        required=True,
        help="reference: build reference dist | monitor: live PSI monitoring"
    )
    args = parser.parse_args()

    if args.mode == "reference":
        build_reference_distribution()
    else:
        run_monitor()