"""
LogiEdge — Five-Metric Model Benchmarking Suite
Component F, Task F2

Benchmarks M1 (FP32), M2 (PTQ INT8), and M3 (Pruned + INT8):
1. Mean inference latency (ms) — 200 iterations after 10 warm-up runs
2. p95 inference latency (ms)
3. Model file size (KB)
4. Overall Accuracy (%) & Class 2 (Critical) Recall (%) on validation set
5. Energy per inference (mJ) using E = P * t (psutil CPU% * TDP)

Saves results to benchmarks/results.json and prints comparison table.
"""

import os
import sys
import time
import json
import psutil
import numpy as np
import tensorflow as tf

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_X_PATH = os.path.join(BASE_DIR, "training", "X_val.npy")
VAL_Y_PATH = os.path.join(BASE_DIR, "training", "y_val.npy")
RESULTS_PATH = os.path.join(BASE_DIR, "benchmarks", "results.json")

MODELS = {
    "M1 (FP32 Baseline)": os.path.join(BASE_DIR, "training", "models", "m1_fp32.tflite"),
    "M2 (PTQ INT8)": os.path.join(BASE_DIR, "training", "models", "m2_ptq_int8.tflite"),
    "M3 (Pruned 35% + INT8)": os.path.join(BASE_DIR, "training", "models", "m3_pruned_int8.tflite"),
}

# ── Parameters ────────────────────────────────────────────────────────────────
WARMUP_RUNS = 10
BENCH_RUNS = 200
LAPTOP_TDP_WATTS = 35.0   # Baseline laptop TDP (Apple M-series typical active load ~30-45W)
PI5_TDP_WATTS = 12.0      # Raspberry Pi 5 peak TDP for comparison in report


def load_tflite_interpreter(model_path):
    """Loads TFLite model and allocates tensors."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return interpreter, input_details, output_details


def run_single_inference(interpreter, input_details, output_details, sample):
    """Runs a single inference handling both FP32 and INT8 quantization."""
    in_scale, in_zero_point = input_details.get("quantization", (0.0, 0))
    sample_prep = sample.reshape(input_details["shape"])

    if input_details["dtype"] == np.int8:
        if in_scale > 0:
            sample_prep = np.round(sample_prep / in_scale + in_zero_point).astype(np.int8)
        else:
            sample_prep = sample_prep.astype(np.int8)
    else:
        sample_prep = sample_prep.astype(np.float32)

    interpreter.set_tensor(input_details["index"], sample_prep)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])

    out_scale, out_zero_point = output_details.get("quantization", (0.0, 0))
    if output_details["dtype"] == np.int8 and out_scale > 0:
        output = (output.astype(np.float32) - out_zero_point) * out_scale

    return output[0]


def evaluate_accuracy_and_recall(interpreter, input_details, output_details, X_val, y_val):
    """Evaluates accuracy and Class 2 (Critical) recall on validation set."""
    predictions = []
    for i in range(len(X_val)):
        prob = run_single_inference(interpreter, input_details, output_details, X_val[i])
        pred_class = int(np.argmax(prob))
        predictions.append(pred_class)

    predictions = np.array(predictions)
    y_true = y_val.astype(int)

    # Accuracy
    accuracy = float(np.mean(predictions == y_true) * 100.0)

    # Class 2 (Critical) recall = TP / (TP + FN)
    crit_mask = (y_true == 2)
    if np.sum(crit_mask) > 0:
        crit_tp = np.sum((predictions == 2) & crit_mask)
        crit_total = np.sum(crit_mask)
        crit_recall = float((crit_tp / crit_total) * 100.0)
    else:
        crit_recall = 100.0

    return accuracy, crit_recall


def benchmark_model(model_name, model_path, X_val, y_val):
    """Benchmarks a single model across all five assignment metrics."""
    print(f"\n[BENCH] Benchmarking {model_name}...")
    file_size_kb = os.path.getsize(model_path) / 1024.0

    interpreter, in_details, out_details = load_tflite_interpreter(model_path)

    # 1. Accuracy & Critical Recall
    acc, crit_recall = evaluate_accuracy_and_recall(interpreter, in_details, out_details, X_val, y_val)

    # 2. Warm-up runs (excluded from timing)
    sample = X_val[0]
    for _ in range(WARMUP_RUNS):
        run_single_inference(interpreter, in_details, out_details, sample)

    # 3. 200 Timed Inferences + CPU% measurement
    latencies = []
    psutil.cpu_percent(interval=None)  # Reset CPU measurement window
    cpu_measurements = []

    for i in range(BENCH_RUNS):
        s_idx = i % len(X_val)
        t0 = time.perf_counter()
        run_single_inference(interpreter, in_details, out_details, X_val[s_idx])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

        if i % 40 == 0:
            cpu_measurements.append(psutil.cpu_percent(interval=None))

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    min_latency = float(np.min(latencies))
    max_latency = float(np.max(latencies))

    # Average CPU % during benchmark
    avg_cpu = float(np.mean(cpu_measurements)) if cpu_measurements else psutil.cpu_percent(interval=None)
    avg_cpu = max(avg_cpu, 2.0)  # floor at 2% to avoid zero on fast M-series burst

    # 4. Energy per inference: E = P * t
    # P (Watts) = (cpu_percent / 100) * TDP
    # E (mJ) = P * (mean_latency_ms / 1000) * 1000 = P * mean_latency_ms
    power_laptop_w = (avg_cpu / 100.0) * LAPTOP_TDP_WATTS
    energy_laptop_mj = float(power_laptop_w * mean_latency)

    power_pi5_w = (avg_cpu / 100.0) * PI5_TDP_WATTS
    energy_pi5_mj = float(power_pi5_w * mean_latency)

    return {
        "file_size_kb": round(file_size_kb, 2),
        "mean_latency_ms": round(mean_latency, 3),
        "p95_latency_ms": round(p95_latency, 3),
        "min_latency_ms": round(min_latency, 3),
        "max_latency_ms": round(max_latency, 3),
        "accuracy_pct": round(acc, 2),
        "critical_recall_pct": round(crit_recall, 2),
        "avg_cpu_pct": round(avg_cpu, 1),
        "energy_laptop_mj": round(energy_laptop_mj, 4),
        "energy_pi5_mj": round(energy_pi5_mj, 4),
    }


def main():
    print("=" * 70)
    print("  LogiEdge — Five-Metric Model Benchmarking Suite")
    print(f"  Warm-up: {WARMUP_RUNS} runs | Timed: {BENCH_RUNS} runs")
    print(f"  TDP reference: Laptop={LAPTOP_TDP_WATTS}W, Raspberry Pi 5={PI5_TDP_WATTS}W")
    print("=" * 70)

    if not os.path.exists(VAL_X_PATH) or not os.path.exists(VAL_Y_PATH):
        print(f"[ERROR] Validation files not found in {os.path.dirname(VAL_X_PATH)}")
        sys.exit(1)

    X_val = np.load(VAL_X_PATH)
    y_val = np.load(VAL_Y_PATH)
    print(f"[DATA] Loaded {len(X_val)} validation samples.")

    results = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[WARN] Skipping {name}: {path} not found.")
            continue
        results[name] = benchmark_model(name, path, X_val, y_val)

    # ── Comparison Table ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Model Variant':<24} | {'Size (KB)':<9} | {'Mean (ms)':<9} | {'p95 (ms)':<9} | {'Acc (%)':<8} | {'Crit Rec (%)':<12} | {'Energy (mJ)':<11}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<24} | {r['file_size_kb']:<9.1f} | {r['mean_latency_ms']:<9.3f} | {r['p95_latency_ms']:<9.3f} | {r['accuracy_pct']:<8.2f} | {r['critical_recall_pct']:<12.1f} | {r['energy_laptop_mj']:<11.4f}")
    print("=" * 80)

    # ── Check SLA and Critical Recall ─────────────────────────────────────────
    print("\n[VERIFICATION AGAINST REQUIREMENTS]")
    sla_seconds = 90.0  # Cold-chain SLA is 90 seconds
    for name, r in results.items():
        sla_pass = "✅ PASS" if (r['mean_latency_ms'] / 1000.0) < sla_seconds else "❌ FAIL"
        recall_pass = "✅ PASS" if r['critical_recall_pct'] >= 95.0 else "❌ FAIL"
        print(f"  {name}:")
        print(f"    - 90s SLA Budget : {r['mean_latency_ms']:.3f} ms vs 90,000 ms -> {sla_pass}")
        print(f"    - Critical Recall: {r['critical_recall_pct']:.1f}% (required >= 95%) -> {recall_pass}")

    # Save to JSON
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] Benchmark results saved to {RESULTS_PATH} ✅")


if __name__ == "__main__":
    main()