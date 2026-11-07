"""
LogiEdge TFLite Conversion — Component D (Task D1) + Component F (M1, M2)
Converts the trained Keras model into two TFLite variants:

  M1 — FP32 Baseline : no quantisation, full float32
  M2 — PTQ INT8      : Post-Training Quantisation with representative dataset
                       Uses ≥200 calibration samples (assignment requirement)

Output files (saved to training/models/):
  m1_fp32.tflite          — FP32 baseline
  m2_ptq_int8.tflite      — PTQ INT8 (this is the deployment model)

The m2_ptq_int8.tflite is also copied to inference/model.tflite
for use by the Docker inference service.
"""

import numpy as np
import tensorflow as tf
import os
import shutil

# ── Paths ──────────────────────────────────────────────────────────────────────
SAVED_MODEL_PATH = "training/models/logibridge_saved_model"
KERAS_MODEL_PATH = "training/models/logibridge_model.keras"
X_TRAIN_PATH     = "training/X_train.npy"

M1_OUTPUT = "training/models/m1_fp32.tflite"
M2_OUTPUT = "training/models/m2_ptq_int8.tflite"
INFERENCE_MODEL = "inference/model.tflite"

CALIBRATION_SAMPLES = 200   # assignment requires ≥200


# ══════════════════════════════════════════════════════════════════════════════
# REPRESENTATIVE DATASET (required for INT8 calibration)
# ══════════════════════════════════════════════════════════════════════════════

def make_representative_dataset(X_calib: np.ndarray):
    """
    Generator function that yields calibration samples.
    TFLite converter uses these to determine the scale/zero-point
    for each layer when quantising to INT8.
    Must yield at least 200 samples (assignment requirement).
    """
    def representative_dataset():
        for i in range(len(X_calib)):
            # Shape must be (1, 6) — batch size of 1
            sample = X_calib[i:i+1].astype(np.float32)
            yield [sample]
    return representative_dataset


# ══════════════════════════════════════════════════════════════════════════════
# M1 — FP32 BASELINE CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def convert_fp32(saved_model_path: str, output_path: str):
    """
    Convert SavedModel to TFLite with NO quantisation.
    This is the M1 FP32 baseline variant.
    """
    print("\n[M1] Converting to FP32 TFLite (no quantisation)...")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    # No optimisations — pure float32
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[M1] Saved to {output_path}")
    print(f"[M1] File size: {size_kb:.1f} KB")
    return size_kb


# ══════════════════════════════════════════════════════════════════════════════
# M2 — PTQ INT8 CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def convert_ptq_int8(saved_model_path: str,
                     output_path: str,
                     X_train: np.ndarray):
    """
    Convert SavedModel to fully quantised INT8 TFLite.
    Uses representative dataset for calibration (≥200 samples).
    This is the M2 PTQ INT8 variant — the deployment model.
    """
    print("\n[M2] Converting to PTQ INT8 TFLite...")
    print(f"     Using {CALIBRATION_SAMPLES} calibration samples")

    # Select calibration samples from training data
    # If we have fewer than 200, repeat to meet the requirement
    if len(X_train) >= CALIBRATION_SAMPLES:
        X_calib = X_train[:CALIBRATION_SAMPLES]
    else:
        # Tile to reach 200 samples
        repeats = (CALIBRATION_SAMPLES // len(X_train)) + 1
        X_calib = np.tile(X_train, (repeats, 1))[:CALIBRATION_SAMPLES]

    print(f"     Calibration set shape: {X_calib.shape}")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

    # Enable DEFAULT optimisation (triggers quantisation)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Provide representative dataset for full INT8 calibration
    converter.representative_dataset = make_representative_dataset(X_calib)

    # Force full INT8 — both weights AND activations
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[M2] Saved to {output_path}")
    print(f"[M2] File size: {size_kb:.1f} KB")
    return size_kb


# ══════════════════════════════════════════════════════════════════════════════
# QUICK ACCURACY CHECK ON TFLITE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def check_tflite_accuracy(tflite_path: str,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           is_int8: bool = False):
    """
    Run inference on validation set using the TFLite interpreter
    and report accuracy.
    """
    print(f"\n[CHECK] Validating {tflite_path}...")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    for i in range(len(X_val)):
        sample = X_val[i:i+1].astype(np.float32)

        if is_int8:
            # Scale float input to INT8 range
            input_scale, input_zero_point = input_details[0]["quantization"]
            if input_scale != 0:
                sample = sample / input_scale + input_zero_point
            sample = sample.astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        if is_int8:
            pred = int(np.argmax(output[0]))
        else:
            pred = int(np.argmax(output[0]))

        if pred == y_val[i]:
            correct += 1

    accuracy = correct / len(y_val)
    print(f"[CHECK] Accuracy: {accuracy * 100:.2f}% ({correct}/{len(y_val)})")
    return accuracy


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  LogiEdge TFLite Conversion (M1: FP32 | M2: PTQ INT8)")
    print("=" * 60)

    # Load training data for calibration
    print("\n[LOAD] Loading training data for calibration...")
    X_train = np.load(X_TRAIN_PATH).astype(np.float32)
    X_val   = np.load("training/X_val.npy").astype(np.float32)
    y_val   = np.load("training/y_val.npy")
    print(f"       X_train: {X_train.shape}")

    os.makedirs("training/models", exist_ok=True)
    os.makedirs("inference", exist_ok=True)

    # ── M1: FP32 ──────────────────────────────────────────────────────────
    m1_size = convert_fp32(SAVED_MODEL_PATH, M1_OUTPUT)
    check_tflite_accuracy(M1_OUTPUT, X_val, y_val, is_int8=False)

    # ── M2: PTQ INT8 ──────────────────────────────────────────────────────
    m2_size = convert_ptq_int8(SAVED_MODEL_PATH, M2_OUTPUT, X_train)
    check_tflite_accuracy(M2_OUTPUT, X_val, y_val, is_int8=True)

    # ── Copy M2 to inference/ folder (used by Docker service) ─────────────
    shutil.copy(M2_OUTPUT, INFERENCE_MODEL)
    print(f"\n[COPY] Deployment model copied to {INFERENCE_MODEL}")

    # ── Size comparison summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Conversion Summary")
    print("=" * 60)
    print(f"  M1 FP32 size  : {m1_size:.1f} KB")
    print(f"  M2 INT8 size  : {m2_size:.1f} KB")
    if m1_size > 0:
        reduction = (1 - m2_size / m1_size) * 100
        print(f"  Size reduction: {reduction:.1f}%")
    print(f"\n  Deployment model → {INFERENCE_MODEL}")
    print("\n[DONE] Conversion complete! ✅")
    print("       Next step: run prune_quantise.py for M3 variant")


if __name__ == "__main__":
    main()