"""
LogiEdge Model Pruning + Quantisation — Component F (Model Variant M3)
Applies 35% structured sparsity pruning using PolynomialDecay schedule,
then converts the pruned model to Full INT8 TFLite via PTQ.

Fix: Rebuilds model using tf.keras (required for tfmot compatibility).
"""

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
X_TRAIN_PATH    = "training/X_train.npy"
Y_TRAIN_PATH    = "training/y_train.npy"
X_VAL_PATH      = "training/X_val.npy"
Y_VAL_PATH      = "training/y_val.npy"
M3_SAVED_MODEL  = "training/models/m3_pruned_saved_model"
M3_OUTPUT       = "training/models/m3_pruned_int8.tflite"

# ── Pruning Configuration ──────────────────────────────────────────────────────
TARGET_SPARSITY     = 0.35
PRUNE_EPOCHS        = 20
BATCH_SIZE          = 16
CALIBRATION_SAMPLES = 200


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    X_train = np.load(X_TRAIN_PATH).astype(np.float32)
    y_train = np.load(Y_TRAIN_PATH)
    X_val   = np.load(X_VAL_PATH).astype(np.float32)
    y_val   = np.load(Y_VAL_PATH)
    print(f"[DATA] Train: {X_train.shape} | Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD BASE MODEL USING tf.keras (required for tfmot compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def build_and_train_base_model(X_train, y_train, X_val, y_val):
    """
    Rebuild the same MLP architecture using tf.keras (Keras 2 API).
    tensorflow_model_optimization requires tf.keras, not standalone Keras 3.
    Re-trains from scratch — takes < 15 seconds.
    """
    print("\n[BUILD] Rebuilding model with tf.keras for tfmot compatibility...")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu",
                              input_shape=(6,), name="hidden_1"),
        tf.keras.layers.Dropout(0.2, name="dropout_1"),
        tf.keras.layers.Dense(16, activation="relu", name="hidden_2"),
        tf.keras.layers.Dropout(0.2, name="dropout_2"),
        tf.keras.layers.Dense(3, activation="softmax", name="output")
    ], name="logibridge_mlp")

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=0
    )

    print("[BUILD] Pre-training base model (fast)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=BATCH_SIZE,
        callbacks=[early_stop], verbose=0
    )

    _, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"[BUILD] Base model accuracy: {acc * 100:.2f}%")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3. APPLY PRUNING WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def apply_pruning(base_model, n_train_samples: int):
    steps_per_epoch = n_train_samples // BATCH_SIZE
    end_step        = steps_per_epoch * PRUNE_EPOCHS

    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=TARGET_SPARSITY,
        begin_step=0,
        end_step=end_step,
        frequency=steps_per_epoch
    )

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        base_model, pruning_schedule=pruning_schedule
    )

    model_for_pruning.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\n[PRUNE] Target sparsity : {TARGET_SPARSITY * 100:.0f}%")
    print(f"[PRUNE] Schedule        : PolynomialDecay (0% → 35%)")
    print(f"[PRUNE] End step        : {end_step}")
    return model_for_pruning


# ══════════════════════════════════════════════════════════════════════════════
# 4. FINE-TUNE WITH PRUNING
# ══════════════════════════════════════════════════════════════════════════════

def fine_tune_with_pruning(model_for_pruning, X_train, y_train, X_val, y_val):
    print(f"\n[PRUNE] Fine-tuning for {PRUNE_EPOCHS} epochs with pruning active...")

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        # PruningSummaries removed — requires TensorBoard
    ]

    model_for_pruning.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=PRUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. STRIP PRUNING + EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def strip_and_evaluate(model_for_pruning, X_val, y_val):
    print("\n[PRUNE] Stripping pruning wrappers...")
    model_stripped = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    model_stripped.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    _, accuracy = model_stripped.evaluate(X_val, y_val, verbose=0)
    print(f"[PRUNE] Pruned model accuracy: {accuracy * 100:.2f}%")

    # Count actual sparsity
    total, zeros = 0, 0
    for layer in model_stripped.layers:
        for w in layer.weights:
            arr = w.numpy()
            total += arr.size
            zeros += np.sum(arr == 0)

    sparsity = zeros / total if total > 0 else 0
    print(f"[PRUNE] Actual sparsity: {sparsity * 100:.1f}% ({zeros}/{total} zeros)")
    return model_stripped, accuracy


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXPORT + QUANTISE TO INT8 TFLITE
# ══════════════════════════════════════════════════════════════════════════════

def export_and_quantise(model_stripped, X_train: np.ndarray):
    print("\n[M3] Saving pruned model as SavedModel...")
    tf.saved_model.save(model_stripped, M3_SAVED_MODEL)
    print(f"[M3] Exported to {M3_SAVED_MODEL}/")

    print("[M3] Applying PTQ INT8...")

    if len(X_train) >= CALIBRATION_SAMPLES:
        X_calib = X_train[:CALIBRATION_SAMPLES]
    else:
        repeats = (CALIBRATION_SAMPLES // len(X_train)) + 1
        X_calib = np.tile(X_train, (repeats, 1))[:CALIBRATION_SAMPLES]

    def representative_dataset():
        for i in range(len(X_calib)):
            yield [X_calib[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(M3_SAVED_MODEL)
    converter.optimizations             = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset    = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type      = tf.int8
    converter.inference_output_type     = tf.int8

    tflite_model = converter.convert()

    with open(M3_OUTPUT, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(M3_OUTPUT) / 1024
    print(f"[M3] Saved to {M3_OUTPUT}")
    print(f"[M3] File size: {size_kb:.1f} KB")
    return size_kb


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  LogiEdge M3: Structured Pruning (35%) + PTQ INT8")
    print("=" * 60)

    tf.random.set_seed(42)
    np.random.seed(42)

    X_train, y_train, X_val, y_val = load_data()

    # Build + pre-train base model using tf.keras (tfmot compatible)
    base_model = build_and_train_base_model(X_train, y_train, X_val, y_val)

    # Apply pruning wrapper
    model_for_pruning = apply_pruning(base_model, len(X_train))

    # Fine-tune with pruning
    fine_tune_with_pruning(model_for_pruning, X_train, y_train, X_val, y_val)

    # Strip + evaluate
    model_stripped, pruned_accuracy = strip_and_evaluate(model_for_pruning, X_val, y_val)

    # Export + INT8 quantise
    m3_size = export_and_quantise(model_stripped, X_train)

    # Summary
    m1_size = os.path.getsize("training/models/m1_fp32.tflite") / 1024
    m2_size = os.path.getsize("training/models/m2_ptq_int8.tflite") / 1024

    print("\n" + "=" * 60)
    print("  All Three Model Variants — Size Comparison")
    print("=" * 60)
    print(f"  M1 FP32 baseline        : {m1_size:.1f} KB")
    print(f"  M2 PTQ INT8             : {m2_size:.1f} KB")
    print(f"  M3 Pruned (35%) + INT8  : {m3_size:.1f} KB")
    print(f"  M3 vs M1 reduction      : {(1 - m3_size/m1_size)*100:.1f}%")

    status = "✅" if pruned_accuracy >= 0.88 else "⚠️ "
    print(f"\n{status} M3 accuracy: {pruned_accuracy*100:.2f}%")
    print("\n[DONE] M3 variant complete! ✅")


if __name__ == "__main__":
    main()