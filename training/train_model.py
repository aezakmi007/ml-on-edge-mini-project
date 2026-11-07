"""
LogiEdge Model Training — Component D, Task D1
Trains a 2-hidden-layer MLP classifier on the LogiEdge dataset.
Architecture: Input(6) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(3, Softmax)
Target: >88% validation accuracy
Saves the trained model as training/models/logibridge_model (Keras SavedModel)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
X_TRAIN_PATH = "training/X_train.npy"
Y_TRAIN_PATH = "training/y_train.npy"
X_VAL_PATH   = "training/X_val.npy"
Y_VAL_PATH   = "training/y_val.npy"
MODEL_SAVE    = "training/models/logibridge_model"
PLOT_PATH     = "training/models/training_history.png"

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS        = 100
BATCH_SIZE    = 16
LEARNING_RATE = 0.001
NUM_CLASSES   = 3
INPUT_FEATURES = 6


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("[DATA] Loading dataset...")
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    X_val   = np.load(X_VAL_PATH)
    y_val   = np.load(Y_VAL_PATH)

    print(f"       X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"       X_val:   {X_val.shape}   | y_val:   {y_val.shape}")
    print(f"       Classes  train → 0:{np.sum(y_train==0)} "
          f"1:{np.sum(y_train==1)} 2:{np.sum(y_train==2)}")
    print(f"       Classes  val   → 0:{np.sum(y_val==0)} "
          f"1:{np.sum(y_val==1)} 2:{np.sum(y_val==2)}")

    return X_train, y_train, X_val, y_val


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(input_dim: int, num_classes: int) -> keras.Model:
    """
    2-hidden-layer MLP as specified in the assignment.
    Architecture: 6 → 32 (ReLU) → 16 (ReLU) → 3 (Softmax)
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,), name="input"),
        keras.layers.Dense(32, activation="relu", name="hidden_1"),
        keras.layers.Dropout(0.2, name="dropout_1"),
        keras.layers.Dense(16, activation="relu", name="hidden_2"),
        keras.layers.Dropout(0.2, name="dropout_2"),
        keras.layers.Dense(num_classes, activation="softmax", name="output")
    ], name="logibridge_mlp")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(model, X_train, y_train, X_val, y_val):
    print("\n[TRAIN] Starting training...")
    print(f"        Epochs={EPOCHS} | Batch={BATCH_SIZE} | LR={LEARNING_RATE}")
    print(f"        Architecture: 6 → 32 (ReLU) → 16 (ReLU) → 3 (Softmax)\n")

    # Early stopping: stop if val_accuracy doesn't improve for 15 epochs
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    return history


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_val, y_val):
    print("\n[EVAL] Evaluating on validation set...")
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"       Validation Loss:     {loss:.4f}")
    print(f"       Validation Accuracy: {accuracy * 100:.2f}%")

    # Per-class metrics
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    class_names = ["Normal (0)", "Warning (1)", "Critical (2)"]

    print("\n[EVAL] Per-class Recall (important for safety):")
    for cls in range(3):
        mask    = y_val == cls
        correct = np.sum(y_pred[mask] == cls)
        total   = np.sum(mask)
        recall  = correct / total if total > 0 else 0.0
        flag    = " ⚠️ BELOW 95%" if cls == 2 and recall < 0.95 else ""
        print(f"       Class {cls} ({class_names[cls]}): "
              f"{correct}/{total} = {recall * 100:.1f}% recall{flag}")

    # Assignment threshold check
    if accuracy >= 0.88:
        print(f"\n✅ PASS: Validation accuracy {accuracy*100:.2f}% >= 88% threshold")
    else:
        print(f"\n❌ FAIL: Accuracy {accuracy*100:.2f}% is below 88%!")
        print("   → Try increasing EPOCHS or adjusting architecture.")

    return accuracy, y_pred


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLOT TRAINING HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"],     label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.axhline(y=0.88, color="red", linestyle="--", label="88% threshold")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history["loss"],     label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"\n[PLOT] Training history saved to {PLOT_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model):
    os.makedirs("training/models", exist_ok=True)
    # Save in native Keras format (for reloading/retraining)
    keras_path = "training/models/logibridge_model.keras"
    model.save(keras_path)
    print(f"[SAVE] Keras model saved to {keras_path}")
    # Export as SavedModel format (required for TFLite conversion)
    saved_model_path = "training/models/logibridge_saved_model"
    model.export(saved_model_path)
    print(f"[SAVE] SavedModel exported to {saved_model_path}/")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  LogiEdge MLP Model Training")
    print("=" * 60)

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Load data
    X_train, y_train, X_val, y_val = load_data()

    # Build model
    model = build_model(INPUT_FEATURES, NUM_CLASSES)
    print("\n[MODEL] Architecture:")
    model.summary()

    # Train
    history = train(model, X_train, y_train, X_val, y_val)

    # Evaluate
    accuracy, y_pred = evaluate(model, X_val, y_val)

    # Plot
    plot_history(history)

    # Save
    save_model(model)

    print("\n[DONE] Training complete! ✅")
    print(f"       Model saved to: {MODEL_SAVE}/")
    print("       Next step: run convert_ptq.py to create TFLite model")


if __name__ == "__main__":
    main()