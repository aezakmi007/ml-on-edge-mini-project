"""
LogiEdge Dataset Generator — Component D, Task D1
Generates labelled training data in BATCH mode (no real-time waiting).
Runs the same simulation + preprocessing logic internally, fast.

Dataset:
  Class 0 — Normal   : --anomaly none       20 min → ~120 windows
  Class 1 — Warning  : --anomaly temp_drift 15 min → ~90 windows
  Class 2 — Critical : --anomaly combined   15 min → ~90 windows

Also computes and saves training_stats.npy (mean + std per feature)
from the Normal class windows — used by the preprocessor at runtime.

Output files (saved to data_pipeline/):
  training_stats.npy   — normalisation stats (mean, std per feature)
Output files (saved to training/):
  X_train.npy          — feature matrix  shape: (N, 6)
  y_train.npy          — label vector    shape: (N,)
  X_val.npy            — validation features (20% split)
  y_val.npy            — validation labels   (20% split)
"""

import numpy as np
from scipy import stats as scipy_stats
from sklearn.model_selection import train_test_split
import os

# ── Simulation Parameters ──────────────────────────────────────────────────────
TEMP_SETPOINT   = 4.0
TEMP_STD        = 0.3
TEMP_DRIFT_RATE = 0.08    # °C per second

VIB_NORMAL_MEAN = 0.45
VIB_NORMAL_STD  = 0.05
VIB_FAULT_MEAN  = 1.20
VIB_FAULT_STD   = 0.15

TEMP_HZ        = 1        # readings per second
VIB_HZ         = 0.5     # readings per second

# ── Window Parameters ──────────────────────────────────────────────────────────
WINDOW_SECONDS = 30
STEP_SECONDS   = 10
FILTER_SIZE    = 5

# ── Class Configuration ────────────────────────────────────────────────────────
# (class_label, anomaly_mode, duration_minutes)
CLASS_CONFIG = [
    (0, "none",       20),   # Normal   → ~120 windows
    (1, "temp_drift", 15),   # Warning  → ~90 windows
    (2, "combined",   15),   # Critical → ~90 windows
]

FEATURE_NAMES = [
    "temp_mean", "temp_std", "temp_roc",
    "vib_rms",   "vib_peak", "vib_kurtosis"
]


# ══════════════════════════════════════════════════════════════════════════════
# BATCH SENSOR SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate_batch(anomaly_mode: str, duration_minutes: int) -> dict:
    """
    Generate synthetic sensor readings for a given anomaly mode.
    Returns dict with 'temp' and 'vib' arrays (time-series).
    """
    total_seconds = duration_minutes * 60
    n_temp = total_seconds * TEMP_HZ
    n_vib  = int(total_seconds * VIB_HZ)

    # ── Temperature readings ───────────────────────────────────────────────
    if anomaly_mode in ("temp_drift", "combined"):
        # Linear drift: starts at setpoint, rises 0.08°C per second
        drift = np.arange(n_temp) * TEMP_DRIFT_RATE
        temp  = np.random.normal(TEMP_SETPOINT, TEMP_STD, int(n_temp)) + drift
    else:
        temp = np.random.normal(TEMP_SETPOINT, TEMP_STD, int(n_temp))

    # ── Vibration readings ─────────────────────────────────────────────────
    if anomaly_mode in ("vibration", "combined"):
        vib = np.abs(np.random.normal(VIB_FAULT_MEAN, VIB_FAULT_STD, n_vib))
    else:
        vib = np.abs(np.random.normal(VIB_NORMAL_MEAN, VIB_NORMAL_STD, n_vib))

    return {"temp": temp.astype(np.float32),
            "vib":  vib.astype(np.float32)}


# ══════════════════════════════════════════════════════════════════════════════
# MOVING AVERAGE FILTER
# ══════════════════════════════════════════════════════════════════════════════

def moving_average(arr: np.ndarray, window: int = FILTER_SIZE) -> np.ndarray:
    """Apply a simple moving average filter to a 1-D array."""
    kernel = np.ones(window) / window
    # 'valid' mode: output length = len(arr) - window + 1
    # pad start with the first value so length is preserved
    padded = np.concatenate([np.full(window - 1, arr[0]), arr])
    return np.convolve(padded, kernel, mode="valid")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(temp_window: np.ndarray,
                     vib_window:  np.ndarray) -> np.ndarray:
    """Extract 6 features from one 30-second window."""
    temp_mean    = np.mean(temp_window)
    temp_std     = np.std(temp_window)
    temp_roc     = (temp_window[-1] - temp_window[0]) / (WINDOW_SECONDS / 60.0)
    vib_rms      = np.sqrt(np.mean(vib_window ** 2))
    vib_peak     = np.max(np.abs(vib_window))
    vib_kurtosis = float(scipy_stats.kurtosis(vib_window)) if len(vib_window) >= 4 else 0.0

    return np.array([temp_mean, temp_std, temp_roc,
                     vib_rms,   vib_peak, vib_kurtosis],
                    dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDING WINDOW FEATURE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_windows(sensor_data: dict) -> np.ndarray:
    """
    Apply moving average filter, then slide a 30-second window
    with 10-second step to extract feature vectors.
    Returns feature matrix of shape (n_windows, 6).
    """
    temp_filtered = moving_average(sensor_data["temp"])
    vib_filtered  = moving_average(sensor_data["vib"])

    temp_win_size = WINDOW_SECONDS * TEMP_HZ         # 30 samples
    vib_win_size  = int(WINDOW_SECONDS * VIB_HZ)     # 15 samples
    temp_step     = STEP_SECONDS * TEMP_HZ            # 10 samples
    vib_step      = int(STEP_SECONDS * VIB_HZ)       # 5 samples

    features = []
    max_steps = (len(temp_filtered) - temp_win_size) // temp_step

    for i in range(max_steps):
        t_start = i * temp_step
        t_end   = t_start + temp_win_size

        v_start = i * vib_step
        v_end   = v_start + vib_win_size

        if t_end > len(temp_filtered) or v_end > len(vib_filtered):
            break

        temp_window = temp_filtered[t_start:t_end]
        vib_window  = vib_filtered[v_start:v_end]

        feat = extract_features(temp_window, vib_window)
        features.append(feat)

    return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: GENERATE FULL DATASET
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  LogiEdge Dataset Generator")
    print("=" * 60)

    all_X = []
    all_y = []
    normal_features = None   # saved separately for training_stats

    for class_label, anomaly_mode, duration_min in CLASS_CONFIG:
        class_names = {0: "Normal", 1: "Warning", 2: "Critical"}
        print(f"\n[GEN] Class {class_label} ({class_names[class_label]}) "
              f"| mode={anomaly_mode} | {duration_min} min")

        # 1. Simulate batch sensor data
        sensor_data = simulate_batch(anomaly_mode, duration_min)

        # 2. Extract sliding window features
        features = generate_windows(sensor_data)
        labels   = np.full(len(features), class_label, dtype=np.int32)

        print(f"      Generated {len(features)} windows "
              f"| feature shape: {features.shape}")
        print(f"      temp_mean range: "
              f"[{features[:, 0].min():.2f}, {features[:, 0].max():.2f}] °C")

        # Save Normal class features for training_stats computation
        if class_label == 0:
            normal_features = features.copy()

        all_X.append(features)
        all_y.append(labels)

    # ── Combine all classes ────────────────────────────────────────────────
    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"\n[GEN] Total dataset: {X.shape[0]} windows × {X.shape[1]} features")
    print(f"      Class distribution: "
          f"0={np.sum(y==0)}, 1={np.sum(y==1)}, 2={np.sum(y==2)}")

    # ── Compute & save training_stats.npy from Normal windows ─────────────
    print("\n[STATS] Computing normalisation stats from Normal class windows...")
    norm_mean = normal_features.mean(axis=0)
    norm_std  = normal_features.std(axis=0)
    norm_std  = np.where(norm_std == 0, 1e-8, norm_std)   # avoid division by zero

    stats = {"mean": norm_mean, "std": norm_std}
    stats_path = "data_pipeline/training_stats.npy"
    np.save(stats_path, stats)
    print(f"[STATS] Saved to {stats_path}")
    for name, m, s in zip(FEATURE_NAMES, norm_mean, norm_std):
        print(f"        {name:15s}  mean={m:8.4f}  std={s:8.4f}")

    # ── Train / Validation split (80/20) ──────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[SPLIT] Train: {X_train.shape[0]} samples | "
          f"Val: {X_val.shape[0]} samples")

    # ── Normalise using training stats ────────────────────────────────────
    X_train_norm = (X_train - norm_mean) / norm_std
    X_val_norm   = (X_val   - norm_mean) / norm_std

    # ── Save datasets ─────────────────────────────────────────────────────
    os.makedirs("training", exist_ok=True)
    np.save("training/X_train.npy", X_train_norm)
    np.save("training/y_train.npy", y_train)
    np.save("training/X_val.npy",   X_val_norm)
    np.save("training/y_val.npy",   y_val)

    print("\n[SAVE] Files written:")
    print(f"       training/X_train.npy  shape={X_train_norm.shape}")
    print(f"       training/y_train.npy  shape={y_train.shape}")
    print(f"       training/X_val.npy    shape={X_val_norm.shape}")
    print(f"       training/y_val.npy    shape={y_val.shape}")
    print(f"       data_pipeline/training_stats.npy")

    print("\n[DONE] Dataset generation complete! ✅")
    print("       Next step: run train_model.py")


if __name__ == "__main__":
    main()