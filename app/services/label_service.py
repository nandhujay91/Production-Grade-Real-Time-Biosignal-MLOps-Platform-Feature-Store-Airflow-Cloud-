import pandas as pd
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


# -------------------------
# Label Logic (UNCHANGED)
# -------------------------
def assign_label(bpm, spo2):
    if pd.isna(bpm) or pd.isna(spo2):
        return "Normal"

    bpm = float(bpm)
    spo2 = float(spo2)

    if bpm < 55 or bpm > 110 or spo2 < 94:
        return "Critical"
    elif bpm < 60 or bpm > 100 or spo2 < 96:
        return "Alert"
    else:
        return "Normal"


# -------------------------
# Dynamic Label Logic (UNCHANGED)
# -------------------------
def assign_label_dynamic(bpm, spo2, bpm_base, spo2_base):
    if pd.isna(bpm) or pd.isna(spo2):
        return "Normal"

    bpm = float(bpm)
    spo2 = float(spo2)

    bpm_dev = abs(bpm - bpm_base)
    spo2_dev = abs(spo2_base - spo2)

    if bpm_dev > 25 or spo2_dev > 5:
        return "Critical"
    elif bpm_dev > 10 or spo2_dev > 2:
        return "Alert"
    else:
        return "Normal"


# -------------------------
# Generate Labeled Dataset
# -------------------------
def generate_labels(session_id, features_path):
    logger.info(f"🧠 Generating labels → {session_id}")

    # -------------------------
    # File validation
    # -------------------------
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")

    # -------------------------
    # Load file safely
    # -------------------------
    try:
        if features_path.endswith(".parquet"):
            df = pd.read_parquet(features_path)
        else:
            df = pd.read_csv(features_path)
    except Exception as e:
        logger.error(f"❌ Failed to read features file: {e}")
        raise

    # -------------------------
    # Handle column mismatch
    # -------------------------
    column_mapping = {
        "imu_x": "imu_x_mean",
        "imu_y": "imu_y_mean",
        "imu_z": "imu_z_mean"
    }

    df = df.rename(columns=column_mapping)

    # -------------------------
    # Validation
    # -------------------------
    required_cols = [
        "bpm",
        "spo2",
        "imu_x_mean",
        "imu_y_mean",
        "imu_z_mean"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Cleaning
    # -------------------------
    df = df.dropna(subset=required_cols)
    df = df[(df["bpm"] > 0) & (df["spo2"] > 0)]

    df["bpm"] = df["bpm"].clip(20, 220)
    df["spo2"] = df["spo2"].clip(70, 100)

    df["imu_x_mean"] = df["imu_x_mean"].clip(-5, 5)
    df["imu_y_mean"] = df["imu_y_mean"].clip(-5, 5)
    df["imu_z_mean"] = df["imu_z_mean"].clip(-5, 5)

    # -------------------------
    # Dynamic baseline
    # -------------------------
    bpm_base = df["bpm"].mean()
    spo2_base = df["spo2"].mean()

    logger.info(f"📊 Baseline → BPM={bpm_base:.2f}, SpO2={spo2_base:.2f}")

    # -------------------------
    # Labeling (UNCHANGED)
    # -------------------------
    df["label"] = df.apply(
        lambda row: assign_label_dynamic(
            row["bpm"],
            row["spo2"],
            bpm_base,
            spo2_base
        ),
        axis=1
    )

    # -------------------------
    # INDUSTRY FIX: Ensure Critical exists
    # -------------------------
    counts = df["label"].value_counts()

    if counts.get("Critical", 0) == 0:
        logger.warning("⚠️ No CRITICAL samples → Injecting synthetic edge cases")

        synthetic = []
        for _ in range(20):
            synthetic.append({
                "bpm": np.random.uniform(130, 180),
                "spo2": np.random.uniform(80, 90),
                "imu_x_mean": np.random.uniform(-2, 2),
                "imu_y_mean": np.random.uniform(-2, 2),
                "imu_z_mean": np.random.uniform(-2, 2),
                "label": "Critical"
            })

        df = pd.concat([df, pd.DataFrame(synthetic)], ignore_index=True)

    # -------------------------
    # Save
    # -------------------------
    output_dir = os.path.dirname(features_path)
    output_path = os.path.join(output_dir, "features_labeled.parquet")

    try:
        df.to_parquet(output_path, index=False)
        logger.info(f"✅ Labels saved → {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save labeled file: {e}")
        raise

    # -------------------------
    # Final Distribution
    # -------------------------
    final_counts = df["label"].value_counts()
    logger.info(f"📊 Final Class Distribution: {final_counts.to_dict()}")

    if final_counts.get("Alert", 0) == 0:
        logger.warning("⚠️ No ALERT samples generated!")

    return output_path