import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from feast import FeatureStore
from feast.data_source import PushMode

from app.utils.signal_utils import (
    bandpass_filter,
    compute_bpm,
    extract_spo2,
    check_quality
)

logger = logging.getLogger(__name__)

# ===============================
# CONFIG
# ===============================
SAMPLING_RATE_EPHY = 256
SAMPLING_RATE_IMU = 64
SAMPLING_RATE_OXYM = 128

WINDOW_SECONDS = 10
STEP_SECONDS = 5

WINDOW_SIZE = SAMPLING_RATE_EPHY * WINDOW_SECONDS
STEP_SIZE = SAMPLING_RATE_EPHY * STEP_SECONDS

FEATURES_DIR = "data/features"

# ===============================
# 🔥 GLOBAL FEATURE STORE (PRODUCTION FIX)
# ===============================
try:
    store = FeatureStore(repo_path="biosignal_feature_repo/feature_repo")
    logger.info("✅ Feast FeatureStore initialized (once)")
except Exception as e:
    logger.error(f"❌ Failed to initialize FeatureStore: {e}")
    store = None


# ===============================
# 🔥 Feast Push (PRODUCTION SAFE)
# ===============================
def push_to_feast(df: pd.DataFrame):
    if store is None:
        logger.error("❌ FeatureStore not initialized")
        return

    try:
        required_cols = [
            "session_id",
            "event_timestamp",
            "bpm",
            "spo2",
            "imu_x_mean",
            "imu_y_mean",
            "imu_z_mean"
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for Feast push: {missing}")

        # Ensure correct types
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
        df["session_id"] = df["session_id"].astype(str)

        logger.info(f"🚀 Pushing {len(df)} rows to Feast")

        store.push(
            "biosignal_push_source",
            df,
            to=PushMode.ONLINE
        )

        logger.info("✅ Features pushed to Feast ONLINE store")

    except Exception as e:
        logger.error(f"❌ Feast push failed: {e}", exc_info=True)


# ===============================
# MAIN FUNCTION
# ===============================
def process_stream(session_id: str, processed_files: list):

    logger.info(f"🚀 STREAM STARTED → session={session_id}")

    session_dir = os.path.join(FEATURES_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    ecg, imu, oxym = None, None, None

    # -------------------------
    # Load signals
    # -------------------------
    for f in processed_files:
        try:
            file_type = f.get("type")
            file_path = f.get("output")

            if file_type == "Ephy":
                ecg = pd.read_csv(file_path).values[:, 0]

            elif file_type == "IMU":
                imu = pd.read_csv(file_path).values[:, :3]

            elif file_type == "Oxym":
                oxym = pd.read_csv(file_path).values[:, 0]

        except Exception as e:
            logger.error(f"❌ Failed loading file {file_path}: {e}", exc_info=True)
            raise

    # -------------------------
    # Validation
    # -------------------------
    if ecg is None or imu is None or oxym is None:
        raise ValueError("Missing required signals (Ephy/IMU/Oxym)")

    if len(ecg) < WINDOW_SIZE:
        raise ValueError("ECG signal too short")

    start = 0
    window_id = 1
    results = []

    base_time = pd.Timestamp.utcnow()

    # -------------------------
    # Streaming loop
    # -------------------------
    while start + WINDOW_SIZE <= len(ecg):

        end = start + WINDOW_SIZE
        ecg_window = ecg[start:end]

        try:
            filtered = bandpass_filter(ecg_window, fs=SAMPLING_RATE_EPHY)
            bpm = float(compute_bpm(filtered, fs=SAMPLING_RATE_EPHY))
            quality = check_quality(filtered)
        except Exception as e:
            logger.warning(f"⚠️ ECG processing failed (window={window_id}): {e}")
            bpm = np.nan
            quality = "Unknown"

        # -------------------------
        # IMU
        # -------------------------
        imu_ratio = SAMPLING_RATE_IMU / SAMPLING_RATE_EPHY
        imu_start = int(start * imu_ratio)
        imu_end = int(end * imu_ratio)

        imu_window = imu[max(0, imu_start):min(len(imu), imu_end)]

        if imu_window is not None and len(imu_window) > 0:
            imu_x_mean = float(np.mean(imu_window[:, 0]))
            imu_y_mean = float(np.mean(imu_window[:, 1]))
            imu_z_mean = float(np.mean(imu_window[:, 2]))
        else:
            imu_x_mean = imu_y_mean = imu_z_mean = 0.0

        # -------------------------
        # SpO2
        # -------------------------
        ox_ratio = SAMPLING_RATE_OXYM / SAMPLING_RATE_EPHY
        ox_start = int(start * ox_ratio)
        ox_end = int(end * ox_ratio)

        ox_window = oxym[max(0, ox_start):min(len(oxym), ox_end)]

        try:
            spo2 = float(extract_spo2(ox_window)) if len(ox_window) > 0 else np.nan
        except Exception as e:
            logger.warning(f"⚠️ SpO2 extraction failed (window={window_id}): {e}")
            spo2 = np.nan

        # -------------------------
        # Validation (UNCHANGED)
        # -------------------------
        if bpm == 0 or np.isnan(bpm) or bpm < 20 or bpm > 220:
            start += STEP_SIZE
            window_id += 1
            continue

        event_time = base_time + pd.Timedelta(seconds=window_id * STEP_SECONDS)

        results.append({
            "session_id": session_id,
            "event_timestamp": event_time,
            "window_id": window_id,
            "bpm": bpm,
            "spo2": spo2,
            "imu_x_mean": imu_x_mean,
            "imu_y_mean": imu_y_mean,
            "imu_z_mean": imu_z_mean,
            "signal_quality": quality
        })

        start += STEP_SIZE
        window_id += 1

    # -------------------------
    # DataFrame
    # -------------------------
    df = pd.DataFrame(results)

    if df.empty:
        raise ValueError("No valid windows generated")

    df = df.dropna()

    # -------------------------
    # Save
    # -------------------------
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(session_dir, f"features_{timestamp}.parquet")

    df.to_parquet(output_path, index=False)

    logger.info(f"✅ FEATURES SAVED → {output_path}")
    logger.info(f"📊 Total valid windows: {len(df)}")

    # -------------------------
    # 🔥 REAL-TIME PUSH
    # -------------------------
    push_to_feast(df)

    return {
        "session_id": session_id,
        "features_path": output_path,
        "total_windows": len(df)
    }