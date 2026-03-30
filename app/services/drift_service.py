import pandas as pd
import os
import logging
from glob import glob

from app.config.settings import DRIFT_THRESHOLD

logger = logging.getLogger(__name__)


# -------------------------
# 🔥 Resolve latest feature file
# -------------------------
def _get_latest_features_file(session_id: str):
    base_path = os.path.join("data", "features", session_id)

    if not os.path.exists(base_path):
        logger.error(f"❌ Session folder not found: {base_path}")
        return None

    files = glob(os.path.join(base_path, "*.parquet"))

    if not files:
        logger.error(f"❌ No parquet files found in: {base_path}")
        return None

    latest_file = max(files, key=os.path.getmtime)
    return latest_file


# -------------------------
# 🔥 Production Drift Detection
# -------------------------
def detect_drift(session_id: str, threshold=DRIFT_THRESHOLD):
    logger.info(f"🔍 Drift check → session={session_id}")

    # -------------------------
    # Resolve file path
    # -------------------------
    features_path = _get_latest_features_file(session_id)

    if features_path is None:
        return {
            "drift_detected": False,
            "score": 0.0,
            "reason": "no_features_file"
        }

    logger.info(f"📂 Using file: {features_path}")

    # -------------------------
    # Load file safely
    # -------------------------
    try:
        df = pd.read_parquet(features_path)
    except Exception as e:
        logger.error(f"❌ Failed to read file: {e}")
        return {
            "drift_detected": False,
            "score": 0.0,
            "reason": "read_error"
        }

    # -------------------------
    # Column validation
    # -------------------------
    required_cols = ["label", "predicted_class"]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.error(f"❌ Missing columns: {missing_cols}")
        return {
            "drift_detected": False,
            "score": 0.0,
            "reason": "missing_columns"
        }

    # -------------------------
    # Clean data
    # -------------------------
    df = df.dropna(subset=required_cols)

    total = len(df)

    if total == 0:
        logger.error("❌ Empty dataset after cleaning")
        return {
            "drift_detected": False,
            "score": 0.0,
            "reason": "empty_data"
        }

    # -------------------------
    # Drift calculation (UNCHANGED LOGIC)
    # -------------------------
    mismatches = (df["label"] != df["predicted_class"]).sum()
    drift_score = mismatches / total

    logger.info({
        "threshold": threshold,
        "total_samples": total,
        "mismatches": int(mismatches),
        "drift_score": round(drift_score, 4)
    })

    # -------------------------
    # Decision
    # -------------------------
    drift_detected = drift_score > threshold

    if drift_detected:
        logger.warning("🚨 DRIFT DETECTED!")
    else:
        logger.info("✅ No significant drift")

    # -------------------------
    # Final result (MATCH predict.py)
    # -------------------------
    return {
        "drift_detected": bool(drift_detected),
        "score": float(round(drift_score, 4)),
        "total_samples": int(total),
        "mismatches": int(mismatches)
    }