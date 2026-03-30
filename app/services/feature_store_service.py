from feast import FeatureStore
import logging
import numpy as np

logger = logging.getLogger(__name__)


# -------------------------
# Load FeatureStore ONCE
# -------------------------
def load_store():
    try:
        store = FeatureStore(
            repo_path="biosignal_feature_repo/feature_repo"
        )
        logger.info("✅ FeatureStore initialized (once)")
        return store
    except Exception as e:
        raise RuntimeError(f"Failed to initialize FeatureStore: {e}")


# Global store (production pattern)
store = load_store()


# -------------------------
# 🔧 Helper: Clean feature values
# -------------------------
def clean_feature(values, name):
    try:
        arr = np.array(values, dtype=float)

        # Remove NaN / inf
        arr = arr[~np.isnan(arr)]
        arr = arr[np.isfinite(arr)]

        if len(arr) == 0:
            logger.warning(f"⚠️ All values invalid for feature: {name}")
            return []

        return arr.tolist()

    except Exception as e:
        logger.error(f"❌ Cleaning failed for {name}: {e}")
        return []


# -------------------------
# Fetch online features
# -------------------------
def get_online_features(session_id: str):
    try:
        logger.info(f"📡 Fetching features for session: {session_id}")

        entity_rows = [{"session_id": session_id}]

        feature_refs = [
            "biosignal_features:bpm",
            "biosignal_features:spo2",
            "biosignal_features:imu_x_mean",
            "biosignal_features:imu_y_mean",
            "biosignal_features:imu_z_mean",
        ]

        response = store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        ).to_dict()

        # -------------------------
        # Validate response
        # -------------------------
        if not response:
            logger.warning("⚠️ FeatureStore returned empty response")
            return None

        if "bpm" not in response:
            logger.warning("⚠️ Missing 'bpm' in FeatureStore response")
            return None

        # -------------------------
        # 🔥 CLEAN FEATURES (CRITICAL FIX)
        # -------------------------
        cleaned = {
            "bpm": clean_feature(response.get("bpm", []), "bpm"),
            "spo2": clean_feature(response.get("spo2", []), "spo2"),
            "imu_x_mean": clean_feature(response.get("imu_x_mean", []), "imu_x_mean"),
            "imu_y_mean": clean_feature(response.get("imu_y_mean", []), "imu_y_mean"),
            "imu_z_mean": clean_feature(response.get("imu_z_mean", []), "imu_z_mean"),
        }

        # -------------------------
        # Final safety check
        # -------------------------
        if len(cleaned["bpm"]) == 0:
            logger.warning("⚠️ No valid BPM values after cleaning")
            return None

        logger.info(
            f"✅ Clean features ready → "
            f"BPM count={len(cleaned['bpm'])}, "
            f"SpO2 count={len(cleaned['spo2'])}"
        )

        return cleaned

    except Exception as e:
        logger.error(f"❌ FeatureStore fetch failed: {e}", exc_info=True)
        return None