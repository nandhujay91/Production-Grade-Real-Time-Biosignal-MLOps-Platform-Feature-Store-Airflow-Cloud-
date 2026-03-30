import logging
import os
from datetime import datetime

from app.services.drift_service import check_drift

# -------------------------
# Logger
# -------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def monitor_drift():
    try:
        logger.info("🔍 Drift monitoring started")

        start_time = datetime.utcnow()

        # -------------------------
        # Configurable path
        # -------------------------
        features_path = os.getenv(
            "FEATURES_PATH",
            "data/latest_features.parquet"
        )

        # -------------------------
        # File check
        # -------------------------
        if not os.path.exists(features_path):
            logger.warning(f"⚠️ Features file not found: {features_path}")
            return {
                "status": "skipped",
                "reason": "features_not_found"
            }

        # -------------------------
        # Drift check
        # -------------------------
        drift_result = check_drift(features_path)

        logger.info(f"📊 Drift result → {drift_result}")

        # -------------------------
        # Alerting (basic)
        # -------------------------
        if drift_result["drift"]:
            logger.warning("🚨 DRIFT DETECTED → Model performance degraded")
        else:
            logger.info("✅ No drift detected")

        end_time = datetime.utcnow()

        return {
            "status": "completed",
            "drift": drift_result,
            "start_time": str(start_time),
            "end_time": str(end_time)
        }

    except Exception as e:
        logger.error(f"❌ Drift monitoring failed: {e}", exc_info=True)

        return {
            "status": "failed",
            "error": str(e)
        }


if __name__ == "__main__":
    monitor_drift()