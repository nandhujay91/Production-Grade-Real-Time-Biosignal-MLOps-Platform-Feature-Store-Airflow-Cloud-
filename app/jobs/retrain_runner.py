import logging
import os
from datetime import datetime

from app.services.drift_service import check_drift
from app.services.retrain_service import retrain_job

# -------------------------
# Logger
# -------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_pipeline():
    try:
        logger.info("🚀 Retrain pipeline started")

        start_time = datetime.utcnow()

        # -------------------------
        # Step 1: Drift detection
        # -------------------------
        features_path = os.getenv(
            "FEATURES_PATH",
            "data/latest_features.parquet"
        )

        if not os.path.exists(features_path):
            logger.warning(f"⚠️ Features file not found: {features_path}")
            return {
                "status": "skipped",
                "reason": "features_not_found"
            }

        drift_result = check_drift(features_path)

        logger.info(f"📊 Drift result → {drift_result}")

        # -------------------------
        # Step 2: Retrain if needed
        # -------------------------
        result = retrain_job(drift_result)

        end_time = datetime.utcnow()

        logger.info(f"🔁 Retrain result → {result}")
        logger.info(f"⏱ Duration → {end_time - start_time}")

        return {
            "status": "completed",
            "drift": drift_result,
            "retrain": result,
            "start_time": str(start_time),
            "end_time": str(end_time)
        }

    except Exception as e:
        logger.error(f"❌ Retrain pipeline failed: {e}", exc_info=True)

        return {
            "status": "failed",
            "error": str(e)
        }


if __name__ == "__main__":
    run_pipeline()