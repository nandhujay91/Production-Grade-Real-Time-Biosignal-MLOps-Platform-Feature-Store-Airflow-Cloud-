import logging
from datetime import datetime

from app.services.train_model import train_model

logger = logging.getLogger(__name__)


# -------------------------
# 🔥 API-friendly retrain trigger
# -------------------------
def retrain_model():
    """
    Triggered directly from API (predict.py)
    """
    logger.info("🔁 Retrain triggered from API")

    start_time = datetime.utcnow()

    try:
        train_model()

        end_time = datetime.utcnow()

        logger.info("✅ Retraining completed (API trigger)")
        logger.info("📦 Model updated in MLflow")

        return {
            "retrained": True,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "duration_seconds": (end_time - start_time).total_seconds()
        }

    except Exception as e:
        end_time = datetime.utcnow()

        logger.error(f"❌ Retraining failed: {e}", exc_info=True)

        return {
            "retrained": False,
            "error": str(e),
            "start_time": str(start_time),
            "end_time": str(end_time)
        }


# -------------------------
# 🔥 Scheduler / Batch retrain (UNCHANGED LOGIC)
# -------------------------
def retrain_job(drift_result: dict):
    """
    Triggered by:
    - scheduler / cron / airflow
    """

    logger.info("🔁 Retrain job started")

    start_time = datetime.utcnow()

    try:
        # -------------------------
        # Validate drift input (UPDATED KEY)
        # -------------------------
        if not isinstance(drift_result, dict) or "drift_detected" not in drift_result:
            logger.warning("⚠️ Invalid drift result input")

            return {
                "retrained": False,
                "reason": "invalid_drift_input",
                "timestamp": str(start_time)
            }

        drift_flag = bool(drift_result.get("drift_detected", False))

        # -------------------------
        # Drift detected → retrain
        # -------------------------
        if drift_flag:
            logger.warning("🚨 Drift detected → Retraining model...")

            train_model()

            end_time = datetime.utcnow()

            logger.info("✅ Retraining completed")
            logger.info("📦 New model trained and registered in MLflow")

            return {
                "retrained": True,
                "start_time": str(start_time),
                "end_time": str(end_time),
                "duration_seconds": (end_time - start_time).total_seconds()
            }

        # -------------------------
        # No drift
        # -------------------------
        else:
            logger.info("✅ No retraining needed")

            end_time = datetime.utcnow()

            return {
                "retrained": False,
                "reason": "no_drift",
                "start_time": str(start_time),
                "end_time": str(end_time)
            }

    except Exception as e:
        end_time = datetime.utcnow()

        logger.error(f"❌ Retraining failed: {e}", exc_info=True)

        return {
            "retrained": False,
            "error": str(e),
            "start_time": str(start_time),
            "end_time": str(end_time)
        }