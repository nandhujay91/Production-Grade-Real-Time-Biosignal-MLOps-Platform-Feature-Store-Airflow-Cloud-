import logging
import os
from datetime import datetime
import pandas as pd

from app.services.validation_service import DataValidator

# -------------------------
# Logger
# -------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_data_quality_check():
    try:
        logger.info("🧪 Data quality check started")

        start_time = datetime.utcnow()

        # -------------------------
        # Configurable path
        # -------------------------
        features_path = os.getenv(
            "FEATURES_PATH",
            "data/latest_features.parquet"
        )

        # -------------------------
        # File existence check
        # -------------------------
        if not os.path.exists(features_path):
            logger.warning(f"⚠️ Features file not found: {features_path}")
            return {
                "status": "skipped",
                "reason": "features_not_found"
            }

        # -------------------------
        # Load dataset
        # -------------------------
        if features_path.endswith(".parquet"):
            df = pd.read_parquet(features_path)
        else:
            df = pd.read_csv(features_path)

        logger.info(f"📊 Loaded dataset → rows={len(df)}")

        # -------------------------
        # Run validation
        # -------------------------
        validator = DataValidator()

        validator.validate_schema(df)
        validator.validate_nulls(df)
        validator.validate_ranges(df)

        logger.info("✅ Data quality check PASSED")

        end_time = datetime.utcnow()

        return {
            "status": "passed",
            "rows": len(df),
            "start_time": str(start_time),
            "end_time": str(end_time)
        }

    except ValueError as ve:
        logger.warning(f"⚠️ Data quality FAILED: {ve}")

        return {
            "status": "failed",
            "error": str(ve)
        }

    except Exception as e:
        logger.error(f"❌ Data quality check crashed: {e}", exc_info=True)

        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    run_data_quality_check()