from fastapi import APIRouter, HTTPException
import pandas as pd
import os
import logging

from app.services.session_service import get_files_by_session
from app.services.signal_processing import process_files
from app.services.stream_processing import process_stream
from app.services.label_service import generate_labels
from app.services.train_model import train_model
from app.services.validation_service import DataValidator

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/stream/{session_id}")
def stream_process(session_id: str):
    try:
        logger.info(f"🚀 STREAM API CALLED → session_id={session_id}")

        # -------------------------
        # 1️⃣ Get RAW files
        # -------------------------
        raw_files = get_files_by_session(session_id)

        # -------------------------
        # 2️⃣ Convert BIN → CSV
        # -------------------------
        processed_files = process_files(session_id, raw_files)

        # -------------------------
        # 3️⃣ Feature Engineering
        # -------------------------
        result = process_stream(session_id, processed_files)

        # -------------------------
        # 🔥 VALIDATION
        # -------------------------
        features_path = result["features_path"]

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        if features_path.endswith(".parquet"):
            df_features = pd.read_parquet(features_path)
        else:
            df_features = pd.read_csv(features_path)

        validator = DataValidator()
        validator.run_validation(df_features)

        # -------------------------
        # 4️⃣ Label generation
        # -------------------------
        labeled_path = generate_labels(session_id, features_path)

        # -------------------------
        # 🔥 CSV → PARQUET (safe)
        # -------------------------
        if labeled_path.endswith(".csv"):
            df = pd.read_csv(labeled_path)

            parquet_path = labeled_path.replace(".csv", ".parquet")
            df.to_parquet(parquet_path, index=False)

            try:
                os.remove(labeled_path)
            except Exception:
                logger.warning("⚠️ Failed to delete CSV after conversion")

        else:
            parquet_path = labeled_path

        result["labeled_features_path"] = parquet_path

        # -------------------------
        # 5️⃣ Auto training
        # -------------------------
        logger.info("🧠 Triggering auto-training...")
        train_model()

        logger.info(f"✅ STREAM COMPLETED → session_id={session_id}")

        return {
            "message": "Streaming + Feature extraction + Labeling + Training completed 🚀",
            **result
        }

    # -------------------------
    # Client errors
    # -------------------------
    except ValueError as ve:
        logger.warning(f"⚠️ Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    # -------------------------
    # Not found
    # -------------------------
    except FileNotFoundError as fe:
        logger.error(f"❌ File error: {fe}")
        raise HTTPException(status_code=404, detail=str(fe))

    # -------------------------
    # Internal errors
    # -------------------------
    except Exception as e:
        logger.error(f"❌ Stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")