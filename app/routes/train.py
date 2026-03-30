from fastapi import APIRouter, HTTPException
import logging
from datetime import datetime

from app.services.train_model import train_model

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/train")
def train():
    try:
        start_time = datetime.utcnow()
        logger.info("🚀 TRAIN API CALLED")

        # -------------------------
        # Train model
        # -------------------------
        train_model()

        end_time = datetime.utcnow()

        logger.info("✅ Training completed successfully")

        return {
            "message": "Model trained successfully 🚀",
            "start_time": str(start_time),
            "end_time": str(end_time),
            "model_registry": "mlflow",
            "status": "Production model updated"
        }

    # -------------------------
    # Client errors (rare here)
    # -------------------------
    except ValueError as ve:
        logger.warning(f"⚠️ Training validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    # -------------------------
    # Internal errors
    # -------------------------
    except Exception as e:
        logger.error(f"❌ Training error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Training failed")