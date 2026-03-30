from fastapi import APIRouter, HTTPException
import logging

from app.services.session_service import get_files_by_session
from app.services.signal_processing import process_files

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/process/{session_id}")
def process_session(session_id: str):
    try:
        logger.info(f"🧠 Processing started → session_id={session_id}")

        # -------------------------
        # Get session files
        # -------------------------
        files = get_files_by_session(session_id)

        # -------------------------
        # Process files
        # -------------------------
        processed = process_files(session_id, files)

        logger.info(f"✅ Processing completed → session_id={session_id}")

        return {
            "message": "Processing completed",
            "session_id": session_id,
            "processed_files": processed,
            "total_files": len(processed)
        }

    # -------------------------
    # Client errors (bad input)
    # -------------------------
    except ValueError as ve:
        logger.warning(f"⚠️ Processing validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    # -------------------------
    # Not found (session/files)
    # -------------------------
    except FileNotFoundError as fe:
        logger.error(f"❌ File error: {fe}")
        raise HTTPException(status_code=404, detail=str(fe))

    # -------------------------
    # Internal errors
    # -------------------------
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")