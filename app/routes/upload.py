from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import logging
from datetime import datetime

from app.utils.validation import validate_file
from app.utils.metadata import log_metadata, log_session

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = "data/raw"
REQUIRED_TYPES = ["Aux", "Ephy", "IMU", "Oxym"]


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload exactly 4 .bin files:
    - Aux
    - Ephy
    - IMU
    - Oxym
    """

    try:
        logger.info("📥 Upload API called")

        # -------------------------
        # Validate number of files
        # -------------------------
        if len(files) != 4:
            raise HTTPException(
                status_code=400,
                detail="Exactly 4 files required (Aux, Ephy, IMU, Oxym)"
            )

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # -------------------------
        # Generate session_id
        # -------------------------
        session_id = datetime.utcnow().strftime("session_%Y%m%d_%H%M%S")

        detected_types = {}
        saved_files = []

        # -------------------------
        # Log session
        # -------------------------
        log_session(session_id)
        logger.info(f"🧾 Session created → {session_id}")

        # -------------------------
        # Process each file
        # -------------------------
        for file in files:
            filename = file.filename

            # Detect file type
            file_type = None
            for t in REQUIRED_TYPES:
                if t.lower() in filename.lower():
                    file_type = t
                    break

            if not file_type:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {filename}"
                )

            # Duplicate check
            if file_type in detected_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Duplicate file type: {file_type}"
                )

            detected_types[file_type] = filename

            # -------------------------
            # Validate file
            # -------------------------
            validate_file(file)

            # -------------------------
            # Save file
            # -------------------------
            filepath = os.path.join(UPLOAD_DIR, f"{session_id}_{filename}")

            try:
                contents = await file.read()
                with open(filepath, "wb") as f:
                    f.write(contents)
            except Exception as e:
                logger.error(f"❌ Failed saving file {filename}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save file: {filename}"
                )

            # Normalize path (Windows safe)
            filepath = filepath.replace("\\", "/")

            # -------------------------
            # Log metadata
            # -------------------------
            log_metadata(session_id, filename, filepath)

            saved_files.append(filepath)

            logger.info(f"✅ Saved file → {filename}")

        # -------------------------
        # Final validation
        # -------------------------
        if set(detected_types.keys()) != set(REQUIRED_TYPES):
            raise HTTPException(
                status_code=400,
                detail="Missing required files (Aux, Ephy, IMU, Oxym)"
            )

        logger.info(f"🎯 Upload completed → session_id={session_id}")

        # -------------------------
        # Response
        # -------------------------
        return {
            "message": "Uploaded successfully",
            "session_id": session_id,
            "files": saved_files,
            "total_files": len(saved_files)
        }

    # -------------------------
    # Client errors
    # -------------------------
    except HTTPException as he:
        raise he

    # -------------------------
    # Unexpected errors
    # -------------------------
    except Exception as e:
        logger.error(f"❌ Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upload failed")