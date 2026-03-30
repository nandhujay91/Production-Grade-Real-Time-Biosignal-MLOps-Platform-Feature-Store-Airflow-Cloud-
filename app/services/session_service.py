import logging
from app.db.database import get_connection

logger = logging.getLogger(__name__)


def get_files_by_session(session_id: str):
    logger.info(f"📂 Fetching files for session: {session_id}")

    conn = None

    try:
        # -------------------------
        # Get DB connection
        # -------------------------
        conn = get_connection()
        cursor = conn.cursor()

        # -------------------------
        # Execute query
        # -------------------------
        cursor.execute(
            """
            SELECT filename, path 
            FROM files 
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,)
        )

        rows = cursor.fetchall()

        # -------------------------
        # Validate results
        # -------------------------
        if not rows:
            logger.warning(f"⚠️ No files found for session: {session_id}")
            raise ValueError(f"No files found for session: {session_id}")

        files = [{"filename": r[0], "path": r[1]} for r in rows]

        # -------------------------
        # Validate completeness
        # -------------------------
        if len(files) != 4:
            logger.warning(
                f"⚠️ Incomplete session: expected 4 files, got {len(files)}"
            )
            raise ValueError(
                f"Incomplete session: expected 4 files, got {len(files)}"
            )

        logger.info(f"✅ Retrieved {len(files)} files for session: {session_id}")

        return files

    except Exception as e:
        logger.error(f"❌ Failed to fetch session files: {e}", exc_info=True)
        raise

    finally:
        # -------------------------
        # Ensure connection closed
        # -------------------------
        if conn:
            try:
                conn.close()
                logger.info("🔒 DB connection closed")
            except Exception:
                logger.warning("⚠️ Failed to close DB connection")