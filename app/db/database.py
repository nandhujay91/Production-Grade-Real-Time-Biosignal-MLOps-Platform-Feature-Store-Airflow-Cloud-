import sqlite3
import os
import logging

from app.config.settings import DB_PATH  # ✅ SINGLE SOURCE OF TRUTH

logger = logging.getLogger(__name__)

# -------------------------
# Ensure DB directory exists
# -------------------------
try:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    logger.info(f"📌 Using DB PATH: {DB_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to create DB directory: {e}")
    raise


# -------------------------
# DB Connection (Production-safe)
# -------------------------
def get_connection():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    except Exception as e:
        logger.error(f"❌ DB connection failed: {e}", exc_info=True)
        raise


# -------------------------
# Initialize DB
# -------------------------
def init_db():
    conn = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        logger.info("🔥 Creating tables if not exists...")

        # -------------------------
        # Sessions Table
        # -------------------------
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # -------------------------
        # Files Table
        # -------------------------
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            filename TEXT,
            path TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
        """)

        conn.commit()

        logger.info("✅ Database initialized successfully")

    except Exception as e:
        logger.error(f"❌ DB initialization failed: {e}", exc_info=True)
        raise

    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                logger.warning("⚠️ Failed to close DB connection")