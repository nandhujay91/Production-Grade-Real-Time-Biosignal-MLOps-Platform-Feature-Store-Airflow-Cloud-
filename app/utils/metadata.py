from datetime import datetime
from app.db.database import get_connection


# 🔥 Create session entry
def log_session(session_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO sessions (session_id, created_at) VALUES (?, ?)",
        (session_id, datetime.now().isoformat())
    )

    conn.commit()
    conn.close()


# 🔥 Log file metadata
def log_metadata(session_id, filename, path):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO files (session_id, filename, path, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, filename, path, datetime.now().isoformat())
    )

    conn.commit()
    conn.close()