import sqlite3
import os

# 🔥 SAME PATH AS DATABASE.PY
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
DB_PATH = os.path.join(BASE_DIR, "data", "metadata", "metadata.db")

print("📌 USING DB PATH:", DB_PATH)


def view_metadata():
    try:
        print("\n🔍 Checking database...\n")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        print("📦 Tables:", tables)

        if not tables:
            print("❌ No tables found!")
            return

        # Sessions
        print("\n📂 Sessions:")
        cursor.execute("SELECT * FROM sessions")
        for row in cursor.fetchall():
            print(row)

        # Files
        print("\n📂 Files:")
        cursor.execute("SELECT * FROM files")
        for row in cursor.fetchall():
            print(row)

        conn.close()

    except Exception as e:
        print("❌ Error:", str(e))


if __name__ == "__main__":
    view_metadata()