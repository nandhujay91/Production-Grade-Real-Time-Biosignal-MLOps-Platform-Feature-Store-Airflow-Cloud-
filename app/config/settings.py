import os

# -------------------------
# ENV
# -------------------------
ENV = os.getenv("ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# -------------------------
# PATHS
# -------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "features")

DB_PATH = os.path.join(DATA_DIR, "metadata", "metadata.db")

# ✅ Ensure DB directory exists (CRITICAL FIX)
DB_DIR = os.path.dirname(DB_PATH)
os.makedirs(DB_DIR, exist_ok=True)

FEATURES_PATH = os.getenv(
    "FEATURES_PATH",
    os.path.join(DATA_DIR, "latest_features.parquet")
)


# -------------------------
# MODEL / MLFLOW
# -------------------------
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "biosignal_model")
MLFLOW_STAGE = os.getenv("MLFLOW_STAGE", "Production")


# -------------------------
# DRIFT
# -------------------------
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", 0.2))


# -------------------------
# VALIDATION
# -------------------------
BPM_MIN = int(os.getenv("BPM_MIN", 20))
BPM_MAX = int(os.getenv("BPM_MAX", 220))

SPO2_MIN = int(os.getenv("SPO2_MIN", 70))
SPO2_MAX = int(os.getenv("SPO2_MAX", 100))