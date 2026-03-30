from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import logging
import mlflow.pyfunc

# Routers
from app.routes.upload import router as upload_router
from app.routes.process import router as process_router
from app.routes.stream import router as stream_router
from app.routes.predict import router as predict_router
from app.routes.train import router as train_router

# DB
from app.db.database import init_db

# Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# -------------------------
# GLOBAL LOGGING
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# -------------------------
# Lifespan (Production-safe)
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Biosignal MLOps API...")

    # -------------------------
    # DB INIT
    # -------------------------
    try:
        logger.info("🔥 Initializing database...")
        init_db()
        logger.info("✅ Database initialized")

    except Exception as e:
        logger.error(f"❌ DB init failed: {e}", exc_info=True)
        raise RuntimeError("Database initialization failed")

    # -------------------------
    # MODEL LOAD
    # -------------------------
    try:
        logger.info("🤖 Loading MLflow model...")

        model = mlflow.pyfunc.load_model("models:/biosignal_model/Production")

        # Store globally
        app.state.model = model

        logger.info("✅ Model loaded successfully")

    except Exception as e:
        logger.error(f"❌ Model load failed: {e}", exc_info=True)
        raise RuntimeError("Model failed to load")

    # -------------------------
    # READY
    # -------------------------
    logger.info("📡 Services ready:")
    logger.info("   → Upload")
    logger.info("   → Process")
    logger.info("   → Stream")
    logger.info("   → Train")
    logger.info("   → Predict")

    yield

    logger.info("🛑 Shutting down API...")


# -------------------------
# APP INIT
# -------------------------
app = FastAPI(
    title="Biosignal MLOps API",
    version="1.0.0",
    lifespan=lifespan
)


# -------------------------
# GLOBAL ERROR HANDLER
# -------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"❌ Unhandled error: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc)
        }
    )


# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# ROUTES
# -------------------------
app.include_router(upload_router, prefix="/api", tags=["Upload"])
app.include_router(process_router, prefix="/api", tags=["Process"])
app.include_router(stream_router, prefix="/api", tags=["Stream"])
app.include_router(train_router, prefix="/api", tags=["Train"])
app.include_router(predict_router, prefix="/api", tags=["Predict"])


# -------------------------
# HEALTH CHECKS
# -------------------------
@app.get("/", tags=["Health"])
def health():
    return {
        "status": "API running 🚀",
        "service": "biosignal-mlops",
        "pipeline": ["upload", "process", "stream", "train", "predict"]
    }


@app.get("/health/ready", tags=["Health"])
def ready():
    return {"ready": True}


@app.get("/health/live", tags=["Health"])
def live():
    return {"alive": True}