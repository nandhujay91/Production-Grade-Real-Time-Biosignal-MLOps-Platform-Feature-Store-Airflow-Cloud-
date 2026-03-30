from fastapi import APIRouter, HTTPException, Request
import numpy as np
import pandas as pd
import logging
import os
import glob

from app.services.feature_store_service import get_online_features
from app.services.validation_service import DataValidator

from app.services.drift_service import detect_drift
from app.services.retrain_service import retrain_model
from app.services.inference_service import load_latest_model, load_calibrator

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/predict/{session_id}")
def predict(session_id: str, request: Request):
    try:
        logger.info(f"🔮 Predict API called → session={session_id}")

        # -------------------------
        # Load model (SAFE)
        # -------------------------
        model = getattr(request.app.state, "model", None)

        if model is None:
            logger.warning("⚠️ Model not in memory → loading latest")
            model = load_latest_model()
            request.app.state.model = model

        # -------------------------
        # Fetch features
        # -------------------------
        features = get_online_features(session_id)

        if not features or "bpm" not in features or not features["bpm"]:
            raise ValueError("No valid features returned from Feature Store")

        # -------------------------
        # SAFE AGGREGATION
        # -------------------------
        def safe_mean(values):
            arr = np.array(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            return float(np.mean(arr)) if len(arr) > 0 else np.nan

        df = pd.DataFrame({
            "bpm": [safe_mean(features.get("bpm", []))],
            "spo2": [safe_mean(features.get("spo2", []))],
            "imu_x_mean": [safe_mean(features.get("imu_x_mean", []))],
            "imu_y_mean": [safe_mean(features.get("imu_y_mean", []))],
            "imu_z_mean": [safe_mean(features.get("imu_z_mean", []))],
        })

        logger.info(f"📥 Aggregated Features:\n{df}")

        # =========================================================
        # 🔥 FIX: FEATURE NAME ALIGNMENT (NO LOGIC CHANGE)
        # =========================================================
        try:
            if hasattr(model, "feature_names_in_"):
                missing_cols = set(model.feature_names_in_) - set(df.columns)

                if missing_cols:
                    logger.warning(f"⚠️ Missing model columns: {missing_cols}")

                df = df.reindex(columns=model.feature_names_in_, fill_value=0)

                logger.info("✅ Feature alignment applied")

        except Exception as e:
            logger.warning(f"⚠️ Feature alignment skipped: {e}")

        # -------------------------
        # Validation
        # -------------------------
        validator = DataValidator()

        try:
            val_result = validator.run_validation(df)
            quality_score = val_result["quality_score"]

        except Exception as val_err:
            logger.warning(f"⚠️ Validation failed: {val_err}")

            return {
                "session_id": session_id,
                "predicted_class": "Normal",
                "confidence": 0.5,
                "risk_level": "⚠️ Uncertain",
                "data_quality": 0.0,
                "fallback": True
            }

        # -------------------------
        # RAW PREDICTION
        # -------------------------
        probs = model.predict(df)
        probs = np.array(probs)

        # -------------------------
        # CALIBRATION (UNCHANGED)
        # -------------------------
        try:
            calibrator = load_calibrator()
            probs = calibrator.predict_proba(df)
            logger.info("🎯 Calibration applied")
        except Exception:
            logger.warning("⚠️ Calibration not available → using raw probabilities")

        # -------------------------
        # Prediction parsing
        # -------------------------
        if probs.ndim > 1:
            pred_idx = int(np.argmax(probs))
            confidence = float(np.max(probs))
        else:
            pred_idx = int(probs[0])
            confidence = 0.8

        # -------------------------
        # Label mapping
        # -------------------------
        class_names = ["Alert", "Critical", "Normal"]

        if 0 <= pred_idx < len(class_names):
            label = class_names[pred_idx]
        else:
            logger.warning(f"⚠️ Invalid prediction index: {pred_idx}")
            label = "Normal"

        # -------------------------
        # Confidence adjustment
        # -------------------------
        confidence = confidence * quality_score
        confidence = min(confidence, 0.95)

        # =========================================================
        # SAVE prediction for drift (SAFE)
        # =========================================================
        try:
            session_dir = os.path.join("data", "features", session_id)

            if os.path.exists(session_dir):

                parquet_files = glob.glob(os.path.join(session_dir, "*.parquet"))

                if parquet_files:
                    latest_file = max(parquet_files, key=os.path.getctime)

                    df_file = pd.read_parquet(latest_file)

                    df_file["predicted_class"] = label

                    df_file.to_parquet(latest_file, index=False)

                    logger.info(f"💾 Prediction saved → {latest_file}")
                else:
                    logger.warning("⚠️ No parquet files found")

            else:
                logger.warning(f"⚠️ Session folder not found: {session_dir}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to save prediction: {e}")

        # -------------------------
        # DRIFT DETECTION
        # -------------------------
        try:
            drift_result = detect_drift(session_id)

            if drift_result.get("drift_detected", False):
                logger.warning(f"🚨 Drift detected → {drift_result}")

                retrain_model()

                logger.info("🔄 Retraining completed")

                model = load_latest_model()
                request.app.state.model = model

                logger.info("✅ Model reloaded")

        except Exception as e:
            logger.warning(f"⚠️ Drift check failed: {e}")

        # -------------------------
        # Risk logic
        # -------------------------
        if quality_score < 0.6:
            risk = "⚠️ Uncertain"
        elif label == "Critical":
            risk = "🚨 Critical"
        elif label == "Alert":
            risk = "⚠️ Watch"
        else:
            risk = "Normal"

        # -------------------------
        # Logging
        # -------------------------
        logger.info(
            f"Prediction → session={session_id}, "
            f"label={label}, confidence={confidence:.4f}, "
            f"quality={quality_score}"
        )

        # -------------------------
        # Response
        # -------------------------
        return {
            "session_id": session_id,
            "predicted_class": label,
            "confidence": round(confidence, 4),
            "risk_level": risk,
            "data_quality": quality_score,
            "features_source": "feast_online_store",
            "model_source": "mlflow_registry",
            "model_stage": "Production"
        }

    except ValueError as ve:
        logger.warning(f"⚠️ Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except FileNotFoundError as fe:
        logger.error(f"❌ File error: {fe}")
        raise HTTPException(status_code=404, detail=str(fe))

    except Exception as e:
        logger.error(f"❌ Predict error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")