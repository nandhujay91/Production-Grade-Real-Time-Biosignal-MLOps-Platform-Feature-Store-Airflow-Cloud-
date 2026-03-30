import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


# -------------------------
# 🔥 Load latest model (MLflow)
# -------------------------
def load_latest_model():
    try:
        import mlflow.pyfunc

        model = mlflow.pyfunc.load_model("models:/biosignal_model/Production")

        logger.info("✅ Loaded latest model from MLflow (Production)")

        return model

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}", exc_info=True)
        raise


# -------------------------
# 🔥 Load calibrator (OPTIONAL)
# -------------------------
def load_calibrator():
    try:
        import joblib

        path = "models/calibrator.pkl"

        if not os.path.exists(path):
            raise FileNotFoundError("Calibrator not found")

        calibrator = joblib.load(path)

        logger.info("🎯 Calibrator loaded successfully")

        return calibrator

    except Exception as e:
        logger.warning(f"⚠️ Calibration unavailable: {e}")
        raise


# -------------------------
# Inference Function
# -------------------------
def run_inference(
    features_path: str,
    model,
    scaler,
    encoder
) -> Dict[str, Any]:

    logger.info(f"🚀 Running inference → {features_path}")

    # -------------------------
    # Validate input path
    # -------------------------
    if not isinstance(features_path, str) or not os.path.exists(features_path):
        logger.error(f"❌ Features file not found: {features_path}")
        raise FileNotFoundError(f"Features file not found: {features_path}")

    # -------------------------
    # Load features safely
    # -------------------------
    try:
        if features_path.endswith(".parquet"):
            df = pd.read_parquet(features_path)
        else:
            df = pd.read_csv(features_path)

        logger.info(f"📊 Loaded features → rows={len(df)}")

    except Exception as e:
        logger.error(f"❌ Failed to load features: {e}", exc_info=True)
        raise

    if df.empty:
        raise ValueError("Empty feature dataset")

    # -------------------------
    # Column normalization
    # -------------------------
    column_mapping = {
        "imu_x": "imu_x_mean",
        "imu_y": "imu_y_mean",
        "imu_z": "imu_z_mean"
    }

    df = df.rename(columns=column_mapping)

    # -------------------------
    # Required columns
    # -------------------------
    required_cols = [
        "bpm",
        "spo2",
        "imu_x_mean",
        "imu_y_mean",
        "imu_z_mean"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"❌ Missing columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    # -------------------------
    # Prepare features
    # -------------------------
    X = df[required_cols].copy()

    if hasattr(scaler, "feature_names_in_"):
        try:
            X = X[scaler.feature_names_in_]
        except Exception as e:
            logger.warning(f"⚠️ Feature order mismatch: {e}")

    # -------------------------
    # Scale features
    # -------------------------
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        logger.error(f"❌ Scaling failed: {e}", exc_info=True)
        raise

    # -------------------------
    # Predict
    # -------------------------
    try:
        probs = model.predict(X_scaled)
        probs = np.array(probs)
    except Exception as e:
        logger.error(f"❌ Model prediction failed: {e}", exc_info=True)
        raise

    # -------------------------
    # Handle prediction format
    # -------------------------
    if probs.ndim == 1:
        pred_indices_all = probs.astype(int)
        probs = np.eye(len(encoder.classes_))[pred_indices_all]
    else:
        pred_indices_all = np.argmax(probs, axis=1)

    pred_labels_all = encoder.inverse_transform(pred_indices_all)

    # -------------------------
    # Save predictions (for drift)
    # -------------------------
    try:
        df["predicted_class"] = pred_labels_all

        if features_path.endswith(".parquet"):
            df.to_parquet(features_path, index=False)
        else:
            df.to_csv(features_path, index=False)

        logger.info("💾 Predictions saved for drift detection")

    except Exception as e:
        logger.warning(f"⚠️ Failed to save predictions: {e}", exc_info=True)

    # -------------------------
    # Session-level aggregation
    # -------------------------
    try:
        avg_probs = np.max(probs, axis=0)

        pred_class_idx = int(np.argmax(avg_probs))
        confidence = float(np.max(avg_probs))

        label = encoder.inverse_transform([pred_class_idx])[0]

    except Exception as e:
        logger.error(f"❌ Aggregation failed: {e}", exc_info=True)
        raise

    # -------------------------
    # Prevent overconfidence
    # -------------------------
    confidence = min(confidence, 0.95)

    # -------------------------
    # Uncertainty flag
    # -------------------------
    uncertainty_flag = confidence < 0.6

    # -------------------------
    # Risk logic (UNCHANGED)
    # -------------------------
    class_names = list(encoder.classes_)

    def get_index(name):
        return class_names.index(name) if name in class_names else None

    alert_idx = get_index("Alert")
    critical_idx = get_index("Critical")

    alert_prob = avg_probs[alert_idx] if alert_idx is not None else 0
    critical_prob = avg_probs[critical_idx] if critical_idx is not None else 0

    if label == "Critical" or critical_prob > 0.3:
        risk = "🚨 Critical"
    elif alert_prob > 0.3:
        risk = "⚠️ Watch"
    else:
        risk = "Normal"

    if uncertainty_flag:
        risk = "⚠️ Uncertain"

    # -------------------------
    # Probability dictionary
    # -------------------------
    prob_dict = {
        class_names[i]: float(avg_probs[i])
        for i in range(len(avg_probs))
    }

    # -------------------------
    # Logging
    # -------------------------
    logger.info({
        "prediction": label,
        "confidence": round(confidence, 4),
        "uncertain": uncertainty_flag
    })

    # -------------------------
    # Final response
    # -------------------------
    return {
        "predicted_class": label,
        "risk_level": risk,
        "confidence": round(confidence, 4),
        "uncertain": uncertainty_flag,
        "probabilities": prob_dict,
        "model_source": "mlflow_registry",
        "model_stage": "Production"
    }