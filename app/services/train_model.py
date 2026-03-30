import os
import pandas as pd
import numpy as np
import joblib
import logging

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

# 🔥 MLflow experiment
mlflow.set_experiment("biosignal_classification")


# -------------------------
# Load all sessions
# -------------------------
def load_all_sessions(data_dir="data/features"):
    all_data = []

    if not os.path.exists(data_dir):
        raise ValueError("Features directory not found!")

    for session in os.listdir(data_dir):
        session_path = os.path.join(data_dir, session)

        csv_path = os.path.join(session_path, "features_labeled.csv")
        parquet_path = os.path.join(session_path, "features_labeled.parquet")

        try:
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
                all_data.append(df)

            elif os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                all_data.append(df)

        except Exception as e:
            logger.warning(f"⚠️ Failed loading session {session}: {e}")

    if not all_data:
        raise ValueError("No labeled data found!")

    return pd.concat(all_data, ignore_index=True)


# -------------------------
# Synthetic Injection
# -------------------------
def inject_synthetic_if_needed(df):
    counts = df["label"].value_counts()
    synthetic = []

    if counts.get("Critical", 0) < 50:
        for _ in range(200):
            synthetic.append([
                np.random.uniform(25, 180),
                np.random.uniform(80, 90),
                0.5, 0.5, 0.6,
                "Critical"
            ])

    if counts.get("Alert", 0) < 50:
        for _ in range(150):
            synthetic.append([
                np.random.uniform(70, 110),
                np.random.uniform(92, 96),
                0.2, 0.2, 0.3,
                "Alert"
            ])

    if synthetic:
        synth_df = pd.DataFrame(
            synthetic,
            columns=["bpm", "spo2", "imu_x_mean", "imu_y_mean", "imu_z_mean", "label"]
        )
        df = pd.concat([df, synth_df], ignore_index=True)

    return df


# -------------------------
# Train model
# -------------------------
def train_model():
    logger.info("🚀 Training models (MLflow production enabled)...")

    df = load_all_sessions()

    column_mapping = {
        "imu_x": "imu_x_mean",
        "imu_y": "imu_y_mean",
        "imu_z": "imu_z_mean"
    }
    df = df.rename(columns=column_mapping)

    required_cols = ["bpm", "spo2", "imu_x_mean", "imu_y_mean", "imu_z_mean"]

    df = inject_synthetic_if_needed(df)

    X = df[required_cols]
    y = df["label"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42, k_neighbors=2)
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight=class_weight_dict
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42
        )
    }

    best_model = None
    best_score = 0
    best_name = ""
    best_run_id = None

    for name, model in models.items():

        with mlflow.start_run(run_name=name) as run:

            logger.info(f"🔹 Training: {name}")

            mlflow.log_param("model_name", name)

            model.fit(X_train_scaled, y_train)

            calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
            calibrated_model.fit(X_train_scaled, y_train)

            y_pred = calibrated_model.predict(X_test_scaled)

            logger.info(f"\n{classification_report(y_test, y_pred)}")

            score = f1_score(y_test, y_pred, average="weighted")

            mlflow.log_metric("f1_score", score)

            mlflow.log_input(
                mlflow.data.from_pandas(df, source="local", name="training_data")
            )

            # 🔥 Pipeline
            pipeline = Pipeline([
                ("scaler", scaler),
                ("model", calibrated_model)
            ])

            signature = infer_signature(X_train, y_pred)

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature
            )

            # 🔥 Log encoder
            encoder_path = "label_encoder.pkl"
            joblib.dump(encoder, encoder_path)
            mlflow.log_artifact(encoder_path)

            if score > best_score:
                best_score = score
                best_model = pipeline
                best_name = name
                best_run_id = run.info.run_id

            mlflow.set_tag("candidate_model", name)

    logger.info(f"🏆 BEST MODEL: {best_name} (F1-score: {best_score:.4f})")

    # -------------------------
    # Register BEST model
    # -------------------------
    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name="biosignal_model"
    )

    # -------------------------
    # Promote to Production
    # -------------------------
    client = MlflowClient()

    client.transition_model_version_stage(
        name="biosignal_model",
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )

    logger.info(f"🚀 Model v{result.version} promoted to Production")

    # -------------------------
    # Local backup
    # -------------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoder, "models/label_encoder.pkl")

    logger.info("✅ Model saved & registered (Production Ready)")


if __name__ == "__main__":
    train_model()