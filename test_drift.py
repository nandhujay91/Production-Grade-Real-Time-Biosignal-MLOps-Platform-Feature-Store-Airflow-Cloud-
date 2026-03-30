from app.services.drift_service import check_drift

features_path = "data/features/session_20260327_184157/features_labeled.csv"

result = check_drift(features_path)

print("\n✅ Drift Result:", result)