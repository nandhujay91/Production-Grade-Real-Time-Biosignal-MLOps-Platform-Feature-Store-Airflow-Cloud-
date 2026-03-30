import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class DataValidator:

    def __init__(self):
        self.required_columns = [
            "bpm",
            "spo2",
            "imu_x_mean",
            "imu_y_mean",
            "imu_z_mean"
        ]

    # -------------------------
    # Schema validation
    # -------------------------
    def validate_schema(self, df: pd.DataFrame) -> None:
        logger.info("🔍 Validating schema...")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input is not a valid DataFrame")

        if df.empty:
            raise ValueError("Input DataFrame is empty")

        missing_cols = [
            col for col in self.required_columns
            if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

    # -------------------------
    # Range validation (SOFT)
    # -------------------------
    def validate_ranges(self, df: pd.DataFrame) -> int:
        logger.info("📏 Validating ranges...")

        try:
            invalid_bpm = ~df["bpm"].between(20, 220)
            invalid_spo2 = ~df["spo2"].between(70, 100)

            invalid_count = int(invalid_bpm.sum() + invalid_spo2.sum())

            if invalid_count > 0:
                logger.warning(
                    f"⚠️ Invalid values detected "
                    f"(bpm/spo2 out of range): count={invalid_count}"
                )

            # ✅ Soft correction (no failure)
            df["bpm"] = df["bpm"].clip(20, 220)
            df["spo2"] = df["spo2"].clip(70, 100)

            return invalid_count

        except Exception as e:
            logger.error(f"❌ Range validation failed: {e}", exc_info=True)
            raise

    # -------------------------
    # Null validation (SOFT)
    # -------------------------
    def validate_nulls(self, df: pd.DataFrame) -> int:
        logger.info("🧹 Checking null values...")

        try:
            null_counts = df[self.required_columns].isnull().sum()
            total_nulls = int(null_counts.sum())

            if total_nulls > 0:
                logger.warning(
                    f"⚠️ Null values found: {null_counts.to_dict()}"
                )

                # ✅ Drop invalid rows (industry standard)
                df.dropna(subset=self.required_columns, inplace=True)

            return total_nulls

        except Exception as e:
            logger.error(f"❌ Null validation failed: {e}", exc_info=True)
            raise

    # -------------------------
    # Run all validations
    # -------------------------
    def run_validation(self, df: pd.DataFrame) -> Dict:
        logger.info("🚦 Running full data validation pipeline")

        try:
            # -------------------------
            # Schema check
            # -------------------------
            self.validate_schema(df)

            # -------------------------
            # Track rows
            # -------------------------
            initial_rows = len(df)

            # -------------------------
            # Apply validations
            # -------------------------
            invalid_count = self.validate_ranges(df)
            null_count = self.validate_nulls(df)

            final_rows = len(df)

            # -------------------------
            # QUALITY SCORE
            # -------------------------
            total_issues = invalid_count + null_count

            if initial_rows == 0:
                quality_score = 0.0
            else:
                quality_score = max(
                    0.0,
                    1 - (total_issues / max(initial_rows, 1))
                )

            quality_score = round(float(quality_score), 3)

            # -------------------------
            # Logging
            # -------------------------
            logger.info(f"📊 Data quality score: {quality_score}")
            logger.info(
                f"📊 Rows before={initial_rows}, after={final_rows}"
            )

            if final_rows == 0:
                logger.warning("⚠️ All rows dropped after validation")

            logger.info("✅ Validation completed (soft mode)")

            # -------------------------
            # Return structured output
            # -------------------------
            return {
                "valid": True,
                "quality_score": quality_score,
                "rows_before": initial_rows,
                "rows_after": final_rows,
                "issues": {
                    "invalid_values": invalid_count,
                    "null_values": null_count
                }
            }

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}", exc_info=True)
            raise