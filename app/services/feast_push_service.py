from feast import FeatureStore
from feast.data_source import PushMode
import pandas as pd
import logging

logger = logging.getLogger(__name__)

store = FeatureStore(repo_path="biosignal_feature_repo/feature_repo")


def push_features_to_feast(df: pd.DataFrame):
    try:
        logger.info("🚀 Pushing features to Feast online store")

        store.push(
            push_source_name="biosignal_push_source",
            df=df,
            to=PushMode.ONLINE
        )

        logger.info("✅ Push successful")

    except Exception as e:
        logger.error(f"❌ Feast push failed: {e}", exc_info=True)
        raise