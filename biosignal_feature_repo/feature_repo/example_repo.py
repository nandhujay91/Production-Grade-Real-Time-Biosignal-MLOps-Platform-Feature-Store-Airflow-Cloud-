from feast import Entity, FeatureView, Field, FileSource, ValueType, PushSource
from feast.types import Float32
from datetime import timedelta

# -------------------------
# Entity
# -------------------------
session = Entity(
    name="session_id",
    value_type=ValueType.STRING,
    join_keys=["session_id"],
)

# -------------------------
# File Source (offline)
# -------------------------
file_source = FileSource(
    path="../../data/features",
    timestamp_field="event_timestamp",
)

# -------------------------
# 🔥 Push Source (REAL-TIME)
# -------------------------
push_source = PushSource(
    name="biosignal_push_source",
    batch_source=file_source,
)

# -------------------------
# Feature View
# -------------------------
biosignal_view = FeatureView(
    name="biosignal_features",
    entities=[session],
    ttl=timedelta(days=1),
    schema=[
        Field(name="bpm", dtype=Float32),
        Field(name="spo2", dtype=Float32),
        Field(name="imu_x_mean", dtype=Float32),
        Field(name="imu_y_mean", dtype=Float32),
        Field(name="imu_z_mean", dtype=Float32),
    ],
    online=True,
    source=push_source,   # 🔥 IMPORTANT
)