
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Float32, Float64, Int32, Int64, String

# ── Entity ────────────────────────────────────────────────────────────────────
station = Entity(
    name="station_id",
    description="AQI monitoring station (e.g. lahore_main)",
)

# ── Offline source (parquet) ──────────────────────────────────────────────────
aqi_file_source = FileSource(
    name="aqi_file_source",
    path="data/aqi_features.parquet",
    timestamp_field="event_timestamp",
)

# ── Push source (fix #2 — used by fetch_data.py store.push()) ─────────────────
aqi_push_source = PushSource(
    name="aqi_push_source",
    batch_source=aqi_file_source,
)

# ── Raw pollutant + weather features (fix #5 — wind_speed + boundary added) ──
raw_features = FeatureView(
    name="aqi_raw",
    entities=[station],
    ttl=timedelta(days=2),
    schema=[
        Field(name="temperature",   dtype=Float32),
        Field(name="humidity",      dtype=Float32),
        Field(name="precipitation", dtype=Float32),
        Field(name="cloud_cover",   dtype=Float32),
        Field(name="wind_speed",    dtype=Float32),   # ← fix #5
        Field(name="boundary",      dtype=Float32),   # ← fix #5
        Field(name="pm25",          dtype=Float32),
        Field(name="pm10",          dtype=Float32),
        Field(name="no2",           dtype=Float32),
        Field(name="so2",           dtype=Float32),
        Field(name="co",            dtype=Float32),
        Field(name="o3",            dtype=Float32),
        Field(name="aqi",           dtype=Int32),
    ],
    source=aqi_push_source,   # supports both batch + push
)

# ── Time / cyclical features ──────────────────────────────────────────────────
time_features = FeatureView(
    name="aqi_time",
    entities=[station],
    ttl=timedelta(days=2),
    schema=[
        Field(name="hour",        dtype=Int32),
        Field(name="day",         dtype=Int32),
        Field(name="month",       dtype=Int32),
        Field(name="year",        dtype=Int32),
        Field(name="day_of_week", dtype=Int32),
        Field(name="sin_hour",    dtype=Float32),
        Field(name="cos_hour",    dtype=Float32),
        Field(name="sin_month",   dtype=Float32),
        Field(name="cos_month",   dtype=Float32),
        Field(name="sin_day",     dtype=Float32),
        Field(name="cos_day",     dtype=Float32),
        Field(name="sin_dow",     dtype=Float32),
        Field(name="cos_dow",     dtype=Float32),
    ],
    source=aqi_file_source,
)

# ── Lag + rolling features ────────────────────────────────────────────────────
lag_features = FeatureView(
    name="aqi_lag",
    entities=[station],
    ttl=timedelta(days=2),
    schema=[
        Field(name="pm25_lag1",          dtype=Float32),
        Field(name="pm25_lag24",         dtype=Float32),
        Field(name="pm25_roll_mean_6",   dtype=Float32),
        Field(name="pm25_roll_mean_24",  dtype=Float32),
        Field(name="aqi_lag1",           dtype=Float32),
        Field(name="aqi_lag24",          dtype=Float32),
        Field(name="aqi_change_rate",    dtype=Float32),
        Field(name="aqi_roll_mean_6",    dtype=Float32),
        Field(name="aqi_roll_mean_24",   dtype=Float32),
    ],
    source=aqi_file_source,
)

# ── Target labels (offline only — not served online) ──────────────────────────
target_features = FeatureView(
    name="aqi_targets",
    entities=[station],
    ttl=timedelta(days=4),
    schema=[
        Field(name="aqi_target_24h", dtype=Float32),
        Field(name="aqi_target_48h", dtype=Float32),
        Field(name="aqi_target_72h", dtype=Float32),
    ],
    source=aqi_file_source,
)
