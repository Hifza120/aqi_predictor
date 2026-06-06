import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from feast import FeatureStore

ROOT = Path(__file__).resolve().parent
FEATURE_REPO = ROOT / "feature_repo"
MODELS_DIR = ROOT / "models"
PARQUET = FEATURE_REPO / "data" / "aqi_features.parquet"

def aqi_label(val):
    if val is None or np.isnan(val):
        return "Unknown"
    if val <= 50:
        return "Good"
    if val <= 100:
        return "Moderate"
    if val <= 150:
        return "Unhealthy (Sensitive Groups)"
    if val <= 200:
        return "Unhealthy"
    if val <= 300:
        return "Very Unhealthy"
    return "Hazardous"

print("=" * 55)
print("STEP 1: Fetching live features from online store...")
print("=" * 55)

store = FeatureStore(repo_path=str(FEATURE_REPO))

try:
    online = store.get_online_features(
        features=[
            "aqi_raw:aqi", "aqi_raw:pm25", "aqi_raw:pm10",
            "aqi_raw:no2", "aqi_raw:so2", "aqi_raw:co", "aqi_raw:o3",
            "aqi_raw:temperature", "aqi_raw:humidity",
            "aqi_raw:precipitation", "aqi_raw:cloud_cover",
            "aqi_raw:wind_speed", "aqi_raw:boundary",
            "aqi_lag:pm25_lag1", "aqi_lag:pm25_lag24",
            "aqi_lag:pm25_roll_mean_6", "aqi_lag:pm25_roll_mean_24",
            "aqi_lag:aqi_lag1", "aqi_lag:aqi_lag24",
            "aqi_lag:aqi_change_rate",
            "aqi_lag:aqi_roll_mean_6", "aqi_lag:aqi_roll_mean_24",
            "aqi_time:hour", "aqi_time:day", "aqi_time:month",
            "aqi_time:year", "aqi_time:day_of_week",
            "aqi_time:sin_hour", "aqi_time:cos_hour",
            "aqi_time:sin_month", "aqi_time:cos_month",
            "aqi_time:sin_day", "aqi_time:cos_day",
            "aqi_time:sin_dow", "aqi_time:cos_dow",
        ],
        entity_rows=[{"station_id": "lahore_main"}],
    ).to_dict()

    live = {k.split(":")[-1]: v[0] for k, v in online.items()}
    print("Online store fetch OK")
    print(f"   Current AQI: {live.get('aqi')} ({aqi_label(live.get('aqi'))})")

except Exception as e:
    print(f"Online store not available: {e}")
    print("Falling back to latest row in parquet...")

    df_hist = pd.read_parquet(PARQUET)
    df_hist = df_hist.sort_values("event_timestamp")
    last = df_hist.iloc[-1].to_dict()
    live = last
    print(f"Using parquet row: {last.get('event_timestamp')}")

now = datetime.now(timezone.utc)

live.setdefault("hour", now.hour)
live.setdefault("day", now.day)
live.setdefault("month", now.month)
live.setdefault("year", now.year)
live.setdefault("day_of_week", now.weekday())
live.setdefault("sin_hour", np.sin(2 * np.pi * now.hour / 24))
live.setdefault("cos_hour", np.cos(2 * np.pi * now.hour / 24))
live.setdefault("sin_month", np.sin(2 * np.pi * now.month / 12))
live.setdefault("cos_month", np.cos(2 * np.pi * now.month / 12))
live.setdefault("sin_day", np.sin(2 * np.pi * now.day / 31))
live.setdefault("cos_day", np.cos(2 * np.pi * now.day / 31))
live.setdefault("sin_dow", np.sin(2 * np.pi * now.weekday() / 7))
live.setdefault("cos_dow", np.cos(2 * np.pi * now.weekday() / 7))

print("\n" + "=" * 55)
print("STEP 2: Running AQI forecasts...")
print("=" * 55)

HORIZONS = ["24h", "48h", "72h"]
results = {}

for horizon in HORIZONS:
    model_path = MODELS_DIR / f"best_model_{horizon}.pkl"
    feat_path = MODELS_DIR / f"features_{horizon}.pkl"

    if not model_path.exists():
        print(f"No model found for {horizon} - run model_training.py first")
        continue

    model = joblib.load(model_path)
    features = joblib.load(feat_path) if feat_path.exists() else None

    if features is None:
        print(f"No feature list for {horizon} - skipping")
        continue

    row = {}
    missing = []

    for f in features:
        val = live.get(f)
        if val is None:
            missing.append(f)
            val = 0.0
        row[f] = val

    if missing:
        print(
            f"  {horizon}: {len(missing)} features missing from online store -> "
            f"defaulted to 0: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    X = pd.DataFrame([row])[features]
    pred = float(model.predict(X)[0])
    pred = max(0, round(pred, 1))
    results[horizon] = pred

print("\n" + "=" * 55)
print("AQI FORECAST - LAHORE")
print("=" * 55)

now_str = now.strftime("%Y-%m-%d %H:%M UTC")
print(f"Generated : {now_str}")
print(f"Current   : {live.get('aqi', 'N/A')} {aqi_label(live.get('aqi'))}")
print()

for horizon, pred in results.items():
    target_time = now + timedelta(hours=int(horizon.replace("h", "")))
    label = aqi_label(pred)
    print(f"+{horizon:<5} ({target_time.strftime('%a %H:%M')}) -> AQI {pred:>6.1f} {label}")

print("=" * 55)
print()

forecast = {
    "generated_at": now.isoformat(),
    "current_aqi": live.get("aqi"),
    "forecasts": results,
}