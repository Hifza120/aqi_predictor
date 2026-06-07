import os
import sys
import traceback
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from feast import FeatureStore

THIS_DIR = Path(__file__).resolve().parent
FEATURE_REPO = THIS_DIR / "feature_repo"

load_dotenv(THIS_DIR / ".env")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

LATITUDE = 31.5497
LONGITUDE = 74.3436
STATION_ID = "lahore_main"
PST_OFFSET = 5

now_utc = datetime.now(timezone.utc)
now_pst = now_utc + timedelta(hours=PST_OFFSET)

utc_hour = now_utc.hour

print(f"UTC time : {now_utc}")
print(f"PST time : {now_pst}")

def safe_get_json(url, **kwargs):
    try:
        r = requests.get(url, timeout=15, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Request failed: {e}")
        return {}

def hourly_value(data, key, hour):
    try:
        values = data.get("hourly", {}).get(key, [])
        if len(values) > hour:
            return values[hour]
        return None
    except Exception:
        return None

# AQICN
print("Fetching AQICN...")
aqicn_json = safe_get_json(
    f"https://api.waqi.info/feed/A471607/?token={AQICN_TOKEN}"
)

if aqicn_json.get("status") != "ok":
    print("AQICN returned non-ok status:", aqicn_json)

aqicn = aqicn_json.get("data", {})

# Weather
print("Fetching weather...")
weather = safe_get_json(
    "https://api.open-meteo.com/v1/forecast",
    params={
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
            "cloud_cover",
            "boundary_layer_height",
        ],
        "forecast_days": 1,
        "timezone": "UTC",
    },
)

# Air Quality
print("Fetching air quality...")
aq = safe_get_json(
    "https://air-quality-api.open-meteo.com/v1/air-quality",
    params={
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "carbon_monoxide",
            "ozone",
            "european_aqi",
        ],
        "forecast_days": 3,
        "timezone": "UTC",
    },
)

combined = {
    "station_id": STATION_ID,
    "event_timestamp": now_utc,
    "aqi": aqicn.get("aqi"),
    "pm25": aqicn.get("iaqi", {}).get("pm25", {}).get("v"),
    "pm10": aqicn.get("iaqi", {}).get("pm10", {}).get("v"),
    "temperature": aqicn.get("iaqi", {}).get("t", {}).get("v"),
    "humidity": aqicn.get("iaqi", {}).get("h", {}).get("v"),
    "wind_speed": hourly_value(weather, "wind_speed_10m", utc_hour),
    "precipitation": hourly_value(weather, "precipitation", utc_hour),
    "cloud_cover": hourly_value(weather, "cloud_cover", utc_hour),
    "boundary": hourly_value(weather, "boundary_layer_height", utc_hour),
    "no2": hourly_value(aq, "nitrogen_dioxide", utc_hour),
    "so2": hourly_value(aq, "sulphur_dioxide", utc_hour),
    "co": hourly_value(aq, "carbon_monoxide", utc_hour),
    "o3": hourly_value(aq, "ozone", utc_hour),
}

df_live = pd.DataFrame([combined])

print("\nData to push:")
print(df_live)

if not FEATURE_REPO.exists():
    raise FileNotFoundError(f"Feature repo not found: {FEATURE_REPO}")

try:
    store = FeatureStore(repo_path=str(FEATURE_REPO))

    store.push(
        push_source_name="aqi_push_source",
        df=df_live,
        to="online",
    )

    print("Features pushed successfully.")

except Exception as e:
    print("\n===== FEAST ERROR =====")
    print(type(e).__name__, str(e))
    traceback.print_exc()
    sys.exit(1)

print("Done.")
