import os
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from feast import FeatureStore

THIS_DIR     = Path(__file__).resolve().parent
FEATURE_REPO = THIS_DIR / "feature_repo"

load_dotenv(THIS_DIR / ".env")
AQICN_TOKEN = os.getenv("AQICN_TOKEN", "52b9bee1d5ccafa0bbe9654db72316ca58e6a4c6")

LATITUDE   = 31.5497
LONGITUDE  = 74.3436
STATION_ID = "lahore_main"
PST_OFFSET = 5

now_utc  = datetime.now(timezone.utc)
now_pst  = now_utc + timedelta(hours=PST_OFFSET)

utc_hour = now_utc.hour
pst_hour = now_pst.hour

print(f"UTC time : {now_utc.strftime('%Y-%m-%d %H:%M')}  (hour={utc_hour})")
print(f"PST time : {now_pst.strftime('%Y-%m-%d %H:%M')}  (hour={pst_hour})")

# ── Fetch AQICN ───────────────────────────────────────────────────────────────
print("\nFetching AQICN data...")
try:
    aqicn_resp = requests.get(
        f"https://api.waqi.info/feed/A471607/?token={AQICN_TOKEN}",
        timeout=10
    )
    aqicn = aqicn_resp.json().get("data", {})
except Exception as e:
    print(f"AQICN fetch failed: {e}")
    aqicn = {}

# ── Fetch Open-Meteo weather ──────────────────────────────────────────────────
print("Fetching Open-Meteo weather (UTC)...")
try:
    weather_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "hourly": [
                "temperature_2m", "relative_humidity_2m",
                "wind_speed_10m", "precipitation",
                "cloud_cover", "boundary_layer_height"
            ],
            "forecast_days": 1,
            "timezone": "UTC",
        },
        timeout=10
    )
    weather = weather_resp.json()
except Exception as e:
    print(f"Weather fetch failed: {e}")
    weather = {}

# ── Fetch Open-Meteo air quality ──────────────────────────────────────────────
print("Fetching Open-Meteo air quality (UTC)...")
try:
    aq_resp = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "hourly": [
                "nitrogen_dioxide", "sulphur_dioxide",
                "carbon_monoxide", "ozone", "european_aqi"
            ],
            "forecast_days": 3,
            "timezone": "UTC",
        },
        timeout=10
    )
    aq = aq_resp.json()
except Exception as e:
    print(f"AQ fetch failed: {e}")
    aq = {}


def _hourly(data, key, hour):
    return data.get("hourly", {}).get(key, [None] * 24)[hour]


# ── STEP 1: Resolve best AQI value (fallback before anything else) ────────────
# Convert AQI to a number if possible
aqi_val = aqicn.get("aqi")

try:
    aqi_val = float(aqi_val)
except (TypeError, ValueError):
    aqi_val = None

aqi_from_openmeteo = _hourly(aq, "european_aqi", utc_hour)

if aqi_val is None or (aqi_from_openmeteo and aqi_val < aqi_from_openmeteo * 0.5):
    aqi_val = aqi_from_openmeteo
    print(f"  AQICN AQI overridden with Open-Meteo: {aqi_from_openmeteo}")

# ── STEP 2: Build combined dict with correct AQI ─────────────────────────────
combined = {
    "station_id":      STATION_ID,
    "event_timestamp": now_utc,
    "aqi":             aqi_val,
    "pm25":            aqicn.get("iaqi", {}).get("pm25", {}).get("v"),
    "pm10":            aqicn.get("iaqi", {}).get("pm10", {}).get("v"),
    "temperature":     aqicn.get("iaqi", {}).get("t",   {}).get("v"),
    "humidity":        aqicn.get("iaqi", {}).get("h",   {}).get("v"),
    "wind_speed":      _hourly(weather, "wind_speed_10m",        utc_hour),
    "precipitation":   _hourly(weather, "precipitation",         utc_hour),
    "cloud_cover":     _hourly(weather, "cloud_cover",           utc_hour),
    "boundary":        _hourly(weather, "boundary_layer_height", utc_hour),
    "no2":             _hourly(aq, "nitrogen_dioxide", utc_hour),
    "so2":             _hourly(aq, "sulphur_dioxide",  utc_hour),
    "co":              _hourly(aq, "carbon_monoxide",  utc_hour),
    "o3":              _hourly(aq, "ozone",            utc_hour),
}

print("\nFetched data:")
for k, v in combined.items():
    print(f"  {k:<20} {v}")

# ── STEP 3: Compute lag features from historical parquet ──────────────────────
print("\nComputing lag features from parquet...")
parquet_path = FEATURE_REPO / "data" / "aqi_features.parquet"

if parquet_path.exists():
    df_hist = pd.read_parquet(parquet_path)
    df_hist = df_hist.sort_values("event_timestamp").reset_index(drop=True)

    last_aqi  = df_hist["aqi"].iloc[-1]
    last_pm25 = df_hist["pm25"].iloc[-1]

    combined["aqi_lag1"]          = last_aqi
    combined["aqi_lag24"]         = df_hist["aqi"].iloc[-24] if len(df_hist) >= 24 else last_aqi
    combined["aqi_roll_mean_6"]   = df_hist["aqi"].iloc[-6:].mean()
    combined["aqi_roll_mean_24"]  = df_hist["aqi"].iloc[-24:].mean()
    combined["aqi_change_rate"]   = (aqi_val - last_aqi) if aqi_val is not None else 0.0

    combined["pm25_lag1"]         = last_pm25
    combined["pm25_lag24"]        = df_hist["pm25"].iloc[-24] if len(df_hist) >= 24 else last_pm25
    combined["pm25_roll_mean_6"]  = df_hist["pm25"].iloc[-6:].mean()
    combined["pm25_roll_mean_24"] = df_hist["pm25"].iloc[-24:].mean()

    print(f"  aqi_lag1          = {combined['aqi_lag1']:.1f}")
    print(f"  aqi_lag24         = {combined['aqi_lag24']:.1f}")
    print(f"  aqi_roll_mean_6   = {combined['aqi_roll_mean_6']:.1f}")
    print(f"  aqi_roll_mean_24  = {combined['aqi_roll_mean_24']:.1f}")
    print(f"  aqi_change_rate   = {combined['aqi_change_rate']:.1f}")
else:
    print(f"Parquet not found at {parquet_path} — lag features will be missing!")

# ── STEP 4: Push to Feast online store ───────────────────────────────────────
df_live = pd.DataFrame([combined])

store = FeatureStore(repo_path=str(FEATURE_REPO))
store.push(
    push_source_name="aqi_push_source",
    df=df_live,
    to="online",
)

print(f"\nLive features pushed to Feast online store!")
print(f"   Station : {STATION_ID}")
print(f"   UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")