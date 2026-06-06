import sys
import os
import pandas as pd
import numpy as np


def _load_raw() -> pd.DataFrame:
    sys.path.insert(0, os.path.dirname(__file__))
    from historic_backfill import get_df_weather
    from historic_airquality import get_df_air

    df_weather = get_df_weather()
    df_air = get_df_air()

    df = df_weather.merge(df_air, on="timestamp")
    return df


def get_features() -> pd.DataFrame:
    df = _load_raw()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_day"] = np.sin(2 * np.pi * df["day"] / 31)
    df["cos_day"] = np.cos(2 * np.pi * df["day"] / 31)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df = df.sort_values("timestamp").reset_index(drop=True)

    df["pm25_lag1"] = df["pm25"].shift(1)
    df["pm25_lag24"] = df["pm25"].shift(24)
    df["pm25_roll_mean_6"] = df["pm25"].rolling(6).mean()
    df["pm25_roll_mean_24"] = df["pm25"].rolling(24).mean()

    df["aqi_lag1"] = df["aqi"].shift(1)
    df["aqi_lag24"] = df["aqi"].shift(24)
    df["aqi_change_rate"] = df["aqi"].diff()
    df["aqi_roll_mean_6"] = df["aqi"].rolling(6).mean()
    df["aqi_roll_mean_24"] = df["aqi"].rolling(24).mean()

    df["aqi_target_24h"] = df["aqi"].shift(-24)
    df["aqi_target_48h"] = df["aqi"].shift(-48)
    df["aqi_target_72h"] = df["aqi"].shift(-72)

    df = df.dropna().reset_index(drop=True)

    df["event_timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["station_id"] = "lahore_main"

    return df


df_features = None
df_with_timestamp = None


def build():
    global df_features, df_with_timestamp

    df = get_features()

    corr = df.corr(numeric_only=True)
    print("\nTop correlations with AQI:")
    print(corr["aqi"].sort_values(ascending=False).head(10))

    df_with_timestamp = df.copy()
    df_features = df.drop(columns=["timestamp"])

    return df_features


if __name__ == "__main__":
    from pathlib import Path

    df_feat = build()

    out = Path(__file__).parent / "feature_repo" / "data" / "aqi_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    df_feat.to_parquet(out, index=False)

    print(f"\nSaved {len(df_feat):,} rows -> {out}")
    print(f"Columns ({len(df_feat.columns)}): {list(df_feat.columns)}")