import requests
import pandas as pd

LATITUDE  = 31.5497
LONGITUDE = 74.3436

_df_air = None  


def get_df_air(start_date="2022-06-01", end_date="2026-05-31") -> pd.DataFrame:
    """Fetch (or return cached) historical air quality dataframe."""
    global _df_air
    if _df_air is not None:
        return _df_air

    print(f"Fetching air quality data {start_date} → {end_date} ...")
    response = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude":  LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_date,
            "end_date":   end_date,
            "hourly": [
                "european_aqi",
                "pm10",
                "pm2_5",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "carbon_monoxide",
                "ozone",
            ]
        }
    )
    data = response.json()

    _df_air = pd.DataFrame({
        "timestamp": data["hourly"]["time"],
        "aqi":  data["hourly"]["european_aqi"],
        "pm25": data["hourly"]["pm2_5"],
        "pm10": data["hourly"]["pm10"],
        "no2":  data["hourly"]["nitrogen_dioxide"],
        "so2":  data["hourly"]["sulphur_dioxide"],
        "co":   data["hourly"]["carbon_monoxide"],
        "o3":   data["hourly"]["ozone"],
    })

    print(f"✅ Air quality: {_df_air.shape[0]:,} rows")
    return _df_air


class _LazyDF:
    def __getattr__(self, _):
        return get_df_air().__getattribute__(_)
    def __len__(self):
        return len(get_df_air())
    def merge(self, *a, **kw):
        return get_df_air().merge(*a, **kw)


df_air = _LazyDF()


if __name__ == "__main__":
    df = get_df_air()
    print(df.head())
    print(df.shape)
