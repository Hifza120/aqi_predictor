import requests
import pandas as pd

LATITUDE  = 31.5497
LONGITUDE = 74.3436

_df_weather = None  


def get_df_weather(start_date="2022-06-01", end_date="2026-05-31") -> pd.DataFrame:
    """Fetch (or return cached) historical weather dataframe."""
    global _df_weather
    if _df_weather is not None:
        return _df_weather

    print(f"Fetching weather data {start_date} → {end_date} ...")
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":  LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_date,
            "end_date":   end_date,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "cloud_cover",
                "wind_speed_10m",
                "boundary_layer_height",
            ]
        }
    )
    data = response.json()

    _df_weather = pd.DataFrame({
        "timestamp":   data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "humidity":    data["hourly"]["relative_humidity_2m"],
        "precipitation": data["hourly"]["precipitation"],
        "cloud_cover": data["hourly"]["cloud_cover"],
        "wind_speed":  data["hourly"]["wind_speed_10m"],
        "boundary":    data["hourly"]["boundary_layer_height"],
    })

    print(f"✅ Weather: {_df_weather.shape[0]:,} rows")
    return _df_weather


class _LazyDF:
    def __getattr__(self, _):
        return get_df_weather().__getattribute__(_)
    def __len__(self):
        return len(get_df_weather())
    def merge(self, *a, **kw):
        return get_df_weather().merge(*a, **kw)


df_weather = _LazyDF()


if __name__ == "__main__":
    df = get_df_weather()
    print(df.head())
    print(df.shape)
