"""Open-Meteo live weather: forecast API (next 2 days) + archive API (backfill).

Free JSON APIs, no key. Same variables as the historical hackathon CSVs
(which are Open-Meteo exports for Delhi 28.65N 77.27E).
All timestamps handled in IST via the API's timezone parameter.
"""
import pandas as pd
import requests

from . import store

LAT, LON = 28.65, 77.27
HOURLY = "temperature_2m,relative_humidity_2m,apparent_temperature,rain,cloud_cover"
COLS = {"temperature_2m": "temp_c", "relative_humidity_2m": "rh_pct",
        "apparent_temperature": "apparent_temp_c", "rain": "rain_mm",
        "cloud_cover": "cloud_pct"}


def _to_df(js: dict) -> pd.DataFrame:
    df = pd.DataFrame(js["hourly"]).rename(columns=COLS)
    df["ts"] = pd.to_datetime(df.pop("time"))
    return df.set_index("ts").sort_index()


def fetch_forecast(days: int = 2, past_days: int = 7) -> pd.DataFrame:
    """Forecast plus the last few days of analysis values — the past_days
    rows bridge the ~5-day lag of the archive API."""
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=dict(latitude=LAT, longitude=LON, hourly=HOURLY,
                    forecast_days=days, past_days=past_days,
                    timezone="Asia/Kolkata"),
        timeout=20,
    )
    r.raise_for_status()
    return _to_df(r.json())


def fetch_archive(start: str, end: str) -> pd.DataFrame:
    """Historical actuals, dates as YYYY-MM-DD (archive lags ~5 days)."""
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=dict(latitude=LAT, longitude=LON, hourly=HOURLY,
                    start_date=start, end_date=end, timezone="Asia/Kolkata"),
        timeout=60,
    )
    r.raise_for_status()
    return _to_df(r.json())


RE_HOURLY = ("shortwave_radiation,direct_normal_irradiance,diffuse_radiation,"
             "wind_speed_10m,wind_speed_100m,temperature_2m,cloud_cover")
RE_COLS = {"shortwave_radiation": "ghi", "direct_normal_irradiance": "dni",
           "diffuse_radiation": "dhi", "wind_speed_10m": "wind10_kmh",
           "wind_speed_100m": "wind100_kmh", "temperature_2m": "temp_c",
           "cloud_cover": "cloud_pct"}


def fetch_re_forecast(days: int = 2, past_days: int = 7) -> pd.DataFrame:
    """Irradiance + hub-height wind for the RE generation digital twin."""
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=dict(latitude=LAT, longitude=LON, hourly=RE_HOURLY,
                    forecast_days=days, past_days=past_days,
                    timezone="Asia/Kolkata"),
        timeout=20,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"]).rename(columns=RE_COLS)
    df["ts"] = pd.to_datetime(df.pop("time"))
    return df.set_index("ts").sort_index()


def get_re_forecast(days: int = 2):
    """Live fetch with cached fallback. Returns (df, meta)."""
    try:
        df = fetch_re_forecast(days)
        df["kind"] = "forecast"
        store.upsert("re_weather", df)
        store.log_fetch("re_weather", True)
        return df.drop(columns="kind"), {"live": True, "asof": pd.Timestamp.now()}
    except Exception as e:
        store.log_fetch("re_weather", False, str(e))
        cached = store.read("re_weather")
        return (cached.drop(columns=["kind", "fetched_at"], errors="ignore"),
                {"live": False, "asof": store.last_good_fetch("re_weather"),
                 "error": str(e)})


def get_forecast(days: int = 2):
    """Live fetch with cached fallback. Returns (df, meta)."""
    try:
        df = fetch_forecast(days)
        df["kind"] = "forecast"
        store.upsert("weather", df)
        store.log_fetch("weather", True)
        return df.drop(columns="kind"), {"live": True, "asof": pd.Timestamp.now()}
    except Exception as e:
        store.log_fetch("weather", False, str(e))
        cached = store.read("weather")
        cached = cached[cached["kind"] == "forecast"]
        asof = store.last_good_fetch("weather")
        return (cached.drop(columns=["kind", "fetched_at"]),
                {"live": False, "asof": asof, "error": str(e)})
