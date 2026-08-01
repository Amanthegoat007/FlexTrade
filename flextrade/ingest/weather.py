"""Open-Meteo live weather: forecast API (next 2 days) + archive API (backfill).

Free JSON APIs, no key. Same variables as the historical hackathon CSVs
(which are Open-Meteo exports for Delhi 28.65N 77.27E).
All timestamps handled in IST via the API's timezone parameter.

Retry policy (added 28 Jul after a real 4-day outage): the scheduled
pipeline fires at 11:00 and repeatedly hit
`getaddrinfo failed` — DNS was not yet up when the task woke the machine,
even though a manual run minutes later worked fine. A single-shot fetch
therefore silently lost the whole day's plan. Every network call now
retries with exponential backoff, which costs nothing when the network is
healthy and rescues the run when it is merely slow to come up.
"""
import time

import pandas as pd
import requests

from . import store

LAT, LON = 28.65, 77.27

RETRIES = 4
BACKOFF_S = 5  # 5s, 10s, 20s, 40s — ~75s of patience in total


def _get(url: str, params: dict, timeout: int = 20) -> dict:
    """GET with retry/backoff. Raises the last error if all attempts fail."""
    last = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # DNS, timeout, 5xx — all worth retrying
            last = e
            if attempt < RETRIES - 1:
                time.sleep(BACKOFF_S * (2 ** attempt))
    raise last
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
    js = _get("https://api.open-meteo.com/v1/forecast",
              dict(latitude=LAT, longitude=LON, hourly=HOURLY,
                   forecast_days=days, past_days=past_days,
                   timezone="Asia/Kolkata"))
    return _to_df(js)


def fetch_archive(start: str, end: str) -> pd.DataFrame:
    """Historical actuals, dates as YYYY-MM-DD (archive lags ~5 days)."""
    js = _get("https://archive-api.open-meteo.com/v1/archive",
              dict(latitude=LAT, longitude=LON, hourly=HOURLY,
                   start_date=start, end_date=end, timezone="Asia/Kolkata"),
              timeout=60)
    return _to_df(js)


RE_HOURLY = ("shortwave_radiation,direct_normal_irradiance,diffuse_radiation,"
             "wind_speed_10m,wind_speed_100m,temperature_2m,cloud_cover")
RE_COLS = {"shortwave_radiation": "ghi", "direct_normal_irradiance": "dni",
           "diffuse_radiation": "dhi", "wind_speed_10m": "wind10_kmh",
           "wind_speed_100m": "wind100_kmh", "temperature_2m": "temp_c",
           "cloud_cover": "cloud_pct"}


def fetch_re_forecast(days: int = 2, past_days: int = 7) -> pd.DataFrame:
    """Irradiance + hub-height wind for the RE generation digital twin."""
    js = _get("https://api.open-meteo.com/v1/forecast",
              dict(latitude=LAT, longitude=LON, hourly=RE_HOURLY,
                   forecast_days=days, past_days=past_days,
                   timezone="Asia/Kolkata"))
    df = pd.DataFrame(js["hourly"]).rename(columns=RE_COLS)
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
