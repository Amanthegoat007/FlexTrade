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


def fetch_forecast_archive(start: str, end: str, lead_days: int = 1) -> pd.DataFrame:
    """What the forecast SAID `lead_days` before each hour, for a past window.

    Every accuracy number this project has published for the load model was
    computed against `kind="actual"` weather, because that is all the store
    held before 2026-07-07. That is perfect foresight: a day-ahead forecast
    issued before the 12:00 gate cannot know tomorrow's temperature, it knows
    tomorrow's temperature FORECAST. Scoring against the actual credits the
    model with weather skill it never had.

    Open-Meteo archives its own past runs, so the honest input is recoverable
    for the whole history rather than only from the day we started saving it.
    Measured at Delhi over 2026-05-20..21, the D-1 forecast differs from the
    analysis by 0.77 C MAE and the D-2 forecast by 1.10 C — error growing with
    lead time is the signature of a real forecast archive, not reanalysis
    relabelled.

    Returned with kind="forecast_d{lead}" so it sits beside the actuals rather
    than overwriting them: the comparison between the two IS the measurement of
    how much of our accuracy was foresight.
    """
    if not 1 <= lead_days <= 7:
        raise ValueError("lead_days must be 1..7 — Open-Meteo archives 7 runs")
    fields = [f"{v}_previous_day{lead_days}" for v in HOURLY.split(",")]
    js = _get("https://historical-forecast-api.open-meteo.com/v1/forecast",
              dict(latitude=LAT, longitude=LON, hourly=",".join(fields),
                   start_date=start, end_date=end, timezone="Asia/Kolkata"),
              timeout=90)
    df = pd.DataFrame(js["hourly"])
    df["ts"] = pd.to_datetime(df.pop("time"))
    # strip the _previous_dayN suffix, then map to our column names
    df = df.rename(columns={c: COLS.get(c.rsplit("_previous_day", 1)[0],
                                        c.rsplit("_previous_day", 1)[0])
                            for c in df.columns if c != "ts"})
    return df.set_index("ts").sort_index()


_FCST_SCHEMA = """
CREATE TABLE IF NOT EXISTS weather_fcst (
    ts TEXT NOT NULL,
    lead_days INTEGER NOT NULL,
    temp_c REAL, rh_pct REAL, apparent_temp_c REAL,
    rain_mm REAL, cloud_pct REAL, fetched_at TEXT,
    PRIMARY KEY (ts, lead_days));
"""


def backfill_forecast_weather(start: str, end: str, lead_days: int = 1,
                              chunk_days: int = 120) -> int:
    """Store the D-`lead_days` forecast for a past window, in chunks.

    Deliberately a SEPARATE TABLE, not kind="forecast_dN" inside `weather`.
    That table's primary key is ts ALONE — not (ts, kind) — so an
    INSERT OR REPLACE carrying a forecast row would silently overwrite the
    ACTUAL reading for the same hour and destroy five years of observations
    one chunk at a time, with no error and nothing in the diff to notice.
    weather_fcst keys on (ts, lead_days) instead, so several lead times can
    coexist and the actuals cannot be touched.

    One request per few months keeps each response small and lets a failure
    cost a chunk rather than the run.
    """
    with store.connect() as con:
        con.executescript(_FCST_SCHEMA)
    total = 0
    cur, stop = pd.Timestamp(start), pd.Timestamp(end)
    while cur <= stop:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days - 1), stop)
        try:
            df = fetch_forecast_archive(f"{cur:%Y-%m-%d}", f"{chunk_end:%Y-%m-%d}",
                                        lead_days).dropna(how="all")
            if len(df):
                df = df.reset_index()
                df["ts"] = df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
                df["lead_days"] = lead_days
                df["fetched_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                cols = [c for c in ("ts", "lead_days", "temp_c", "rh_pct",
                                    "apparent_temp_c", "rain_mm", "cloud_pct",
                                    "fetched_at") if c in df.columns]
                with store.connect() as con:
                    con.executemany(
                        f"INSERT OR REPLACE INTO weather_fcst ({','.join(cols)}) "
                        f"VALUES ({','.join('?' * len(cols))})",
                        df[cols].where(pd.notna(df[cols]), None)
                        .itertuples(index=False, name=None))
                total += len(df)
            print(f"  {cur:%Y-%m-%d} -> {chunk_end:%Y-%m-%d}: {len(df)} hours",
                  flush=True)
        except Exception as e:
            print(f"  {cur:%Y-%m-%d} -> {chunk_end:%Y-%m-%d}: FAILED "
                  f"{type(e).__name__}: {str(e)[:80]}", flush=True)
        cur = chunk_end + pd.Timedelta(days=1)
    return total


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
