"""Day-ahead load forecast using the trained model from load_forecast/.

Reuses the exact feature pipeline of load_forecast/02_train_model.py
(imported by file path since the module name starts with a digit) and
feeds it fresh data from the live SQLite store: SLDC load history +
Open-Meteo actuals/forecast.
"""
import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import store

LF_DIR = Path(__file__).resolve().parent.parent.parent / "load_forecast"
WEATHER_COLS = ["temp_c", "rh_pct", "apparent_temp_c", "rain_mm", "cloud_pct"]

_spec = importlib.util.spec_from_file_location("lf_train", LF_DIR / "02_train_model.py")
lf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lf)

FEATURES = lf.FEATURES
build_features = lf.build_features


def _booster() -> lgb.Booster:
    return lgb.Booster(model_file=str(LF_DIR / "output" / "model.txt"))


def _history_frame(target: date) -> pd.DataFrame:
    """15-min frame from 35 days before target through target 23:45."""
    start = pd.Timestamp(target) - pd.Timedelta(days=35)
    end = pd.Timestamp(target) + pd.Timedelta(hours=23, minutes=45)

    load = store.read("load_5min", since=str(start))["delhi"]
    load = load.resample("15min").mean().rename("load_mw")
    # rolling features need a contiguous history; fill telemetry gaps
    hist_idx = pd.date_range(start, pd.Timestamp(target) - pd.Timedelta(minutes=15),
                             freq="15min")
    load = load.reindex(hist_idx)
    n_missing = int(load.isna().sum())
    if n_missing > len(load) * 0.05:
        print(f"warning: {n_missing} of {len(load)} load blocks gap-filled")
    load = load.interpolate(limit_direction="both").ffill().bfill()

    w = store.read("weather", since=str(start))
    w = w[~w.index.duplicated(keep="first")]
    actual = w[w["kind"] == "actual"][WEATHER_COLS]
    fcst = w[w["kind"] == "forecast"][WEATHER_COLS]
    weather = actual.combine_first(fcst)  # prefer actuals, patch with forecast
    weather = weather.resample("15min").interpolate(limit=8)

    idx = pd.date_range(start, end, freq="15min")
    frame = pd.concat([load, weather], axis=1).reindex(idx)
    frame[WEATHER_COLS] = frame[WEATHER_COLS].ffill(limit=8)
    return frame


def forecast_day(target: date | None = None) -> pd.DataFrame:
    """Predict all 96 blocks of `target` (default: tomorrow)."""
    target = target or (date.today() + timedelta(days=1))
    frame = _history_frame(target)
    feats = build_features(frame)
    day = feats[feats.index.date == target]
    missing = day[FEATURES].isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        raise ValueError(f"missing features for {target}: {missing.to_dict()}")
    out = pd.DataFrame(index=day.index)
    out["forecast_load_mw"] = _booster().predict(day[FEATURES])
    return out


if __name__ == "__main__":
    f = forecast_day()
    print(f.describe().round(1))
    print(f.head())
