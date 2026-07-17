"""
Step 1: Data preparation for FlexTrade load forecasting.

- Converts the 40MB SLDC Excel (5-min Delhi load) to a clean parquet file.
- Cleans obvious sensor errors (zeros, spikes) and resamples to 15-min blocks
  (the IEX DAM market time-block resolution).
- Loads hourly weather and interpolates it to 15-min resolution.
- Saves a single merged modelling table: data/model_table.parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)


def load_sldc() -> pd.DataFrame:
    cache = DATA / "sldc_5min.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    print("Reading SLDC Excel (one-time, ~2-4 min)...")
    df = pd.read_excel(
        RAW / "sldc_data.csv.xlsx",
        sheet_name="sldc_data",
        usecols=["date", "timeslot", "delhi", "brpl", "bypl", "ndpl", "ndmc", "mes"],
    )
    ts = pd.to_datetime(
        df["date"].astype(str).str.slice(0, 10) + " " + df["timeslot"].astype(str),
        errors="coerce",
    )
    df = df.assign(ts=ts).dropna(subset=["ts"]).drop(columns=["date", "timeslot"])
    df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")
    df = df.apply(pd.to_numeric, errors="coerce")
    df.to_parquet(cache)
    return df


def clean_load(df: pd.DataFrame) -> pd.Series:
    """Delhi total load in MW at 15-min resolution, cleaned."""
    load = df["delhi"].copy()
    # Delhi load realistically stays within ~1.5-9 GW; zeros/negatives are telemetry drops
    load[(load < 1000) | (load > 9500)] = np.nan
    # Kill isolated spikes: > 20% jump vs both neighbours within 5 minutes
    d_prev = load.diff().abs() / load.shift()
    d_next = load.diff(-1).abs() / load.shift(-1)
    load[(d_prev > 0.20) & (d_next > 0.20)] = np.nan

    load = load.resample("15min").mean()
    # Interpolate short gaps only (up to 2 hours); long outages stay NaN
    load = load.interpolate(limit=8, limit_direction="both")
    load.name = "load_mw"
    return load


def load_weather() -> pd.DataFrame:
    w = pd.read_csv(RAW / "Load Forecasting dataset C&I Consumers.csv",
                    skiprows=2, encoding="utf-8")
    w.columns = ["ts", "temp_c", "rh_pct", "apparent_temp_c", "rain_mm", "cloud_pct"]
    w["ts"] = pd.to_datetime(w["ts"])
    w = w.set_index("ts").sort_index()
    # Weather file is UTC; SLDC timestamps are IST. Shift weather to IST.
    w.index = w.index + pd.Timedelta(hours=5, minutes=30)
    w = w.resample("15min").interpolate(limit=8)
    return w


def main():
    sldc = load_sldc()
    print(f"SLDC raw: {sldc.shape[0]:,} rows, {sldc.index.min()} -> {sldc.index.max()}")

    load = clean_load(sldc)
    weather = load_weather()
    print(f"Weather:  {weather.shape[0]:,} rows, {weather.index.min()} -> {weather.index.max()}")

    table = pd.concat([load, weather], axis=1).dropna(subset=["load_mw"])
    table = table.loc[weather.index.min(): weather.index.max()]
    table[["temp_c", "rh_pct", "apparent_temp_c", "rain_mm", "cloud_pct"]] = (
        table[["temp_c", "rh_pct", "apparent_temp_c", "rain_mm", "cloud_pct"]]
        .ffill(limit=8)
    )
    table.to_parquet(DATA / "model_table.parquet")
    print(f"Saved model table: {table.shape[0]:,} rows x {table.shape[1]} cols")
    print(table.describe().round(1))


if __name__ == "__main__":
    main()
