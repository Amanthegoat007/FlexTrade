"""One-time history bootstrap into data/flextrade.db.

1. Seed load_5min + weather from the existing hackathon files
   (via load_forecast/data/*.parquet built earlier).
2. Backfill SLDC load from where the file ends -> yesterday (live scrape).
3. Backfill weather actuals over the same gap (Open-Meteo archive API).
4. Backfill 13 months of IEX DAM prices (one request/day, polite pause).

Re-runnable: already-stored days are skipped.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import iex, sldc, store, weather

LOAD_FORECAST_DATA = Path(__file__).resolve().parent.parent / "load_forecast" / "data"
PRICE_HISTORY_MONTHS = 13


def seed_from_files():
    sldc_pq = LOAD_FORECAST_DATA / "sldc_5min.parquet"
    if sldc_pq.exists() and store.read("load_5min").empty:
        df = pd.read_parquet(sldc_pq)
        n = store.upsert("load_5min", df[["delhi", "brpl", "bypl", "ndpl", "ndmc", "mes"]])
        print(f"seeded load_5min from parquet: {n:,} rows")

    table_pq = LOAD_FORECAST_DATA / "model_table.parquet"
    if table_pq.exists() and store.read("weather").empty:
        w = pd.read_parquet(table_pq)[
            ["temp_c", "rh_pct", "apparent_temp_c", "rain_mm", "cloud_pct"]
        ].resample("1h").first().dropna()
        w["kind"] = "actual"
        n = store.upsert("weather", w)
        print(f"seeded weather from parquet: {n:,} rows")


def fill_gaps():
    yesterday = date.today() - timedelta(days=1)

    have_load = store.read("load_5min")
    load_end = have_load.index.max().date() if len(have_load) else date(2021, 6, 21)
    if load_end < yesterday:
        print(f"backfilling SLDC load {load_end + timedelta(days=1)} -> {yesterday}")
        sldc.backfill_load(load_end + timedelta(days=1), yesterday)

    w = store.read("weather")
    w_actual_end = (w[w["kind"] == "actual"].index.max().date()
                    if len(w) else date(2021, 6, 1))
    archive_end = date.today() - timedelta(days=6)  # archive API lags ~5 days
    if w_actual_end < archive_end:
        print(f"backfilling weather archive {w_actual_end} -> {archive_end}")
        arch = weather.fetch_archive(str(w_actual_end), str(archive_end))
        arch = arch.dropna()
        arch["kind"] = "actual"
        print(f"  weather archive: {store.upsert('weather', arch):,} rows")

    start = date.today() - timedelta(days=30 * PRICE_HISTORY_MONTHS)
    print(f"backfilling IEX DAM prices {start} -> today (skips stored days)")
    iex.backfill_prices(start)


if __name__ == "__main__":
    seed_from_files()
    fill_gaps()
    for t in ["load_5min", "weather", "dam_price"]:
        df = store.read(t)
        rng = f"{df.index.min()} -> {df.index.max()}" if len(df) else "EMPTY"
        print(f"{t:10s} {len(df):>9,} rows   {rng}")
