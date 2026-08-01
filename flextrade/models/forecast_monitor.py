"""Realized forecast accuracy — how did our *actual issued* forecasts do?

Backtests replay history with today's model; this monitor is stricter: it
scores the forecasts we actually shipped. Every daily pipeline run writes
plan_<date>.csv (the 96-block load + price forecast the bid sheet was
built from). Once the delivery day has passed, actuals exist in the store
— so each plan can be settled against reality:

    load : realized MAPE vs SLDC actual load
    price: realized MAPE + shape correlation vs actual DAM MCP

This is the metric a customer would hold us to (a forecast issued at bid
time, scored on what happened), and it can only be produced by a system
that has genuinely been running — a backtest cannot fake it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ingest import store  # noqa: E402

OUT = HERE.parent / "output"


def realized() -> pd.DataFrame:
    plans = sorted(OUT.glob("plan_????-??-??.csv"))
    if not plans:
        return pd.DataFrame()

    load = store.read("load_5min")["delhi"].resample("15min").mean()
    dam = store.read("dam_price")["mcp_rs_mwh"]

    rows = []
    for f in plans:
        day = pd.Timestamp(f.stem.replace("plan_", "")).date()
        plan = pd.read_csv(f, parse_dates=["ts"], index_col="ts")
        actual_load = load[load.index.date == day]
        actual_mcp = dam[dam.index.date == day]
        row = {"delivery_day": str(day), "issued": f.stat().st_mtime}

        if "forecast_load_mw" in plan and len(actual_load) >= 48:
            j = plan["forecast_load_mw"].to_frame().join(
                actual_load.rename("actual"), how="inner").dropna()
            if len(j):
                row["load_mape_pct"] = round(float(
                    (np.abs(j["forecast_load_mw"] - j["actual"]) / j["actual"])
                    .mean() * 100), 2)
                row["load_blocks"] = len(j)
        if "forecast_mcp" in plan and len(actual_mcp) >= 48:
            j = plan["forecast_mcp"].to_frame().join(
                actual_mcp.rename("actual"), how="inner").dropna()
            if len(j) >= 8:
                row["price_mape_pct"] = round(float(
                    (np.abs(j["forecast_mcp"] - j["actual"])
                     / np.maximum(j["actual"], 100)).mean() * 100), 2)
                row["price_corr"] = round(float(
                    np.corrcoef(j["forecast_mcp"], j["actual"])[0, 1]), 3)
                row["price_blocks"] = len(j)
        if len(row) > 2:
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df["issued"] = pd.to_datetime(df["issued"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
        df.to_csv(OUT / "realized_accuracy.csv", index=False)
    return df


if __name__ == "__main__":
    df = realized()
    if not len(df):
        print("No settled plan days yet — plans exist only for days whose "
              "actuals haven't arrived, or none were issued.")
    else:
        print("Realized accuracy of forecasts actually issued (scored vs actuals):")
        print(df.to_string(index=False))
