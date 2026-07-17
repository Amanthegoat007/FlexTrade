"""FlexTrade daily cycle — what runs at ~11:00 IST before DAM gate closure.

  1. Refresh live data: weather forecast, today's IEX DAM, latest SLDC
     load curves, realtime snapshot.
  2. Forecast tomorrow's 96 blocks: Delhi load + DAM MCP.
  3. LP-optimize the BESS schedule on the price forecast.
  4. Emit the DAM bid sheet (output/bid_sheet_<date>.csv) and the full
     plan (output/plan_<date>.csv) that the dashboard displays.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import iex, sldc, store, weather
from models import load_model, price_model
from optimize.dispatch import Bess, bid_sheet, optimize_dispatch

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def refresh_live() -> dict:
    status = {}
    _, status["weather"] = weather.get_forecast(days=2)
    _, status["re_wx"] = weather.get_re_forecast(days=2)
    _, status["iex_dam"] = iex.get_today()
    _, status["iex_rtm"] = iex.get_rtm_today()
    _, status["sldc"] = sldc.get_realtime()
    # pull any missing recent day curves (keeps lags fresh)
    last = store.read("load_5min").index.max().date()
    if last < date.today() - timedelta(days=1):
        sldc.backfill_load(last + timedelta(days=1))
    for k, v in status.items():
        print(f"  {k:8s} {'LIVE' if v['live'] else 'CACHED':6s} asof {v['asof']}")
    return status


def plan_tomorrow(bess: Bess = Bess()) -> pd.DataFrame:
    target = date.today() + timedelta(days=1)
    print(f"planning delivery day {target}")

    load_fc = load_model.forecast_day(target)
    price_fc = price_model.forecast_day(target)
    plan = load_fc.join(price_fc)

    sched, exp_pnl = optimize_dispatch(plan["forecast_mcp"], bess)
    plan = plan.join(sched[["charge_mw", "discharge_mw", "soc_mwh", "bess_mw"]])
    bids = bid_sheet(sched, plan["forecast_mcp"])

    plan.to_csv(OUT / f"plan_{target}.csv", index_label="ts")
    plan.to_csv(OUT / "plan_latest.csv", index_label="ts")
    bids.to_csv(OUT / f"bid_sheet_{target}.csv", index=False)
    bids.to_csv(OUT / "bid_sheet_latest.csv", index=False)

    n_trades = (bids["side"] != "-").sum()
    print(f"  expected P&L Rs {exp_pnl:,.0f} | {n_trades} block bids "
          f"| peak load fc {plan['forecast_load_mw'].max():,.0f} MW")
    return plan


if __name__ == "__main__":
    refresh_live()
    plan_tomorrow()
