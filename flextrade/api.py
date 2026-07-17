"""FlexTrade Forecast-as-a-Service API.

The standalone monetizable data product from the business model (§3.3):
price, demand, and renewable generation forecasts plus dispatch
optimization, sold via tiered API keys (§8 pricing: Starter /
Professional / Enterprise). Usage is metered per key per endpoint — the
basis for usage-based billing.

Run:  uvicorn api:app --port 8100
Docs: http://localhost:8100/docs
"""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ingest import iex, store  # noqa: E402
from models import load_model, price_model, re_model  # noqa: E402
from optimize.dispatch import Bess, bid_sheet, optimize_dispatch  # noqa: E402

app = FastAPI(
    title="FlexTrade Forecast-as-a-Service",
    description="Price, demand and RE generation forecasts for India's "
                "power markets. Tiered access per the FlexTrade pricing model.",
    version="0.1.0",
)

# demo keys per pricing tier (§8 of the business model)
API_KEYS = {
    "demo-starter": {"tier": "Starter", "endpoints": {"prices"}},
    "demo-professional": {"tier": "Professional",
                          "endpoints": {"prices", "load", "price_forecast",
                                        "renewables"}},
    "demo-enterprise": {"tier": "Enterprise", "endpoints": "*"},
}


def _meter(key: str, endpoint: str) -> None:
    with store.connect() as con:
        con.execute("CREATE TABLE IF NOT EXISTS api_usage "
                    "(key TEXT, endpoint TEXT, at TEXT)")
        con.execute("INSERT INTO api_usage VALUES (?,?,?)",
                    (key, endpoint, datetime.now().isoformat(timespec="seconds")))


def auth(endpoint: str):
    def check(x_api_key: str = Header(description="e.g. demo-professional")):
        acct = API_KEYS.get(x_api_key)
        if acct is None:
            raise HTTPException(401, "unknown API key")
        if acct["endpoints"] != "*" and endpoint not in acct["endpoints"]:
            raise HTTPException(403, f"endpoint '{endpoint}' requires a higher "
                                     f"tier than {acct['tier']}")
        _meter(x_api_key, endpoint)
        return acct
    return check


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.round(2).reset_index()
    df.columns = ["ts", *df.columns[1:]]
    df["ts"] = df["ts"].astype(str)
    return df.to_dict("records")


@app.get("/v1/prices/dam", tags=["market data"])
def dam_prices(acct=Depends(auth("prices"))):
    """Today's IEX Day-Ahead Market clearing prices (96 blocks, live)."""
    df, meta = iex.get_today()
    return {"live": meta["live"], "blocks": _records(df)}


@app.get("/v1/prices/rtm", tags=["market data"])
def rtm_prices(acct=Depends(auth("prices"))):
    """Today's IEX Real-Time Market prices (live)."""
    df, meta = iex.get_rtm_today()
    return {"live": meta["live"], "blocks": _records(df)}


@app.get("/v1/forecast/load", tags=["forecasts"])
def load_forecast(target: date | None = Query(None), acct=Depends(auth("load"))):
    """Day-ahead Delhi load forecast, 96 x 15-min blocks (MW)."""
    df = load_model.forecast_day(target)
    return {"target": str(df.index[0].date()), "model": "LightGBM, MAPE 4.98%",
            "blocks": _records(df)}


@app.get("/v1/forecast/price", tags=["forecasts"])
def price_forecast(target: date | None = Query(None),
                   acct=Depends(auth("price_forecast"))):
    """Day-ahead DAM MCP forecast, 96 blocks (Rs/MWh)."""
    df = price_model.forecast_day(target)
    return {"target": str(df.index[0].date()), "model": "LightGBM log-MCP",
            "blocks": _records(df)}


@app.get("/v1/forecast/renewables", tags=["forecasts"])
def re_forecast(target: date | None = Query(None),
                acct=Depends(auth("renewables"))):
    """Day-ahead solar + wind generation forecast for the reference
    100 MW portfolio (digital twin on live NWP weather)."""
    df = re_model.forecast_day(target)
    return {"target": str(df.index[0].date()),
            "portfolio": "50 MW solar + 50 MW wind (Delhi NCR)",
            "blocks": _records(df)}


@app.get("/v1/dsm/report", tags=["renewable scheduling"])
def dsm(day: date | None = Query(None), acct=Depends(auth("dsm"))):
    """Deviation-settlement comparison for a past day: FlexTrade forecast
    vs naive persistence — the penalty saved is the billable value."""
    df, flex, naive = re_model.dsm_comparison(day)
    return {"day": str(day or (date.today() - timedelta(days=1))),
            "flextrade": flex, "naive_persistence": naive,
            "penalty_saved_rs": round(naive["penalty_rs"] - flex["penalty_rs"], 2)}


class DispatchRequest(BaseModel):
    power_mw: float = 20.0
    energy_mwh: float = 40.0
    round_trip_eff: float = 0.90
    degradation_rs_mwh: float = 200.0
    target: date | None = None


@app.post("/v1/optimize/dispatch", tags=["optimization"])
def dispatch(req: DispatchRequest, acct=Depends(auth("dispatch"))):
    """LP-optimal BESS schedule + DAM bid sheet for the target day
    (Enterprise tier — pairs with the revenue-share model)."""
    prices = price_model.forecast_day(req.target)["forecast_mcp"]
    bess = Bess(power_mw=req.power_mw, energy_mwh=req.energy_mwh,
                round_trip_eff=req.round_trip_eff,
                degradation_rs_mwh=req.degradation_rs_mwh)
    sched, pnl = optimize_dispatch(prices, bess)
    bids = bid_sheet(sched, prices)
    bids = bids.astype(object).where(bids.notna(), None)  # NaN -> JSON null
    return {"target": str(prices.index[0].date()),
            "expected_pnl_rs": round(pnl, 0),
            "schedule": _records(sched),
            "bid_sheet": bids.to_dict("records")}


@app.get("/v1/usage", tags=["account"])
def usage(x_api_key: str = Header()):
    """Metered API calls for this key — the usage-based billing feed."""
    with store.connect() as con:
        df = pd.read_sql("SELECT endpoint, COUNT(*) n FROM api_usage "
                         "WHERE key=? GROUP BY endpoint", con,
                         params=(x_api_key,))
    return {"key": x_api_key, "calls": df.to_dict("records")}
