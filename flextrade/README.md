# FlexTrade — AI Energy Trading & Optimization Platform (Hackathon Build)

Implements the FlexTrade business model (`../FlexTrade_Business_Model.pdf`):
an AI-driven trading & optimization platform for India's power markets,
serving **all five customer segments** on **live grid & market data**.

## Business model → running code

| Business model element | Where it runs |
|---|---|
| DAM price forecasting + bidding optimization (§2) | `models/price_model.py`, `optimize/dispatch.py` |
| RTM market intelligence (§2) | `ingest/iex.py::get_rtm_today`, Trader tab |
| Renewable scheduling / DSM minimization (§2, RE developers §4) | `models/re_model.py` — digital twin + DSM calculator on real NWP forecast error |
| BESS dispatch optimization (§4) | LP optimizer + bid sheet, BESS tab |
| DISCOM / C&I demand intelligence (§4) | Load model (4.98% MAPE) + predictive peak windows |
| **Forecast-as-a-Service** (§3.3, primary stream) | `api.py` — REST API, tiered keys (Starter/Professional/Enterprise per §8), metered usage |
| **Asset optimization revenue share** (§3.4) | BESS tab: fee = share% × (FlexTrade P&L − customer baseline P&L), measured by the backtest |
| Transaction fee / brokerage (§3.2) | Bid sheet = execution artifact; fee per MWh routed |
| Data & market intelligence (§3.7) | 13 months of scraped IEX history + DAM/RTM spread analytics |

Closed loop: **Predict → Decide → Trade → Prove**, running on **live Indian
grid & market data** with cached fallback.

```
                LIVE SOURCES                        MODELS                 DECISION
┌──────────────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────┐
│ Open-Meteo forecast API (JSON)   │──▶│ Load model (LightGBM,    │──▶│ LP dispatch      │
│ Delhi SLDC realtime + day curves │──▶│   MAPE ~5%)              │   │ optimizer (PuLP) │
│ IEX DAM snapshot (96 blocks/day) │──▶│ Price model (LightGBM)   │──▶│ → DAM bid sheet  │
└──────────────────────────────────┘   └──────────────────────────┘   └──────────────────┘
          │  SQLite cache (data/flextrade.db)                                 │
          └────────────────────▶  Streamlit dashboard (app.py)  ◀─────────────┘
```

## Live data sources (all verified working)

| Source | What | How |
|---|---|---|
| [Open-Meteo](https://open-meteo.com) forecast + archive APIs | Delhi weather, hourly | official free JSON API |
| [Delhi SLDC](https://www.delhisldc.org) | realtime load (5-sec page) + historical 5-min day curves | HTML scrape (`Loaddata.aspx?mode=DD/MM/YYYY`) |
| [IEX India](https://www.iexindia.com) | DAM 96-block prices, any delivery day | server-rendered table scrape (`?dp=SELECT_RANGE&fromDate=…`) |

Every fetcher: live fetch → on failure, latest cached rows from SQLite +
"CACHED" badge in the dashboard. The demo cannot die on stage.

## Run it

```bash
pip install -r requirements.txt

# one-time (~30 min): seeds DB from hackathon files, backfills SLDC load
# to yesterday, weather archive, and 13 months of IEX DAM prices
python bootstrap_history.py

# train the price model + build the backtest proof
python models/price_model.py
python backtest/backtest.py

# the daily cycle (run any time before DAM gate closure ~12:00 IST):
# live refresh → forecast tomorrow (load + price) → LP schedule → bid sheet
python run_pipeline.py

# operator dashboard (5 segment tabs)
streamlit run app.py

# Forecast-as-a-Service API  →  http://localhost:8100/docs
uvicorn api:app --port 8100
```

(The load model itself is trained in `../load_forecast/` — see that README.)

## What each piece proves

- **ingest/** — real integration with grid/market data, resilient by design
- **models/load_model.py** — genuine day-ahead Delhi load forecast for
  *tomorrow*, using only bid-time-valid features (lags ≥ 48 h + live
  weather forecast). Test MAPE 4.98 %, R² 0.957.
- **models/price_model.py** — DAM MCP forecast per block, trained on 13
  months of scraped IEX history (price lags ≥ 1 day are bid-time valid
  because D+1 prices clear at ~13:00 on D).
- **optimize/dispatch.py** — LP (PuLP/CBC): SoC dynamics, split
  efficiencies, power/energy limits, degradation cost, optional
  peak-shaving constraint. Emits the 96-block DAM bid sheet.
- **backtest/backtest.py** — replays history: schedule on *forecast*
  prices, settle at *actual* prices; benchmarked against greedy and the
  perfect-foresight upper bound.
- **app.py** — operator dashboard: live badges, tomorrow's plan, bid
  sheet download, backtest P&L, model accuracy. Sidebar re-sizes the BESS
  and re-solves the LP interactively.

## Demo script (3 minutes)

1. Open dashboard → point at the **LIVE badges**: SLDC load ticking, IEX
   MCP curve for today, weather API.
2. "At 11 AM the platform forecasts tomorrow's 96 blocks — load *and*
   price — and solves an LP for the battery schedule."
3. Show the schedule chart: buy the midday solar trough, sell the evening
   peak; SoC stays feasible; degradation is costed.
4. Download the **bid sheet** — "this is what goes to IEX at gate closure."
5. Backtest panel: "over the last N days this strategy would have made
   ₹X lakh on a 20 MW/40 MWh asset — Y % of the theoretical maximum."
6. Drag the sidebar sliders — 100 MW/6 h utility-scale asset → LP re-solves
   in seconds. "Same engine scales across the portfolio."
