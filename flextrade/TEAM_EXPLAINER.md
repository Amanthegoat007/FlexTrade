# FlexTrade — Complete Technical Explainer (for the team)

The system is one loop, run every day before the IEX day-ahead-market gate
closes (~12:00 IST):

> **PREDICT** tomorrow's load, prices, and RE generation →
> **DECIDE** the optimal battery schedule (LP optimizer) →
> **TRADE** = emit the 96-block DAM bid sheet →
> **PROVE** = dashboard, backtest P&L, metered API.

---

## 1. Data: what is live, what is not

| Source | What we get | Live? | How |
|---|---|---|---|
| Delhi SLDC (delhisldc.org) | Delhi load right now (updates every few sec) + full 5-min load curve of any past day | **LIVE** (scrape) | realtime page + `Loaddata.aspx?mode=DD/MM/YYYY` |
| IEX India (iexindia.com) | DAM & RTM clearing prices, 96×15-min blocks, any delivery date | **LIVE** (scrape) | the new site server-renders the table into HTML; `?dp=SELECT_RANGE&fromDate=…` gives history |
| Open-Meteo forecast API | Tomorrow's hourly weather for Delhi (temp, humidity, rain, cloud, irradiance GHI/DNI/DHI, wind @100 m) | **LIVE** (official free JSON API) | `api.open-meteo.com/v1/forecast`, `timezone=Asia/Kolkata` |
| Open-Meteo archive API | Past actual weather (ERA5), lags ~5 days; `past_days=7` on the forecast API bridges the gap | **LIVE** | `archive-api.open-meteo.com` |
| Open-Meteo Previous Runs API | What the weather model predicted *one day earlier* for each hour (`*_previous_day1`) | **LIVE** | used to measure honest day-ahead RE forecast error |
| `sldc_data.csv.xlsx` (hackathon file) | Delhi 5-min load Jun 2021 → Jun 2026, ~504k rows | static seed | loaded once into the DB, then extended live to yesterday |
| "Load Forecasting dataset C&I Consumers.csv" | Hourly Delhi weather 2021–2026 (it is an Open-Meteo export; contains NO load despite the name) | static seed | training features for the load model |
| `DAM_Market Snapshot.xlsx` | One day of DAM prices | static | superseded — we scraped 13 months ourselves |
| Solar/Wind 5-year CSVs | Historical irradiance / wind weather | static | reference for the RE twin; live RE forecasts come from Open-Meteo |

Every live fetcher writes successful pulls to SQLite (`data/flextrade.db`).
If a fetch fails, it serves the last cached rows and the dashboard badge
flips from **LIVE** to **CACHED hh:mm**. That is why the demo cannot die
on stage Wi-Fi.

**Not real / clearly-labeled simplifications:** the RE plant is a modeled
reference portfolio (50 MW solar + 50 MW wind digital twin), not real plant
telemetry; the DSM penalty uses one simplified slab instead of the full
CERC/SERC slab table; bids are simulated (we never place real orders).

---

## 2. Model 1 — Delhi load forecast (`load_forecast/`)

**Problem it solves:** every trading decision starts with "how much power
will be needed tomorrow, block by block." DISCOMs use it for procurement,
the platform uses it for peak prediction and as a price-model input.

**Algorithm: LightGBM** (gradient-boosted decision trees).
Why this and not the LSTM mentioned in the deck: our data is *tabular*
(lags + calendar + weather), where boosted trees are consistently as good
or better than neural nets, train in ~1 minute on a laptop, don't need GPU,
handle missing data natively, and give feature importances you can show a
judge. (Honest answer for Q&A: "we benchmarked complexity vs benefit; LSTM
is a Phase-2 experiment, LightGBM is the production baseline.")

**Target:** Delhi total load in MW, per 15-min block (96/day) — 15 min
because that is the IEX settlement block, so forecast and market align 1:1.

**Cleaning:** drop values outside 1,000–9,500 MW (telemetry drops; Delhi's
real record peak is ~8.6 GW so the band is safe), kill isolated spikes
(>20% jump vs both neighbours), resample 5-min→15-min mean, interpolate
gaps up to 2 h only.

**The leakage rule (most important design decision):** DAM bids for
delivery day D+1 close at ~12:00 on day D. At that moment you do NOT yet
know the full load of day D. So the model is only allowed features that
exist at bid time:

- Load lags ≥ 48 h: same block on D−1, D−2, D−6, D−13 (`lag_2d, lag_3d, lag_7d, lag_14d`)
- Rolling stats of the 7 days ending 2 days back (`roll7d_mean/max/min`)
- Same-block average of the last 4 same weekdays (weekly shape)
- Calendar: block index 0–95, sin/cos of hour and day-of-year (so 23:45 is
  "next to" 00:00 and Dec next to Jan), day-of-week, month, weekend flag,
  Indian public holidays (`holidays` library)
- Weather of the target day — allowed because a weather *forecast* exists
  at bid time; in training we proxy it with actual weather:
  temp, temp² (AC load is non-linear), cooling-degree `max(T−24,0)`,
  heating-degree `max(14−T,0)`, temp×humidity (discomfort), rain, cloud,
  apparent temp, 24 h mean temp (thermal inertia)

**27 features total. Hyperparameters:** up to 3000 trees, learning rate
0.03, 127 leaves, min 50 samples/leaf, 80% row & column subsampling, L2 =
1.0, early stopping on validation (stopped at 284 trees).

**Evaluation — chronological split, never shuffled** (shuffling time series
leaks the future into training):
train Jun 2021→Jun 2025 · validation 6 months · **test = last 6 months
(Dec 2025–Jun 2026), completely unseen.**

| Split | MAPE | RMSE | R² |
|---|---|---|---|
| Train | 2.19% | 112 MW | 0.993 |
| Validation | 4.70% | 338 MW | 0.931 |
| **Test** | **4.98%** | **262 MW** | **0.957** |

**What each metric means:**
- **MAPE** (mean absolute % error): "on average we're off by 4.98%." The
  headline number — comparable across assets of different size. State
  forecasters typically run 2–5% day-ahead, so ~5% from public data only
  is credible.
- **RMSE** (root mean squared error, in MW): punishes big misses more than
  small ones. 262 MW on a 4,000–8,000 MW system ≈ 4–6%.
- **R²**: share of load variance the model explains (1.0 = perfect). 0.957
  = the model tracks almost all the daily/seasonal structure.
- Train ≪ test error is normal (memorization headroom); what matters is
  validation ≈ test, which shows we didn't overfit the tuning.

---

## 3. Model 2 — DAM price forecast (`flextrade/models/price_model.py`)

**Problem it solves:** the battery makes money from price *spreads*. To
schedule tomorrow you need tomorrow's 96 clearing prices (MCP, ₹/MWh).

**Training data:** we scraped **13 months of real IEX DAM history**
(Jun 2025 → Jul 2026, ~37,500 blocks) from the IEX website — no synthetic
prices anywhere.

**Algorithm:** LightGBM again, but the target is **log(MCP)** and we
exponentiate predictions back. Why log: MCP ranges ₹1,000–₹10,000 (the
regulatory price cap); training on the log scale makes a 10% error at
₹2,000 count the same as a 10% error at ₹9,000, instead of the model only
caring about expensive blocks.

**Bid-time validity for prices differs from load:** prices for delivery
day D clear at ~13:00 on D−1. So when bidding for D+1 at noon on D, the
newest known price day is D itself → **price lags ≥ 1 day are legal**
(load lags still need ≥ 2 days).

**25 features:** calendar set (same as load model) + price lags 1/2/7 days
+ 7-day rolling mean/max/std + previous-day mean/max/min + purchase-bid
and sell-bid volumes lagged 1 day + their gap (bid pressure ≈ supply/demand
balance) + Delhi load lag 2d and its weekly mean + temperature and
cooling-degree for the target day.

| Split | MAPE | RMSE | Correlation |
|---|---|---|---|
| Train | 6.2% | 442 ₹/MWh | 0.983 |
| Validation | 30.4% | 1372 ₹/MWh | 0.953 |
| **Test (last 60 days)** | **22.6%** | **1626 ₹/MWh** | **0.919** |

**Why 22.6% is fine (and the honest way to present it):** Indian DAM
prices are brutally volatile — evening blocks pin at the ₹10,000 cap on
heatwave days and crash to ₹1,000 in solar hours. Published day-ahead
price-forecast errors for IEX are typically 15–25%. More importantly, the
optimizer doesn't need the exact price level — it needs the *shape* (which
blocks are cheap, which are expensive). Correlation 0.92 says the shape is
right, and the backtest (below) proves it: we capture 93.5% of the profit
a perfect price forecast would earn. **Metric that matters = ₹ captured,
not MAPE.**

---

## 4. Model 3 — RE generation digital twin (`flextrade/models/re_model.py`)

**Problem it solves:** RE developers must submit day-ahead generation
schedules; deviating beyond the tolerance band incurs **deviation
settlement (DSM) penalties**. Better forecasts = lower penalties. This is
the "Renewable Scheduling" segment of the business model.

**Approach — physics, not ML** (defensible: this is how the industry
models plants without telemetry):

- **Solar (PVWatts-style), 50 MW:** cell temperature
  `T_cell = T_air + GHI·(NOCT−20)/800` with NOCT = 45 °C;
  power = capacity × DC/AC ratio 1.25 × (GHI/1000) ×
  (1 − 0.35%/°C × (T_cell − 25)), × 96% inverter efficiency, clipped at
  50 MW. Inputs: GHI (global horizontal irradiance, W/m²) and air temp.
- **Wind, 50 MW:** wind speed at 100 m hub height (from the API directly),
  standard cubic power curve — 0 below cut-in 3 m/s, rises with v³ to
  rated at 12 m/s, flat to cut-out 25 m/s, 0 above.

**The honest-forecast trick:** to measure real forecast error we compare
the twin driven by *yesterday's model run for today*
(Open-Meteo Previous Runs API, `_previous_day1` variables — the genuine
day-ahead forecast) against the twin driven by today's analysis weather.
No simulated noise anywhere.

**DSM calculator (simplified single slab, parameters configurable):**
tolerance = ±10% of *scheduled* MW per block (floor: 2% of capacity so
night blocks aren't penalized), penalty ₹1,500/MWh on energy outside the
band. Benchmark = "naive persistence" (schedule = same block yesterday),
which is what an unsophisticated operator does. Example result (12 Jul):
FlexTrade ₹36.0k vs naive ₹63.5k → **₹27.5k/day penalty saved on 100 MW**
— that saving is the billable value (Forecast-as-a-Service / revenue
share).

---

## 5. The decision engine — LP dispatch optimizer (`flextrade/optimize/dispatch.py`)

**Problem it solves:** given 96 forecast prices and a battery, when to buy,
when to sell, how hard — respecting physics. This is what makes it a
*trading platform* rather than a forecasting notebook.

**Method: Linear Programming** (PuLP with the CBC open-source solver —
solves in under a second, guaranteed optimal for the model).

- **Decision variables per block t:** charge_t and discharge_t (MW, grid
  side, 0…P_max), SoC_t (MWh stored).
- **Constraints:**
  - SoC dynamics: `SoC_{t+1} = SoC_t + 0.25h·(charge·η − discharge/η)`
    where η = √(round-trip eff) = √0.90 ≈ 0.949 applied on each direction;
  - SoC between 5% and 100% of capacity;
  - end-of-day SoC ≥ starting SoC (you can't book profit by just draining
    the battery you started with);
  - optional peak-shaving mode: charging forbidden in flagged peak blocks.
- **Objective:** maximize
  `Σ 0.25·price·(discharge − charge) − 0.25·deg·(charge+discharge)`,
  i.e., arbitrage profit minus a **degradation cost of ₹200/MWh of
  throughput** — a proxy for battery-life consumption so the optimizer
  doesn't cycle for pennies.
- **Reference asset (all adjustable in the dashboard sidebar):** 20 MW /
  40 MWh (2-hour battery), 90% round-trip efficiency, start at 50% SoC.
- **Output → bid sheet:** per block, BUY (charging), SELL (discharging) or
  idle, volume in MW, and a price limit = forecast ±10% safety margin.
  This CSV is literally what would be submitted to IEX at gate closure.

**Why LP and not rules:** we keep a rule-based "static EMS" as the
baseline — charge in the day's cheapest price quartile, discharge in the
most expensive, simulated chronologically so it stays physically feasible.
The LP beats it by 34% (below) because it sizes positions, respects SoC,
prices degradation, and finds multi-cycle opportunities.

---

## 6. The proof — backtest (`flextrade/backtest/backtest.py`)

**Methodology (this is the honest part — memorize it):** for each of the
last 61 days (15 May – 14 Jul 2026): (1) forecast the 96 prices using only
information legal at bid time, (2) build the LP schedule **on the
forecast**, (3) settle that fixed schedule **at the actual cleared
prices**. Exactly what would have happened if the platform had been live.

Two benchmarks: the static-EMS greedy rule (same forecast), and
**perfect foresight** = LP run on actual prices — the theoretical maximum
no real system can beat. Any end-of-day SoC deficit is charged back
conservatively at the day's max price so no strategy can cheat by
liquidating inventory.

| KPI | Value | Meaning |
|---|---|---|
| Cumulative P&L (61 d) | **₹1.60 Cr** | what the 20 MW/40 MWh BESS would have earned with FlexTrade |
| Static EMS baseline | ₹1.20 Cr | same battery, rule-based dispatch |
| **Uplift** | **+₹40.9 lakh (+34%)** | the value FlexTrade *adds* — the revenue-share billing base |
| Perfect-foresight bound | ₹1.72 Cr | ceiling with a crystal ball |
| **Capture ratio** | **93.5%** | fraction of the theoretical max we actually capture — the single best summary of forecast+optimizer quality |
| Annualized | ~₹9.6 Cr/yr | **caveat: extrapolates a summer window with ₹10,000-cap evenings; quote ₹1.6 Cr/61 days as the hard number** |

**Revenue-share meter:** business model §3.4 says FlexTrade takes a % of
incremental profit. Fee = share% × uplift → at 20%: **₹8.2 lakh** for the
61-day window. The backtest is the billing meter.

---

## 7. Dashboard KPI glossary (`app.py`, port 8765)

**Header:** Delhi load now (live SLDC MW) · Grid frequency (50 Hz ± —
proximity of supply/demand, ancillary-services context) · DAM / RTM avg
MCP today (live IEX) · Peak load forecast D+1 (max of our 96 blocks).

**BESS tab:** Expected arbitrage P&L (LP objective on forecast prices) ·
Energy traded (MWh discharged) · Avg spread captured (avg sell price −
avg buy price; > ~₹1,100/MWh needed to clear efficiency+degradation costs)
· backtest KPIs (table above) · downloadable bid sheet.

**RE tab:** RE energy forecast D+1 (MWh from the twin) · DSM penalty
FlexTrade vs naive · Penalty saved/day.

**DISCOM & C&I tab:** Forecast peak + peak window (the 5% highest blocks
— the *predictive* peak-shaving promise from the deck: know the peak
before it happens) · model accuracy cards.

**Trader tab:** DAM vs RTM curves and per-block RTM−DAM spread (mean
positive spread ⇒ buy-DAM/sell-RTM bias — the market-intelligence product).

**Business tab:** revenue-stream mapping, live API examples, API usage
metering table (the usage-billing feed).

---

## 8. Forecast-as-a-Service API (`api.py`, port 8100, `/docs`)

FastAPI. Demo keys map to the PDF's pricing tiers: `demo-starter` → market
data only; `demo-professional` → + load/price/RE forecasts;
`demo-enterprise` → + dispatch optimization and DSM reports. Wrong tier →
HTTP 403. Every call inserted into an `api_usage` table = metered billing.

---

## 9. Limitations (answer these before the judges ask)

1. Price MAPE 22.6% — volatile market; shape (corr 0.92) is what monetizes; 93.5% capture is the proof.
2. RE plant is a digital twin, not real telemetry — standard industry practice pre-integration; the forecast error is still real NWP error.
3. DSM single slab — real CERC/SERC slabs are step-wise; parameterized, swap-in is trivial.
4. Annualized ₹9.6 Cr overweights summer — quote the 61-day hard number.
5. SLDC/IEX access is scraping, not contracted feeds — exactly the partnership the business model names (§9); the architecture treats them as pluggable adapters.
6. No ancillary-services module — no public Indian AS price feed; roadmap Phase 3.
7. Weather "forecast" in load-model *training* is actual weather (standard proxy); live operation uses the real Open-Meteo forecast.
