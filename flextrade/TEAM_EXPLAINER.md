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
| IEX India (iexindia.com) | DAM, RTM **and GDAM** clearing prices, 96×15-min blocks, any delivery date | **LIVE** (scrape) | the new site server-renders the table into HTML; `?dp=SELECT_RANGE&fromDate=…` gives history |
| Delhi SLDC frequency curve | 5-min system frequency (drives the DSM rate) | **LIVE**, today only | image-map tooltips; date param is ignored by the site, so history must be sampled |
| Delhi SLDC BESS page | BRPL Kilokari battery: MW, kVAr, SoC% | **LIVE**, instantaneous | `bess.aspx`; no history endpoint — sampled by `poll_bess.py` |
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

**DSM calculator — now the real CERC mechanism** (`models/dsm.py`,
structure per the CERC Deviation Settlement and Related Matters
Regulations, 2024). Three things make it genuine rather than a made-up
penalty:

1. **Normal Rate (Reg. 14)** = ⅓ I-DAM ACP + ⅓ RTM ACP + ⅓ ancillary
   charge. We compute the first two from live IEX DAM and RTM prices —
   so the penalty is priced off the actual market, typically
   ₹6,000–7,000/MWh. The ancillary third has no public 15-min feed
   (TRAS is settled internally by NLDC), so it is proxied by RTM and
   flagged `ancillary_proxied` in every result.
2. **Frequency-linked rate (Reg. 8)** — the charge is a *percentage* of
   NR set by system frequency: 115% at/below 49.90 Hz, 100% at 50.00 Hz,
   50% at 50.05 Hz, 0% through 50.10 Hz, −10% above. Encoded at 0.01 Hz
   resolution in `RATE_CURVE`. Economic logic: under-injecting when the
   grid is already short costs the most; over-injecting into a
   high-frequency grid is penalised too.
3. **Deviation basis (Reg. 6(2))** — for a wind-solar seller, deviation
   is a percentage of **available capacity**, not of the schedule. Note
   also Reg. 8(5): a standalone BESS settles at general-seller rates
   (10% of schedule or 100 MW, whichever is lower) — that is how this
   platform's own battery would be charged.

**Frequency data:** scraped from SLDC's frequency curve, which renders as
a chart *image* — but the HTML image-map carries every point as a tooltip
(`Time Slot: 14:40 / Freq Bawana: (49.89)`), so the series is recovered
exactly. Important caveat baked into the code: the page accepts a date
parameter but **ignores it** and always returns today's curve (verified
by requesting two dates and getting byte-identical values). So frequency
history can only be accumulated by sampling daily — never backfilled —
and any day we haven't sampled falls back to 50.00 Hz with
`frequency_observed: False` shown in the UI.

Benchmark = "naive persistence" (schedule = same block yesterday).
Example (20 Jul, 100 MW portfolio): FlexTrade net DSM ₹12,474 vs naive
₹19,244 → **₹6,770/day saved**, priced at a real NR of ₹6,801/MWh. That
saving is the billable value (Forecast-as-a-Service / revenue share).

---

## 4b. Probabilistic forecasting & risk-aware bidding

**The problem with a point forecast.** The LP optimizes against one price
curve as if it were certain, while the price model has ~23% MAPE. A DAM
bid is a *here-and-now decision under uncertainty* — you commit volumes
before the price is known — so the honest formulation is stochastic.

**Quantile models** (`price_model.train_quantiles`). LightGBM's pinball
objective gives P10/P50/P90 per block instead of one number. Trained on
log(MCP), so intervals are multiplicative: wide in the volatile evening
peak, tight in quiet night blocks.

**Conformal calibration — and why it was needed.** The raw quantile band
captured only **50.8%** of actual prices against a nominal 80%. The cause
was not a coding error but a genuine **regime shift**: the test window's
mean MCP is ₹5,062 vs ₹3,742 in training (+35%), and cap-pinned blocks
tripled from 10% to 30%. Models fitted on cheaper history predict low.
Conformalized Quantile Regression (CQR, Romano et al. 2019) fixes this
without retraining — score each calibration point by how far outside the
band it fell, then widen the band by the empirical quantile of those
scores. Done in log space, so the correction is multiplicative. Result:
**50.8% → 81.5% coverage** against an 80% target, mean width ₹1,631 →
₹2,166/MWh. Honest caveat: the coverage guarantee assumes exchangeability,
which time series violate — treat it as well-calibrated in practice, not
as a proof.

**Stochastic dispatch** (`optimize/stochastic.py`). Scenarios are drawn
from the quantile band through a Gaussian copula so prices stay correlated
across blocks (an expensive day is expensive all day). We then solve for
**one** schedule that maximizes a blend of expected profit and **CVaR**
(the mean of the worst 10% of scenarios), linearized by the standard
Rockafellar–Uryasev construction. λ=0 is risk-neutral, λ=1 maximizes the
worst tail alone; the dashboard exposes λ as a slider because risk
appetite is the asset owner's preference, not something to fit.

**What the 57-day backtest actually showed** — report this honestly,
because the single-day preview was far more flattering:

| | total | mean/day | worst day | P10 day | volatility | capture |
|---|---|---|---|---|---|---|
| point forecast LP | ₹1.49 Cr | ₹261,426 | ₹150,984 | ₹211,091 | ₹39,028 | 93.1% |
| stochastic (λ=0) | ₹1.50 Cr | ₹263,235 | ₹145,619 | ₹225,923 | ₹39,313 | 93.7% |
| **CVaR (λ=0.5)** | **₹1.51 Cr** | **₹264,085** | ₹148,376 | ₹222,553 | ₹38,275 | **94.1%** |

So: mean/day **+1.0%**, P10 day **+5.4%**, volatility **−1.9%**, capture
93.1% → 94.1% — but worst day **−1.7%**, i.e. slightly *worse*. At n=57
the "worst day" is a single observation and far too noisy to claim
anything from.

**The interesting part is why the gain is small**, and it is not an
implementation bug — we verified the schedules genuinely differ (74–123
MWh of reallocated energy) and the scenarios have real spread (peak block
₹4,456–10,000 vs night block ₹1,257–1,963). The reason is structural:
**Delhi's day/night spread is so large and so reliable** — night around
₹1,300, evening pinned at the ₹10,000 cap — that *when* to charge and
discharge is obvious almost regardless of the price level. The uncertainty
affects how much you earn, not what you should do. Risk-aware optimization
would matter far more in a market with narrower, less predictable spreads.
That is a legitimate and interesting finding to present, and it is much
stronger than pretending a 1% gain is a breakthrough.

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
positive spread ⇒ buy-DAM/sell-RTM bias — the market-intelligence
product). Plus the **GDAM** panel: green power clears in its own segment,
so the GDAM−DAM spread tells an RE developer whether to sell green or
plain. Live example (21 Jul): mean spread **+₹137/MWh**, swinging from
−₹4,499 to +₹5,475 across the day — i.e. *which segment you sell into
matters more than most developers assume*. The panel also shows the green
sell-bid fuel mix (wind vs other-RE vs hydro), which IEX publishes only
in the GDAM table.

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
3. DSM follows the CERC 2024 *structure* (NR formula, frequency curve, Reg. 6(2) basis), but final notified slabs and SERC variants differ in detail — every band is a constant in `models/dsm.py`. Decision support, not settlement accounting. Frequency defaults to 50.00 Hz on days we hadn't sampled it.
4. Annualized ₹9.6 Cr overweights summer — quote the 61-day hard number.
5. SLDC/IEX access is scraping, not contracted feeds — exactly the partnership the business model names (§9); the architecture treats them as pluggable adapters.
6. No ancillary-services module — no public Indian AS price feed; roadmap Phase 3.
7. Weather "forecast" in load-model *training* is actual weather (standard proxy); live operation uses the real Open-Meteo forecast.

---

## 10. Operations modules (added 24 Jul) — RTM re-opt, degradation physics, C&I

**Intraday RTM re-optimization (`optimize/rtm_reopt.py`).** After the DAM
clears (13:00 on D−1) the position is *financially* firm, but the battery
still has physical flexibility around it. IEX's Real-Time Market (48
half-hourly auctions, gate ~1 h before delivery) prices that flexibility.
The LP re-optimizes every remaining block of today: physical dispatch is the
decision, the *deviation from the DAM position* is what gets traded at RTM
prices (DAM revenue is sunk and excluded), SoC starts from the position
implied by executing the plan so far, and end-of-day SoC must return to the
planned level so tomorrow's plan stays feasible. Expected RTM price =
actual cleared RTM where available, else today's DAM × observed RTM/DAM
ratio (today's blocks → trailing 7 days → 1.0, provenance always labelled).
First real run (24 Jul, 00:01): ₹39k incremental uplift over 92 blocks,
ratio 0.79 from 270 trailing blocks.

**Physics-based degradation (`optimize/degradation.py`).** Cycle life
follows a Wöhler power law L(d) = L₁₀₀·d^−k — deep cycles consume more life
per MWh than shallow ones, so the old flat ₹200/MWh proxy *understated*
cost and over-cycled the asset. We rainflow-count the SoC trajectory (ASTM
E1049 four-point method, 5-assertion selftest) and price each cycle with
LFP datasheet-typical parameters: L₁₀₀ = 6,000 full cycles, k = 1.1, capex
₹1.5 Cr/MWh, 70% of capex attributed to cycling (the rest is calendar aging
that happens anyway). Result: true cost ≈ ₹843/MWh of throughput (~₹1,700
per discharged MWh) — 4× the proxy — and the DoD-dependent (nonconvex) cost
enters the LP by fixed-point calibration of a flat rate, converging in 2
iterations. Honest headline: Delhi's day/night spread still clears the
hurdle (net ₹2–3.7 L/day on real prices), and the uplift vs the proxy
schedule is small precisely because the spread is wide — say that before
the judges do.

**C&I peak shaving (`optimize/peak_shave.py`).** Attacks three lines of a
Delhi HT-industrial bill with a behind-the-meter battery: demand charge
(₹250/kVA/month) via peak cut, ToD peak surcharge (14–17 h & 22–01 h, +20%)
via shifting, off-peak rebate (04–10 h, −20%) via charging. Joint LP:
min[ToD energy + month-projected demand charge], no-export constraint,
degradation at the physics-calibrated ₹843/MWh. Demo: illustrative 5 MW
two-shift factory with a 14–17 h process peak, 2 MW/4 MWh BESS → 1.26 MW
peak cut, ₹11.4k/day net (~₹42 L/yr). Both tariff values and the profile
are labelled indicative — a pilot's meter CSV drops straight in, and
peak-shaving economics are profile-shaped, so the pilot is the real test.

**BESS validation fairness gate (`validate/bess_validate.py`).** Days with
<90% telemetry coverage are labelled PARTIAL and excluded from headline
uplift: sampling that catches only the evening discharge misses the real
battery's charging cost and would flatter it (23 Jul: observed-blocks
revenue ₹2.76 L looked huge for exactly this reason). First fair full-
coverage day accrues automatically now the 5-min poller runs 24/7.

All three surface on the web app's **Operations** page (`/operations`),
exported via `export_web.py → modules.json → /api/modules`.

---

## 11. National coverage + info buttons (added 24 Jul)

**MERIT national layer (`ingest/states.py`).** meritindia.in (Ministry of
Power) exposes POST `/StateWiseDetails/BindCurrentStateStatus
{"StateCode": ...}` → live Demand Met / Own Generation / Import per state,
and `/Dashboard/BindAllIndiaMap` → all-India demand + generation by fuel
(thermal/gas/nuclear/hydro/RE/storage/other/transnational). We discovered
23 state codes by crawling each `/state-data/<slug>` page for its hidden
StateCode field, verified every one with a live call, and **cross-checked
against independent feeds**: Delhi ≈ our SLDC number, Gujarat within 0.5%
of sldcguj.com fetched a minute apart, Himachal correctly shows negative
import (hydro exporter in July). Fetched concurrently (~4 s for all 23),
stored in `state_live` + `national_snapshot` tables every pipeline run —
a growing 23-state demand history nobody hands out as a download.

**Deep adapters beyond MERIT.** Gujarat: sldcguj.com's public homepage
server-renders live frequency + "Gujarat Catered" demand + DAM rate —
scraped directly, no login (`fetch_gujarat_realtime`). Rajasthan: both
endpoints reverse-engineered — `read-sftp` JSON with the tag table taken
from their own page JS (03046004 freq, 03046001 **DSM rate paise/unit**,
03046008 load, 03046009 generation…) currently 500s upstream (their own
homepage widget is broken too; the adapter reports health honestly), plus
a **working** ~151-plant RE injection DataTable by QCA/substation (upstream
timestamps observed stale — surfaced, not hidden). Dead ends recorded:
Maharashtra publishes public SCADA as a JPEG (numeric feeds behind an
"Authorized HO Users" login — partnership route), WRLDC's GetLiveData
webmethod answers but returns null without a browser session, SRLDC's
indexPageDataInEvery5min 500s, vidyutpravah.in timed out repeatedly.

**Why this matters for the pitch:** "Bloomberg for India's power markets"
requires India, not Delhi. One verified government source now gives every
big market live (UP 27.8 GW, MH 20.2 GW, TN 19.1 GW, GJ 16.1 GW, RJ 12.8
GW…), the app shows the national fuel mix including **storage (PSP+BESS)
generation — the very market we optimize**, and the registry tells the
truth about depth vs breadth per state.

**ⓘ info buttons (`client/src/lib/glossary.js`, `InfoTip` in ui.jsx).**
~50-term glossary (DAM, RTM, MCP, SoC, DoD, FCE, CVaR, CQR, DSM, NR,
ToD, QCA, ISGS, X-factor…), each with full form + one-plain-sentence
definition. Hover or keyboard-focus the ⓘ next to any KPI; works in both
themes. Anyone — a judge, a customer, a teammate — can now decode every
short form on screen without asking.

---

## 12. Model lab — how the accuracy improved (24 Jul)

Discipline: one change per experiment, identical chronological splits,
adoption only if the test metric improves AND the mechanism is
explainable. Full leaderboards in `output/model_lab.json` (rendered on
the Methodology page); lab code `models/model_lab.py`.

**Load 4.98% → 4.33% MAPE** (test = 6 unseen months). Adopted stack:
(1) thermal-inertia + growth features — prev-day temperature, 3-day heat
streak, evening cooling-degree interaction, lag ratios, per-block EWM
baseline (AC load depends on how hot it HAS BEEN); (2) recency sample
weights, half-life 180 d (Delhi load grows YoY — old regimes weigh less);
(3) tuned LightGBM (255 leaves, lr 0.015, early stopping); (4) 3-seed
ensemble. Rejected by evidence: relative-target transform (5.14% — the
4-week baseline drags stale weeks in). Production: `load_forecast/
02_train_model.py` + `model_meta.json`; live wrapper averages the
ensemble automatically.

**Price: cap-hurdle two-stage** (P(cap≥₹9.5k) classifier × below-cap
log regression, expectation-combined). Indian DAM is right-censored at
the ₹10,000 cap — a single regression smears it. Test: MAPE 22.0→20.4%,
corr 0.924→0.933, **evening MAPE 15.3→11.4%**, cap-block recall 49→78%.
Beat recency weights and cross-market features (both tried; combos added
complexity, not accuracy). Single `predict_hurdle()` used by forecast_day
AND the backtest — they can never diverge. Walk-forward capture ratio:
93.5% → **93.8%**.

**Honest CVaR flip.** With the sharper point model, scenario-based
CVaR/stochastic dispatch now UNDERPERFORMS the point LP (mean −5.2%,
worst day −28.9% over 55 days): scenarios are drawn from quantile models
that didn't get the hurdle treatment, so they inject noise around a
better centre. Default is point-LP (λ=0); CVaR machinery stays built and
measured, awaiting quantile-hurdle unification. Say it before asked.

**CQR guard self-calibrated.** After retraining with the regime now in
history, the raw P10–P90 band covers 82.5% vs 80% nominal — conformal
margin ×1.00. The guard recomputes at every retrain; it corrected
50.8%→81.5% during the May shift and will fire again at the next one.

**Realized ledger (`models/forecast_monitor.py`).** Forecasts actually
ISSUED (plan_<date>.csv, filed before delivery) scored after the fact:
load 3.66% / 2.58% MAPE, price corr 0.954 with 15.85% MAPE on 24 Jul.
Stricter than any backtest — only a genuinely-running system produces it.

---

## 13. BESS operator pain-points solved (24 Jul) — 5 new modules

Built to answer "what other real problems do BESS systems have?" Each runs
on data we already collect and surfaces on the web app.

**Sizing & Bankability (`models/sizing.py`, Sizing page).** The first
question every customer asks. Arbitrage revenue is linear in power at fixed
duration, so we precompute per-MW daily revenue for 1h/2h/4h over a full
YEAR of real IEX DAM prices (LP with perfect foresight x measured 93.8%
capture ratio), then scale instantly for any plant. Bankability = 10,000-
draw bootstrap of the daily distribution -> P50/P75/P90 annual revenue,
the pessimistic numbers lenders lend against. 20 MW/2h example: ~Rs 33 L/MW/yr
P50. Interactive: pick MW / duration / capex -> revenue, payback, cycles.
DAM-only; RTM/DSM/ancillary/C&I are the labelled stacked upside.

**Warranty & Availability Guard (`models/warranty.py`, Operations).**
Aggressive arbitrage voids warranties (max cycles/day, SoC window). Audits
BOTH the real BRPL telemetry and our own next-day plan against the warranty
envelope. Pre-trade compliance gate: tomorrow's plan flagged (2.03 FCE vs
1.5 limit, SoC to 100% vs 95% ceiling) -> the warranty-safe LP can re-solve
with cycle/SoC caps as hard constraints. Coverage-honest: partial sampling
UNDERcounts cycles, so those days are flagged, never cleared. (Also added a
sanity gate in ingest/bess.py after a mis-parsed "89,100,000%" SoC row got
in — now rejected at ingestion.)

**Three-way DAM+RTM+DSM co-optimization (`optimize/threeway.py`, Operations).**
The intellectually novel one, strengthens the patent. Once DAM is firm, a
deviation can settle via RTM OR via DSM at the Normal Rate (1/3 DAM + 1/3
RTM + 1/3 ancillary) — and these prices invert across the day. No Indian
platform treats DSM as a priced third channel. Deliberately conservative &
COMPLIANT: both deviation legs hard-capped at the +/-10% tolerance band
(an early version capped only the credit side and the LP instantly found
the gaming strategy — sell in RTM, deliver nothing, settle the shortfall
through DSM — exactly the conduct CERC polices; the band cap forbids it).
The chosen deviations are re-settled by the exact versioned dsm.py engine
as independent verification. On 23 Jul's price swings: RTM-only Rs 122k ->
three-way Rs 131.5k (DSM adds Rs 9k across 24 blocks).

**Thermal derating (`optimize/thermal.py`, Operations).** Batteries derate
above 35 C and their HVAC is parasitic load peaking when prices do. Applies
a linear 100%->80% power derate (35-45 C) + 2% aux load to the committed
plan using stored temperature. Honest "what the heat costs today" line.

**Frequency-response readiness (`models/freq_response.py`, Operations).**
Monetizes our UNIQUE sampled frequency history (SLDC serves no archive —
this dataset exists only because our poller built it). Simulates a droop-
controlled battery (dead band +/-0.03 Hz, full response +/-0.15 Hz) and
reports the duty cycle: called on ~52% of samples, ~63 MWh/day. Explicitly
a readiness report, NOT a revenue claim (no public ancillary price exists).

**Bug caught & fixed during this build (matters for the patent claim).**
The RTM re-optimizer (and its three-way copy) anchored SoC on the day-start
value and forced a return to it using only future blocks. Once the morning
discharge had drained the battery and the refill blocks were locked, the LP
was forced to BUY energy to refill -> a spurious NEGATIVE "uplift", which is
impossible (following the plan is always available). Fixed by anchoring on
the DAM plan's OWN stored SoC trajectory and targeting the plan's end-of-day
SoC, so the zero-deviation baseline is always feasible and uplift is provably
>= 0. Verified across the day: Rs 25.3k @ 06:00 (real opportunity), Rs 0 at
midday (plan already optimal) — never negative.
