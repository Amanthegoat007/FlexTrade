# Data discovery — what would actually make the multi-state layer sellable

Written 2 Aug 2026. Every endpoint below was **probed live**, not collected from
documentation. Findings are split into what works today, what is documented but
broken, and what is genuinely unavailable — because a source list that does not
say which is which is worth nothing.

---

## 1. The diagnosis, before the shopping list

The multi-state layer is not weak because we failed to find sources. It is weak
for three specific reasons, and only one of them is a discovery problem:

| Problem | Detail | Is more discovery the fix? |
|---|---|---|
| **Resolution** | MERIT publishes **daily** energy by procurement source. One number per state per day. | **No** — the data does not exist at 15-min in any public archive we found. |
| **Depth** | ~340 usable days per state, one seasonal cycle. Every metric sits on a single summer. | **No** — this is calendar time. |
| **No state-level price** | IEX clears **one pan-India price**. There is no per-state price to forecast, so "state trading" has nothing to trade against. | **Partly** — area prices exist but are not currently reachable (see §3). |

A daily state energy number **cannot drive a 15-minute market**. It sizes a
market and screens opportunities; it does not trade. That is the honest ceiling
of the current multi-state layer and no amount of extra daily data lifts it.

### The thing that actually changes this

We are **already accruing the missing dataset**. The MERIT poller has been
writing 23 states at a **15-minute cadence** since 24 July:

```
state_live: 23,277 rows · 23 states · 9.6 days · 15.0 min cadence
projection: +1 month  →  ~73k rows   (40 days)
            +3 months →  ~218k rows  (100 days)
            +6 months →  ~436k rows  (190 days)
```

At ~100 days this becomes a real pooled **intraday** multi-state panel — the
same shape that gets Delhi to 4.33%. **The binding constraint is calendar time,
not discovery.** Which means the highest-value engineering action is not adding
sources: it is making sure that poller never misses a block, because every gap
is a day of history that cannot be bought back.

---

## 2. Verified working (probed 2 Aug 2026)

### Vidyut PRAVAH — Ministry of Power live dashboard
Undocumented but real JSON controllers behind `vidyutpravah.in`:

| Endpoint | Method | Returns | Verified |
|---|---|---|---|
| `/PXDashboard/BindTopStatisticsFromJS` | POST | national demand (GW), yesterday's demand, frequency, UI rate | ✅ 1,512 B |
| `/PXDashboard/BindCurrentDateTimeForJson` | POST | the current **15-minute block** stamp | ✅ `{"FromTime":"15:00","ToTime":"15:15","CurrentDate":"02 AUG 2026"}` |
| `/PXDashboard/BindStatePricesFromJS` | POST | per-state price (IDs `MHA_Price`, `DL_Price`, … exist in the DOM) | ⚠️ returns `[]` — feed appears idle |
| `/PXDashboard/GetShortageDetailDataForJson` | POST | shortage detail | ❌ 500 on every parameter tried |

**Value:** the block stamp confirms MoP publishes at 15-min granularity, and the
per-state price DOM slots prove a state price feed is *intended* to exist. Worth
re-probing periodically — if `BindStatePricesFromJS` ever populates, it is the
single most valuable field on this page for us.

### CEA — official documented API
`https://cea.nic.in/api-for-central-electricity-authority-data/` lists 12 keyless
endpoints, including `psp_energy.php`, `psp_peak.php`,
`installed_capacity_statewise.php`, `power_generation.php`, `renewable_energy.php`.

**Status: documented, keyless, and currently returning `504 Gateway Timeout`** on
every endpoint, from a warmed session with a 120 s timeout. This is an upstream
outage, not a block on us. It is the best-quality state-wise structural data
India publishes and should be retried on a schedule rather than written off.

---

## 3. Probed and NOT usable today

| Source | What we tried | Result |
|---|---|---|
| **WRLDC** `OnlinestateTest1.aspx/GetRealTimeData_state_Wise` | POST `{date:'YYYY-MM-DD'}`, 6 date formats | Endpoint **parses** dates (HTTP 500 on `dd-mm-yyyy`, 200 on ISO) but returns **0 rows** for every date tried. Page appears deprecated. |
| **WRLDC** `onlinestate.aspx` | direct GET | ASP.NET WebForms + UpdatePanel; data is not in the initial HTML, needs a ViewState postback. Scrapeable with effort. |
| **IEX area prices** `/day-ahead-market/area-price` | same params as the working market-snapshot scraper | 200 but **81 KB SPA shell with zero data** — unlike market-snapshot, which server-renders 257 KB. Area prices are the one real route to a state-level price signal, so this is the highest-value blocked item. |
| **Grid-India** `grid-india.in` | GET | SSL handshake failure |
| **India-WRIS reservoir** `indiawris.gov.in/Dataset/Reservoir` | POST, contract fully mapped by walking its own 400s | **Contract known, data not served.** It is a Spring endpoint taking `stateName`, `districtName`, `agencyName`, `startdate`, `enddate`, `download`, `page`, `size` as **query params** (not a JSON body). With all of them supplied it returns `{"statusCode":500,"message":"Data NOT Fetch"}` for every state and date range tried — probably needs a real `districtName` rather than an empty one. Worth one more pass with a district list, since reservoir storage is the single feature most likely to fix Himachal. |

---

## 4. Ranked recommendations

**1. Protect the 15-min MERIT poller above everything else.** It is silently
building the only intraday multi-state dataset that will exist. A missed week is
a week that cannot be recovered. Alert on gaps, not just on failures.

**2. Add exogenous drivers we cannot accrue ourselves.** Our models see demand,
weather and price. They do not see *supply-side* state, which is what actually
moves Indian prices:

- **CWC / India-WRIS reservoir storage** (weekly, per reservoir) — directly drives
  hydro availability. Would plausibly fix Himachal, the one state we currently
  serve baseline-only because its driver is unobserved.
- **CEA daily coal stock** — thermal availability, the dominant price driver.
- **CEA/NLDC unit outage data** — a large unit tripping is a step change in price
  that no calendar feature can anticipate.

These are *low-frequency* but high-signal, and they are the class of feature most
likely to move the price model off 20% MAPE, because the residual there is supply
shocks rather than demand error.

**2a. The bid-margin study is the template for the rest.** The first supply-side
question we answered with the new order ledger — "is our ±10% bid margin right?" —
turned out to be worth **+10% realised P&L**, and it needed no new data at all,
only a way to measure a decision we had never measured. Before buying new feeds,
it is worth asking which existing decisions are still set by assumption.

**3. Re-probe CEA and `BindStatePricesFromJS` on a schedule.** Both are real and
both are currently idle. Cheap to retry, high payoff if they wake up.

**4. Do not promise state-level trading until a state-level price exists.** With
one pan-India clearing price, "trade any state" is not a product. Sell the
multi-state layer as **market screening and demand intelligence**, which is what
the data honestly supports, and keep trading claims to the assets and markets we
can actually settle against.

---

## 5. What this means for the pitch

The defensible claim today is: *one state proven at 15-minute resolution, a
national screening layer at daily resolution, and a running 15-minute collector
that turns the second into the first over the next quarter.* That is a real moat —
nobody else is accruing this panel — and it is honest about what is not ready.
