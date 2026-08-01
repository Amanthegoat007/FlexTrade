# FlexTrade — Demo Runbook (Esyasoft Global Innovation 2026, ~5 Aug)

One document to run the demo from. Numbers below are the *hard* numbers as
of 24 Jul — refresh the marked ones on demo morning (they update
themselves in the app; this file is your spoken script).

---

## T-1 day checklist (do the evening before)

- [ ] **Open the app and confirm NO red banner at the top.** The health
      banner fires whenever the last pipeline run failed, the plan isn't
      for tomorrow, or no run succeeded in 36 h. Green = everything below
      is trustworthy. (Added after a real 4-day silent outage — see below.)
- [ ] `Get-ScheduledTask FlexTrade-*` → all three tasks Ready
      (DailyPipeline 11:00 · BessPoller 5 min · StatesPoller 15 min)
- [ ] `Start-ScheduledTask FlexTrade-DailyPipeline` manually once; check
      `logs\pipeline.log` ends with `pipeline OK`
- [ ] Open http://localhost:8090 — every topbar badge LIVE
- [ ] `python models\state_model.py` — states should be **training-ready**
      (gate crosses ~31 Jul); if so, run the first per-state trainings and
      note the MAPEs for the Multi-State talk track
- [ ] `python validate\bess_validate.py 7` — quote ONLY days tagged FAIR
- [ ] **Forecast Lab page loads with all four tabs populated.** If a panel
      shows "not trained", retrain it: `python models\rtm_model.py`,
      `models\load_quantile.py`, `models\peak_model.py`. The DSM tab is
      refreshed by the pipeline itself (`forecast_lab` stage) — check
      `logs\pipeline.log` for `forecast lab: 2/2 ok`.
- [ ] Charge laptop, hotspot as backup network, browser zoom ~110%
- [ ] **Publish a fresh snapshot for anyone not in the room**:
      `python build_static_site.py` → drag `dist-static` to
      <https://app.netlify.com/drop>. 1.3 MB, works with your laptop off.
      Doubles as a demo-day fallback: if the venue network dies, open the
      published URL on a phone hotspot. See `DEPLOY.md`.

## Demo morning (before leaving)

- [ ] Trigger the pipeline once more so plan/date = today+1
- [ ] Hard-refresh the app; confirm "data as of" timestamp is this morning
- [ ] If venue wifi dies mid-demo: badges flip to CACHED with age shown —
      that is a *designed* feature; say so, don't apologize

---

## The 7-minute talk track (one page per minute)

**1 · Overview — "This is live, right now."**
Delhi load, grid frequency, DAM/RTM/GDAM prices, all-India ~212 GW — all
fetched live from SLDC/IEX/MERIT (Ministry of Power). Point at the Data
Freshness table: every timestamp computed from the data itself.
> Key line: "Nothing on this screen is a mock. Kill our network and the
> badges tell you the age of every number."

**2 · Overview bottom — the real battery.**
BRPL Kilokari, India's first utility-scale standalone BESS — 20 MW/40 MWh,
the exact spec we optimize. We sample its telemetry every 5 min because
SLDC keeps no history; we watched it do a real full discharge (81%→4% SoC
at ~19.6 MW) on 23–24 Jul.
> Key line: "Our reference asset isn't hypothetical — it's operating 15 km
> from here and we validate against it block-by-block."

**3 · Trading Desk — Predict → Decide → Trade.**
Load model: **4.33% test MAPE** (LightGBM 3-seed ensemble, recency-weighted,
bid-time-valid features only — lags ≥48 h because the DAM gate is 12:00;
improved from 4.98% in the controlled model-lab, leaderboard on the
Methodology page). Price model: **cap-hurdle two-stage** — P(cap) classifier
× below-cap regression — MAPE 20.4%, **shape correlation 0.93**, evening
MAPE 11.4% (was 15.3%; evenings are where the money is). P10–P90 band is
CQR-guarded: **82.5% coverage vs 80% target** (during the May regime shift
the guard corrected 50.8%→81.5% — we say this unprompted). Backtest, 55
days walk-forward settled at actual prices: **₹1.45 Cr**, capture **93.8%**
of perfect foresight, +29% vs a static rule-based EMS. And the **realized
ledger**: forecasts we actually issued scored 2.6–3.7% load MAPE and 0.95
price correlation — a customer's yardstick, not a backtest.
> Key line: "Every backtest number is settled at actual clearing prices
> with information available at bid time. No hindsight anywhere."

**4 · Operations — the moats (run this page mid-afternoon so RTM has live divergence).**
RTM re-optimization: same battery, second revenue stream (uplift ≥ 0 by
construction — the plan is always an available baseline). **Three-way
DAM+RTM+DSM co-optimization** — the deviation itself is a priced decision;
on the 23 Jul price swings the DSM channel added ₹9k on top of RTM, and it
is re-settled by the exact CERC engine as proof. *To our knowledge nobody
in India productizes DSM as a third settlement channel — this is the patent's
technical-combination heart.* Degradation: rainflow + Wöhler physics says
true cycling cost is **~₹843/MWh throughput, 4× the naive proxy** — and the
strategy still clears it. **Warranty & Availability Guard**: audits the real
BRPL battery AND our own plan against cycle/SoC warranty limits — "we protect
the asset while monetizing it." **Thermal derating** quantifies what the heat
costs. **Frequency-response readiness** monetizes our unique sampled frequency
data (~52% call rate). C&I peak shaving: 1.26 MW peak cut, **₹11.4k/day
(~₹42 L/yr)**.
> Key line: "We capped the DSM deviation legs at the compliance band on
> purpose — an earlier LP found the gaming strategy and we forbade it. We
> optimize inside the rules, not around them."

**4b · Sizing & Bankability — the salable product.**
Pick MW / duration / capex → P50 and **P90 annual revenue** (the lender's
number) + payback, from a full year of real IEX prices × our measured 93.8%
capture. 20 MW / 2h ≈ ₹33 L/MW/yr P50. Everything else (RTM, DSM, ancillary,
C&I) is labelled stacked upside not counted here.
> Key line: "This turns our backtest into a bankability study a lender can
> underwrite — every one of our five customer personas starts here."

**5 · Renewables & DSM — the regulation engine.**
Versioned CERC profiles (2022 frozen / 2024 in force), dated amendments
auto-apply by settlement date (the 1 Apr 2026 tolerance tightening is IN
the engine), per-technology settlement, Alerts & Revision before gate
closure. Provenance table marks every rule confirmed vs proxied.
> Key line: "The regulation is versioned in code the way tax software
> versions tax law — and 13 self-test assertions guard it."

**6 · Multi-State — the scale story.**
23 states verified live via MERIT; cross-checked against independent
feeds (Gujarat within 0.5%). Import-dependence view: Kerala imports ~97%,
Bihar ~88% — those balance sheets live on purchase-price volatility;
that's the Forecast-as-a-Service pipeline. Forecast-readiness table shows
per-state history accruing with an honest training gate.
> Key line: "Delhi is our reference implementation, not our market. The
> recipe replicates wherever the data poller has been running."

**7 · Methodology / Roadmap — close.**
HLD/LLD, every hyperparameter justified, SOTA citations (M5, Lago 2021,
Romano 2019, Rockafellar–Uryasev 2000), limitations listed before anyone
asks. Roadmap ties to the business model's phasing.
> Close: "Forecast-first revenue today, optimization uplift as trust
> builds, an India-wide data asset accruing every 5 minutes — that's the
> Bloomberg-for-power-markets path."

---

## Hard Q&A (answers we've already earned)

- **"Price MAPE 20% is high."** Indian DAM is cap-pinned and violently
  volatile; levels are hard, *shape* is tradable — corr 0.93 and 93.8%
  capture prove the money survives the error. The cap-hurdle stage models
  the ₹10,000 censoring explicitly (evening MAPE 11.4%), and conformal
  bands price the residual risk honestly.
- **"Your RTM day-ahead model loses to persistence."** It does, on the
  *level*, and the leaderboard says so — persistence is the champion at
  that horizon. The model still calls the *direction* of the DAM→RTM
  spread far better (73.4% vs 63.8%), and direction is what decides
  whether to trade. Only the intraday model is promoted into the
  optimizer, where it beats the hour-ratio it replaced on both: WAPE
  26.6% vs 33.0%, direction 76.6% vs 60.2%.
- **"Why WAPE instead of MAPE for RTM?"** Because RTM clears at ₹0 on real
  blocks — the 1st percentile is ₹23 — so a per-block percentage divides
  by near-zero and reports nonsense. WAPE divides once at the end by total
  value, which is what a desk means by "how far off were we".
- **"Your state model used to lose to a naive baseline."** It did, and the
  fix was not tuning: per-state champion selection was *measured* worse
  than either pure strategy (17.5% vs 11.1% naive, 16.2% model), because
  one year of daily history makes per-state windows too noisy to select
  on. We replaced it with a fixed equal-weight combination plus each
  state's own generation mix as features. Served sMAPE is now 8.78% vs
  11.14% naive, and the worst state fell from 35% to 16%.
- **"Why does Himachal get no model?"** A structural rule, not a
  post-hoc exclusion: hydro > 50% of energy met AND coefficient of
  variation > 40%. HP is 67.6% hydro with 55.2% CV — its procurement
  tracks reservoir state that appears in no feature we have. The rule is
  applied to every state equally and today HP is the only one it catches.
- **"Does the DSM forecast need grid frequency?"** No, and that surprised
  us too. CERC de-linked deviation charges from frequency in the 2022
  regulations and the notified 2024 text does not reinstate it, so the
  7 days of frequency we hold is not a blocker. The Normal Rate is priced
  off DAM/RTM — which is exactly why the RTM model feeds that page.
- **"Why no 'optimal schedule bias' recommendation?"** Because the optimum
  landed at the edge of the sweep. Our engine implements the general-seller
  Normal Rate without the over-injection caps the real regulation carries,
  so "schedule lower" always looks free. We publish the sensitivity curve
  and withhold the recommendation rather than advise gaming a settlement
  off an incomplete rule.
- **"Is the DSM engine legally exact?"** Structure yes, with cited
  provenance per rule; ancillary leg proxied by RTM (no public feed) and
  flagged in every result. Decision support today, settlement-grade after
  counsel review — said before you asked.
- **"Why is CVaR off by default?"** Because we measured it. With the old
  point model it lifted the P10 day +5.4%; after the cap-hurdle upgrade
  sharpened the centre, scenarios drawn from the (un-hurdled) quantile
  models inject noise and cost −5.2% mean. We switched our own feature off
  when the data said so — the machinery stays built for the quantile-hurdle
  unification. That answer wins more trust than any +% would.
- **"Real battery beats you?"** Only fair-coverage days count (≥90%
  telemetry); partial-day comparisons flatter whoever's charging hours
  went unobserved — our validator refuses to headline them.
- **"What's defensible?"** The combination (conformal quantiles + CVaR +
  versioned regulation engine + rule-driven revision alerts) plus data
  assets nobody serves: sampled frequency, BESS telemetry, 23-state
  15-min demand history. Patent counsel engaged for a provisional
  (Section 3(k) strategy: claim the technical combination, not the math).
- **"Scraping is fragile."** Correct — that's the partnership pitch (§9
  of the business model): adapters are pluggable for licensed feeds, and
  the cache layer means a source outage degrades gracefully on stage.

## Numbers to re-check on demo morning
all-India GW · Delhi load · today's DAM avg/max · plan expected P&L ·
BESS SoC and state · states training-ready count (+ first real per-state
MAPEs if the gate has crossed) · fair-day BESS validation uplift if one
exists.

---

## If something looks stale on demo day

`python run_pipeline.py` — it is now self-healing and will fix most things
by itself: it waits for DNS, retries the plan twice while re-pulling
inputs, and backfills any missing load/price history. It prints
`pipeline OK` or exits non-zero naming the stage that failed.

**The red banner is the single source of truth.** If it is absent, the
plan on screen is for tomorrow and the last run succeeded.

## Reliability story (worth telling if asked "is this production-grade?")

On 27-28 Jul the 11:00 scheduled run died twice with
`getaddrinfo failed` — Windows fires the task the moment the machine wakes,
before Wi-Fi/DNS is up, while a manual run 25 minutes later worked fine.
Four days of plans were lost, and **nothing on screen said so**. Separately,
Delhi SLDC restyled its realtime page (`Delhi Load` → `DELHI LOAD`, inline
values → header/value tables) and the scraper silently served cache.

What that produced, all now in the codebase and tested:
- **Retry with backoff** on every Open-Meteo call, and a `wait_for_network()`
  gate (up to 5 min) before the pipeline starts.
- **Plan retry** — a failed plan re-pulls inputs and tries again twice.
- **`pipeline_health.json`** written at every stage → surfaced in
  `meta.json` → red banner in the UI. Silence is no longer possible.
- **Table-based SLDC parsing** scoped to the right table (a flat header
  lookup had grabbed a DISCOM's 270 MW as Delhi's system load) plus a
  **plausibility guard** (2,000-9,500 MW) so a wrong number raises instead
  of being displayed.
- **Strict-JSON export** — a NaN would have emitted bare `NaN`, which
  browsers refuse to parse; every artifact is now sanitised and validated.

The honest framing: *we found these because the system runs unattended
every day, and each one is now a test.* That is a better answer than
claiming nothing ever broke.
