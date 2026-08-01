/* Full forms + plain-language definitions for every abbreviation used on
   the app. Wired into <InfoTip> / Stat's `info` prop so any KPI can be
   understood by someone who has never seen a power market. */

export const GLOSSARY = {
  DAM: ["Day-Ahead Market", "IEX auction cleared at ~13:00 for every 15-min block of the next day. Our primary trading venue."],
  RTM: ["Real-Time Market", "48 half-hourly IEX auctions through the delivery day itself; gate closes ~1 hour before delivery. Used for intraday re-optimization."],
  GDAM: ["Green Day-Ahead Market", "DAM variant where only renewable energy is traded; clears alongside DAM with a green premium/discount."],
  MCP: ["Market Clearing Price", "The single price (₹/MWh) where aggregate buy and sell bids intersect for a block. What everyone pays/receives."],
  MCV: ["Market Clearing Volume", "Total MW traded in a block at the clearing price."],
  IEX: ["Indian Energy Exchange", "India's main power exchange, where DAM/RTM/GDAM trade."],
  BESS: ["Battery Energy Storage System", "Grid-scale battery. Our reference asset is 20 MW / 40 MWh — the same spec as BRPL's real Kilokari unit."],
  SoC: ["State of Charge", "How full the battery is, in % or MWh. All dispatch schedules must keep it between 5% and 100%."],
  DoD: ["Depth of Discharge", "How deep a charge-discharge cycle swings, as a fraction of capacity. Deeper cycles consume disproportionately more battery life."],
  FCE: ["Full-Cycle Equivalents", "Total cycling normalised to full-depth cycles; 2 half-depth cycles ≈ 1 FCE. How battery usage is counted."],
  LP: ["Linear Program", "Exact mathematical optimization (PuLP/CBC solver) that picks the profit-maximal charge/discharge schedule subject to physics constraints."],
  CVaR: ["Conditional Value-at-Risk", "The average of the worst α% of outcomes. Our risk-aware optimizer maximises a blend of expected profit and CVaR (Rockafellar–Uryasev, 2000)."],
  CQR: ["Conformalized Quantile Regression", "Calibration method (Romano et al., 2019) that widens forecast bands until they empirically cover what they claim. Recomputed every retrain — during the May regime shift it corrected our coverage from 51% to 81.5%; currently the raw band passes (82.5%) so the margin is zero."],
  DSM: ["Deviation Settlement Mechanism", "CERC's penalty system for deviating from your scheduled generation/drawal. What our DSM engine computes, per 15-min block."],
  CERC: ["Central Electricity Regulatory Commission", "National power regulator; writes the DSM regulations (2022 and 2024 frameworks) our settlement engine implements."],
  DERC: ["Delhi Electricity Regulatory Commission", "Delhi's state regulator; sets the retail ToD tariffs used in the C&I peak-shaving module."],
  SLDC: ["State Load Despatch Centre", "The state grid control room. Delhi's publishes the live load, frequency and BESS telemetry we scrape."],
  NLDC: ["National Load Despatch Centre", "National grid operator (Grid-India). Ancillary-service prices settle here, with no public feed."],
  RLDC: ["Regional Load Despatch Centre", "Regional grid operators (NRLDC/WRLDC/SRLDC…). Issue the settlement statements DSM reconciliation would ingest."],
  NR: ["Normal Rate", "The DSM reference price under CERC 2024: ⅓ DAM + ⅓ RTM + ⅓ ancillary price (ancillary proxied by RTM here — no public feed, flagged)."],
  "OD/UD": ["Over-Drawal / Under-Drawal", "Taking more/less power from the grid than scheduled. Priced by the DSM."],
  ToD: ["Time-of-Day tariff", "Retail tariff that surcharges peak-hour energy (+20%) and rebates off-peak (−20%) — the price signal the C&I battery follows."],
  "C&I": ["Commercial & Industrial", "Business electricity consumers — factories, malls, data centres. One of our five customer segments."],
  DISCOM: ["Distribution Company", "The utility that delivers power to consumers (e.g. BRPL in Delhi). Another customer segment."],
  RE: ["Renewable Energy", "Solar + wind. Our digital twin forecasts RE plants' output for DSM management."],
  NWP: ["Numerical Weather Prediction", "Physics-based weather forecast models. We use the previous day's run — what was actually knowable at bid time — for honest forecast error."],
  MAPE: ["Mean Absolute Percentage Error", "Average |forecast − actual| / actual. Load model: 4.33% on unseen test data."],
  RMSE: ["Root Mean Squared Error", "Square root of average squared error; penalises big misses more than MAPE."],
  "R²": ["Coefficient of Determination", "Share of variance the model explains; 1.0 = perfect. Load model: 0.957."],
  P10: ["10th Percentile", "Value the outcome falls below only 10% of the time — the bottom of our forecast band."],
  P50: ["50th Percentile (Median)", "The central forecast; equally likely to be above or below."],
  P90: ["90th Percentile", "Value the outcome stays below 90% of the time — the top of the band. P10–P90 spans an 80% confidence band."],
  kVAr: ["Kilovolt-Ampere reactive", "Reactive power — supports grid voltage, moves no net energy. Published in the BESS telemetry."],
  Hz: ["Hertz", "Grid frequency. India targets 50.00 Hz; deviations drive DSM charge rates."],
  ISGS: ["Own Generation (state)", "MERIT's 'own generation' figure — power a state produced itself vs imported (ISGS historically: Inter-State Generating Station share)."],
  MERIT: ["Merit Order Despatch portal", "Ministry of Power site (meritindia.in) publishing live per-state demand, own generation and imports — our national coverage layer."],
  QCA: ["Qualified Coordinating Agency", "Aggregator that forecasts and schedules a pool of RE plants and carries their DSM exposure — a FlexTrade customer persona."],
  GSS: ["Grid Sub-Station", "Substation where an RE plant injects; Rajasthan's feed reports plant injection per GSS."],
  "X-factor": ["Deviation denominator factor", "CERC glide-path multiplier that progressively tightens how wind/solar deviation % is computed (FY28→FY32); 1.0 through FY27."],
  FaaS: ["Forecast-as-a-Service", "Selling forecasts via metered API — our primary revenue stream (business model §3.3)."],
  LFP: ["Lithium Iron Phosphate", "Dominant grid-battery chemistry; long cycle life (~6,000 full cycles), used for our degradation parameters."],
  LGBM: ["LightGBM", "Gradient-boosted decision tree library — state of the art on tabular time series (M5 competition) — powering our load and price models."],
  "η": ["Round-Trip Efficiency", "Energy out ÷ energy in over a full cycle (90% here); √η charged on each direction."],
  "λ": ["Risk-Aversion Weight", "Dial blending expected profit vs CVaR in the stochastic optimizer; 0 = risk-neutral."],
  IST: ["Indian Standard Time", "UTC +05:30. All market blocks and timestamps on this app are IST."],
  MW: ["Megawatt", "Power — the rate of energy flow. 1 MW ≈ ~800 Indian homes' average draw."],
  MWh: ["Megawatt-hour", "Energy — 1 MW sustained for 1 hour. Battery capacity is measured in MWh."],
  P_L: ["Profit & Loss", "Revenue minus costs over a period; 'expected P&L' uses forecast prices, 'settled' uses actuals."],
};

/* "DAM, MCP" -> tooltip text with full forms + definitions */
export function gloss(keys) {
  return String(keys).split(",").map((k) => k.trim()).filter((k) => GLOSSARY[k])
    .map((k) => {
      const [full, def] = GLOSSARY[k];
      return `${k} — ${full}. ${def}`;
    }).join("\n\n");
}
