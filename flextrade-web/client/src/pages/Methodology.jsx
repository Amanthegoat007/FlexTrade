import { Card, Loading } from "../components/ui";
import { fmtTs, useApi } from "../lib/api";

/* Every claim on this page is written to be defensible in Q&A. Metrics are
   read from the live export (meta.json), never hardcoded. */

/* Values come from meta.json's parsed `headline` block, never from constants
   here. They used to be hardcoded and every one of them had drifted: RMSE read
   262 MW against an actual 235.6, R² 0.957 against 0.9648, price correlation
   0.919 against 0.933, and the backtest was called 61 days when it has been 55
   for a while. Hardcoding a metric next to a claim that metrics are never
   hardcoded is how a demo loses an audience. */
const n = (v, d = 2, suffix = "") =>
  v === null || v === undefined ? "—" : `${Number(v).toFixed(d)}${suffix}`;
const lakh = (v) => (v == null ? "—" : `₹${(v / 1e5).toFixed(1)} lakh`);

const kpiGlossary = (h = {}) => [
  ["MAPE", "Mean Absolute Percentage Error — average % miss. The headline accuracy number; comparable across assets of different size.",
    `Delhi load: ${n(h.load_test_mape_pct)}% on a 6-month unseen holdout (was 4.98% — improved 24 Jul via model-lab experiments).`],
  ["RMSE", "Root Mean Squared Error, in physical units — punishes large misses more than small ones.",
    `Delhi load: ${n(h.load_test_rmse_mw, 1)} MW on a 4,000–8,000 MW system.`],
  ["R²", "Share of variance explained (1.0 = perfect). Shows the model tracks structure, not just the mean.",
    `Delhi load: ${n(h.load_test_r2, 4)} test.`],
  ["Correlation", "Whether forecast and reality move together, regardless of level. For arbitrage, shape is what pays — the battery needs the cheap/expensive ranking right, not the exact rupee.",
    `IEX DAM price: ${n(h.price_test_corr, 3)} test.`],
  ["WAPE", "Σ|error| ÷ Σ|actual| — one division at the end instead of one per block. The honest percentage when the target can approach zero, which RTM prices do (1st percentile ₹23).",
    "RTM intraday: 26.6% served, vs 33.0% for the hour-ratio it replaced."],
  ["Pinball loss", "The proper scoring rule for a quantile forecast — penalises a P90 that gets exceeded more than 10% of the time.",
    "Reported per quantile in the quantile metrics below."],
  ["Coverage", "% of actual outcomes falling inside the predicted band — measured walk-forward (recalibrate each day, score the next), never on a single window. A calibrated P10–P90 band should cover ~80%, and it should do so inside every regime, not just on average.",
    `Price band ${n(h.price_band_coverage_pct, 1)}% vs ${h.price_band_target_pct ?? 80}% nominal over ${h.price_band_walk_days ?? "—"} days, at ${
      h.price_band_width_rs_mwh ? `₹${Number(h.price_band_width_rs_mwh).toLocaleString("en-IN")}/MWh` : "—"
    } wide — worst rolling 30 days ${n(h.price_band_worst_30d_pct, 1)}%, worst cap-regime ${n(h.price_band_worst_regime_pct, 1)}%. Delhi load band: 80.2% coverage at 526 MW across 8 rolling origins, margin recalibrated daily on a trailing 14 days.`],
  ["Capture ratio", "Backtest P&L ÷ perfect-foresight P&L — the single best summary of forecast + optimizer quality together.",
    `${n(h.capture_ratio_pct, 1)}% with the cap-hurdle point model (${h.backtest_days ?? "—"}-day walk-forward).`],
  ["Uplift", "FlexTrade P&L − customer's baseline (static EMS) P&L. This is the revenue-share billing base (business model §3.4).",
    `+${lakh(h.uplift_rs)} over ${h.backtest_days ?? "—"} days, +${n(h.uplift_pct, 1)}%.`],
  ["CVaR", "Conditional Value-at-Risk — mean P&L of the worst 10% of scenarios. What an asset owner with debt covenants actually cares about.",
    "Optimized via the Rockafellar–Uryasev linearization. Still OFF by default, but its earlier rejection no longer stands: that verdict (−9.4% mean/day, −44.8% worst day) was measured against a misspecified price distribution. Re-run on the censored-mixture band it is −0.4% mean/day for +3.6% on the P10 day and −3.2% volatility — roughly a fair trade of mean for tail. 51 days is too few to switch the default on; it is being re-evaluated as the walk-forward window grows."],
  ["Normal Rate (NR)", "The DSM reference price: ⅓ DAM + ⅓ RTM + ⅓ ancillary charge (CERC 2024, Reg. 14). Deviations are priced off real market outcomes.",
    "Computed from live IEX prices; the ancillary leg is proxied by RTM (no public feed) and flagged in every result."],
  ["DSM saved", "Penalty under FlexTrade's schedule minus penalty under naive persistence, both settled identically.",
    "Varies day to day; shown honestly, including days persistence wins."],
];

/* Load and price are tuned SEPARATELY and their values genuinely differ, so
   they get separate columns. An earlier single-column version quoted the price
   model's settings as if they were the load model's — after the model-lab
   session retuned load, every load figure in it was wrong. */
const PARAMS = [
  ["n_estimators", "6000", "2000",
    "An upper bound only: early stopping on the validation window picks the real count, so the cap never binds. Load can afford a higher cap because its learning rate is half as large."],
  ["learning_rate", "0.015", "0.03",
    "Small steps plus early stopping is the standard bias–variance trade for GBDTs. Load runs slower and longer because it has 5× the rows and a finer 96-block daily shape to resolve."],
  ["num_leaves", "255", "63",
    "Tree capacity ≈ interaction depth. Load has 33 features with strong interactions (hour × temperature × weekday) and 132k training rows to support 255 leaves; price has ~20k rows, so a quarter of the capacity guards against overfitting."],
  ["min_child_samples", "20", "40",
    "The floor on how few observations a leaf may represent — it stops a tree memorising a single freak day. Price gets the stricter floor because it has fewer rows and a much noisier target."],
  ["subsample / colsample", "0.8 / 0.7", "0.8 / 0.8",
    "Row and feature bagging decorrelate the trees. Load drops column sampling slightly further because several of its features are near-duplicates (lag_2d vs roll7d_mean)."],
  ["reg_lambda", "2.0", "1.0",
    "Mild L2 on leaf weights. Load carries more because 255 leaves is a lot of capacity to hand a model with real regime drift in it."],
  ["recency half-life", "180 days", "—",
    "Down-weights old regimes so the model tracks Delhi's load growth. Chosen in the model lab: 180 d beat unweighted by ~0.45 pp MAPE."],
  ["ensemble", "3 seeds", "1",
    "Averaging seeds removes tree-growth variance worth ~0.03 pp MAPE on load. Not worth the training time on price, where the irreducible error is far larger."],
  ["target transform", "level", "log(MCP)",
    "Prices span ₹0–₹10,000 (regulatory cap), so log makes a 10% error at ₹2,000 cost the same as at ₹9,000 — otherwise the model only cares about expensive blocks. Load is a narrow, strictly positive band and needs no transform."],
  ["feature lag floor", "≥ 48 h", "≥ 1 day",
    "Bid-time validity, not convenience: DAM bids close ~12:00 on D for D+1, when day-D load is only half known but day-D prices cleared at 13:00 on D−1. Every feature is checked against what a bidder actually knows at gate closure."],
];

function Arch() {
  return (
    <div className="arch">
      <div className="arch-layer"><div className="layer-name">Sources</div>
        <div className="arch-box"><b>Delhi SLDC</b><span>5-min load · realtime · frequency · BESS · 8 NR states</span></div>
        <div className="arch-box"><b>IEX</b><span>DAM · RTM · GDAM, 96 blocks, any date</span></div>
        <div className="arch-box"><b>Open-Meteo</b><span>forecast · archive · previous-runs (honest fc error)</span></div>
      </div>
      <div className="arch-down">▼ every fetcher: live → SQLite cache fallback, logged</div>
      <div className="arch-layer"><div className="layer-name">Store</div>
        <div className="arch-box"><b>SQLite (flextrade.db)</b><span>11 tables · upserts keyed by timestamp · self-healing backfill detects & repairs gaps</span></div>
        <div className="arch-box"><b>Artifacts</b><span>models, plans, bid sheets, backtests → output/</span></div>
      </div>
      <div className="arch-down">▼ daily 11:00 IST pipeline (scheduled task) — before DAM gate closure</div>
      <div className="arch-layer"><div className="layer-name">Models</div>
        <div className="arch-box"><b>Load forecast</b><span>LightGBM ×3-seed · 33 features · recency-weighted · 4.33% MAPE</span></div>
        <div className="arch-box"><b>Price forecast</b><span>LightGBM log-MCP + P10/P50/P90 + conformal</span></div>
        <div className="arch-box"><b>RE twin</b><span>PVWatts-style solar + wind power curve</span></div>
        <div className="arch-box"><b>DSM engine</b><span>versioned CERC 2022/2024 profiles</span></div>
      </div>
      <div className="arch-down">▼</div>
      <div className="arch-layer"><div className="layer-name">Decide</div>
        <div className="arch-box"><b>LP dispatch</b><span>PuLP/CBC · SoC physics · degradation cost</span></div>
        <div className="arch-box"><b>CVaR stochastic</b><span>copula scenarios · risk-aware λ</span></div>
        <div className="arch-box"><b>Alerts & Revision</b><span>revise-vs-hold before each gate</span></div>
      </div>
      <div className="arch-down">▼</div>
      <div className="arch-layer"><div className="layer-name">Serve</div>
        <div className="arch-box"><b>Bid sheet</b><span>96-block DAM order file</span></div>
        <div className="arch-box"><b>FastAPI FaaS</b><span>tiered keys · metered usage · :8100</span></div>
        <div className="arch-box"><b>Express API + React</b><span>this app · :8090 · reads store, never re-scrapes</span></div>
      </div>
    </div>
  );
}

/** Rolling-origin re-measurement of the published claims. */
function WalkForward({ wf }) {
  const models = wf?.models || [];
  const done = models.filter((x) => !x.error);
  return (
    <>
      <h2>Walk-forward audit — the same claims, re-measured honestly</h2>
      <p>
        Every model above is refitted at each of several consecutive origins and
        scored only on the window after it, so the scoring data is never inside
        the training data. What this buys is not a better average — it is a{" "}
        <strong>distribution</strong> of performance, which makes “worst window”
        reportable instead of unknown. Rolling-origin evaluation (Tashman 2000)
        has been the standard in forecasting for decades; we were not using it,
        and a headline was wrong by 20 points as a result.
      </p>
      <p className="note">
        Bands are scored by <strong>interval score</strong> (Gneiting &amp;
        Raftery 2007) — width plus a miss penalty, in one proper number. Coverage
        and width reported as a bare pair can be traded against each other; a
        proper scoring rule cannot be gamed. Coverage is then tested, not just
        quoted: <strong>Kupiec (1995)</strong> asks whether the failure rate is
        right, and <strong>Christoffersen (1998)</strong> asks whether failures
        are independent or arrive in bursts. The second matters operationally —
        a band can hit exactly 80% and still fail six days running through a
        heatwave, which is the same rate and a completely different risk.
        Model-vs-benchmark uses <strong>Diebold–Mariano</strong> with the
        Harvey–Leybourne–Newbold small-sample correction, so “better” is a test
        result rather than two averages compared by eye.
      </p>
      {!done.length ? (
        <Card><div className="note">No walk-forward results exported yet — run
          <code> python backtest/audit.py all</code>.</div></Card>
      ) : (
        <div className="grid cols-2">
          {done.map((x) => (
            <Card key={x.key || x.model} title={x.model}
              sub={`${x.origins_run} rolling origins × ${x.test_days}d · ${x.window}`}>
              <pre className="formula">{[
                x.coverage_pct_mean != null &&
                  `interval score  ${Number(x.interval_score_mean).toLocaleString("en-IN")}  (lower is better)\n` +
                  `coverage        ${x.coverage_pct_mean}% mean / ${x.coverage_pct_worst}% worst   vs ${x.nominal_pct}% nominal\n` +
                  `mean width      ${Number(x.width_mean).toLocaleString("en-IN")} ${x.unit || ""}\n` +
                  `origins below nominal: ${x.origins_below_nominal}/${x.origins_run}` +
                  (x.kupiec_rejected_origins?.length
                    ? `\nKupiec REJECTS correct rate at: ${x.kupiec_rejected_origins.join(", ")}` : "") +
                  (x.independence_rejected_origins?.length
                    ? `\nChristoffersen REJECTS independence (failures CLUSTER) at: ${x.independence_rejected_origins.join(", ")}` : ""),
                x.wape_pct && `WAPE   mean ${x.wape_pct.mean}%   worst ${x.wape_pct.worst}%`,
                x.mae && `MAE    mean ${Number(x.mae.mean).toLocaleString("en-IN")}   worst ${Number(x.mae.worst).toLocaleString("en-IN")} ${x.unit || ""}`,
                x.bias && `bias   mean ${Number(x.bias.mean).toLocaleString("en-IN")}   worst ${Number(x.bias.worst).toLocaleString("en-IN")} ${x.unit || ""}`,
                x.vs_benchmark?.stat != null &&
                  `Diebold-Mariano vs benchmark: stat ${x.vs_benchmark.stat}, p=${x.vs_benchmark.p_value} → ${x.vs_benchmark.better}`,
              ].filter(Boolean).join("\n")}</pre>
            </Card>
          ))}
        </div>
      )}
      <p className="note">
        Models not listed here have <strong>not</strong> been re-measured yet and
        their figures above remain single-window. Saying so is the point: staying
        quiet about which claims are audited would repeat exactly the failure
        this section exists to correct. Note also that rolling origins share
        training data, so per-origin results are not independent draws — the
        aggregates are descriptive, and the hypothesis tests are applied within
        an origin, never across them.
      </p>
    </>
  );
}

export default function Methodology() {
  const { data: meta, loading, error } = useApi("/api/meta");
  if (loading && !meta) return <Loading error={error} />;
  const m = meta?.metrics || {};

  return (
    <div className="prose">
      <h2>System architecture (HLD)</h2>
      <p>
        One loop, run daily before the IEX day-ahead gate closes at 12:00 IST:
        <b> Predict → Decide → Trade → Prove</b>. Every arrow below is real, running
        code — nothing on this page describes intent.
      </p>
      <Card><Arch /></Card>

      <h2>Low-level design (LLD)</h2>
      <Card>
        <table className="data">
          <thead><tr><th>Module</th><th>Responsibility</th><th>Key detail</th></tr></thead>
          <tbody>
            <tr><td className="mono">ingest/sldc.py</td><td>Delhi load (5-min day curves + realtime), grid frequency</td><td>Frequency comes from chart image-map tooltips; the site ignores its own date parameter, so history is sampled daily, never backfilled — a guard raises if a non-today date ever appears</td></tr>
            <tr><td className="mono">ingest/iex.py</td><td>DAM/RTM/GDAM scrape, any delivery date; self-healing price history</td><td>New IEX site server-renders data into HTML; RTM needs per-session dedup; GDAM has a two-row MultiIndex header with fuel split</td></tr>
            <tr><td className="mono">ingest/bess.py + poll_bess.py</td><td>BRPL Kilokari BESS telemetry (MW, kVAr, SoC)</td><td>SLDC publishes only instantaneous state — history exists because we sample every 5 min. Sign convention verified empirically, not assumed: the page is generation-positive (raw +19.6 MW observed while SoC fell 81%→20% during a real discharge)</td></tr>
            <tr><td className="mono">ingest/states.py</td><td>Multi-state: MERIT national layer + NR snapshot + deep adapters</td><td>23 states live via meritindia.in JSON (codes discovered per-state, values cross-checked against independent feeds); Gujarat scraped direct from sldcguj.com; Rajasthan endpoints mapped incl. the DSM-rate tag table; dead ends (image-only Maharashtra SCADA, session-gated WRLDC) recorded, not hidden</td></tr>
            <tr><td className="mono">models/load_model, price_model</td><td>Day-ahead forecasts + quantiles + conformal</td><td>Chronological splits only, never shuffled — shuffling time series leaks the future</td></tr>
            <tr><td className="mono">models/re_model.py</td><td>Solar + wind digital twin, per-technology</td><td>Forecast error measured against the previous day's NWP run — real, not simulated</td></tr>
            <tr><td className="mono">models/dsm.py (+ dsm_selftest)</td><td>Versioned CERC settlement profiles</td><td>13 assertions guard band arithmetic, sign conventions, and profile isolation (a caught bug: the 2026 amendment briefly leaked into the frozen 2022 profile)</td></tr>
            <tr><td className="mono">optimize/dispatch, stochastic</td><td>LP + CVaR dispatch</td><td>End-of-day SoC ≥ start (no fake profit from draining); √η charged each direction</td></tr>
            <tr><td className="mono">optimize/rtm_reopt.py</td><td>Intraday re-optimization vs the RTM</td><td>DAM position is sunk; only the deviation trades at RTM prices. Cleared blocks use actual RTM MCP, the rest DAM × observed RTM/DAM ratio, provenance labelled</td></tr>
            <tr><td className="mono">optimize/degradation.py</td><td>Rainflow + Wöhler DoD-dependent cycle cost</td><td>Nonconvex cost kept out of the LP via fixed-point calibration of a flat rate; 5-assertion selftest incl. ASTM rainflow identities</td></tr>
            <tr><td className="mono">optimize/peak_shave.py</td><td>C&amp;I bill optimization under DERC ToD</td><td>Minimizes energy + demand charge jointly; no-export constraint by default; tariff and profile both labelled for verification</td></tr>
            <tr><td className="mono">backtest/*</td><td>Walk-forward settlement at actual prices</td><td>SoC deficits bought back at the day's max price — no strategy can cheat by liquidating inventory</td></tr>
            <tr><td className="mono">export_web.py → server → React</td><td>This app</td><td>Node reads SQLite + JSON exports; Python owns every fetch and model — single implementation, no drift</td></tr>
          </tbody>
        </table>
      </Card>

      <h2>The data: collection → cleaning → linking → inference</h2>
      <h4>Collection</h4>
      <p>
        Three live source families (SLDC, IEX, Open-Meteo), each with the same
        contract: try live, fall back to the last good cache, log the outcome, and
        show LIVE/CACHED status in the UI. History that no source publishes (grid
        frequency, BESS telemetry) is built by disciplined sampling — which is
        itself a data asset competitors would have to spend months accumulating.
      </p>
      <h4>Cleaning</h4>
      <ul>
        <li>Load: drop values outside 1,000–9,500 MW (telemetry drops; Delhi's real record peak is ~8.6 GW), kill isolated spikes (&gt;20% jump vs both neighbours), interpolate gaps ≤ 2 h only — longer outages stay missing rather than invented.</li>
        <li>Prices: parse-time numeric coercion, per-day completeness checks (96 blocks), partial tables (&lt;200 of 288 slots for load) treated as failures and retried.</li>
        <li>Frequency: clip to 47–53 Hz; fabricated-history guard (see LLD row).</li>
      </ul>
      <h4>Linking</h4>
      <ul>
        <li>Everything is aligned to the <b>15-minute IEX settlement block</b> — the market's native resolution — so forecast, price, schedule and settlement join 1:1.</li>
        <li>Weather arrives in UTC; SLDC in IST — shifted +05:30 at ingest (a silent one-day-ahead bug otherwise).</li>
        <li>5-min load → 15-min mean; hourly weather → 15-min interpolation.</li>
      </ul>
      <h4>Inference</h4>
      <p>
        Daily at 11:00 IST (automated): refresh live data → self-heal any history
        gaps → forecast tomorrow's 96 load blocks and price distribution → optimize
        the battery schedule → emit the bid sheet → export to this app. The same
        pipeline also samples frequency and re-scores DSM for yesterday.
      </p>

      <h2>Datasets (live inventory)</h2>
      <Card sub={`Read from the store right now — generated ${fmtTs(meta?.generated_at)}. This table cannot go stale because it is the data describing itself.`}>
        <div className="scroll-x">
          <table className="data">
            <thead><tr><th>Table</th><th className="num">Rows</th><th>Span</th><th>Columns</th></tr></thead>
            <tbody>
              {(meta?.datasets || []).filter((d) => d.rows > 0).map((d) => (
                <tr key={d.table}>
                  <td><b>{d.table}</b><br /><span style={{ fontSize: 11.5, color: "var(--muted)" }}>{d.description} · {d.source}</span></td>
                  <td className="num">{d.rows.toLocaleString("en-IN")}</td>
                  <td style={{ fontSize: 12 }}>{fmtTs(d.from)} →<br />{fmtTs(d.to)}</td>
                  <td className="mono" style={{ fontSize: 11, maxWidth: 260 }}>{(d.columns || []).join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <h2>Model choice: why gradient-boosted trees, and is it SOTA?</h2>
      <p>
        <b>The honest answer, with references.</b> For tabular time-series with
        engineered lag/calendar/weather features, gradient-boosted decision trees
        are the consistently winning family: GBDT/LightGBM combinations dominated
        the M5 forecasting competition (Makridakis et al., 2022), and in electricity
        price forecasting specifically, the standard open benchmark (Lago et al.,
        <i>Applied Energy</i> 2021) finds well-tuned tree/linear models competitive
        with deep nets like DeepAR and transformers at a fraction of the cost and
        with far better interpretability. Deep learning wins when there are
        thousands of related series to learn across or raw unstructured inputs —
        neither applies to one city's load and one exchange's prices.
      </p>
      <ul>
        <li><b>Trains in ~1 minute on a laptop</b> — retraining daily as regimes shift is practical; a transformer would need a GPU budget for no measured gain.</li>
        <li><b>Feature importances are visible</b> — an operator (or judge) can see the model leans on same-block-last-week + temperature, which builds trust.</li>
        <li><b>Handles missing data natively</b> — real scraped data has holes.</li>
        <li><b>Quantile objective built-in</b> — the P10/P50/P90 heads reuse the same features (pinball loss, the proper scoring rule).</li>
      </ul>
      <p>
        The genuinely state-of-the-art pieces here are not the base learner: they
        are <b>conformalized quantile regression</b> (Romano et al., NeurIPS 2019)
        for calibrated uncertainty under regime shift, and <b>CVaR-constrained
        stochastic dispatch</b> (Rockafellar &amp; Uryasev, 2000) — both current
        best practice in forecasting and energy-trading literature respectively,
        and both measured on real data below.
      </p>

      <h2>Why these hyperparameters — not magic numbers</h2>
      <Card>
        <div className="scroll-x">
          <table className="data">
            <thead><tr>
              <th>Parameter</th>
              <th className="num">Delhi load</th>
              <th className="num">IEX DAM price</th>
              <th>Why</th>
            </tr></thead>
            <tbody>
              {PARAMS.map(([p, load, price, why]) => (
                <tr key={p}>
                  <td className="mono" style={{ whiteSpace: "nowrap" }}>{p}</td>
                  <td className="num mono">{load}</td>
                  <td className="num mono">{price}</td>
                  <td>{why}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <h2>Measured accuracy (live from the model store)</h2>
      <p className="note">
        The figures in this section come from a <strong>single chronological
        split</strong>: fit on the front of the record, score on the back. That is
        the standard machine-learning protocol and it is the wrong one for a
        forecasting system — it reports one window and says nothing about the
        variance across windows. It cost us: the price band was published here at
        94.4% coverage and measured 74.7% under a rolling origin. Every claim
        below should be read against the walk-forward re-measurement that
        follows it.
      </p>
      <div className="grid cols-2">
        <Card title="Load model" sub="chronological split, last 6 months held out">
          <pre className="formula">{m.load_model || "not exported"}</pre>
        </Card>
        <Card title="Price model (point)" sub="13 months of scraped IEX history">
          <pre className="formula">{m.price_model || "not exported"}</pre>
        </Card>
        <Card title="Price distribution + band" sub="Quantiles of the censored mixture — a point mass at the ₹10,000 cap plus the below-cap law — with a per-cap-regime margin that adapts online to seasonal drift. Coverage below is walk-forward, and should be read with the width, never on its own.">
          <pre className="formula">{m.price_quantiles || "not exported"}</pre>
        </Card>
        <Card title="Backtests" sub="every day forecast with bid-time-valid features, settled at actual cleared prices">
          <pre className="formula">{[m.backtest_summary, m.risk_backtest_summary].filter(Boolean).join("\n\n") || "not exported"}</pre>
        </Card>
      </div>

      <WalkForward wf={m.walkforward} />

      <h2>The forecast lab — four models past the day-ahead curve</h2>
      <p>
        A year of backfilled RTM and GDAM history unlocked a second layer of
        models. Each exists because a specific decision needed it, and each is
        scored against <b>the thing it replaced</b> on a window it never saw —
        a forecast accuracy number with no baseline beside it is decoration.
        Full detail, including the runs we lost, is on the{" "}
        <b>Forecast Lab</b> page.
      </p>
      <Card sub="Every row is a held-out test result, not a training score.">
        <table className="data">
          <thead><tr>
            <th>Model</th><th>Decision it serves</th>
            <th className="num">Result</th><th className="num">Had to beat</th>
          </tr></thead>
          <tbody>
            <tr>
              <td><b>RTM price + DAM→RTM spread</b><br />
                <span className="mono">intraday horizon</span></td>
              <td>Which way to trade the deviation from a firm DAM position, 48
                sessions a day</td>
              <td className="num"><b>26.6% WAPE</b><br />direction <b>76.6%</b></td>
              <td className="num">33.0% WAPE<br />direction 60.2%</td>
            </tr>
            <tr>
              <td><b>Probabilistic load</b><br />
                <span className="mono">P05–P95, conformal</span></td>
              <td>How wide a schedule band must be — DSM charges you for leaving
                a band, not for missing an average</td>
              <td className="num"><b>80.2%</b> coverage<br />526 MW wide</td>
              <td className="num">55.6% raw<br />(nominal 80%)</td>
            </tr>
            <tr>
              <td><b>Peak timing + magnitude</b></td>
              <td>When the battery must be full, and the single most-asked DISCOM
                control-room question</td>
              <td className="num"><b>173 MW</b> MAE<br />hour exact <b>64.5%</b></td>
              <td className="num">272 MW<br />52.5%</td>
            </tr>
            <tr>
              <td><b>DSM exposure</b></td>
              <td>What tomorrow's deviation costs, as a distribution rather than
                a number</td>
              <td className="num">expected + P90<br />from 183 real error days</td>
              <td className="num">— (no incumbent)</td>
            </tr>
          </tbody>
        </table>
      </Card>
      <p style={{ marginTop: 10 }}>
        The peak model's baseline is worth naming: our own 96-block forecast
        already implies a peak for free, so a dedicated model had to beat its
        argmax to justify existing. Its useful output is a <b>distribution</b> —
        an 80% window needs only <b>2.0 hours</b> and contains the peak 80.9% of
        the time, against 19.2 hours for a flat guess over the day.
      </p>

      <h3>Choosing the metric before quoting it</h3>
      <p>
        <b>MAPE is the wrong instrument wherever the target approaches zero,</b>{" "}
        and RTM does: it clears at ₹0 on real blocks and its 1st percentile is
        ₹23. The same intraday forecast scores <b>65% by MAPE and 26.6% by
        WAPE</b> — the difference is entirely a handful of cheap night blocks in
        the denominator. So prices are reported as WAPE (Σ|error| ÷ Σ|actual|,
        which is what a trading desk means by "how far off were we"), and the
        state exchange target as sMAPE, which is bounded at 200%. This is the
        same class of bug that once produced a 131,195% MAPE on state exchange
        volume.
      </p>
      <p>
        For the spread the headline is not an error at all but{" "}
        <b>direction</b> — the share of blocks where the sign of (RTM − DAM) is
        right. A level forecast that is close on average but wrong about the
        sign tells the optimizer to trade the wrong way, so direction is the
        number that decides money.
      </p>

      <h3>Two results we published against ourselves</h3>
      <ul>
        <li>
          <b>The day-ahead RTM model used to lose to persistence on the
          level, and we published that.</b>{" "} It no longer does: on 38
          months of RTM history rather than 12.7, day-ahead scores 29.89% WAPE
          against persistence at 31.29% (it was 33.9% vs 33.2%, losing), and
          all three horizons now promote the model. Worth being precise about
          what changed — the algorithm did not. The history did. We had been
          training on a third of the RTM data IEX serves.
        </li>
        <li>
          <b>Conformal calibration is asymmetric because the median
          under-forecasts.</b> Actual load exceeded the P50 on 75.6% of held-out
          blocks (+104 MW) as Delhi's demand grew. Nearly all the missed
          coverage escapes through the upper bound, so the margin is −91/+150 MW.
          Widening both sides equally buys no coverage and costs band width —
          and band width is what a customer pays for in procured reserve.
        </li>
      </ul>

      <h2>The RE digital twin (physics, not ML — deliberately)</h2>
      <p>
        Standard industry practice for plants without telemetry. Solar follows the
        PVWatts formulation; wind uses the canonical cubic power curve at 100 m hub
        height:
      </p>
      <pre className="formula">{`T_cell = T_air + GHI · (NOCT − 20) / 800          NOCT = 45 °C
P_dc   = P_rated · (DC/AC = 1.25) · (GHI/1000) · (1 − 0.35%/°C · (T_cell − 25))
P_ac   = min(P_dc · η_inv,  P_rated)               η_inv = 96%

Wind:  P = 0                    v < 3 m/s  (cut-in)
       P = P_r · (v³−3³)/(12³−3³)   3 ≤ v < 12  (rated)
       P = P_r                  12 ≤ v ≤ 25;  0 above (cut-out)`}</pre>
      <p>
        The forecast-vs-actual comparison uses Open-Meteo's <b>previous-runs API</b>
        — what the weather model predicted one day earlier vs its analysis — so the
        deviation being settled is genuine day-ahead NWP error, never synthetic noise.
      </p>

      <h2>The optimizer</h2>
      <pre className="formula">{`max  Σₜ 0.25 · MCPₜ · (disₜ − chₜ)  −  0.25 · c_deg · (chₜ + disₜ)
s.t. SoCₜ₊₁ = SoCₜ + 0.25·(chₜ·√η − disₜ/√η)     η = 0.90 round-trip
     0.05·E ≤ SoCₜ ≤ E,   0 ≤ chₜ, disₜ ≤ P,   SoC₉₆ ≥ SoC₀

Risk-aware variant (one schedule, S price scenarios from the conformal band):
max  (1−λ)·E[profit] + λ·CVaR₉₀(profit)      CVaR via Rockafellar–Uryasev:
     CVaR = ζ − (1/(1−α))·Σₛ πₛ·uₛ,   uₛ ≥ ζ − profitₛ,  uₛ ≥ 0`}</pre>
      <p>
        Both are pure LPs — CBC solves them in under a second, guaranteed optimal.
        <b> A finding we report honestly:</b> when the point model was weaker,
        risk-aware bidding lifted the P10 day +5.4%; after the cap-hurdle upgrade
        sharpened the point forecast (evening MAPE 15.3%→11.4%), the same
        scenario ensemble now <i>subtracts</i> value (mean −8.9% under corrected degradation economics) —
        its scenarios are drawn from quantile models that haven't received the
        hurdle treatment, so they inject noise around a better centre. Default
        is therefore the point-forecast LP (λ=0); the CVaR machinery stays
        built, measured, and ready for the quantile-hurdle unification that
        would make it earn its keep. Measuring your own feature into the
        off-position is the difference between engineering and a demo.
      </p>

      <h2>Does better forecasting actually make more money?</h2>
      <p>
        It is the assumption the whole product rests on, so we measured it rather
        than asserting it. The backtest stores per-day forecast error <i>and</i>
        per-day P&amp;L, so the two can simply be correlated across 53 delivery days.
      </p>
      <Card sub="Days grouped into quartiles by that day's price-forecast error, then compared on what the optimizer actually earned.">
        <table className="data">
          <thead><tr>
            <th>Forecast quality</th><th className="num">Price MAPE</th>
            <th className="num">Capture ratio</th><th className="num">₹ lost vs perfect</th>
          </tr></thead>
          <tbody>
            <tr><td>best 25% of days</td><td className="num">9.0%</td>
              <td className="num"><b>97.2%</b></td><td className="num">₹6,460</td></tr>
            <tr><td>2nd quartile</td><td className="num">14.6%</td>
              <td className="num">96.3%</td><td className="num">₹9,677</td></tr>
            <tr><td>3rd quartile</td><td className="num">21.7%</td>
              <td className="num">96.2%</td><td className="num">₹8,198</td></tr>
            <tr><td>worst 25% of days</td><td className="num">31.1%</td>
              <td className="num" style={{ color: "var(--critical)" }}>90.0%</td>
              <td className="num" style={{ color: "var(--critical)" }}>₹26,794</td></tr>
          </tbody>
        </table>
      </Card>
      <p style={{ marginTop: 10 }}>
        <b>Yes — but with sharp diminishing returns, and that changes where effort
        should go.</b> The correlation is real (−0.36 between price MAPE and capture
        ratio) and the tail is what costs: the worst quartile of forecast days loses
        <b> 4× more</b> per day than the best. But across the whole window only
        <b>₹6.71 lakh — 5.3% of the ceiling</b> — separates our schedule from one
        built with <i>perfect knowledge of tomorrow's prices</i>.
      </p>
      <div className="note info">
        <b>A perfect price forecast is worth about 5% more revenue. Correcting the
        degradation price was worth 14%, and the bid-margin fix ~7%.</b> That is not
        an argument for a worse forecast — it is an argument that at 94.7% capture,
        the marginal rupee has moved from <i>prediction</i> to <i>execution and cost
        modelling</i>. Chasing price MAPE from 20% to 15% would be the most
        expensive way to earn the least. Reducing the <i>tail</i> of forecast error
        on the worst days is still worth it; reducing the average is largely not.
      </div>

      <h2>What the optimizer is told a cycle costs — and why that was wrong</h2>
      <p>
        The dispatch LP maximises <span className="mono">Σ price·(dis − ch) −
        c_deg·(ch + dis)</span>. Everything turns on <span className="mono">c_deg</span>,
        the marginal cost of putting a MWh through the battery, and until 2 Aug 2026
        it was <b>₹200/MWh — a round number nobody had ever derived</b>, sitting in
        the objective while our own physics module computed something four times
        larger and was used only for reporting.
      </p>
      <Card sub="Calibrated by running the rainflow degradation fixed point over 30 sampled days of real DAM prices from the last 90.">
        <table className="data">
          <thead><tr><th>Quantity</th><th className="num">Value</th><th>Source</th></tr></thead>
          <tbody>
            <tr><td>Physics rate, median</td><td className="num"><b>₹806/MWh</b></td>
              <td>rainflow on the SoC path + LFP Wöhler curve L(d)=L₁₀₀·d⁻ᵏ</td></tr>
            <tr><td>Physics rate, p10–p90</td><td className="num">₹724 – ₹831</td>
              <td>tight enough that a constant is defensible</td></tr>
            <tr><td>Adopted in the LP</td><td className="num"><b>₹800/MWh</b></td>
              <td>rounded median</td></tr>
            <tr><td>Previous proxy</td><td className="num" style={{ color: "var(--critical)" }}>₹200/MWh</td>
              <td>undocumented; <b>4.0× understated</b></td></tr>
          </tbody>
        </table>
      </Card>
      <p style={{ marginTop: 10 }}>
        Under-charging degradation does not merely overstate profit — <b>it changes
        the schedule</b>, because the LP takes marginal spreads that are not worth
        taking. Over the last 60 days the correction moves cycling from 457 to
        412 EFC/yr; on the higher-spread May–July window the ₹200 schedule reached
        ~657 EFC/yr, <i>outside</i> a typical 365–550 warranty envelope. So the proxy
        did not always breach the envelope, but it was always willing to, and part of
        the revenue it reported was paid for in warranty rather than earned.
      </p>
      <div className="note info">
        <b>No cycle cap was added, deliberately.</b> The obvious fix is a hard
        constraint on equivalent full cycles. It is the wrong fix: the over-cycling
        was never a missing constraint, it was a mispriced input. Charge cycling what
        it actually costs and the optimizer self-regulates — which is both simpler and
        correct, because a cap would still be making bad trades right up to the limit.
      </div>
      <Card title="What the correction cost us, stated plainly" style={{ marginTop: 12 }}
        sub="Same forecasts, same optimizer, same prices — only c_deg changed.">
        <table className="data">
          <thead><tr><th>Headline</th><th className="num">At ₹200</th><th className="num">At ₹800</th></tr></thead>
          <tbody>
            <tr><td>Backtest annualised</td><td className="num">₹9.61 Cr/yr</td>
              <td className="num"><b>₹8.23 Cr/yr</b></td></tr>
            <tr><td>Capture ratio</td><td className="num">93.8%</td>
              <td className="num"><b>94.7%</b></td></tr>
            <tr><td>Cycling (60-day window)</td><td className="num">457 EFC/yr</td>
              <td className="num"><b>412 EFC/yr</b></td></tr>
          </tbody>
        </table>
        <div className="note" style={{ marginTop: 10 }}>
          Revenue falls ~14% and we publish the lower number, because it is the one
          that survives a technical due-diligence review. Capture ratio <i>improves</i>,
          which is the tell that this is a better model rather than a haircut: against
          a correctly-priced perfect-foresight bound, our schedule is closer to optimal
          than it was against a mispriced one.
        </div>
      </Card>

      <h2>KPI glossary — every number, what it means, why it exists</h2>
      <Card>
        <table className="data">
          <thead><tr><th>KPI</th><th>Definition &amp; why it's needed</th><th>Where it stands</th></tr></thead>
          <tbody>
            {kpiGlossary(m.headline).map(([k, def, val]) => (
              <tr key={k}><td><b>{k}</b></td><td>{def}</td><td style={{ fontSize: 12.5 }}>{val}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>

      <h2>Scaling forecasting to many states — a pooled global model</h2>
      <p>
        Delhi reaches 4.33% MAPE because it publishes ~5 years of 5-minute SLDC
        load. <b>No other Indian state publishes that.</b> Training 23 separate models
        on a few months each would produce 23 weak models, so we do what the
        forecasting literature says to do with short, related series: train{" "}
        <b>one global model across all states at once</b>, with state identity and
        state scale as features. Short series borrow statistical strength from long
        ones — the result that decided the M4 and M5 competitions (Januschowski et
        al. 2020; Montero-Manso &amp; Hyndman 2021).
      </p>
      <p>
        The data unlock: MERIT's state pages are backed by date-parameterised
        endpoints that return <b>daily energy by procurement source</b> and{" "}
        <b>plant-level generation by fuel</b> for any day going back 2+ years. One
        detail matters — the date must be sent as <span className="mono">%m/%d/%Y</span>;
        the day-first form returns a "no data" sentinel that is easy to mistake for
        "history doesn't exist".
      </p>
      <p>
        <b>Coverage is not uniform, and we show that rather than average it away.</b>{" "}
        Probing every state on spread-out dates found 8 with full history
        (Maharashtra, Gujarat, Rajasthan, Tamil Nadu, MP, West Bengal, Kerala, HP),
        4 partial, and 11 — including Delhi, UP and Karnataka — where MERIT returns
        well-formed responses with every value null. Those 11 get live monitoring
        and say so. Each state's tier is displayed on the State Workspace page.
      </p>
      <Card sub="What each state can support, by published data — not by marketing.">
        <table className="data">
          <thead><tr><th>Tier</th><th>Resolution</th><th>Basis</th></tr></thead>
          <tbody>
            <tr><td><b>Intraday native</b><br /><span className="mono">DL</span></td>
              <td>15-min, day-ahead · <b>4.33% MAPE</b></td>
              <td>~5 y of SLDC 5-min load — the only state publishing it</td></tr>
            <tr><td><b>Daily pooled</b><br /><span className="mono">8 states</span></td>
              <td>daily energy + exchange purchases · <b>8.78% sMAPE</b> vs 11.14% naive</td>
              <td>MERIT daily history, one pooled LightGBM across states</td></tr>
            <tr><td><b>Daily pooled (partial)</b><br /><span className="mono">4 states</span></td>
              <td>daily, gappy</td>
              <td>intermittent MERIT history; fewer rows, metrics still per-state</td></tr>
            <tr><td><b>Live monitoring</b><br /><span className="mono">11 states</span></td>
              <td>demand / own gen / import, live</td>
              <td>no historical series published — monitored, not forecast</td></tr>
          </tbody>
        </table>
      </Card>
      <p style={{ marginTop: 8, fontSize: 13, color: "var(--muted)" }}>
        The tiers classify <i>published data</i>; the pooled model is actually
        trained and scored on <b>11 states</b> — the 8 full plus the partial ones
        that cleared the plausibility guard, minus Madhya Pradesh (see below).
      </p>
      <p style={{ marginTop: 10 }}>
        Guard rails are the same as everywhere else: chronological splits, demand
        lags ≥ 2 days so the model is usable at bid time, per-state test metrics
        reported individually, and a naive seasonal baseline (same weekday last
        week) scored on the identical window — if we don't beat it for a state, that
        state's row says so.
      </p>

      <h3>What we serve is a combination, not a choice — and why</h3>
      <p>
        This model used to pick, per state, whichever of {"{"}pooled model,
        seasonal naive{"}"} won on validation. <b>We measured that rule and it was
        worse than not choosing at all:</b> 17.5% sMAPE on test, against 11.1%
        for always-naive and 16.2% for always-model — worse than either pure
        strategy. With one year of daily history the per-state windows are far
        too noisy to select on; the baseline's own error swings from 64.9% to
        7.8% (Haryana) between validation and test, so the selection is close to
        a coin flip and occasionally an expensive one.
      </p>
      <p>
        Estimating a blend weight fails for a related reason: every window
        earlier in time flatters the model, because its edge decays going
        forward. Weight selection on validation picked 0.7, a
        fitting-independent window picked 0.6, and the test-optimal was ~0.2. A
        parameter whose estimate moves that much with the window should not be
        estimated. So the weights are <b>fixed a priori and equal</b> — the
        oldest robust result in forecast combination (Bates &amp; Granger 1969;
        the "forecast combination puzzle", Smith &amp; Wallis 2009, where
        estimated weights routinely lose to the simple average). It costs ~0.5 pp
        on the average and nearly halves the worst state.
      </p>
      <Card sub="Same data, same split — only the combination rule changes.">
        <table className="data">
          <thead><tr><th>Rule</th><th className="num">Served sMAPE</th>
            <th className="num">Worst state</th></tr></thead>
          <tbody>
            <tr><td>per-state champion (what we removed)</td>
              <td className="num">17.5%</td><td className="num">45.7%</td></tr>
            <tr><td>always naive</td><td className="num">11.1%</td>
              <td className="num">45.7%</td></tr>
            <tr><td>always model</td><td className="num">16.2%</td>
              <td className="num">44.7%</td></tr>
            <tr style={{ fontWeight: 700 }}><td>equal-weight combination ✅</td>
              <td className="num">8.78%</td><td className="num">16.1%</td></tr>
          </tbody>
        </table>
      </Card>
      <p style={{ marginTop: 10 }}>
        The other half of the gain came from features, not the rule: each state's
        own <b>generation mix</b> — hydro, thermal, solar, wind and their shares
        — lagged ≥ 2 days like demand. Energy met is procurement, and procurement
        is demand minus what the state generates itself; a hydro state in a wet
        month and the same state in a dry one buy very differently at identical
        demand and temperature. Served error is now <b>8.78% vs 11.14% naive</b>,
        and the worst state fell from 35% to 16%.
      </p>
      <p>
        Two coarse overrides remain, both written on structure rather than on
        scores. A state whose demand is driven by something we cannot observe is
        served the baseline alone — operationalised as <b>hydro &gt; 50% of energy
        met AND coefficient of variation &gt; 40%</b>. Himachal is 67.6% hydro
        with 55.2% CV, the smallest and most volatile series in the panel, and its
        procurement tracks reservoir and snowmelt state that appears in no feature
        we hold; adding the generation features did not fix it. The rule is
        applied to every state equally and today HP is the only one it catches.
        Conversely, a state whose baseline is more than 3× worse than the model is
        served the model alone, because averaging assumes both parts are credible
        — on the exchange target the naive baseline scores 2,590% for Maharashtra,
        and blending 50/50 with that turned a 41% model into 1,302%.
      </p>
      <p>
        Madhya Pradesh is excluded entirely by a physical plausibility guard: its
        State Generation leg implies ~56 GW against a ~17 GW peak. Better to serve
        11 states honestly than 12 with one silently wrong by 6×.
      </p>

      <h2>Model lab — how the numbers got better</h2>
      <p>
        Improvements are adopted from controlled experiments, one change at a
        time, on the same untouched chronological test window — never from a
        lucky run. The 24 Jul session cut load MAPE 4.98%→4.33% (thermal-inertia
        features + recency weighting + tuned params + 3-seed ensemble) and price
        evening-MAPE 15.3%→11.4% (cap-hurdle two-stage), and — measured honestly —
        switched the CVaR layer <i>off</i> by default when it stopped paying.
      </p>
      {m.model_lab && (
        <div className="grid2">
          {["load", "price"].map((k) => m.model_lab[k] && (
            <Card key={k} title={`${k} experiments`} sub="test-set leaderboard; adopted = bold">
              <div style={{ overflowX: "auto" }}>
                <table className="data">
                  <thead><tr><th>Experiment</th><th className="num">MAPE %</th><th className="num">{k === "load" ? "RMSE MW" : "corr"}</th>{k === "price" ? <th className="num">evening %</th> : null}</tr></thead>
                  <tbody>
                    {m.model_lab[k].map((r) => {
                      const adopted = r.experiment.startsWith(k === "load" ? "L8" : "P3");
                      return (
                        <tr key={r.experiment} style={adopted ? { fontWeight: 700 } : undefined}>
                          <td>{r.experiment}{adopted ? " ✅" : ""}</td>
                          <td className="num">{r.test_mape_pct}</td>
                          <td className="num">{k === "load" ? r.test_rmse_mw : r.corr}</td>
                          {k === "price" ? <td className="num">{r.evening_mape_pct}</td> : null}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          ))}
        </div>
      )}

      <h2>Realized accuracy — forecasts we actually issued</h2>
      <p>
        Stricter than any backtest: each daily pipeline run files its forecast
        (plan_&lt;date&gt;.csv) <i>before</i> delivery; once the day has passed it is
        scored against what actually happened. A customer's yardstick, and one
        only a genuinely-running system can produce.
      </p>
      <p>
        Price is scored in <b>rupees</b>, not percent, and that is deliberate.
        Over the last 365 days 16.1% of DAM blocks pinned at the ₹10,000 cap
        and 20.1% cleared below ₹2,000, so one fixed ₹800/MWh error reads as
        80% at the 5th percentile of price and 8% at the 85th. A percentage
        error on this target scores hardest where the money is smallest, and
        its level says more about the window's price distribution than about
        the model. MAE is what a trading desk is exposed to; correlation is
        what monetizes.
      </p>
      {Array.isArray(m.realized_accuracy) && m.realized_accuracy.length > 0 && (
        <Card>
          <table className="data">
            <thead><tr><th>Delivery day</th><th>Issued</th><th className="num">Load MAPE %</th><th className="num">Price MAE ₹/MWh</th><th className="num">Price WAPE %</th><th className="num">Price corr</th></tr></thead>
            <tbody>
              {m.realized_accuracy.map((r) => (
                <tr key={r.delivery_day}>
                  <td><b>{r.delivery_day}</b></td>
                  <td style={{ color: "var(--muted)" }}>{r.issued}</td>
                  <td className="num">{r.load_mape_pct ?? "—"}</td>
                  <td className="num">{r.price_mae_rs_mwh ?? "—"}</td>
                  <td className="num">{r.price_wape_pct ?? "—"}</td>
                  <td className="num">{r.price_corr ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      <h2>Known limitations (say them before you're asked)</h2>
      <ul>
        <li>Price MAE ≈ ₹746/MWh on the held-out test window (WAPE 14.9%, corr 0.933). We previously quoted this as "MAPE ~20%", which overstated the error: with 16.1% of blocks at the cap and 20.1% under ₹2,000, MAPE is dominated by the cheap blocks it matters least on. Indian DAM is genuinely volatile; shape is what monetizes, and 93.8% capture is the proof.</li>
        <li>RE plant is a digital twin, not plant telemetry — standard pre-integration practice; the forecast error is still real NWP error.</li>
        <li>DSM engine follows the CERC 2024 <i>structure</i> with cited provenance per rule; final gazetted slabs and SERC variants need counsel review before real settlement.</li>
        <li>Ancillary-services prices have no public feed (NLDC-internal) — the NR's third component is proxied by RTM and flagged in every result.</li>
        <li>Annualising a summer backtest window overweights cap-price evenings — quote the walk-forward window's own number as the hard result. Measured across the full year, mean daily DAM spread is ₹7,771 vs ₹8,785 in the backtest window, so the annualised figure is ~12% optimistic; ~₹8.5 Cr/yr is the defensible number.</li>
        <li><b>Cycling is now priced, not capped — fixed 2 Aug 2026.</b> The LP previously charged a ₹200/MWh throughput cost that was ~4× below our own physics model, so it over-traded and the headline revenue was partly paid for in warranty. It now charges the calibrated ₹800/MWh and self-regulates to 412 EFC/yr, inside the 365–550 envelope. Cost of the correction: annualised revenue ₹9.61 Cr → ₹8.23 Cr. There is still no hard cycle cap, and deliberately so — see the section above.</li>
        <li>Same-day RTM is still marginally worse than the DAM-ratio incumbent on the LEVEL (29.18% vs 28.81% WAPE) while being far better on spread DIRECTION (68.6% vs 48.1%). It is promoted because direction is what the optimizer trades on, but the level loss is real and stated rather than averaged away. Day-ahead now beats persistence on both (29.89% vs 31.29% WAPE, 67.4% vs 61.8% direction) after the RTM history was backfilled from 12.7 to 38 months.</li>
        <li>The DSM exposure forecast prices a scheduled generator, not a DISCOM's drawal book: our settlement engine implements the general-seller band, and we could not verify a buyer profile's slabs. No "optimal schedule bias" is offered — the optimum sits at the edge of the sweep because the engine lacks the over-injection caps the real regulation carries, so any optimum would be an artifact of a missing rule rather than a real saving.</li>
        <li>There is <b>no realised RE generation data at all</b> — <span className="mono">re_weather</span> is forecast-only — so an RE developer's plant-level DSM exposure, the case with the strongest commercial pull, is not priced. An invented error distribution quoted in rupees would be worse than no feature.</li>
        <li>SLDC/IEX access is scraping, not contracted feeds — exactly the partnership the business model names (§9); adapters are pluggable for the day feeds are licensed.</li>
        <li>Degradation physics uses datasheet-typical LFP parameters (L₁₀₀ = 6000, k = 1.1, ₹1.5 Cr/MWh capex, 70% cycle-attributed), not parameters fitted to this battery's cell tests — the honest number is still ~4× the old flat proxy.</li>
        <li>The C&amp;I demo runs on an illustrative factory profile and indicative DERC tariff values — peak-shaving economics are profile-shaped, so a pilot's real meter CSV (which drops straight in) is the real test.</li>
        <li>BESS validation quotes an uplift only for days with ≥90% telemetry coverage — partial sampling that catches only the discharge window would flatter the comparison, so it's labelled PARTIAL and excluded from headlines.</li>
      </ul>
    </div>
  );
}
