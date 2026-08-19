/* Forecast Lab — the four models built after the RTM backfill unlocked them.
   Every panel shows what the model scores AND what it had to beat, because a
   forecast accuracy number with no baseline beside it is decoration. */
import { Card, InfoTip, Loading, PageHeader, Stat, Tabs } from "../components/ui";
import { FanChart, HBar } from "../components/charts";
import { useApi } from "../lib/api";
import WalkForward, { wfModel } from "../components/WalkForward";

const pct = (v, d = 1) => (v == null ? "—" : `${Number(v).toFixed(d)}%`);
const rs = (v) =>
  v == null ? "—" : `₹${Math.round(Number(v)).toLocaleString("en-IN")}`;


const bandWf = (meta) => wfModel(meta, "load band");

const TABS = [
  { id: "rtm", label: "RTM & Spread", icon: "⇄", hint: "intraday" },
  { id: "band", label: "Probabilistic Load", icon: "◍", hint: "P05–P95" },
  { id: "peak", label: "Peak Forecast", icon: "▲", hint: "when & how big" },
  { id: "dsm", label: "DSM Exposure", icon: "⚖", hint: "₹ at risk" },
];

export default function Forecasts() {
  const { data, loading, error } = useApi("/api/forecasts");
  const { data: meta } = useApi("/api/meta");
  if (loading && !data) return <Loading error={error} />;

  const rtm = data?.rtm || {};
  const band = data?.load_quantiles || {};
  const peak = data?.peak || {};
  const dsm = data?.dsm_exposure || {};

  return (
    <div>
      <PageHeader eyebrow="Forecast Lab"
        title="Four models beyond the day-ahead curve"
        lead="Each of these was built because a specific decision needed it — which
              way to trade intraday, how wide a schedule band has to be, when to
              have the battery full, and what deviation will cost. Each is scored
              against the thing it replaced, on a window it never saw." />

      <Tabs tabs={TABS}>
        {(tab) => {
          if (tab === "rtm") return <RtmPanel rtm={rtm} />;
          if (tab === "band") return <BandPanel band={band} />;
          if (tab === "peak") return <PeakPanel peak={peak} />;
          return <DsmPanel dsm={dsm} />;
        }}
      </Tabs>
    </div>
  );
}

/* ------------------------------------------------------------------ RTM */
const HORIZON_NOTE = {
  intraday: "Re-optimising during the delivery day: today's DAM has cleared and recent RTM sessions are known. This is the model the intraday optimizer trades on.",
  sameday: "Later today: the DAM is known, but no RTM session near the block has cleared yet.",
  dayahead: "Bidding the DAM at 12:00 on D-1. Tomorrow's DAM has NOT cleared, so it cannot be used — the baselines are anchored on our own DAM forecast, not the actual price.",
};

function RtmPanel({ rtm }) {
  if (rtm.error) return <Card><div className="note">{rtm.error}</div></Card>;
  const hz = rtm.horizons || {};
  const intra = hz.intraday || {};
  const rows = Object.entries(hz);

  return (
    <>
      <div className="grid4">
        <Stat label="Intraday WAPE" value={pct(intra.served_wape_pct, 2)}
          hint={`vs ${pct(intra.incumbent_wape_pct, 2)} for the hour-ratio it replaced`}
          infoText={rtm.metric_note} />
        <Stat label="Spread direction" value={pct(intra.served_direction_pct)}
          hint={`vs ${pct(intra.incumbent_direction_pct)} incumbent · 50% coin flip`}
          infoText={rtm.why_direction} />
        <Stat label="Champion" value={intra.champion === "model" ? "model" : intra.champion}
          hint="chosen on validation, never on test" />
        <Stat label="Held-out window" value={`${intra.n_test?.toLocaleString("en-IN") || "—"}`}
          unit="blocks" hint={`${intra.test_from || "?"} → ${intra.test_to || "?"}`} />
      </div>

      <WalkForward meta={meta} match="rtm" />

      <Card title="Every horizon, and what it must beat"
        sub="The three horizons have genuinely different information, so each is trained and scored separately. A champion is picked per horizon on the validation window; where the baseline wins we serve the baseline and say so.">
        <div className="scroll-x">
          <table className="data">
            <thead><tr>
              <th>Horizon</th><th className="num">Served WAPE</th>
              <th className="num">Model WAPE</th><th className="num">Incumbent</th>
              <th className="num">Direction</th><th>Champion</th>
            </tr></thead>
            <tbody>
              {rows.map(([h, s]) => (
                <tr key={h}>
                  <td><b>{h}</b><br />
                    <span style={{ color: "var(--muted)", fontSize: 11.5 }}>
                      {HORIZON_NOTE[h]}</span></td>
                  <td className="num"><b>{pct(s.served_wape_pct, 2)}</b></td>
                  <td className="num" style={{ color: "var(--muted)" }}>{pct(s.model_wape_pct, 2)}</td>
                  <td className="num" style={{ color: "var(--muted)" }}>{pct(s.incumbent_wape_pct, 2)}</td>
                  <td className="num">{pct(s.served_direction_pct)}</td>
                  <td><span className={`pill ${s.champion === "model" ? "verified" : "hold"}`}>
                    {s.champion}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="note info" style={{ marginTop: 12 }}>
          <b>Why WAPE and not MAPE.</b> {rtm.metric_note}
        </div>
        <div className="note" style={{ marginTop: 8 }}>
          <b>Day-ahead loses to persistence on the level and we show it.</b> The
          model is clearly better at calling the <i>direction</i> of the spread
          (73% vs 64%) but not at the level, so persistence is the champion
          there. Only the intraday model is promoted into the optimizer.
        </div>
      </Card>

      {rtm.metrics_text && (
        <Card title="Full training report" sub="Verbatim, including the runs we lost.">
          <pre className="code-block">{rtm.metrics_text}</pre>
        </Card>
      )}
    </>
  );
}

/* ----------------------------------------------------------------- band */
function BandPanel({ band }) {
  if (band.error) return <Card><div className="note">{band.error}</div></Card>;
  const rows = (band.tomorrow || []).map((r) => ({
    ...r, t: String(r.ts || "").slice(11, 16),
  }));
  const m = band.conformal?.margins_mw?.["0.1-0.9"] || {};
  const sv = band.served || {};   // parsed from the metrics report, not hardcoded

  return (
    <>
      <div className="grid4">
        <Stat label="Band coverage" value={sv.coverage_pct ? `${sv.coverage_pct}%` : "—"}
          hint="nominal 80% · measured on held-out data"
          infoText={`Share of held-out blocks where actual load fell inside the served P10–P90 band. Raw quantile regression achieved only ${sv.raw_coverage_pct ?? "—"}%; conformal calibration fixed it.`} />
        <Stat label="Mean band width" value={sv.width_mw?.toLocaleString("en-IN") ?? "—"} unit="MW"
          hint="the number a customer pays for in procured reserve" />
        <Stat label="Conformal margin" value={`−${m.lo ?? "—"} / +${m.hi ?? "—"}`} unit="MW"
          infoText="Asymmetric on purpose: the median under-forecasts, so nearly all the missed coverage escapes through the upper bound. Widening both sides equally would buy no coverage and cost band width." />
        <Stat label="Calibrated on" value={band.conformal?.calibration_n?.toLocaleString("en-IN") || "—"}
          unit="blocks" hint={`${band.conformal?.calibration_from || ""} → ${band.conformal?.calibration_to || ""}`} />
      </div>

      <Card title="Tomorrow's load band" info="P05–P95"
        sub="The product a DISCOM actually schedules against. Under the DSM framework you are charged for leaving a band, not for missing an average — so the band is the deliverable and the point forecast is just its middle.">
        {rows.length ? (
          <FanChart data={rows} xKey="t" medianKey="p50" loKey="p10" hiKey="p90"
            height={300}
            extra={[{ key: "p05", name: "P05", color: "var(--s6)", dash: "3 3" },
                    { key: "p95", name: "P95", color: "var(--s6)", dash: "3 3" }]} />
        ) : <div className="note">No band exported for tomorrow yet.</div>}
      </Card>

      <div className="note crit">
        <b>The median under-forecasts, and we publish it.</b> Actual load exceeded
        the P50 on 75.6% of held-out blocks (+104 MW mean) as Delhi's demand grew
        through the window. That matters in one direction only — under-scheduling
        is what draws deviation charges — so the conformal margin corrects it
        asymmetrically rather than leaving an operator to discover it.
      </div>

      {band.metrics_text && (
        <>
        <WalkForward meta={meta} match="load band">
          <p className="muted" style={{ marginTop: 10 }}>
            Coverage is the claim being sold, so it is tested rather than
            asserted. Kupiec (1995) asks whether the miss RATE matches nominal;
            Christoffersen (1998) asks whether the misses are independent or
            arrive in clusters. Both verdicts are printed below, including where
            they reject — a band that fails a test it never ran is worse than
            one that fails a test it published.
          </p>
          <div className="grid cols-3" style={{ marginTop: 6 }}>
            <Stat label="Coverage across origins"
              value={pct(bandWf(meta)?.coverage_pct_mean, 1)}
              hint={`worst origin ${pct(bandWf(meta)?.coverage_pct_worst, 1)} vs ${pct(bandWf(meta)?.nominal_pct, 0)} nominal`} />
            <Stat label="Kupiec rejects"
              value={`${bandWf(meta)?.kupiec_rejected_origins?.length ?? 0}/${bandWf(meta)?.origins_run ?? 0}`}
              hint="origins where the miss rate differs from nominal" />
            <Stat label="Christoffersen rejects"
              value={`${bandWf(meta)?.independence_rejected_origins?.length ?? 0}/${bandWf(meta)?.origins_run ?? 0}`}
              hint="origins where misses cluster rather than arrive independently" />
          </div>
        </WalkForward>

        <Card title="Calibration report" sub="Raw vs symmetric vs asymmetric vs Mondrian vs trailing-window conformal, on the untouched test window. The served band recalibrates daily on a trailing 14 days — swept on the rolling-origin audit, where 14d beat 45d and the static margin on interval score, mean coverage and worst-origin coverage alike.">
          <pre className="code-block">{band.metrics_text}</pre>
        </Card>
        </>
      )}
    </>
  );
}

/* ----------------------------------------------------------------- peak */
function PeakPanel({ peak }) {
  if (peak.error) return <Card><div className="note">{peak.error}</div></Card>;
  const t = peak.tomorrow || {};
  const mag = peak.magnitude || {};
  const tim = peak.timing || {};
  const probs = Object.entries(t.hour_probabilities || {})
    .map(([h, p]) => ({ name: `${String(h).padStart(2, "0")}:00`, value: p }));

  return (
    <>
      <div className="grid4">
        <Stat label="Tomorrow's peak" value={t.peak_mw ? Math.round(t.peak_mw).toLocaleString("en-IN") : "—"}
          unit="MW" hint={`±${Math.round(mag.model?.mae || 0)} MW typical error`} />
        <Stat label="Most likely hour" value={t.peak_hour != null ? `${String(t.peak_hour).padStart(2, "0")}:00` : "—"}
          hint={`${pct(t.peak_hour_confidence_pct)} confidence`} />
        <Stat label="80% window" value={(t.window80_hours || []).map((h) => `${h}:00`).join(", ") || "—"}
          hint={`${peak.window80_hours || "—"} hours on average · contains the peak ${pct(peak.window80_hit_pct)} of the time`}
          infoText="The smallest set of hours holding 80% of the probability. A flat guess over 24 hours would need 19.2 hours for the same confidence." />
        <Stat label="Timing accuracy" value={pct(tim.model?.within1_pct)}
          hint={`within ±1h · exact ${pct(tim.model?.exact_pct)}`} />
      </div>

      <div className="grid2">
        <Card title="Where tomorrow's peak probably lands" info="probability"
          sub="A distribution, not a single hour — a control room hedges across a window, and the width of that window is the information a point estimate throws away.">
          {probs.length ? <HBar data={probs} height={240} valLabel="probability %" />
            : <div className="note">No hour distribution exported.</div>}
        </Card>
        <Card title="Beaten baselines" info="held-out"
          sub="The 96-block load forecast already implies a peak for free, so a dedicated peak model has to beat its argmax to justify existing.">
          <div className="scroll-x">
            <table className="data">
              <thead><tr><th>Method</th><th className="num">Peak MAE</th>
                <th className="num">Exact hour</th><th className="num">±1h</th></tr></thead>
              <tbody>
                {["model", "blockfc", "persist_d2", "sameweekday"].map((k) => (
                  <tr key={k}>
                    <td>{k === "model" ? <b>dedicated model</b>
                      : k === "blockfc" ? "block-forecast argmax" : k}</td>
                    <td className="num">{mag[k]?.mae != null ? `${Math.round(mag[k].mae)} MW` : "—"}</td>
                    <td className="num">{pct(tim[k]?.exact_pct)}</td>
                    <td className="num">{pct(tim[k]?.within1_pct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="note" style={{ marginTop: 10 }}>
            Delhi's peak hour is multi-modal — 10:00 on 27.6% of days, 15:00 on
            20.0%, 23:00 on 14.0% — an afternoon cooling regime and a winter
            evening regime that swap places. A model that learns "peaks are in
            the evening" is wrong a third of the year.
          </div>
        </Card>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ DSM */
function DsmPanel({ dsm }) {
  if (dsm.error) return <Card><div className="note">{dsm.error}</div></Card>;
  const e = dsm.exposure || {};
  const s = dsm.bias_sensitivity || {};
  const curve = (s.curve || []).map((r) => ({
    name: `${r.bias_pct > 0 ? "+" : ""}${r.bias_pct}%`,
    value: Math.round(r.expected_payable_rs),
  }));

  return (
    <>
      <div className="grid4">
        <Stat label="Expected DSM charge" value={rs(e.expected_payable_rs)}
          hint={`${e.entity_peak_mw || "—"} MW entity · ${e.profile || ""}`}
          infoText="Payable charge only. Credits earned by deviating the profitable way are deliberately NOT netted off — see the note below." />
        <Stat label="P90 bad day" value={rs(e.p90_payable_rs)}
          hint={`worst in sample ${rs(e.worst_payable_rs)}`} />
        <Stat label="Blocks outside band" value={pct(e.expected_pct_outside_band)}
          hint="expected, per day" />
        <Stat label="Normal Rate" value={rs(e.normal_rate_mean_rs_mwh)} unit="/MWh"
          hint={`RTM leg: ${e.rtm_basis || "—"}`}
          infoText="CERC 2024 Reg 14 prices deviation off the average of DAM, RTM and ancillary prices — which is why the RTM model feeds this page." />
      </div>

      <Card title="What the exposure is built from" sub={e.scenario_basis}>
        <div className="note info">
          <b>No frequency forecast is involved, and that is not an omission.</b> DSM
          was frequency-linked for years, and the obvious design is to predict grid
          frequency and price deviation off it. CERC <b>de-linked</b> deviation
          charges from frequency in the 2022 regulations and nothing in the notified
          2024 text reinstates it — so the 7 days of frequency data we hold is not a
          blocker for this feature at all.
        </div>
        <div className="note crit" style={{ marginTop: 10 }}>
          <b>We do not ship a "best schedule bias", on purpose.</b> The intended
          product was "bias your schedule by X% and save ₹Y". The optimum sits at
          whichever end of the sweep we stop at, because the general-seller Normal
          Rate as implemented credits favourable deviation with no cap — a real
          settlement caps it precisely so this is not free money. An optimum drawn
          from an incomplete rule would be advice to game a settlement, so the curve
          below is published as information and the recommendation is withheld until
          the over-injection limits are sourced.
        </div>
      </Card>

      <div className="grid2">
        <Card title="Schedule-bias sensitivity" info="not a recommendation"
          sub="Expected payable charge if the schedule is shifted off the forecast. Shows the real trade-off: bias down and you pay less but leave the band far more often.">
          {curve.length ? <HBar data={curve} height={260} valLabel="expected payable ₹"
            color="var(--s3)" /> : <div className="note">No sweep exported.</div>}
        </Card>
        <Card title="Worst error days in the sample" info="measured"
          sub="Real days from the load model's held-out window, replayed against tomorrow's prices.">
          <div className="scroll-x">
            <table className="data">
              <thead><tr><th>Error day</th><th className="num">Payable</th>
                <th className="num">Blocks out</th><th className="num">MAE</th></tr></thead>
              <tbody>
                {(e.worst_scenarios || []).map((w) => (
                  <tr key={w.error_day}>
                    <td>{w.error_day}</td>
                    <td className="num">{rs(w.payable_rs)}</td>
                    <td className="num">{w.blocks_outside_band}</td>
                    <td className="num">{w.mae_mw?.toFixed(1)} MW</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      <div className="note">
        <b>What this does not price.</b> An RE developer's plant is the case with
        the strongest commercial pull, and it is missing here: we hold zero rows of
        realised RE generation, so a plant's error distribution would have to be
        invented. An invented distribution priced in rupees is worse than no
        feature. It ships when the generation feed does.
      </div>
    </>
  );
}
