import { useState } from "react";

import { Card, InfoTip, Loading, PageHeader, Stat } from "../components/ui";
import { TimeSeries } from "../components/charts";
import { fmtINR, fmtTs, useApi } from "../lib/api";

export default function Renewables() {
  const { data: dsm, loading, error } = useApi("/api/dsm");
  const { data: re } = useApi("/api/re-state");
  const { data: dstate } = useApi("/api/dsm-state");
  const [profile, setProfile] = useState("CERC_2024");

  if (loading && !dsm) return <Loading error={error} />;

  const p = dsm?.[profile] || {};
  const flex = p.flextrade || {};
  const naive = p.naive || {};
  const blocks = (p.blocks || []).map((b) => ({
    t: String(b.ts).slice(11, 16),
    Scheduled: b.schedule_mw, Actual: b.actual_mw, Naive: b.naive_mw,
    Solar: b.solar_actual_mw, Wind: b.wind_actual_mw,
  }));
  const alerts = dsm?.alerts || {};

  return (
    <>
      <PageHeader eyebrow="Renewables & DSM"
        title="Forecast the plant, settle the deviation"
        lead="A solar + wind digital twin scored on real day-ahead weather error, settled block-by-block
              through a versioned CERC engine (2022 & 2024), with an alerts engine that flags a
              profitable schedule revision before the gate closes." />
      <StateDSMPanel d={dstate} />

      <RealREPanel re={re} />

      <h2 className="section-title">Deviation Settlement Mechanism — settlement day {dsm?.settlement_day}</h2>
      <div className="note info" style={{ marginBottom: 12 }}>
        Reference portfolio: 50 MW solar + 50 MW wind <b>digital twin</b> (Delhi NCR).
        Be clear about which half is real. The <b>weather error is real</b>: the
        schedule is what the weather model forecast <i>one day earlier</i>
        (Open-Meteo previous-runs API) and the settlement uses the analysis that
        followed, so the deviation is genuine day-ahead NWP error, not noise we
        invented. The <b>plant is not real</b> — both schedule and "actual"
        generation are produced by the same deterministic physics twin, because we
        hold <b>zero rows of measured RE generation</b>. That means this prices the
        deviation a weather miss causes, and excludes everything a real plant adds:
        outages, soiling, curtailment and inverter clipping, which in practice
        dominate. Read it as a lower bound on exposure. Solar and wind are settled
        separately — different tolerance bands, and different X-factor glide paths
        from FY28.
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        {["CERC_2024", "CERC_2022"].map((k) => (
          <button key={k} className="btn"
            style={profile === k ? { background: "var(--band)", borderColor: "var(--s1)" } : {}}
            onClick={() => setProfile(k)}>
            {k.replace("_", " ")} {k === "CERC_2024" ? "· in force" : "· superseded"}
          </button>
        ))}
      </div>

      <div className="grid cols-4">
        <Stat label="Net DSM (FlexTrade forecast)" info="DSM, MAPE" value={fmtINR(flex.net_dsm_rs)}
          hint={`${flex.blocks_outside_band ?? "—"} blocks outside band · MAE ${flex.mae_mw?.toFixed?.(1) ?? "—"} MW`} />
        <Stat label="Net DSM (naive persistence)" infoText="Persistence — the no-model baseline: schedule tomorrow = what happened at the same block yesterday. If our forecast can't beat this, it adds nothing." value={fmtINR(naive.net_dsm_rs)}
          hint="schedule = same block yesterday" />
        <Stat label="Saved by forecasting" value={fmtINR(p.saved_rs)}
          deltaDir={p.saved_rs > 0 ? "up" : "down"}
          delta={p.saved_rs > 0 ? "forecast beats persistence" : "persistence won this day"}
          hint="this varies day to day — honest variance, shown as-is" />
        <Stat label="Normal Rate (mean)" info="NR, DAM, RTM" value={fmtINR(flex.mean_normal_rate_rs_mwh)} unit="/MWh"
          hint={profile === "CERC_2024" ? "⅓ DAM + ⅓ RTM + ⅓ ancillary (ancillary proxied by RTM)" : "mean(DAM, RTM) per block"} />
      </div>

      {blocks.length > 0 && (
        <div className="grid cols-2" style={{ marginTop: 14 }}>
          <Card title="Schedule vs actual" sub="the deviation being settled">
            <TimeSeries data={blocks} xKey="t" height={260} yLabel="MW"
              series={[
                { key: "Scheduled", name: "Scheduled (day-ahead)", color: "var(--s1)" },
                { key: "Actual", name: "Actual", color: "var(--muted)", dash: "4 3" },
              ]} />
          </Card>
          <Card title="Actual generation by technology" sub="solar and wind settled independently">
            <TimeSeries data={blocks} xKey="t" height={260} yLabel="MW"
              series={[
                { key: "Solar", name: "Solar", color: "var(--s4)", type: "area" },
                { key: "Wind", name: "Wind", color: "var(--s5)", type: "area" },
              ]} />
          </Card>
        </div>
      )}

      <h2 className="section-title">Alerts &amp; Revision Engine</h2>
      <div className="grid cols-3">
        <Stat label="Gate lead time" info="RTM" value={`≥ ${alerts.lead_minutes ?? 15} min`}
          hint="DSM spec success metric (Table 5)" />
        <Stat label="Blocks flagged" value={alerts.items?.length ?? 0}
          hint="REVISE = schedule change pays; HOLD = exposure exists but revision won't fix it" />
        <Stat label="Benefit if revised" info="DSM" value={fmtINR(alerts.total_benefit_rs)}
          hint={alerts.asof ? `evaluated ${fmtTs(alerts.asof)}` : ""} />
      </div>
      <Card style={{ marginTop: 14 }}
        sub={`Deliberately conservative: only fires past a materiality threshold, because over-triggering on forecast noise destroys operator trust. Schedule basis: ${alerts.schedule_basis || "—"}.`}>
        {alerts.items?.length ? (
          <table className="data">
            <thead><tr><th>Block</th><th>Action</th><th className="num">Scheduled MW</th><th className="num">Forecast MW</th><th className="num">Benefit</th><th>Reason</th></tr></thead>
            <tbody>
              {alerts.items.map((a, i) => (
                <tr key={i}>
                  <td>{a.block}</td>
                  <td><span className={`pill ${a.action.toLowerCase()}`}>{a.action}</span></td>
                  <td className="num">{a.scheduled_mw}</td>
                  <td className="num">{a.forecast_mw}</td>
                  <td className="num">{fmtINR(a.benefit_rs)}</td>
                  <td style={{ fontSize: 12, color: "var(--muted)" }}>{a.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div style={{ color: "var(--muted)" }}>
            No material deviation flagged for the remaining blocks — normal at night
            for a solar-heavy portfolio.
          </div>
        )}
      </Card>

      <h2 className="section-title">Regulatory provenance</h2>
      <Card sub="Built as a configurable rule engine because the regulation keeps changing — every band, rate and effective date is a named constant, applied automatically by settlement date.">
        <table className="data">
          <thead><tr><th>Rule</th><th>Status</th><th>Detail</th></tr></thead>
          <tbody>
            <tr><td>2022 volume bands</td><td><span className="pill verified">confirmed ×3 sources</span></td>
              <td>WS over-injection 0–5% @ 100% NR, 5–10% @ 90%, &gt;10% uncompensated; under-injection free to 10%, then 10% of NR; buyer 10–15% @ 120%, &gt;15% @ 150%</td></tr>
            <tr><td>2024 Normal Rate</td><td><span className="pill verified">in force since Sep 2024</span></td>
              <td>NR = ⅓ I-DAM + ⅓ RTM + ⅓ ancillary; ancillary has no public feed → proxied by RTM, flagged in every result</td></tr>
            <tr><td>Tolerance tightening</td><td><span className="pill verified">effective 01 Apr 2026</span></td>
              <td>Solar/hybrid ±10% → ±5%; wind ±15% → ±10%; new WS projects treated as general sellers</td></tr>
            <tr><td>X-factor glide path</td><td><span className="pill verified">FY28 → FY32</span></td>
              <td>Deviation denominator blends capacity → schedule; X = 1.0 confirmed unchanged through FY27</td></tr>
            <tr><td>Frequency-linked rates</td><td><span className="pill identified">NOT confirmed</span></td>
              <td>Traces to a CERC consultation memo, not gazetted text — off by default, opt-in for scenario exploration only</td></tr>
            <tr><td>Third Amendment (daily NR)</td><td><span className="pill identified">draft, May 2026</span></td>
              <td>Proposed daily-average ACP instead of block-wise — implemented as opt-in pending notification</td></tr>
          </tbody>
        </table>
        <div className="note crit" style={{ marginTop: 12 }}>
          Decision support, not settlement accounting: before any rupee of real
          settlement, every number must be verified against the CERC gazette text
          by counsel — which is exactly what the DSM spec's own risk table (§6) requires.
        </div>
      </Card>
    </>
  );
}

/* The REAL RE forecast — trained and scored on measured generation from
   MERIT's plant-level feed, as opposed to the physics twin above which
   simulates a hypothetical plant. This is the answer to "everything here
   rests on a simulation": it no longer does. */
function RealREPanel({ re }) {
  if (!re) return null;
  if (re.error) return <Card title="Measured RE forecast"><div className="note">{re.error}</div></Card>;
  const solar = re.solar_mwh || {};
  const wind = re.wind_mwh || {};
  const rows = [["Solar", solar], ["Wind", wind]];

  return (
    <>
      <h2 className="section-title">
        Measured RE forecast — real generation, not a twin
        <InfoTip text="Trained and scored on daily solar and wind output that actually happened, from MERIT's plant-level endpoint. The digital twin above prices a hypothetical plant; this prices real fleets in real states." />
      </h2>
      <div className="note info">
        <b>This is the part that is not simulated.</b> The DSM settlement above uses
        a physics twin, because we hold no plant-level telemetry. But MERIT publishes{" "}
        <b>measured daily solar and wind output per state</b>, and this model is
        trained and scored on that — generation that actually happened, scored
        against generation that actually happened. Rajasthan alone averages
        43,444 MWh/day of solar.
      </div>

      <div className="grid2">
        {rows.map(([name, v]) => (
          <Card key={name} title={`${name} — served error`} info="WAPE"
            sub={v.basis || "pooled across states, chronological split"}>
            {v.error ? <div className="note">{v.error}</div> : (
              <>
                <div className="grid3">
                  <Stat label="Served" value={`${v.served_wape_pct}%`} unit="WAPE"
                    hint={`vs ${v.naive_wape_pct}% seasonal-naive`} />
                  <Stat label="Beat naive" value={v.states_beating_naive || "—"}
                    hint={`${v.n_states} states, ${v.test_days}-day held-out window`} />
                  <Stat label="Training rows" value={(v.n_train_rows || 0).toLocaleString("en-IN")}
                    hint={`${v.history_from} → ${v.history_to}`} />
                </div>
                <div className="scroll-x">
                  <table className="data">
                    <thead><tr>
                      <th>State</th><th className="num">Served</th>
                      <th className="num">Model</th><th className="num">Naive</th>
                      <th>What we serve</th><th className="num">Mean MWh/day</th>
                    </tr></thead>
                    <tbody>
                      {(v.per_state || []).map((r) => (
                        <tr key={r.code}>
                          <td><b>{r.name}</b></td>
                          <td className="num"><b>{r.served_wape_pct}%</b></td>
                          <td className="num" style={{ color: "var(--muted)" }}>{r.model_wape_pct}%</td>
                          <td className="num" style={{ color: "var(--muted)" }}>{r.naive_wape_pct}%</td>
                          <td><span className={`pill ${String(r.served).startsWith("blend") || r.served === "model" ? "verified" : "hold"}`}>{r.served}</span></td>
                          <td className="num">{r.mean_mwh?.toLocaleString("en-IN")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </Card>
        ))}
      </div>

      <div className="note">
        <b>Two physics bugs were found building this, and both mattered.</b> The first
        version fed the model <span className="mono">wind_speed_10m_max</span> — but a
        turbine's power goes as the <b>cube of hub-height (100 m) speed</b>, so it was
        being asked to learn a height extrapolation and a cubic at once (38.0% → 34.0%
        WAPE once fixed). Worse, weather came from the state <i>capital</i> while the
        fleet sits hundreds of kilometres away: Chennai for a fleet at Muppandal, which
        is in the Palghat Gap, a monsoon wind tunnel with an unrelated regime. Using
        fleet coordinates took wind to 29.4%, and <b>Tamil Nadu from 21.8% to 12.7%</b> —
        the state with the worst geographic error gained the most, which is the
        confirmation the diagnosis was right.
      </div>
    </>
  );
}

/* Real per-state DSM. Everything else on this page prices a twin; this prices
   a genuine 11 GW schedule against genuine drawal, for the only state in India
   that publishes both. */
function StateDSMPanel({ d }) {
  if (!d) return null;
  if (d.error) return <Card title="Per-state DSM"><div className="note">{d.error}</div></Card>;
  const s = d.settlement || {};
  const fv = d.forecast_value || {};
  if (s.error) return <Card title="Per-state DSM"><div className="note">{s.error}</div></Card>;
  const cr = (v) => (v == null ? "-" : `\u20b9${(v / 1e7).toFixed(1)} Cr`);

  return (
    <>
      <h2 className="section-title">
        Deviation, priced on a real state
        <InfoTip text="Uttar Pradesh's load despatch centre is the only one in India publishing schedule, drawal and deviation together. Everything else on this page prices a simulated plant; this prices a real book." />
      </h2>

      <div className="note info">
        <b>This is the market that is compelled to buy.</b> A generator chooses whether
        to hedge. A DISCOM does not choose whether to deviate — it is charged for it
        under the CERC framework whatever it does. UP runs a{" "}
        <b>{s.mean_schedule_mw?.toLocaleString("en-IN")} MW</b> inter-state schedule with a
        mean absolute deviation of <b>{s.mean_abs_deviation_mw?.toLocaleString("en-IN")} MW</b>{" "}
        against a tolerance band of just {s.band_mw_typical} MW — outside the band on{" "}
        <b>{s.outside_band_pct}%</b> of snapshots.
      </div>

      <div className="grid4">
        <Stat label="Mean |deviation|" value={s.mean_abs_deviation_mw?.toLocaleString("en-IN")} unit="MW"
          hint={`band is ${s.band_mw_typical} MW — min(10% of schedule, 100 MW)`} />
        <Stat label="Outside the band" value={`${s.outside_band_pct}%`}
          hint={`of ${s.snapshots} snapshots · over-drawing ${s.over_drawing_pct}%`} />
        <Stat label="Normal Rate" value={`\u20b9${s.mean_normal_rate_rs_mwh?.toLocaleString("en-IN")}`}
          unit="/MWh" hint="CERC Reg 14: mean of DAM, RTM and ancillary"
          infoText="Built from our live DAM and RTM feeds, so the charge moves with the market. The ancillary leg has no public feed and is proxied by RTM, which is flagged in every result." />
        <Stat label="Exposure, upper bound" value={cr(s.est_payable_per_year_rs)} unit="/yr"
          hint={`scaled x${s.sample_scale_factor} from ${s.blocks_sampled} priced blocks`}
          infoText="An upper bound, not a settlement. See the note below for why it is biased high." />
      </div>

      <div className="note crit">
        <b>Read this as the top of a range, not a bill.</b> {s.caveat}
      </div>

      {fv.scenarios && (
        <Card title="What cutting deviation is worth" info="DSM"
          sub="Deviation charges scale with the excess beyond the band, so a proportional cut in deviation is a proportional cut in charge. The percentage is the customer's assumption about their own improvement — not our claim about it.">
          <div className="scroll-x">
            <table className="data">
              <thead><tr><th>Deviation reduced by</th><th className="num">Annual saving</th></tr></thead>
              <tbody>
                {fv.scenarios.map((x) => (
                  <tr key={x.deviation_reduced_pct}>
                    <td>{x.deviation_reduced_pct}%</td>
                    <td className="num"><b>{cr(x.annual_saving_rs)}</b></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="note" style={{ marginTop: 10 }}>
            Both columns inherit the upper-bound bias above. The point is the
            <i> shape</i> of the opportunity, not the rupee figure: for a state of
            this size, single-digit percentage improvements in scheduling are worth
            tens of crores a year, which is why deviation is the line item a DISCOM
            will fund a forecast to reduce.
          </div>
        </Card>
      )}
    </>
  );
}
