import { useState } from "react";

import { Card, Loading, PageHeader, Stat } from "../components/ui";
import { TimeSeries } from "../components/charts";
import { fmtINR, fmtTs, useApi } from "../lib/api";

export default function Renewables() {
  const { data: dsm, loading, error } = useApi("/api/dsm");
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
      <h2 className="section-title">Deviation Settlement Mechanism — settlement day {dsm?.settlement_day}</h2>
      <div className="note info" style={{ marginBottom: 12 }}>
        Reference portfolio: 50 MW solar + 50 MW wind digital twin (Delhi NCR).
        The schedule is what the weather model forecast <i>one day earlier</i>
        (Open-Meteo previous-runs API) — real day-ahead forecast error, nothing simulated.
        Solar and wind are settled separately: they carry different tolerance bands
        and, from FY28, different X-factor glide paths.
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
