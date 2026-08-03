import { HBar } from "../components/charts";
import { Card, InfoTip, Loading, PageHeader, Stat } from "../components/ui";
import { fmtMW, useApi } from "../lib/api";

const REGION_ORDER = ["Northern", "Western", "Southern", "Eastern", "North-Eastern"];

export default function States() {
  const { data: meta } = useApi("/api/meta");
  const { data: stress } = useApi("/api/stress");
  const { data: live, loading, error } = useApi("/api/live", { refreshMs: 120_000 });
  const { data: reg } = useApi("/api/states");

  if (loading && !live) return <Loading error={error} />;

  const nrStates = live?.northern_region?.states || [];
  const india = live?.india || reg?.india;
  const national = india?.national || {};
  const meritStates = india?.states || [];
  const registry = reg?.registry || [];
  const verified = registry.filter((s) => s.status === "verified").length;
  const delhiLoad = live?.delhi?.delhi_load || 0;
  const nrTotalGW = (nrStates.reduce((a, s) => a + (s.load_mw || 0), 0) + delhiLoad) / 1000;
  const biggest = [...meritStates].sort((a, b) => (b.demand_mw || 0) - (a.demand_mw || 0))[0];

  const fuelMix = [
    ["Thermal", national.thermal_mw], ["Hydro", national.hydro_mw],
    ["Renewable", national.renewable_mw], ["Nuclear", national.nuclear_mw],
    ["Storage", national.storage_mw], ["Gas", national.gas_mw],
    ["Other", national.other_mw],
  ].filter(([, v]) => v != null).map(([name, value]) => ({ name, value }));

  const gujDirect = reg?.gujarat_direct;
  const rajDirect = reg?.rajasthan_direct;

  return (
    <>
      <PageHeader eyebrow="Multi-State India"
        title="Delhi is the reference, not the market"
        lead="Live demand, own generation and import for 23 states via the Ministry of Power's MERIT
              portal, the all-India fuel mix, and deep per-state adapters — plus which states most
              need forecasting, and where the data is ready to train on." />
      <div className="note info" style={{ marginBottom: 14 }}>
        Coverage is layered and honest: <b>MERIT</b> (Ministry of Power) gives live
        demand / own generation / import for 23 states in one verified source;
        the <b>Northern Region table</b> adds schedule-vs-drawal detail for 8 states;
        <b> deep per-state SLDC adapters</b> (Delhi full history, Gujarat direct) add
        what MERIT can't — frequency, DSM rates, plant telemetry. A state is
        marked <b>verified</b> only after a live fetch returned believable numbers.
      </div>

      <div className="grid cols-4">
        <Stat label="All-India demand met" info="MERIT, MW"
          value={national.demand_met_mw ? (national.demand_met_mw / 1000).toFixed(1) : "—"} unit="GW"
          hint="live, meritindia.in (Ministry of Power)" />
        <Stat label="States live" value={verified} info="SLDC"
          hint={`of ${registry.length} in the registry — all verified by live fetch`} />
        <Stat label="Largest market now" value={biggest?.name || "—"}
          hint={biggest ? `${fmtMW(biggest.demand_mw)} — ${(biggest.demand_mw / Math.max(delhiLoad, 1)).toFixed(1)}× Delhi` : ""} />
        <Stat label="RE on the national grid" info="RE"
          value={national.renewable_mw ? (national.renewable_mw / 1000).toFixed(1) : "—"} unit="GW"
          hint={national.storage_mw ? `+ ${fmtMW(national.storage_mw)} storage (PSP + BESS)` : ""} />
      </div>

      {fuelMix.length > 0 && (
        <Card title="All-India generation mix — right now" info="MERIT"
          style={{ marginTop: 14 }}
          sub="Who is producing the country's power this instant. Storage = pumped hydro + grid batteries — the market FlexTrade optimizes.">
          <HBar height={240} valLabel="MW"
            data={[...fuelMix].sort((a, b) => a.value - b.value)} />
        </Card>
      )}

      {meritStates.length > 0 && (
        <Card title="Live state positions — all India" info="MERIT, ISGS"
          style={{ marginTop: 14 }}
          sub="Demand met, own generation and import per state (MW). Negative import (Himachal) = net exporter. ~30 s upstream refresh.">
          <div className="scroll-x">
            <table className="data">
              <thead><tr><th>State</th><th>Region</th><th className="num">Demand met</th><th className="num">Own generation</th><th className="num">Import</th><th className="num">Import share</th></tr></thead>
              <tbody>
                {REGION_ORDER.flatMap((region) =>
                  meritStates
                    .filter((s) => s.grid_region === region)
                    .sort((a, b) => (b.demand_mw || 0) - (a.demand_mw || 0))
                    .map((s) => (
                      <tr key={s.code}>
                        <td><b>{s.name}</b> <span className="mono" style={{ color: "var(--muted)", fontSize: 11 }}>{s.code}</span></td>
                        <td style={{ color: "var(--muted)" }}>{region}</td>
                        <td className="num"><b>{s.demand_mw?.toLocaleString("en-IN") ?? "—"}</b></td>
                        <td className="num">{s.own_gen_mw?.toLocaleString("en-IN") ?? "—"}</td>
                        <td className="num">{s.import_mw?.toLocaleString("en-IN") ?? "—"}</td>
                        <td className="num">{s.demand_mw && s.import_mw != null
                          ? `${Math.round((100 * s.import_mw) / s.demand_mw)}%` : "—"}</td>
                      </tr>
                    )))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <StressPanel st={stress} />

      <CoveragePanel cov={meta?.metrics?.collection} />

      <h2 className="section-title">Deep adapters — beyond MERIT</h2>
      <div className="grid cols-2">
        <Card title="Gujarat SLDC — direct" info="SLDC, DAM"
          sub="sldcguj.com server-renders live values on its public homepage — scraped directly, no login. Cross-checks MERIT within ~0.5%.">
          {gujDirect && !gujDirect.error ? (
            <div className="grid cols-3">
              <Stat label="Gujarat catered" value={gujDirect.demand_mw?.toLocaleString("en-IN")} unit="MW" />
              <Stat label="Frequency" value={gujDirect.frequency_hz} unit="Hz" />
              <Stat label="DAM rate" value={gujDirect.dam_rate_rs_unit} unit="₹/unit" />
            </div>
          ) : (
            <div className="note">direct fetch unavailable: {gujDirect?.error || "no data yet"} — MERIT still covers Gujarat</div>
          )}
        </Card>
        <Card title="Rajasthan SLDC — direct" info="DSM, QCA, GSS"
          sub="Endpoints fully reverse-engineered: a dynamic-data JSON (frequency, DSM rate in paise/unit, load, generation) and a ~151-plant RE injection table by substation.">
          {rajDirect && !rajDirect.error ? (
            <div className="grid cols-3">
              <Stat label="Rajasthan load" value={rajDirect.load_mw?.toLocaleString("en-IN")} unit="MW" />
              <Stat label="Frequency" value={rajDirect.frequency_hz} unit="Hz" />
              <Stat label="DSM rate" value={rajDirect.dsm_rate_paise_unit} unit="paise/unit" />
            </div>
          ) : (
            <div className="note">
              {rajDirect?.note || "endpoint mapped; currently down upstream"} <br />
              <span style={{ color: "var(--muted)", fontSize: 12 }}>
                (their own homepage widget is equally broken — the adapter reports
                its health instead of pretending; MERIT covers Rajasthan demand meanwhile)
              </span>
            </div>
          )}
        </Card>
      </div>

      {(reg?.import_dependence?.rows?.length > 0) && (
        <Card title="Who needs forecasts most — import dependence" info="FaaS, DAM" style={{ marginTop: 14 }}
          sub={`States that import most of their power live at the mercy of purchase-cost volatility — the ideal Forecast-as-a-Service customers. ${reg.import_dependence.note}.`}>
          <HBar height={320} valLabel="Import share (%)"
            data={[...reg.import_dependence.rows].sort((a, b) => a.import_share_pct - b.import_share_pct)
              .map((r) => ({ name: r.name, value: r.import_share_pct }))} />
          <div className="scroll-x">
            <table className="data">
              <thead><tr><th>State</th><th className="num">Demand (MW)</th><th className="num">Import (MW)</th><th className="num">Import share</th><th className="num">Exposure proxy (₹/day)</th></tr></thead>
              <tbody>
                {reg.import_dependence.rows.slice(0, 10).map((r) => (
                  <tr key={r.code}>
                    <td><b>{r.name}</b></td>
                    <td className="num">{r.demand_mw?.toLocaleString("en-IN")}</td>
                    <td className="num">{r.import_mw?.toLocaleString("en-IN")}</td>
                    <td className="num"><b>{r.import_share_pct}%</b></td>
                    <td className="num">{r.exposure_rs_day ? `₹${(r.exposure_rs_day / 1e7).toFixed(1)} Cr` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {(reg?.forecast_readiness?.rows?.length > 0) && (
        <Card title="Per-state forecast readiness" info="MAPE, LGBM" style={{ marginTop: 14 }}
          sub={`The Delhi recipe (${reg.forecast_readiness.recipe_proof}) replicates per state as history accrues — the states poller samples all 23 every 15 min. Training gate: ${reg.forecast_readiness.gate}; no toy models below it.`}>
          <div className="scroll-x">
            <table className="data">
              <thead><tr><th>State</th><th className="num">Samples</th><th className="num">Days held</th><th className="num">Coverage</th><th>Status</th><th>Ready by</th></tr></thead>
              <tbody>
                {reg.forecast_readiness.rows.map((r) => (
                  <tr key={r.code}>
                    <td><b>{r.name}</b></td>
                    <td className="num">{r.samples?.toLocaleString("en-IN")}</td>
                    <td className="num">{r.days}</td>
                    <td className="num">{r.coverage_pct}%</td>
                    <td><span className={`pill ${r.status === "training-ready" ? "verified" : "identified"}`}>{r.status}</span></td>
                    <td>{r.ready_eta || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {nrStates.length > 0 && (
        <Card title="Northern Region detail — schedule vs drawal" info="OD/UD, SLDC" style={{ marginTop: 14 }}
          sub="Published on Delhi SLDC's real-time page; the schedule/drawal gap is exactly what the DSM prices. Negative values (Himachal) = net hydro exporter.">
          <HBar height={300} valLabel="Load (MW)"
            data={nrStates.map((s) => ({ name: s.state, value: s.load_mw })).sort((a, b) => a.value - b.value)} />
          <div className="scroll-x">
            <table className="data">
              <thead><tr><th>State</th><th className="num">Schedule</th><th className="num">Drawl</th><th className="num">OD/UD <InfoTip terms="OD/UD" /></th><th className="num">Load (MW)</th></tr></thead>
              <tbody>
                {[...nrStates].sort((a, b) => b.load_mw - a.load_mw).map((s) => (
                  <tr key={s.state}>
                    <td><b>{s.state}</b></td>
                    <td className="num">{s.schedule_mw?.toLocaleString("en-IN")}</td>
                    <td className="num">{s.drawl_mw?.toLocaleString("en-IN")}</td>
                    <td className="num">{s.od_ud_mw?.toLocaleString("en-IN")}</td>
                    <td className="num"><b>{s.load_mw?.toLocaleString("en-IN")}</b></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <h2 className="section-title">State adapter registry</h2>
      <Card sub="Adding a state = one adapter + one registry entry, validated against a live fetch before being trusted. The pattern is the product; Delhi is the reference implementation. Sources that resisted (Maharashtra's image-only SCADA, WRLDC's session-gated API) are recorded as honestly as the ones that worked.">
        <div className="scroll-x">
          <table className="data">
            <thead><tr><th>Code</th><th>State</th><th>Region</th><th className="num">Peak load</th><th>Status</th><th>Sources & notes</th></tr></thead>
            <tbody>
              {registry.map((s) => (
                <tr key={s.code}>
                  <td className="mono">{s.code}</td>
                  <td><b>{s.name}</b></td>
                  <td>{s.grid_region}</td>
                  <td className="num">{s.peak_load_gw ? `${s.peak_load_gw.toFixed(1)} GW` : "—"}</td>
                  <td><span className={`pill ${s.status}`}>{s.status}</span></td>
                  <td style={{ fontSize: 12, color: "var(--muted)", maxWidth: 420 }}>{s.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}

/* Collection coverage. The multi-state layer rests entirely on accrual — no
   upstream source has a history endpoint — so a gap is permanently
   unrecoverable and a SYSTEMATIC gap is worse than a random one. Row count
   hides that; coverage by hour does not. */
function CoveragePanel({ cov }) {
  if (!cov || cov.error) return null;
  const dead = cov.hours_with_no_data || [];
  const thin = cov.hours_under_half_covered || [];
  const byHour = cov.coverage_by_hour_pct || {};
  const bars = Array.from({ length: 24 }, (_, h) => ({
    name: `${String(h).padStart(2, "0")}`,
    value: Math.min(Math.round(byHour[h] ?? 0), 200),
  }));
  const bad = dead.length > 0;
  return (
    <>
      <h2 className="section-title">Collection coverage — can this data be trusted?</h2>
      <div className={`note ${bad ? "crit" : "info"}`}>
        <b>{bad ? "⚠ There are hours we never collect." : "Coverage is even across the day."}</b>{" "}
        {cov.snapshots?.toLocaleString("en-IN")} snapshots over {cov.span_hours} h
        against {cov.expected_at_15min?.toLocaleString("en-IN")} expected at a
        15-minute cadence.
        {dead.length > 0 && (
          <> <b>No data at all</b> in the {dead.map((h) => `${String(h).padStart(2,"0")}:00`).join(", ")} hours
          {dead.includes(5) || dead.includes(6) || dead.includes(7)
            ? " — which is the morning ramp, one of the most dynamic periods in the system." : "."}</>
        )}
        {thin.length > 0 && <> Under half covered at {thin.map((h) => `${String(h).padStart(2,"0")}:00`).join(", ")}.</>}
        <div style={{ marginTop: 6 }}>
          {cov.hours_lost_to_gaps} h lost to gaps, worst single gap {cov.worst_gap_hours} h.
          Every source here is snapshot-only, so a missed block cannot be
          backfilled — which is why we publish this instead of a row count. An
          intraday model trained on a biased sample would inherit the bias
          silently.
        </div>
      </div>
      <Card title="Snapshots captured per hour of day" info="15-min target"
        sub="100% = the full 4 snapshots an hour. Above 100% means the collector ran more often than scheduled; zero means the machine was asleep.">
        <HBar data={bars} height={300} valLabel="% of target cadence" />
      </Card>
    </>
  );
}

/* State Grid Stress. Competitors sell forecasts; this sells RISK, and it is
   only possible because we hold coal, outages, demand and RE at state level in
   one place. The index answers "which state has to buy, and how exposed is it
   when it does" — the question that decides who needs us. */
function StressPanel({ st }) {
  if (!st) return null;
  if (st.error) return <Card title="State Grid Stress"><div className="note">{st.error}</div></Card>;
  const rows = (st.states || []).filter((r) => r.stress !== null && r.stress !== undefined);
  if (!rows.length) return null;
  const bandColor = (b) => (b === "very tight" ? "var(--critical)"
    : b === "tight" ? "var(--s6)" : b === "normal" ? "var(--s1)" : "var(--muted)");
  const top = rows[0];

  return (
    <>
      <h2 className="section-title">
        State Grid Stress — who has to buy, and how exposed they are
        <InfoTip text="Built from four state-attributed datasets we hold in one place: import dependence (MERIT, 15-min), coal days-of-stock and unit outages (CEA, both backfilled over a year), and RE generation. It measures market EXPOSURE, not a forecast." />
      </h2>

      <div className="note info">
        <b>A forecast says how much a state will draw. This says whether it can cover it.</b>{" "}
        Import dependence is the base — a state buying 80% of its power is exposed
        to every price spike, one buying 10% is not — and thin coal or units out
        make the same dependence riskier. Right now the tightest is{" "}
        <b>{top.name}</b>: {top.why}.
        <div style={{ marginTop: 6, color: "var(--muted)" }}>
          Demand as of {st.demand_asof} · coal {st.coal_day} · outages {st.outage_day}
        </div>
      </div>

      <div className="scroll-x">
        <table className="data">
          <thead><tr>
            <th>State</th><th className="num">Stress</th><th>Band</th>
            <th className="num">Imports</th><th className="num">MW exposed</th>
            <th className="num">Coal days</th><th className="num">Own fleet out</th>
            <th>Why</th>
          </tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.code}>
                <td><b>{r.name}</b></td>
                <td className="num"><b style={{ color: bandColor(r.band) }}>{r.stress}</b></td>
                <td><span className="pill" style={{ background: "var(--wash)", color: bandColor(r.band) }}>{r.band}</span></td>
                <td className="num">{r.import_dependence_pct != null ? `${r.import_dependence_pct}%` : "—"}</td>
                <td className="num">{r.exposed_mw != null ? Math.round(r.exposed_mw).toLocaleString("en-IN") : "—"}</td>
                <td className="num">{r.days_of_stock != null ? r.days_of_stock.toFixed(1) : "—"}</td>
                <td className="num">{r.outage_rate_pct != null ? `${r.outage_rate_pct}%` : "—"}</td>
                <td style={{ fontSize: 12, color: "var(--muted)" }}>{r.why}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="note crit" style={{ marginTop: 12 }}>
        <b>Read the limits, they are real.</b> The index is validated at NATIONAL
        level, where we have price history: the tightest quintile saw 28.1% of
        blocks pin at the price cap against 6.1% in the loosest, monotonic across
        all five bands. <b>Per state it is not yet backtested</b> — state-level
        prices only became available to us on 3 Aug via the area-price feed, which
        has no history endpoint, so that validation is still accruing.
        <div style={{ marginTop: 6 }}>
          Coal and outage coverage is also partial, and for a structural reason:
          CEA groups its first column by OWNER, not state — "IPP" (73 GW) and
          "NTPC" (55 GW) are the largest entries — so only plants listed under a
          state's own name are attributed to it. A plant-to-state lookup would
          close that; CEA publishes none, so it is shown as missing rather than
          guessed.
        </div>
      </div>
    </>
  );
}
