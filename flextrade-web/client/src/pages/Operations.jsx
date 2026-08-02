/* Operations — the three modules that run ON TOP of the day-ahead plan:
   intraday RTM re-optimization, physics-based degradation, C&I peak
   shaving. Every number here comes from /api/modules (Python-computed). */
import { TimeSeries } from "../components/charts";
import { Card, Loading, PageHeader, Stat, Tabs } from "../components/ui";
import { fmtINR, useApi } from "../lib/api";

const hhmm = (ts) => String(ts).slice(11, 16);

function RtmSection({ rtm }) {
  if (!rtm || rtm.error) {
    return <Card title="Intraday RTM re-optimization"
      sub={rtm?.error ? `unavailable: ${rtm.error}` : "no data"} />;
  }
  if (rtm.status !== "ok") {
    return <Card title="Intraday RTM re-optimization"
      sub={`${rtm.status} — as of ${rtm.asof}. The engine re-opens with tomorrow's delivery day.`} />;
  }
  const sched = (rtm.schedule || []).map((r) => ({ ...r, t: hhmm(r.ts) }));
  const trades = sched.filter((r) => r.side !== "-");
  return (
    <>
      <h2>Intraday RTM re-optimization</h2>
      <p className="sub" style={{ marginTop: -6 }}>
        After the DAM clears, the position is financially firm — but the battery still has
        physical flexibility around it. Every 30 minutes IEX's Real-Time Market lets us trade
        that flexibility: the LP below re-optimizes the remaining blocks of{" "}
        <b>{rtm.delivery_day}</b> (as of {rtm.asof}), starting from the SoC implied by the
        committed plan, and must hand tomorrow the planned end-of-day position. Basis:{" "}
        {rtm.dam_basis}. Price expectation: {rtm.blocks_actual_rtm} blocks at actual cleared
        RTM prices, {rtm.blocks_projected} projected from today's DAM curve scaled by an{" "}
        <b>hour-of-day</b> RTM/DAM ratio ({rtm.ratio_basis}).
      </p>
      {rtm.ratio_dispersion?.n_blocks && (
        <div className="note" style={{ marginBottom: 12 }}>
          <b>How firm is that projection?</b> Measured over{" "}
          {rtm.ratio_dispersion.n_blocks.toLocaleString("en-IN")} paired blocks, the
          RTM/DAM ratio has a median of {rtm.ratio_dispersion.median} but a 5th–95th
          percentile of <b>{rtm.ratio_dispersion.p05}–{rtm.ratio_dispersion.p95}</b>, and its
          hourly median swings {rtm.ratio_dispersion.hour_min}–{rtm.ratio_dispersion.hour_max}.
          RTM clears above DAM in ~43% of blocks, so this is a calibrated
          <i> expectation</i>, not a forecast — an RTM price model is the next build,
          and the uplift below should be read with that spread in mind.
        </div>
      )}
      <div className="grid4">
        <Stat label="Expected RTM uplift" info="RTM, P_L" value={fmtINR(rtm.expected_rtm_uplift_rs)}
          hint="incremental, on top of DAM revenue" />
        <Stat label="Tradeable blocks left" info="RTM" value={rtm.tradeable_blocks}
          hint="≥60 min out (RTM gate closure)" />
        <Stat label="RTM trades proposed" value={rtm.n_trades} />
        <Stat label="SoC now" info="SoC" value={rtm.soc_now_mwh} unit="MWh"
          hint="implied by executing the DAM plan so far" />
      </div>
      <Card title="Expected RTM price — rest of day"
        sub="actual cleared blocks where available, ratio-projected otherwise">
        <TimeSeries data={sched} xKey="t" yLabel="₹/MWh"
          series={[{ key: "rtm_price", name: "expected RTM MCP", color: "var(--s1)" }]} />
      </Card>
      <Card title="Proposed RTM trades vs DAM position"
        sub="positive = extra sell into RTM, negative = buy; the DAM line is the committed position">
        <TimeSeries data={sched} xKey="t" yLabel="MW"
          series={[
            { key: "rtm_trade_mw", name: "RTM trade", color: "var(--s2)", type: "bar" },
            { key: "dam_net_mw", name: "DAM position", color: "var(--s3)", dash: "5 3" },
          ]} />
      </Card>
      {trades.length > 0 && (
        <Card title="RTM order ticket" sub="what would be submitted at the next gate">
          <div style={{ overflowX: "auto" }}>
            <table className="data">
              <thead><tr><th>Block</th><th>Side</th><th>MW</th><th>Exp. price ₹/MWh</th><th>SoC after (MWh)</th></tr></thead>
              <tbody>
                {trades.slice(0, 14).map((r) => (
                  <tr key={r.ts}>
                    <td>{r.t}</td>
                    <td><b style={{ color: r.side === "SELL" ? "var(--good)" : "var(--s1)" }}>{r.side}</b></td>
                    <td>{Math.abs(r.rtm_trade_mw).toFixed(1)}</td>
                    <td>{Math.round(r.rtm_price).toLocaleString("en-IN")}</td>
                    <td>{r.soc_mwh}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {trades.length > 14 ? <div className="sub">…{trades.length - 14} more</div> : null}
          </div>
        </Card>
      )}
    </>
  );
}

function DegradationSection({ deg }) {
  if (!deg || deg.error) {
    return <Card title="Physics-based degradation" sub={deg?.error || "no data"} />;
  }
  const sched = (deg.schedule || []).map((r) => ({ ...r, t: hhmm(r.ts) }));
  const uplift = deg.net_rs - (deg.proxy200?.net_rs ?? deg.net_rs);
  return (
    <>
      <h2>Physics-based degradation</h2>
      <p className="sub" style={{ marginTop: -6 }}>
        Cycle life follows a Wöhler power law — L(d) = L₁₀₀·d<sup>−k</sup> — so deep cycles
        consume more life per MWh than shallow ones. We rainflow-count the SoC trajectory
        (ASTM E1049) and price each cycle from LFP parameters (L₁₀₀ = {deg.params?.cycle_life_100},
        k = {deg.params?.kp}, capex ₹{(deg.params?.capex_rs_per_mwh / 1e7).toFixed(1)} Cr/MWh,
        {" "}{Math.round((deg.params?.cycle_share ?? 0) * 100)}% attributed to cycling). The DoD-dependent
        cost is nonconvex, so the LP iterates a fixed point on a calibrated flat rate.
        Shown for {deg.day} actual DAM prices.
      </p>
      <div className="grid4">
        <Stat label="True cycling cost" info="DoD, LFP, FCE" value={`₹${Math.round(deg.converged_rate_rs_mwh).toLocaleString("en-IN")}`}
          unit="/MWh" hint="converged rate — vs the old ₹200 proxy" />
        <Stat label="Physics cost of the day" info="FCE" value={fmtINR(deg.physics_cost_rs)}
          hint={`${deg.full_cycle_equivalents} full-cycle equivalents`} />
        <Stat label="Net P&L (physics-aware)" info="P_L" value={fmtINR(deg.net_rs)}
          hint="gross arbitrage − true degradation" />
        <Stat label="Uplift vs ₹200 proxy" value={fmtINR(uplift)}
          hint={`proxy schedule cycled ${deg.proxy200?.fce} FCE`} />
      </div>
      <div className="grid2">
        <Card title="Marginal cost vs depth of discharge"
          sub="₹ per discharged MWh — why the optimizer should prefer shallower cycles">
          <TimeSeries data={deg.marginal_curve || []} xKey="dod_pct" yLabel="₹/MWh"
            series={[{ key: "rs_per_mwh", name: "marginal cost", color: "var(--s4)", type: "bar" }]} />
        </Card>
        <Card title="Physics-aware schedule" sub={`SoC trajectory on ${deg.day} — the path that gets rainflow-counted`}>
          <TimeSeries data={sched} xKey="t" yLabel="MWh / MW"
            series={[
              { key: "soc_mwh", name: "SoC (MWh)", color: "var(--s1)", type: "area" },
              { key: "bess_mw", name: "net dispatch (MW)", color: "var(--s2)" },
            ]} />
        </Card>
      </div>
      <Card title="Fixed-point convergence" sub="LP flat rate ← rainflow physics cost, until stable">
        <table className="data">
          <thead><tr><th>Flat rate ₹/MWh</th><th>Gross ₹</th><th>Physics cost ₹</th><th>Net ₹</th><th>FCE</th></tr></thead>
          <tbody>
            {(deg.iterations || []).map((it, i) => (
              <tr key={i}>
                <td>{Math.round(it.rate_rs_mwh).toLocaleString("en-IN")}</td>
                <td>{Math.round(it.gross_rs).toLocaleString("en-IN")}</td>
                <td>{Math.round(it.physics_cost_rs).toLocaleString("en-IN")}</td>
                <td>{Math.round(it.net_rs).toLocaleString("en-IN")}</td>
                <td>{it.fce}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </>
  );
}

function CniSection({ cni }) {
  if (!cni || cni.error) {
    return <Card title="C&I peak shaving" sub={cni?.error || "no data"} />;
  }
  const sched = (cni.schedule || []).map((r) => ({ ...r, t: hhmm(r.ts) }));
  const b = cni.baseline, o = cni.optimized;
  return (
    <>
      <h2>C&amp;I peak shaving</h2>
      <p className="sub" style={{ marginTop: -6 }}>
        A behind-the-meter battery ({cni.bess?.power_mw} MW / {cni.bess?.energy_mwh} MWh)
        attacks three lines of a Delhi C&amp;I bill: the demand charge on billed peak,
        the ToD peak surcharge (14:00–17:00 &amp; 22:00–01:00, +20%), and the off-peak
        rebate (04:00–10:00, −20%). <b>Profile: {cni.profile_note}.</b>{" "}
        <b>Tariff: {cni.tariff_note}.</b>
      </p>
      <div className="grid4">
        <Stat label="Net saving" info="ToD, C&I" value={fmtINR(cni.saving_rs_day)} unit="/day"
          hint={`${cni.saving_pct}% of bill · ≈ ${fmtINR(cni.saving_rs_day * 365)}/yr`} />
        <Stat label="Peak cut" value={cni.peak_cut_mw} unit="MW"
          hint={`${b?.billed_peak_mw} → ${o?.billed_peak_mw} MW billed`} />
        <Stat label="Demand-charge saving" infoText="Demand charge — C&I consumers pay for their PEAK demand (Rs/kVA/month) on top of energy. Cutting the peak with a battery cuts this line directly." value={fmtINR((b?.demand_rs ?? 0) - (o?.demand_rs ?? 0))}
          unit="/day" hint="₹250/kVA/month, daily share" />
        <Stat label="Degradation cost" value={fmtINR(cni.degradation_rs)} unit="/day"
          hint={deg?.converged_rate_rs_mwh
            ? `physics-calibrated ₹${Math.round(deg.converged_rate_rs_mwh).toLocaleString("en-IN")}/MWh — already netted out`
            : "physics-calibrated rate — already netted out"} />
      </div>
      <Card title="Factory load vs grid draw"
        sub="the gap is the battery: charging in the off-peak rebate window, discharging into the process peak">
        <TimeSeries data={sched} xKey="t" yLabel="MW"
          series={[
            { key: "load_mw", name: "factory load", color: "var(--s3)", type: "area", fillOpacity: 0.08 },
            { key: "grid_mw", name: "grid draw (billed)", color: "var(--s1)" },
            { key: "bess_mw", name: "battery (+ = discharge)", color: "var(--s2)", dash: "4 3" },
          ]} />
      </Card>
      <Card title="ToD price signal the battery follows" sub="₹/MWh effective energy rate by block">
        <TimeSeries data={sched} xKey="t" yLabel="₹/MWh" height={180}
          series={[{ key: "tod_rate_rs_mwh", name: "ToD rate", color: "var(--s4)" }]} />
      </Card>
    </>
  );
}

function ThreeWaySection({ tw }) {
  if (!tw || tw.error) {
    return <Card title="Three-way DAM + RTM + DSM co-optimization" sub={tw?.error || "no data"} />;
  }
  if (tw.status !== "ok") {
    return <Card title="Three-way DAM + RTM + DSM co-optimization"
      sub={`${tw.status} — reopens with tomorrow's delivery day`} />;
  }
  const sched = (tw.schedule || []).map((r) => ({ ...r, t: hhmm(r.ts) }));
  const used = sched.filter((r) => r.dsm_over_mw > 0.01 || r.dsm_under_mw > 0.01);
  return (
    <>
      <h2>Three-way DAM + RTM + DSM co-optimization</h2>
      <p className="sub" style={{ marginTop: -6 }}>
        Once the DAM position is firm, a deviation from it can be closed <b>two</b> ways:
        trade it in the RTM, or settle it through the DSM at the Normal Rate (⅓ DAM + ⅓ RTM
        + ⅓ ancillary). These prices invert across the day, so the deviation itself becomes a
        priced decision. To our knowledge no Indian platform treats DSM as a third settlement
        channel. <b>{tw.lp_rates_note}.</b> {tw.ancillary_note}.
      </p>
      <div className="grid4">
        <Stat label="RTM-only uplift" info="RTM" value={fmtINR(tw.rtm_only_uplift_rs)}
          hint="single-channel intraday re-optimization" />
        <Stat label="Three-way uplift" info="DSM, NR" value={fmtINR(tw.threeway_uplift_rs)}
          hint="RTM + DSM channels co-optimized" />
        <Stat label="DSM channel adds" value={fmtINR(tw.dsm_leg_added_rs)}
          hint={`across ${tw.dsm_blocks_used} blocks, within the compliance band`} />
        <Stat label="Exact-engine check" infoText="The chosen deviation profile is re-settled by the same versioned CERC engine (models/dsm.py) used on the Renewables page — an independent verification of the LP's DSM leg." value={fmtINR(tw.exact_engine_dsm_rs)}
          hint="re-settled by the versioned CERC engine" />
      </div>
      <Card title="Where the deviation settles"
        sub="RTM trade (traded) vs DSM deviation (settled at Normal Rate), both inside the ±10% compliance band">
        <TimeSeries data={sched} xKey="t" yLabel="MW"
          series={[
            { key: "rtm_trade_mw", name: "RTM trade", color: "var(--s2)", type: "bar" },
            { key: "dsm_under_mw", name: "DSM shortfall", color: "var(--s4)", type: "bar" },
            { key: "dsm_over_mw", name: "DSM over-inject", color: "var(--s3)", type: "bar" },
          ]} />
      </Card>
      <Card title="Price channels — RTM vs Normal Rate"
        sub="the two settle-prices the LP arbitrages; they cross on spike evenings">
        <TimeSeries data={sched} xKey="t" yLabel="₹/MWh" height={200}
          series={[
            { key: "rtm_price", name: "RTM MCP", color: "var(--s1)" },
            { key: "normal_rate", name: "DSM Normal Rate", color: "var(--s5)", dash: "5 3" },
          ]} />
      </Card>
    </>
  );
}

function WarrantySection({ w }) {
  if (!w || w.error) return <Card title="Warranty & Availability Guard" sub={w?.error || "no data"} />;
  const days = (w.days || []).map((d) => ({ ...d, t: d.day.slice(5) }));
  const sm = w.summary || {};
  const chk = w.plan_check;
  const t = w.terms || {};
  return (
    <>
      <h2>Warranty &amp; Availability Guard</h2>
      <p className="sub" style={{ marginTop: -6 }}>
        Aggressive arbitrage can quietly void a battery warranty — max cycles/day, an approved
        SoC window, throughput caps. This audits both the <b>real BRPL Kilokari telemetry</b> and
        our own next-day plan against the warranty envelope, so violations surface before a claim,
        not at one. Terms: ≤ {t.max_cycles_per_day} cycles/day, SoC {t.soc_min_pct}–{t.soc_max_pct}%
        ({t.note}).
      </p>
      <div className="grid4">
        <Stat label="Days observed" info="BESS" value={sm.days_observed ?? "—"}
          hint={`${sm.days_with_reliable_cycle_count ?? 0} with full-coverage cycle counts`} />
        <Stat label="Mean cycles/day" info="FCE" value={sm.mean_fce_reliable_days ?? "—"}
          hint={`limit ${t.max_cycles_per_day}/day — reliable-coverage days only`} />
        <Stat label="Observed SoC range" value={`${sm.worst_soc_min_pct ?? "—"}–${sm.worst_soc_max_pct ?? "—"}%`}
          hint={`warranty window ${t.soc_min_pct}–${t.soc_max_pct}%`} />
        <Stat label="Total violations" value={sm.total_violations ?? "—"}
          hint="across all observed telemetry days" />
      </div>
      {chk && (
        <Card title="Pre-trade check — tomorrow's plan"
          sub="the compliance gate our own schedule must pass before it is bid">
          <div className={`note ${chk.compliant ? "info" : ""}`}>
            <b>{chk.compliant ? "✓ COMPLIANT" : "⚠ VIOLATIONS"}</b> — planned {chk.fce} cycles,
            SoC {chk.soc_min_pct}–{chk.soc_max_pct}%.
            {chk.violations?.length ? <> {chk.violations.join("; ")}. The warranty-safe LP can
              re-solve with cycle/SoC caps as hard constraints.</> : " Within the warranty envelope."}
          </div>
        </Card>
      )}
      {days.length > 0 && (
        <Card title="Daily cycling (real telemetry)"
          sub={`full-cycle equivalents per day, rainflow-counted. ${sm.telemetry_note}`}>
          <TimeSeries data={days} xKey="t" yLabel="FCE"
            series={[{ key: "fce", name: "cycles/day", color: "var(--s1)", type: "bar" }]}
            refLines={[{ y: t.max_cycles_per_day, label: "warranty limit", color: "var(--critical)" }]} />
        </Card>
      )}
    </>
  );
}

function ThermalSection({ th }) {
  if (!th || th.error) return <Card title="Thermal derating" sub={th?.error || "no data"} />;
  return (
    <>
      <h2>Thermal derating — what Delhi's heat costs</h2>
      <p className="sub" style={{ marginTop: -6 }}>
        Grid batteries derate above ~35 °C and their HVAC is parasitic load that peaks exactly
        when prices do. Applied to the committed plan for {th.day} using stored temperature.
        {" "}<b>{th.assumptions}.</b>
      </p>
      <div className="grid4">
        <Stat label="Peak temperature" value={th.temp_max_c} unit="°C"
          hint={`${th.hours_above_35c} h above 35 °C`} />
        <Stat label="Worst derate factor" value={`×${th.min_derate_factor}`}
          hint="available power at the hottest block" />
        <Stat label="Auxiliary (HVAC) load" value={th.aux_mwh} unit="MWh"
          hint={`costs ${fmtINR(th.aux_cost_rs)} at market price`} />
        <Stat label="Heat cost today" info="P_L" value={fmtINR(th.heat_cost_rs)}
          hint={`${fmtINR(th.revenue_ideal_rs)} → ${fmtINR(th.revenue_thermal_rs)}`} />
      </div>
    </>
  );
}

function FreqSection({ fr }) {
  if (!fr || fr.error) return <Card title="Frequency-response readiness" sub={fr?.error || "no data"} />;
  return (
    <>
      <h2>Frequency-response readiness</h2>
      <p className="sub" style={{ marginTop: -6 }}>
        India's SRAS/TRAS ancillary markets are operationalizing and batteries are the ideal
        provider — but no public price feed exists yet. Using the grid-frequency history we
        <b> sample ourselves</b> ({fr.samples} readings over {fr.days_sampled} days; no public
        archive exists), we simulate a {fr.droop?.power_mw} MW droop-controlled battery and
        report the duty cycle it would have served. <b>{fr.note}.</b>
      </p>
      <div className="grid4">
        <Stat label="Grid frequency" info="Hz" value={fr.freq_mean_hz} unit="Hz mean"
          hint={`range ${fr.freq_min_hz}–${fr.freq_max_hz} Hz`} />
        <Stat label="Called on" value={`${fr.pct_samples_called}%`} unit="of samples"
          hint={`under-frequency ${fr.pct_under_frequency}% of the time`} />
        <Stat label="Mean response when called" value={fr.mean_response_when_called_mw} unit="MW"
          hint={`peak ${fr.max_response_mw} MW`} />
        <Stat label="Duty energy" value={fr.energy_mwh_per_day} unit="MWh/day"
          hint={`busiest hours ${fr.busiest_hours?.join(", ")}`} />
      </div>
    </>
  );
}

const TABS = [
  { id: "intraday", label: "Intraday Settlement", icon: "⇄", hint: "RTM · DSM" },
  { id: "asset", label: "Asset Protection", icon: "🛡", hint: "3" },
  { id: "revenue", label: "New Revenue", icon: "＋", hint: "C&I · FR" },
];

export default function Operations() {
  const { data, error } = useApi("/api/modules", { refreshMs: 300_000 });
  if (!data) return <Loading error={error} />;
  return (
    <div>
      <PageHeader eyebrow="Operations Suite"
        title="Beyond the day-ahead trade"
        lead="Everything that runs on top of the committed DAM plan — a second settlement in
              the RTM and DSM, the physics that protects the battery, and entirely new revenue
              streams from the same asset. Every number is Python-computed on live data." />
      <Tabs tabs={TABS}>
        {(active) => (
          <>
            {active === "intraday" && (
              <>
                <RtmSection rtm={data.rtm} />
                <hr className="hr-soft" />
                <ThreeWaySection tw={data.threeway} />
              </>
            )}
            {active === "asset" && (
              <>
                <WarrantySection w={data.warranty} />
                <hr className="hr-soft" />
                <DegradationSection deg={data.degradation} />
                <hr className="hr-soft" />
                <ThermalSection th={data.thermal} />
              </>
            )}
            {active === "revenue" && (
              <>
                <CniSection cni={data.cni} />
                <hr className="hr-soft" />
                <FreqSection fr={data.freq_response} />
              </>
            )}
          </>
        )}
      </Tabs>
    </div>
  );
}
