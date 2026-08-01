/* Sizing & Bankability — the first question every customer asks
   ("what size battery, and will it pay back?") answered interactively
   from a year of real IEX prices. All revenue math is precomputed
   per-MW in Python (models/sizing.py); this page only scales it. */
import { useState } from "react";

import { TimeSeries } from "../components/charts";
import { Card, InfoTip, Loading, PageHeader, Stat } from "../components/ui";
import { fmtINR, useApi } from "../lib/api";

const fmtCr = (rs) => `₹${(rs / 1e7).toFixed(2)} Cr`;

export default function Sizing() {
  const { data, error } = useApi("/api/modules", { refreshMs: 600_000 });
  const [mw, setMw] = useState(20);
  const [dur, setDur] = useState("2h");
  const [capexCr, setCapexCr] = useState(1.5); // ₹ Cr per MWh

  if (!data) return <Loading error={error} />;
  const s = data.sizing;
  if (!s || s.error) return <Card title="Sizing & Bankability" sub={s?.error || "run models/sizing.py to precompute"} />;

  const b = s.bankability?.[dur];
  const hours = parseFloat(dur);
  const energyMwh = mw * hours;
  const capexRs = energyMwh * capexCr * 1e7;
  const p50 = (b?.annual_p50_rs_mw ?? 0) * mw;
  const p75 = (b?.annual_p75_rs_mw ?? 0) * mw;
  const p90 = (b?.annual_p90_rs_mw ?? 0) * mw;
  const paybackP50 = p50 > 0 ? capexRs / p50 : null;
  const paybackP90 = p90 > 0 ? capexRs / p90 : null;
  const monthly = (s.monthly?.[dur] || []).map((m) => ({
    month: m.month, rs_day: Math.round(m.rs_mw_day * mw),
  }));

  return (
    <div>
      <PageHeader eyebrow="Sizing & Bankability"
        title="What size battery — and will it pay back?"
        lead={`Revenue from a dispatch LP on ${s.window?.n_days} days of actual IEX DAM prices,
               scaled by FlexTrade's measured ${Math.round(s.capture_ratio * 1000) / 10}% capture ratio,
               with physics-calibrated degradation inside the optimization. P50/P90 from a
               10,000-draw bootstrap — the quantiles a lender underwrites.`} />

      <Card title="Configure the asset" info="BESS, MWh">
        <div className="grid4">
          <label style={{ display: "block" }}>
            <div className="sub">Power (MW)</div>
            <input type="range" min="1" max="200" value={mw}
              onChange={(e) => setMw(+e.target.value)} style={{ width: "100%" }} />
            <b>{mw} MW</b>
          </label>
          <label style={{ display: "block" }}>
            <div className="sub">Duration</div>
            <select value={dur} onChange={(e) => setDur(e.target.value)}>
              {Object.keys(s.bankability || {}).map((d) => (
                <option key={d} value={d}>{d} ({mw * parseFloat(d)} MWh)</option>
              ))}
            </select>
          </label>
          <label style={{ display: "block" }}>
            <div className="sub">Capex (₹ Cr / MWh)</div>
            <input type="number" min="0.5" max="5" step="0.1" value={capexCr}
              onChange={(e) => setCapexCr(+e.target.value)} style={{ width: 90 }} />
            <div className="sub">tender range ₹1.3–1.8</div>
          </label>
          <div>
            <div className="sub">Total capex</div>
            <b style={{ fontSize: 22 }}>{fmtCr(capexRs)}</b>
            <div className="sub">{energyMwh} MWh installed</div>
          </div>
        </div>
      </Card>

      <div className="grid4">
        <Stat label="P50 annual revenue" info="P50, DAM"
          value={fmtINR(p50, { compact: true })} unit="/yr"
          hint={`₹${(b?.annual_p50_rs_mw / 1e5).toFixed(1)} L per MW`} />
        <Stat label="P90 annual revenue" info="P90"
          value={fmtINR(p90, { compact: true })} unit="/yr"
          hint="the lender's number — exceeded 90% of bootstrap years" />
        <Stat label="Payback (P50 / P90)"
          value={paybackP50 ? `${paybackP50.toFixed(1)} / ${paybackP90.toFixed(1)}` : "—"}
          unit="yrs" hint="DAM arbitrage alone — see upside note below" />
        <Stat label="Cycling" info="FCE, DoD"
          value={b?.fce_per_day ?? "—"} unit="cycles/day"
          hint={`${(b?.throughput_mwh_day_mw * mw).toFixed(0)} MWh/day throughput`} />
      </div>

      <Card title={`Seasonality — daily revenue by month (${mw} MW, ${dur})`}
        sub="mean achievable ₹/day per calendar month of the window — summer evenings carry the year">
        <TimeSeries data={monthly} xKey="month" yLabel="₹/day"
          series={[{ key: "rs_day", name: "mean daily revenue", color: "var(--s1)", type: "bar" }]} />
      </Card>

      <Card title="Why this is conservative — the stacked upside"
        sub="Everything below is additive revenue this platform already computes but does NOT count above.">
        <table className="data"><tbody>
          <tr><td style={{ width: 260 }}><b>Intraday RTM re-optimization</b></td>
            <td style={{ color: "var(--muted)" }}>second settlement of the same flexibility — see Operations (live daily uplift)</td></tr>
          <tr><td><b>DSM channel optimization</b></td>
            <td style={{ color: "var(--muted)" }}>deviation settled at Normal Rate when it beats RTM, inside the compliance band — see Operations</td></tr>
          <tr><td><b>Ancillary services (SRAS/TRAS)</b></td>
            <td style={{ color: "var(--muted)" }}>market operationalizing; our frequency-response readiness report quantifies the duty cycle today</td></tr>
          <tr><td><b>C&amp;I demand-charge stacking</b></td>
            <td style={{ color: "var(--muted)" }}>behind-the-meter configurations add ToD + demand-charge savings</td></tr>
        </tbody></table>
      </Card>

      <Card title="Read before quoting" info="P90, FaaS">
        <ul>
          {(s.caveats || []).map((c, i) => <li key={i}>{c}</li>)}
        </ul>
      </Card>
    </div>
  );
}
