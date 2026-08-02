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
  const { data: bank } = useApi("/api/bankability");
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

      <FinanceSection bank={bank} />

      <Card title="Read before quoting" info="P90, FaaS">
        <ul>
          {(s.caveats || []).map((c, i) => <li key={i}>{c}</li>)}
        </ul>
      </Card>
    </div>
  );
}

/* Project finance. Sizing answers "how much revenue"; this answers the question
   a lender actually asks — whether the asset services its debt in the worst
   year — in DSCR, IRR and LCOS. */
function FinanceSection({ bank }) {
  if (!bank) return <Card><div className="note">Loading the finance model...</div></Card>;
  if (bank.error) return <Card><div className="note">{bank.error}</div></Card>;
  const a = bank.assumptions || {};
  const st = bank.stacked || {};
  const bd = st.revenue_breakdown || {};
  const cr = (v) => (v == null ? "-" : `\u20b9${(v / 1e7).toFixed(2)} Cr`);
  const cp = bank.capacity_payment_for_bankability_rs_per_mw_yr;

  return (
    <>
      <h2 className="section-title">Project finance - what a lender reads</h2>
      <div className="grid4">
        <Stat label="Equity IRR" value={bank.equity_irr_pct != null ? `${bank.equity_irr_pct}%` : "-"}
          hint="DAM arbitrage only, post-debt, post-tax"
          infoText="Return on the equity cheque after debt service and tax. This is what an investment committee decides on." />
        <Stat label="Minimum DSCR" value={bank.min_dscr ?? "-"}
          hint={`avg ${bank.avg_dscr ?? "-"} - covenant 1.20x`}
          infoText="Debt Service Coverage Ratio: cash available for debt service divided by debt service, in the WORST year. The minimum is the covenant, and lenders size debt to it." />
        <Stat label="LCOS" value={bank.lcos_rs_per_mwh ? `\u20b9${bank.lcos_rs_per_mwh.toLocaleString("en-IN")}` : "-"}
          unit="/MWh" hint="levelised cost of storage"
          infoText="PV of all lifetime costs divided by PV of energy discharged - the one number that compares a battery against any other source of flexibility." />
        <Stat label="Capacity fade" value={`${bank.annual_fade_pct}%`} unit="/yr"
          hint={`from ${bank.cycle_life_at_depth?.toLocaleString("en-IN")} cycles to 80% at depth ${a.mean_cycle_depth}`}
          infoText="Derived from the same Wohler curve that sets the degradation cost the optimizer is charged - not a slide assumption." />
      </div>

      <div className={`note ${bank.bankable ? "info" : "crit"}`}>
        <b>{bank.bankable ? "Clears the covenant." : "DAM arbitrage alone does not fund this project."}</b>{" "}
        On {cr(bank.base_annual_revenue_rs)}/yr of arbitrage revenue against {cr(bank.capex_rs)} of
        capex, minimum DSCR is <b>{bank.min_dscr}</b> against a 1.20x covenant. {bank.headroom_note}.
        <div style={{ marginTop: 6 }}>
          That is the honest answer and we publish it rather than tuning the model until it
          agrees. It is also why nearly every funded Indian BESS to date closes on a capacity
          or tariff-based contract rather than on merchant arbitrage.
        </div>
      </div>

      <div className="grid2">
        <Card title="Revenue stacking" info="DSCR"
          sub="Only the DAM leg is proven from our own operating record. RTM is measured on a short book; DSM is an assumption. Kept separate so the two are never conflated.">
          <table className="data">
            <tbody>
              <tr><td>DAM arbitrage <span className="pill verified">proven</span></td>
                <td className="num">{cr(bd.dam_rs)}</td></tr>
              <tr><td>RTM re-optimization <span className="pill identified">short book</span></td>
                <td className="num">{cr(bd.rtm_rs)}</td></tr>
              <tr><td>DSM savings <span className="pill hold">assumption</span></td>
                <td className="num">{cr(bd.dsm_rs)}</td></tr>
              <tr style={{ fontWeight: 700 }}><td>Stacked total</td>
                <td className="num">{cr(bd.total_rs)}</td></tr>
            </tbody>
          </table>
          <div className="note" style={{ marginTop: 10 }}>
            Stacked: equity IRR <b>{st.equity_irr_pct}%</b>, min DSCR <b>{st.min_dscr}</b> -{" "}
            {st.bankable ? "clears" : "still short of"} the 1.20x covenant.
            {cp ? <> The gap closes with a capacity payment of{" "}
              <b>\u20b9{(cp / 1e5).toFixed(1)} lakh/MW/yr</b>, which is the number a developer
              takes into a tender.</> : null}
          </div>
        </Card>
        <Card title="Revenue sensitivity" info="DSCR"
          sub="The first thing a credit committee does is break your revenue assumption and see what survives.">
          <div className="scroll-x">
            <table className="data">
              <thead><tr><th className="num">Revenue</th><th className="num">Equity IRR</th>
                <th className="num">Min DSCR</th><th>Verdict</th></tr></thead>
              <tbody>
                {(bank.sensitivity || []).map((r) => (
                  <tr key={r.revenue_delta_pct}
                    style={r.revenue_delta_pct === 0 ? { background: "var(--band)" } : undefined}>
                    <td className="num">{r.revenue_delta_pct > 0 ? "+" : ""}{r.revenue_delta_pct}%</td>
                    <td className="num">{r.equity_irr_pct}%</td>
                    <td className="num">{r.min_dscr}</td>
                    <td><span className={`pill ${r.bankable ? "verified" : "hold"}`}>
                      {r.bankable ? "funds" : "fails covenant"}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      <Card title="What is measured and what is assumed" style={{ marginTop: 14 }}
        sub="A finance model is only as honest as its parameter list, so here is the whole list.">
        <div className="scroll-x">
          <table className="data">
            <thead><tr><th>Parameter</th><th className="num">Value</th><th>Source</th></tr></thead>
            <tbody>
              <tr><td>Cycles per year</td><td className="num">{a.cycles_per_year}</td>
                <td><span className="pill verified">measured</span> from the dispatch LP</td></tr>
              <tr><td>Mean cycle depth</td><td className="num">{a.mean_cycle_depth}</td>
                <td><span className="pill verified">measured</span> rainflow on the schedule</td></tr>
              <tr><td>Capex</td><td className="num">{`\u20b9${(a.capex_rs_per_mwh / 1e7).toFixed(2)} Cr/MWh`}</td>
                <td>indicative tender range 1.3-1.8 Cr</td></tr>
              <tr><td>GST</td><td className="num">{a.gst_pct}%</td><td>statutory</td></tr>
              <tr><td>O&amp;M</td><td className="num">{a.om_pct_of_capex}%/yr</td>
                <td>assumption, escalating {a.om_escalation_pct}%</td></tr>
              <tr><td>Debt : equity</td><td className="num">{a.debt_share_pct}:{100 - a.debt_share_pct}</td>
                <td>assumption</td></tr>
              <tr><td>Interest / tenor</td><td className="num">{a.interest_pct}% / {a.debt_tenor_years}y</td>
                <td>assumption</td></tr>
              <tr><td>Tax rate</td><td className="num">{a.tax_rate_pct}%</td>
                <td>India new regime, {a.depreciation_pct_wdv}% WDV depreciation</td></tr>
              <tr><td>Availability</td><td className="num">{a.availability_pct}%</td><td>assumption</td></tr>
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
