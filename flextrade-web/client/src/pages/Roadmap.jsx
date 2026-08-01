import { Card, PageHeader } from "../components/ui";

const NOW = [
  ["Live multi-source ingestion", "SLDC + IEX (DAM/RTM/GDAM) + Open-Meteo, cache fallback, self-healing history"],
  ["Load & price forecasting", "LightGBM + quantiles + conformal calibration, bid-time-valid features"],
  ["LP & CVaR dispatch", "risk-aware bidding with a λ dial, 96-block bid sheets"],
  ["DSM module", "versioned CERC 2022/2024 engine + Alerts & Revision, per-technology settlement"],
  ["Real-asset validation", "BRPL Kilokari BESS telemetry sampled 24/7, head-to-head vs our schedule"],
  ["Multi-state: 23 states live", "MERIT national layer (demand/own-gen/import + all-India fuel mix) + NR schedule-vs-drawal detail + deep adapters (Delhi full, Gujarat direct)"],
  ["Forecast-as-a-Service", "FastAPI with tiered keys and metered usage (business model §3.3)"],
  ["Intraday RTM re-optimization", "after DAM clears, the LP re-optimizes residual flexibility against expected RTM prices for the remaining blocks — see Operations"],
  ["Physics-based degradation", "rainflow cycle counting with DoD-dependent Wöhler cost, fixed-point calibrated into the LP — see Operations"],
  ["C&I peak shaving", "demand-charge + ToD optimization under the DERC schedule (illustrative profile, pluggable meter data) — see Operations"],
  ["Three-way DAM+RTM+DSM co-optimization", "the deviation itself is a priced decision — RTM trade vs DSM Normal-Rate settlement, inside the compliance band, verified by the exact CERC engine — see Operations"],
  ["Warranty & availability guard", "audits real BRPL telemetry and our own next-day plan against cycle/SoC warranty limits; pre-trade compliance gate — see Operations"],
  ["Thermal derating", "temperature-dependent power derate + HVAC parasitic load on the committed plan — see Operations"],
  ["Sizing & bankability calculator", "interactive P50/P90 annual revenue + payback from a year of real prices — see Sizing & Bankability"],
  ["Frequency-response readiness", "droop-response duty-cycle report on our own sampled frequency history — see Operations"],
];

const NEXT = [
  ["Warranty-safe dispatch mode", "feed the warranty cycle/SoC limits into the dispatch LP as hard constraints, so the bid sheet is compliant by construction — engine hooks already exist.", "guard already live"],
  ["Ancillary revenue model", "when a public SRAS/TRAS price feed or a partner appears, the frequency-response readiness report becomes a revenue line.", "readiness report live"],
  ["Deep state adapters", "MERIT gives every state's demand; depth (frequency, DSM rates, plant telemetry) needs each SLDC. Rajasthan's endpoints are mapped (upstream currently down); Maharashtra needs the partnership route (login-gated SCADA).", "Rajasthan first"],
  ["State demand history & forecasts", "state_live now accumulates 23-state history every pipeline run — once weeks deep, the Delhi load-model recipe replicates per state.", "data accruing since 24 Jul"],
  ["RE quantile forecasts", "Ensemble/quantile bands on the digital twin so the Alerts engine can weigh revision risk probabilistically, like the price side already does.", ""],
  ["Settlement reconciliation", "Ingest actual RLDC settlement statements and reconcile against our estimates — the DSM spec's ≥99% accuracy metric.", "needs pilot customer data"],
];

const LATER = [
  ["VPP aggregation", "Pool many small BESS/C&I assets into one dispatchable portfolio with shared settlement."],
  ["Portfolio workspace", "Many assets, many states, one risk book — netting, limits, P&L attribution."],
  ["Autonomous bidding", "Human-approved → semi-auto → autonomous exchange submission, in that order, with audit trails."],
  ["National market coupling", "When India couples exchanges, cross-exchange price arbitrage becomes a product overnight."],
];

export default function Roadmap() {
  return (
    <div className="prose">
      <PageHeader eyebrow="Roadmap"
        title="Where this goes"
        lead="The build order follows the business model's phasing: prove value with forecasting
              and analytics first, deepen into optimization and settlement, then monetize flow —
              transaction fees and revenue share — as trust accumulates." />
      <h2 style={{ marginTop: 0 }}>Running today</h2>
      <Card>
        <table className="data"><tbody>
          {NOW.map(([t, d]) => (
            <tr key={t}><td style={{ width: 240 }}><b>✅ {t}</b></td><td style={{ color: "var(--muted)" }}>{d}</td></tr>
          ))}
        </tbody></table>
      </Card>

      <h2>Next — committed</h2>
      <Card>
        <table className="data"><tbody>
          {NEXT.map(([t, d, s]) => (
            <tr key={t}>
              <td style={{ width: 240 }}><b>{t}</b>{s ? <><br /><span className="pill identified">{s}</span></> : null}</td>
              <td style={{ color: "var(--muted)" }}>{d}</td>
            </tr>
          ))}
        </tbody></table>
      </Card>

      <h2>Later — the platform vision</h2>
      <Card sub="'A Bloomberg-like intelligence and AI trading platform for India's power markets' — business model §12.">
        <table className="data"><tbody>
          {LATER.map(([t, d]) => (
            <tr key={t}><td style={{ width: 240 }}><b>{t}</b></td><td style={{ color: "var(--muted)" }}>{d}</td></tr>
          ))}
        </tbody></table>
      </Card>

      <h2>Dashboards planned</h2>
      <ul>
        <li><b>Operator console</b> — SoC, alarms, dispatch overrides, audit log (the BESS control room).</li>
        <li><b>Settlement workbench</b> — estimated vs actual RLDC statements, block-drilldown, dispute flags.</li>
        <li><b>Portfolio risk</b> — exposure by market/state/asset, CVaR heatmaps, limit monitors.</li>
        <li><b>C&I savings report</b> — monthly demand-charge avoidance, ToD arbitrage, ESG lines.</li>
        <li><b>Market intelligence</b> — DAM/RTM/GDAM spreads, congestion patterns, bid-stack analytics.</li>
      </ul>
    </div>
  );
}
