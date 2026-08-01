/* Solutions — who FlexTrade is for. The five business-model customer
   segments (+ consultants/lenders), each mapped to the pain it solves and
   the live parts of the platform that serve it, then the revenue-stream
   map and the Forecast-as-a-Service surface. */
import { Link } from "react-router-dom";

import { Card, PageHeader } from "../components/ui";

const PERSONAS = [
  {
    icon: "🔋", accent: "var(--s2)", name: "BESS Operators",
    role: "Standalone & co-located battery owners",
    pain: "Maximize revenue across DAM, RTM and DSM without over-cycling the asset or voiding its warranty.",
    does: [
      ["Day-ahead bid sheets from LP + risk-aware dispatch", "/trading"],
      ["Intraday RTM & three-way DSM co-optimization", "/operations"],
      ["Warranty guard + physics-based degradation costing", "/operations"],
      ["Validated head-to-head vs a real 20 MW battery", "/"],
    ],
    foot: ["Primary segment — full stack live", "/trading"],
  },
  {
    icon: "☀", accent: "var(--s4)", name: "RE Developers",
    role: "Solar, wind & hybrid generators / QCAs",
    pain: "DSM penalties erode margins when generation deviates from the schedule; forecasting is the whole game.",
    does: [
      ["Per-technology CERC 2022/2024 DSM settlement", "/renewables"],
      ["Solar + wind digital twin with honest NWP error", "/renewables"],
      ["Alerts & Revision engine before gate closure", "/renewables"],
      ["Rajasthan/Tamil Nadu RE-heavy state coverage", "/states"],
    ],
    foot: ["DSM engine + twin live", "/renewables"],
  },
  {
    icon: "🏭", accent: "var(--s1)", name: "DISCOMs",
    role: "Distribution utilities",
    pain: "Accurate demand forecasts and cost-optimal procurement decide whether the day runs at a profit or a loss.",
    does: [
      ["4.33% MAPE day-ahead load forecast", "/"],
      ["23-state live demand & import-dependence view", "/states"],
      ["Schedule-vs-drawal (OD/UD) exposure by state", "/states"],
      ["DAM/RTM/GDAM price intelligence", "/trading"],
    ],
    foot: ["Forecasting proven on Delhi", "/states"],
  },
  {
    icon: "🏢", accent: "var(--s6)", name: "C&I Consumers",
    role: "Factories, malls, data centres",
    pain: "Demand charges and time-of-day tariffs inflate the power bill; a behind-the-meter battery can cut both.",
    does: [
      ["Peak-shaving optimization under DERC ToD", "/operations"],
      ["Demand-charge + energy-charge joint LP", "/operations"],
      ["Pluggable to a pilot's real meter data", "/operations"],
      ["Verified ₹/day saving as the billing base", "/operations"],
    ],
    foot: ["Peak-shaving workspace live", "/operations"],
  },
  {
    icon: "📈", accent: "var(--s7)", name: "Traders & QCAs",
    role: "Power-market participants",
    pain: "Edge comes from better price forecasts and cross-market intelligence than the next desk.",
    does: [
      ["Conformal P10–P90 price bands per block", "/trading"],
      ["DAM / RTM / GDAM spread analytics", "/trading"],
      ["13 months of scraped price history", "/methodology"],
      ["Forecast-as-a-Service REST API, metered", "/methodology"],
    ],
    foot: ["Forecast-as-a-Service is the primary line", "/methodology"],
  },
  {
    icon: "📋", accent: "var(--s5)", name: "Consultants & Lenders",
    role: "Advisory, DPRs, project finance",
    pain: "Bankability studies need P50/P90 revenue tied to real prices, not spreadsheet guesses.",
    does: [
      ["Interactive P50/P75/P90 revenue & payback", "/sizing"],
      ["Walk-forward backtest as the ROI case study", "/trading"],
      ["Every parameter documented & defensible", "/methodology"],
      ["Capex-configurable per project quote", "/sizing"],
    ],
    foot: ["Sizing & Bankability calculator live", "/sizing"],
  },
];

const REVENUE = [
  ["Forecast-as-a-Service", "primary", "Metered REST API — load, price, RE forecasts by tier. The wedge product."],
  ["SaaS subscription", "live", "Per-seat access to this platform — monitoring, trading desk, DSM."],
  ["Asset optimization revenue-share", "live", "Fee = % of the uplift our LP earns over the customer's baseline."],
  ["Transaction / brokerage fee", "live", "Bid sheets are the execution artifact — a fee per MWh routed."],
  ["Data & market intelligence", "live", "Price history, spreads, 23-state demand — the Bloomberg-terminal line."],
  ["Consulting & advisory", "live", "Bankability studies and ROI cases straight from the backtest engine."],
  ["Ancillary & capacity services", "roadmap", "Frequency-response readiness today; a revenue line when SRAS/TRAS opens."],
];

function Persona({ p }) {
  return (
    <div className="persona" style={{ "--accent": p.accent }}>
      <div className="p-icon">{p.icon}</div>
      <div>
        <h3>{p.name}</h3>
        <div className="p-role">{p.role}</div>
      </div>
      <div className="p-pain">{p.pain}</div>
      <ul>
        {p.does.map(([t, to]) => (
          <li key={t}><Link to={to}>{t}</Link></li>
        ))}
      </ul>
      <div className="p-foot">
        <span className="pill verified">✓ live</span>
        <Link to={p.foot[1]}>{p.foot[0]} →</Link>
      </div>
    </div>
  );
}

export default function Solutions() {
  return (
    <div>
      <PageHeader eyebrow="Who it's for"
        title="One platform, five markets"
        lead="FlexTrade serves every side of India's power markets — the battery owner, the
              renewable developer, the utility, the factory and the trader — from a single
              live data-and-optimization core. Delhi is the reference; the model scales." />

      <div className="grid cols-3" style={{ marginBottom: 26 }}>
        {PERSONAS.map((p) => <Persona key={p.name} p={p} />)}
      </div>

      <h2 className="section-title">How value becomes revenue</h2>
      <Card sub="The business model's seven streams, mapped to what is running in this build. Forecast-as-a-Service is the wedge; the rest deepen as trust accumulates.">
        {REVENUE.map(([name, status, desc]) => (
          <div className="rev-row" key={name}>
            <span className="rev-name">{name}</span>
            <span className="rev-desc">{desc}</span>
            <span className={`pill ${status === "roadmap" ? "identified" : status === "primary" ? "sell" : "verified"}`}>
              {status === "primary" ? "★ primary" : status === "roadmap" ? "roadmap" : "live"}
            </span>
          </div>
        ))}
      </Card>

      <h2 className="section-title">Forecast-as-a-Service — the API</h2>
      <Card sub="Tiered, metered access to the same models this dashboard runs on. Starter = market data; Professional = forecasts; Enterprise = optimization + DSM.">
        <div className="code-block">
          <span className="c-comment"># Starter — market data</span>{"\n"}
          curl -H <span className="c-key">"X-API-Key: demo-starter"</span> \{"\n"}
          {"     "}http://localhost:8100/v1/prices/dam{"\n\n"}
          <span className="c-comment"># Professional — day-ahead forecasts</span>{"\n"}
          curl -H <span className="c-key">"X-API-Key: demo-professional"</span> \{"\n"}
          {"     "}http://localhost:8100/v1/forecast/load{"\n\n"}
          <span className="c-comment"># Enterprise — dispatch optimization + DSM</span>{"\n"}
          curl -X POST -H <span className="c-key">"X-API-Key: demo-enterprise"</span> \{"\n"}
          {"     "}-d '{"{"}"power_mw": 50, "energy_mwh": 100{"}"}' \{"\n"}
          {"     "}http://localhost:8100/v1/optimize/dispatch
        </div>
      </Card>

      <Card style={{ marginTop: 20, borderColor: "color-mix(in srgb, var(--s1) 30%, var(--border))" }}
        sub="business model §12">
        <div style={{ fontSize: 16, fontWeight: 600, lineHeight: 1.5 }}>
          “A Bloomberg-like intelligence and AI trading platform for India's power markets.”
        </div>
        <div style={{ color: "var(--muted)", fontSize: 13, marginTop: 6 }}>
          Prove value with forecasting and analytics first, deepen into optimization and
          settlement, then monetize flow — the phasing every stream above follows.
        </div>
      </Card>
    </div>
  );
}
