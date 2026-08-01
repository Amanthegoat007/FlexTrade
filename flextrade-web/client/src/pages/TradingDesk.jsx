import { Card, Loading, PageHeader, Stat } from "../components/ui";
import { FanChart, TimeSeries } from "../components/charts";
import { fmtINR, useApi } from "../lib/api";

export default function TradingDesk() {
  const { data: plan, loading, error } = useApi("/api/plan");
  const { data: bt } = useApi("/api/backtest");

  if (loading && !plan) return <Loading error={error} />;
  if (!plan?.blocks?.length) return <Loading error="No plan exported yet — run the daily pipeline." />;

  const blocks = plan.blocks.map((b) => ({
    ...b, t: String(b.ts).slice(11, 16),
    bess: (b.discharge_mw || 0) - (b.charge_mw || 0),
  }));
  const quants = (plan.price_quantiles || []).map((q) => ({
    t: String(q.ts).slice(11, 16), q10: q.q10, q50: q.q50, q90: q.q90,
  }));
  const bids = (plan.bid_sheet || []).filter((b) => b.side && b.side !== "-");

  // cumulative backtest series
  let accP = 0, accG = 0, accF = 0;
  const btRows = (bt?.arbitrage?.daily || []).map((d) => {
    accF += d.pnl_lp || 0; accG += d.pnl_greedy || 0; accP += d.pnl_perfect || 0;
    return {
      date: String(d.date).slice(5), FlexTrade: Math.round(accF / 1e5),
      "Static EMS": Math.round(accG / 1e5), "Perfect foresight": Math.round(accP / 1e5),
    };
  });
  const t = bt?.arbitrage?.totals;
  const uplift = t ? t.pnl_lp - t.pnl_greedy : null;
  const rt = bt?.risk?.totals;

  return (
    <>
      <PageHeader eyebrow="Trading Desk"
        title="Predict → decide → trade"
        lead="Tomorrow's price forecast with a calibrated uncertainty band, the LP-optimal 96-block
              bid sheet built on it, and the walk-forward backtest that proves the money survives
              the forecast error." />
      <h2 className="section-title">Day-ahead plan — delivery {plan.delivery_day}</h2>
      <div className="grid cols-4">
        <Stat label="Expected P&L" info="P_L, LP, DAM" value={fmtINR(plan.expected_pnl_rs, { compact: true })}
          hint="settled at forecast prices, 20 MW / 40 MWh reference asset" />
        <Stat label="Block bids" info="DAM, MCV" value={bids.length} hint="of 96 blocks; rest idle" />
        <Stat label="Peak load forecast" value={plan.peak_load_mw?.toLocaleString("en-IN")} unit="MW" />
        <Stat label="Forecast band" info="P10, P90, CQR" value={quants.length ? "P10–P90" : "—"}
          hint="CQR-guarded: 82.5% empirical coverage vs 80% target" />
      </div>

      {quants.length > 0 && (
        <Card title="Price forecast with uncertainty band" style={{ marginTop: 14 }}
          sub="Quantile LightGBM + conformal calibration. The band is wide in the volatile evening peak and tight overnight — that asymmetry is real market structure, not noise.">
          <FanChart data={quants} xKey="t" height={280} />
        </Card>
      )}

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <Card title="Load & price forecast" sub="the two model outputs the optimizer consumes">
          <TimeSeries data={blocks} xKey="t" height={240}
            series={[{ key: "forecast_load_mw", name: "Load fc (MW)", color: "var(--s1)" }]} />
          <TimeSeries data={blocks} xKey="t" height={180}
            series={[{ key: "forecast_mcp", name: "MCP fc (₹/MWh)", color: "var(--s5)", type: "area" }]} />
        </Card>
        <Card title="LP-optimal BESS schedule" sub="bars: MW (positive = sell). line: state of charge (MWh). Physically feasible by construction — SoC never breaches its bounds.">
          <TimeSeries data={blocks} xKey="t" height={240}
            series={[{ key: "bess", name: "Net MW", color: "var(--s1)", type: "bar" }]} />
          <TimeSeries data={blocks} xKey="t" height={180}
            series={[{ key: "soc_mwh", name: "SoC (MWh)", color: "var(--s4)" }]} />
        </Card>
      </div>

      <h2 className="section-title">DAM bid sheet</h2>
      <Card sub="The actual trade artifact — what a trader submits before the 12:00 IST gate. Price limits carry a ±10% safety margin around the forecast so a small miss doesn't strand the bid.">
        <div className="scroll-x" style={{ maxHeight: 360, overflowY: "auto" }}>
          <table className="data">
            <thead><tr><th>Block</th><th>Time</th><th>Side</th><th className="num">Volume (MW)</th><th className="num">Price limit (₹/MWh)</th></tr></thead>
            <tbody>
              {bids.map((b) => (
                <tr key={b.block}>
                  <td className="num">{b.block}</td>
                  <td>{b.time_block}</td>
                  <td><span className={`pill ${b.side.toLowerCase()}`}>{b.side}</span></td>
                  <td className="num">{b.volume_mw}</td>
                  <td className="num">{b.price_limit_rs_mwh?.toLocaleString("en-IN")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <h2 className="section-title">Proof — 61-day backtest</h2>
      <div className="grid cols-4">
        <Stat label="FlexTrade LP" info="LP" value={t ? fmtINR(t.pnl_lp, { compact: true }) : "—"}
          hint="forecast-built schedules settled at actual prices" />
        <Stat label="Uplift vs static EMS" value={uplift ? fmtINR(uplift, { compact: true }) : "—"}
          delta={t ? `+${Math.round((t.pnl_lp / t.pnl_greedy - 1) * 100)}% vs rule-based` : ""} deltaDir="up"
          hint="the revenue-share billing base" />
        <Stat label="Capture ratio" infoText="Capture ratio — the share of the theoretical maximum (trading with perfect knowledge of tomorrow's prices) that our forecast-based schedule actually earned. 93-94% means forecast errors cost only ~6-7%." value={t ? `${(100 * t.pnl_lp / t.pnl_perfect).toFixed(1)}%` : "—"}
          hint="share of the perfect-foresight ceiling actually captured" />
        <Stat label="Risk-aware (CVaR λ=0.5)" value={rt ? fmtINR(rt.pnl_cvar, { compact: true }) : "—"}
          hint="currently OFF by default: after the point-model upgrade it underperforms (−5.2% mean) — measured honestly, see Methodology" />
      </div>
      {btRows.length > 0 && (
        <Card title="Cumulative P&L (₹ lakh)" style={{ marginTop: 14 }}
          sub="No leakage: every day is forecast with bid-time-valid features only, then settled at the prices that actually cleared.">
          <TimeSeries data={btRows} xKey="date" height={300}
            series={[
              { key: "Perfect foresight", name: "Perfect foresight (bound)", color: "var(--muted)", dash: "5 4" },
              { key: "FlexTrade", name: "FlexTrade LP", color: "var(--s1)" },
              { key: "Static EMS", name: "Static EMS baseline", color: "var(--s5)" },
            ]} />
        </Card>
      )}
    </>
  );
}
