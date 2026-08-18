/* Reserves & Regulation — the two constraints that decide whether a merchant
   BESS in India is buildable, priced from a year of real IEX DAM prices.

   Both models answer a question with a REQUIRED price rather than a forecast
   revenue, which is the only honest form available: India publishes ancillary
   VOLUME but not ancillary PRICE, so any rupee figure for TRAS revenue would
   be invented. Instead we compute what TRAS would have to clear at for holding
   reserve to beat pure arbitrage, and let the reader compare it to whatever
   price they can obtain. Nothing here inherits an error from a price we cannot
   see. */
import { Card, InfoTip, Loading, PageHeader, Stat } from "../components/ui";
import { useApi } from "../lib/api";

const inr = (n) => `₹${Math.round(n).toLocaleString("en-IN")}`;
const lakh = (n) => `₹${(n / 1e5).toFixed(1)} L`;

export default function Reserves() {
  const { data: tras, error: e1 } = useApi("/api/tras");
  const { data: ists, error: e2 } = useApi("/api/ists-rule");
  if (!tras || !ists) return <Loading error={e1 || e2} />;

  const base = tras.arbitrage_only_annual_rs_per_mw;

  return (
    <div>
      <PageHeader eyebrow="Reserves & Regulation"
        title="What the rules cost, in rupees per MW per year"
        sub={`Measured on ${tras.window.n_days} days of IEX DAM prices, ${tras.window.from} → ${tras.window.to}. A ${tras.asset.power_mw} MW / ${tras.asset.energy_mwh} MWh asset.`} />

      {/* ---------------------------------------------------------- TRAS --- */}
      <h2>Ancillary reserve (TRAS) — the price it would have to clear at</h2>
      <p>
        Holding reserve means not selling that MW into the day-ahead market.
        The cost is therefore the arbitrage given up, which we can measure
        exactly. The revenue is a TRAS clearing price, which is
        <b> not published anywhere we can reach</b> — so instead of guessing it,
        we invert the question and report the price at which participation
        starts to pay.
      </p>
      <div className="grid4">
        <Stat label="Arbitrage-only baseline"
          value={lakh(base)} hint="per MW per year, no reserve held" />
        {tras.reserve_levels.map((r) => (
          <Stat key={r.reserve_frac}
            label={`Breakeven TRAS @ ${(r.reserve_frac * 100).toFixed(0)}% reserve`}
            value={`${inr(r.breakeven_tras_rs_per_mw_h)}/MW/h`}
            hint={`gives up ${r.forgone_pct}% of arbitrage`} />
        ))}
      </div>

      <Card title="Reserve ladder" sub="Higher reserve forgoes more arbitrage, so it demands a higher clearing price to be worth holding.">
        <div style={{ overflowX: "auto" }}>
          <table className="data">
            <thead><tr>
              <th>Reserve held</th>
              <th className="num">Arbitrage kept</th>
              <th className="num">Forgone</th>
              <th className="num">Forgone %</th>
              <th className="num">Breakeven TRAS</th>
              <th className="num">Throughput</th>
            </tr></thead>
            <tbody>
              {tras.reserve_levels.map((r) => (
                <tr key={r.reserve_frac}>
                  <td><b>{(r.reserve_frac * 100).toFixed(0)}%</b> ({r.reserve_mw} MW)</td>
                  <td className="num">{lakh(r.arbitrage_annual_rs_per_mw)}</td>
                  <td className="num">{lakh(r.forgone_arbitrage_rs_per_mw_yr)}</td>
                  <td className="num">{r.forgone_pct}%</td>
                  <td className="num"><b>{inr(r.breakeven_tras_rs_per_mw_h)}</b>/MW/h</td>
                  <td className="num">{Math.round(r.throughput_mwh_per_mw_yr).toLocaleString("en-IN")} MWh</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ marginTop: 12 }}>{tras.how_to_read}</p>
      </Card>

      <Card title="Why there is no TRAS revenue number on this page">
        <p>{tras.not_available.tras_cleared_price}</p>
        <p>
          We could have assumed a price. Every rupee downstream would then have
          inherited that assumption, and a reader could not tell which part was
          measured. The reserve ladder above is measured end to end — feasibility,
          forgone arbitrage and throughput all come out of the same dispatch LP
          that produces the bid sheet.
        </p>
      </Card>

      {/* ---------------------------------------------------------- ISTS --- */}
      <h2>ISTS charge waiver — what the RE-charging condition costs</h2>
      <p>
        The waiver on inter-state transmission charges requires a storage asset
        to charge from renewables rather than freely from the grid. That is a
        real dispatch constraint: the battery can only fill when the sun or wind
        is producing, not when the market is cheapest. This prices that
        constraint alone — <b>not</b> the mandated RE plant, which is a separate
        project with its own PPA and land questions.
      </p>
      <div className="grid4">
        <Stat label="Unrestricted (grid charging)"
          value={lakh(ists.unrestricted_annual_rs_per_mw)} hint="per MW per year" />
        <Stat label="Best restricted mix"
          value={lakh(ists.best_restricted.annual_rs_per_mw)}
          hint={`${ists.best_restricted.mix}, ${ists.best_restricted.re_mw_per_bess_mw}:1 RE:BESS`} />
        <Stat label="Cost of the constraint"
          value={`${ists.headline_pct}%`} hint="best case, revenue given up" />
        <Stat label="Site"
          value={`${ists.site.lat}N ${ists.site.lon}E`} hint={ists.site.note} />
      </div>

      <Card title="RE sizing sweep" sub="More RE relieves the constraint with diminishing returns — and the surplus it strands is reported rather than quietly counted as revenue.">
        <div style={{ overflowX: "auto" }}>
          <table className="data">
            <thead><tr>
              <th>Mix</th><th className="num">RE per BESS MW</th>
              <th className="num">Annual ₹/MW</th><th className="num">vs unrestricted</th>
              <th className="num">Charged</th><th className="num">RE unused</th>
            </tr></thead>
            <tbody>
              {ists.scenarios.map((s, i) => {
                const best = s.mix === ists.best_restricted.mix
                  && s.re_mw_per_bess_mw === ists.best_restricted.re_mw_per_bess_mw;
                return (
                  <tr key={i} style={best ? { fontWeight: 700 } : undefined}>
                    <td>{s.mix}{best ? " ✅" : ""}</td>
                    <td className="num">{s.re_mw_per_bess_mw}:1</td>
                    <td className="num">{lakh(s.annual_rs_per_mw)}</td>
                    <td className="num">{s.vs_unrestricted_pct}%</td>
                    <td className="num">{Math.round(s.charged_mwh_per_mw_yr).toLocaleString("en-IN")} MWh</td>
                    <td className="num" style={{ color: "var(--muted)" }}>
                      {Math.round(s.re_unused_mwh_per_mw_yr).toLocaleString("en-IN")} MWh</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <h2>Caveats — stated before you ask</h2>
      <Card title="Ancillary reserve (TRAS)">
        <ul>{tras.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
      </Card>
      <Card title="ISTS waiver">
        <ul>{ists.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
      </Card>
    </div>
  );
}
