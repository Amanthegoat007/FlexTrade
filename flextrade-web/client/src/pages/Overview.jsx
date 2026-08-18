import { Card, Loading, PageHeader, Stat } from "../components/ui";
import { HBar, TimeSeries } from "../components/charts";
import { ago, fmtINR, fmtMW, fmtTs, useApi } from "../lib/api";

export default function Overview() {
  const { data: live, loading, error } = useApi("/api/live", { refreshMs: 120_000 });
  const { data: meta } = useApi("/api/meta");
  const { data: plan } = useApi("/api/plan");
  const { data: loadHist } = useApi("/api/load/recent?days=3");
  const { data: bessHist } = useApi("/api/bess/history?hours=48", { refreshMs: 300_000 });

  if (loading && !live) return <Loading error={error} />;

  const delhi = live?.delhi || {};
  const bess = live?.bess || {};
  const india = live?.india;
  const states = live?.northern_region?.states || [];
  const totalNR = states.reduce((a, s) => a + (s.load_mw || 0), 0) + (delhi.delhi_load || 0);

  const priceRows = (live?.dam?.blocks || []).map((b, i) => ({
    ts: String(b.ts).slice(11, 16),
    DAM: b.mcp_rs_mwh,
    RTM: live?.rtm?.blocks?.[i]?.mcp_rs_mwh ?? null,
    GDAM: live?.gdam?.blocks?.[i]?.mcp_rs_mwh ?? null,
  }));

  const loadRows = (loadHist?.rows || []).map((r) => ({
    ts: r.block.slice(5, 16), delhi_mw: r.delhi_mw,
  }));

  const bessRows = (bessHist?.rows || []).map((r) => ({
    ts: String(r.ts).slice(5, 16), soc: r.soc_pct, mw: r.discharge_mw,
  }));

  return (
    <>
      <PageHeader eyebrow="Live Overview"
        title="India's power markets, right now"
        lead="Three different scopes sit on this page and each tile says which one it is:
              DELHI for the state we forecast at 15-minute resolution, ALL-INDIA for the
              national position, and IEX for exchange prices, which clear as a single
              pan-India price. Everything is fetched live and every timestamp is computed
              from the data itself." />
      <h2 className="section-title">Live grid &amp; market state</h2>
      <div className="note info" style={{ marginBottom: 12 }}>
        <b>Scope:</b> the first two tiles are <b>Delhi</b> (state-level, from Delhi SLDC).
        The two price tiles are <b>IEX</b> — a single clearing price for the whole
        country, not a Delhi price. The last tile is <b>all-India</b> demand from MERIT.
      </div>
      <div className="grid cols-5">
        <Stat label="DELHI · load now" value={delhi.delhi_load?.toLocaleString("en-IN") ?? "—"} unit="MW"
          info="MW, SLDC" hint={`schedule ${fmtMW(delhi.schedule)} · drawl ${fmtMW(delhi.drawl)}`} />
        <Stat label="Grid frequency" value={delhi.frequency?.toFixed?.(2) ?? "—"} unit="Hz"
          info="Hz, DSM"
          hint="50.00 Hz reference · measured at Delhi"
          infoText="Read at Delhi SLDC, but the Indian grid is synchronous — frequency is a national quantity and is essentially the same everywhere on the interconnected system at any instant." />
        <Stat label="IEX DAM · avg today" value={live?.dam ? fmtINR(live.dam.avg_mcp) : "—"} unit="/MWh"
          info="DAM, MCP, IEX"
          hint={live?.dam ? `range ${fmtINR(live.dam.min_mcp)} – ${fmtINR(live.dam.max_mcp)} · pan-India` : "pan-India price"}
          infoText="IEX clears one Market Clearing Price for the whole country in each 15-minute block, splitting into bid areas only when transmission congests. This is NOT a Delhi-specific price." />
        <Stat label="IEX RTM · avg today" value={live?.rtm ? fmtINR(live.rtm.avg_mcp) : "—"} unit="/MWh"
          info="RTM, GDAM"
          hint={live?.gdam ? `GDAM avg ${fmtINR(live.gdam.avg_mcp)} · pan-India` : "pan-India price"} />
        {india?.national?.demand_met_mw ? (
          <Stat label="ALL-INDIA · demand met" value={Math.round(india.national.demand_met_mw / 100) / 10} unit="GW"
            info="MERIT"
            hint={`${(india?.states || []).length} states live via MERIT · Northern Region ${Math.round(totalNR / 100) / 10} GW`} />
        ) : (
          <Stat label="NORTHERN REGION · demand" value={Math.round(totalNR / 100) / 10} unit="GW"
            info="MERIT"
            hint={`${states.length + 1} NR states incl. Delhi · all-India feed unavailable`}
            infoText="The all-India figure from MERIT is not answering right now, so this tile falls back to the Northern Region total and relabels itself rather than presenting a regional number as a national one." />
        )}
      </div>

      <h2 className="section-title">Tomorrow, decided today</h2>
      <div className="grid cols-3">
        <Stat label="Delivery day planned" value={plan?.delivery_day || "—"}
          info="DAM, IST" hint="Bid sheet generated before the 12:00 IST DAM gate" />
        <Stat label="Expected arbitrage P&L" value={plan ? fmtINR(plan.expected_pnl_rs, { compact: true }) : "—"}
          info="P_L, LP" hint="LP-optimal schedule on the price forecast" />
        <Stat label="Peak load forecast" value={plan?.peak_load_mw?.toLocaleString("en-IN") ?? "—"} unit="MW"
          info="MAPE, LGBM" hint="Delhi, day-ahead, 4.33% test MAPE" />
      </div>

      {meta?.collection?.total_rows > 0 && (
        <>
          <h2 className="section-title">The data asset</h2>
          <div className="grid cols-3">
            <Stat label="Rows collected" info="rows"
              value={meta.collection.total_rows.toLocaleString("en-IN")}
              hint={`across ${meta.collection.table_count} tables, counted from the store at export`} />
            <Stat label="Deepest series" info="5-min"
              value={(meta.collection.tables?.[0]?.rows ?? 0).toLocaleString("en-IN")}
              hint={`${meta.collection.tables?.[0]?.table ?? "—"} — Delhi load at 5-minute resolution`} />
            <Stat label="Collectors running" value="3"
              hint="AWS Lambda (Mumbai, 15-min) · GitHub Actions · laptop" />
          </div>
          <p className="muted" style={{ marginTop: 8 }}>
            Every upstream here is snapshot-only — MERIT, the state SLDCs, the
            Vidyut PRAVAH area price and NPP all publish "now" and no history.
            None of this can be bought or backfilled later: it exists because
            something was running at the time, and a missed block is gone
            permanently. That is why the count is read from the store on every
            export rather than quoted from memory.
          </p>
        </>
      )}

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <Card title="Today's IEX clearing prices" sub="₹/MWh per 15-min block, live scrape. One pan-India price per block across DAM, RTM and GDAM — not a Delhi price.">
          <TimeSeries data={priceRows} height={250} yLabel="₹/MWh"
            series={[
              { key: "DAM", name: "DAM", color: "var(--s1)" },
              { key: "RTM", name: "RTM", color: "var(--s5)" },
              { key: "GDAM", name: "GDAM", color: "var(--s4)" },
            ]} />
        </Card>
        <Card title="Delhi load — last 3 days" sub="Delhi SLDC 5-min feed, averaged to 15-min blocks. This is the one state we forecast at intraday resolution.">
          <TimeSeries data={loadRows} height={250} yLabel="MW"
            series={[{ key: "delhi_mw", name: "Delhi load", color: "var(--s1)", type: "area" }]} />
        </Card>
      </div>

      <h2 className="section-title">Live asset — BRPL Kilokari BESS (Delhi)</h2>
      <div className="note info" style={{ marginBottom: 12 }}>
        A 20 MW / 40 MWh grid-scale standalone battery in Delhi, one of the first of
        its kind in India. Telemetry is published by Delhi SLDC and sampled by
        FlexTrade every 5 minutes — a real operating asset, not a simulation. It is
        the reference asset for every P&amp;L figure on this site; we observe it, we
        do not operate it.
      </div>
      <div className="grid cols-4">
        <Stat label="State" value={bess.discharge_mw > 0.05 ? "Discharging" : bess.discharge_mw < -0.05 ? "Charging" : "Idle"}
          info="BESS" hint={bess.ts ? `sampled ${ago(bess.ts)}` : ""} />
        <Stat label="Net power" value={bess.discharge_mw?.toFixed?.(2) ?? "—"} unit="MW"
          info="MW, kVAr" hint="positive = exporting to grid" />
        <Stat label="State of charge" value={bess.soc_pct?.toFixed?.(0) ?? "—"} unit="%"
          info="SoC, MWh" hint={bess.soc_mwh != null ? `≈ ${bess.soc_mwh.toFixed(1)} MWh of 40` : ""} />
        <Stat label="Samples collected" value={bessHist?.rows?.length ?? "—"}
          info="SLDC" hint="last 48 h · SLDC publishes no history, we build it" />
      </div>
      {bessRows.length > 3 && (
        <div className="grid cols-2" style={{ marginTop: 14 }}>
          <Card title="Observed SoC" sub="state of charge, %">
            <TimeSeries data={bessRows} height={200}
              series={[{ key: "soc", name: "SoC %", color: "var(--s4)" }]} />
          </Card>
          <Card title="Observed dispatch" sub="MW, positive = discharge into grid">
            <TimeSeries data={bessRows} height={200}
              series={[{ key: "mw", name: "Net MW", color: "var(--s1)" }]} />
          </Card>
        </div>
      )}

      <h2 className="section-title">Data freshness</h2>
      <Card sub="Every timestamp on this page is computed from the data itself — nothing is hardcoded.">
        <div className="scroll-x">
          <table className="data">
            <thead>
              <tr><th>Dataset</th><th>Rows</th><th>Coverage from</th><th>Latest record</th><th>Age</th></tr>
            </thead>
            <tbody>
              {(meta?.datasets || [])
                .filter((d) => d.rows > 0 && d.description)
                .map((d) => (
                  <tr key={d.table}>
                    <td><b>{d.table}</b><br /><span style={{ color: "var(--muted)", fontSize: 11.5 }}>{d.description}</span></td>
                    <td className="num">{d.rows.toLocaleString("en-IN")}</td>
                    <td>{fmtTs(d.from)}</td>
                    <td>{fmtTs(d.to)}</td>
                    <td>{ago(d.to)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </Card>

      {states.length > 0 && (
        <>
          <h2 className="section-title">Northern Region — live state loads (8 states, excl. Delhi)</h2>
          <Card sub="Published on Delhi SLDC's real-time page; one fetch covers eight states.">
            <HBar height={280} valLabel="Load (MW)"
              data={states
                .map((s) => ({ name: s.state, value: s.load_mw }))
                .sort((a, b) => a.value - b.value)} />
          </Card>
        </>
      )}
    </>
  );
}
