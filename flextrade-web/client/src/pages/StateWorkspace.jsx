/* State Workspace — pick any of the 23 states and see its whole picture:
   live position, a year of daily history, the pooled forecaster's accuracy
   for THAT state, its RE mix and its exchange exposure. This is the page
   that turns "we forecast Delhi" into "we forecast India". */
import { useMemo, useState } from "react";

import { HBar, TimeSeries } from "../components/charts";
import { Card, InfoTip, Loading, PageHeader, Stat } from "../components/ui";
import { fmtMW, useApi } from "../lib/api";

const fmtGWh = (mwh) =>
  mwh === null || mwh === undefined ? "—" : `${(mwh / 1000).toFixed(1)} GWh`;

export default function StateWorkspace() {
  const { data: sf, loading, error } = useApi("/api/state-forecast");
  const { data: live } = useApi("/api/live", { refreshMs: 120_000 });
  const { data: reg } = useApi("/api/states");
  const [code, setCode] = useState("RJ");

  const profiles = sf?.profiles || [];
  const profile = useMemo(
    () => profiles.find((p) => p.code === code) || profiles[0],
    [profiles, code]);

  if (loading && !sf) return <Loading error={error} />;
  if (sf?.error || !profiles.length) {
    return (
      <div>
        <PageHeader eyebrow="State Workspace" title="Per-state intelligence"
          lead="The pooled 23-state forecaster has not been trained yet." />
        <Card sub="Run the MERIT history backfill, then models/state_forecast.py.">
          <div className="note">{sf?.error || "no state profiles exported yet"}</div>
        </Card>
      </div>
    );
  }

  const energy = sf?.energy_met_mwh || {};
  const exch = sf?.exchange_mwh || {};
  const perState = (energy.per_state || []).find((s) => s.code === profile.code);
  const perStateX = (exch.per_state || []).find((s) => s.code === profile.code);
  const liveRow = (live?.india?.states || []).find((s) => s.code === profile.code);
  const regRow = (reg?.registry || []).find((s) => s.code === profile.code);

  const series = (profile.series || []).map((r) => ({
    ...r, t: r.day.slice(5),
  }));
  const hasRE = series.some((r) => r.solar_mwh || r.wind_mwh);

  // rank this state among peers by forecast accuracy
  // sMAPE, not MAPE. MAPE is unbounded when the target nears zero and the
  // exchange series does: rendering it put 816,144% on this page.
  const ranked = [...(energy.per_state || [])]
    .filter((s) => s.smape_pct === s.smape_pct)
    .sort((a, b) => a.smape_pct - b.smape_pct);
  const rank = ranked.findIndex((s) => s.code === profile.code) + 1;

  return (
    <div>
      <PageHeader eyebrow="State Workspace"
        title={`${profile.name} — full intelligence`}
        lead="Every state gets the same treatment Delhi does: live position, real history,
              a trained day-ahead forecast with its own measured accuracy, RE mix and
              exchange exposure. Pick a state to switch the whole page.">
        <select value={code} onChange={(e) => setCode(e.target.value)}
          style={{ fontSize: 14, fontWeight: 600, padding: "8px 12px", minWidth: 210 }}>
          {profiles.map((p) => (
            <option key={p.code} value={p.code}>
              {p.name} · {(p.mean_energy_mwh / 1000).toFixed(0)} GWh/d
            </option>
          ))}
        </select>
      </PageHeader>

      <div className="grid4">
        <Stat label="Demand right now" info="MERIT, MW"
          value={liveRow ? fmtMW(liveRow.demand_mw) : "—"}
          hint={liveRow ? `own gen ${fmtMW(liveRow.own_gen_mw)} · import ${fmtMW(liveRow.import_mw)}` : "live feed"} />
        <Stat label="Typical daily energy" value={fmtGWh(profile.mean_energy_mwh)}
          infoText="Mean energy served per day across the stored history — the state's scale, which the pooled model uses as a feature."
          hint={`peak ${fmtGWh(profile.peak_energy_mwh)} · ${profile.days_history} days held`} />
        <Stat label="Day-ahead error, daily energy" info="sMAPE"
          value={perState ? `${perState.smape_pct}%` : "—"}
          hint={perState
            ? `vs ${perState.naive_smape_pct}% seasonal-naive${rank ? ` · rank ${rank}/${ranked.length}` : ""}`
            : "not in test window"}
          infoText="Symmetric MAPE, bounded at 200%. Used instead of plain MAPE because a state's exchange purchases can approach zero, where MAPE reports millions of percent." />
        <Stat label="Exchange purchases" info="FaaS"
          value={profile.exchange_share_pct != null ? `${profile.exchange_share_pct}%` : "—"}
          hint="share of energy bought on the power exchange — our addressable market" />
      </div>

      <Card title={`Daily energy served — last ${series.length} days`} info="MERIT"
        sub="Real history from the Ministry of Power's MERIT portal (daily energy by procurement source). This is what the pooled forecaster trains on.">
        <TimeSeries data={series} xKey="t" yLabel="MWh/day" height={260}
          series={[{ key: "energy_mwh", name: "energy served", color: "var(--s1)", type: "area" }]} />
      </Card>

      <div className="grid2">
        <Card title="Power-exchange purchases" info="DAM, FaaS"
          sub="Energy this state bought on the exchange each day — the volume a price forecast directly monetizes.">
          <TimeSeries data={series} xKey="t" yLabel="MWh/day" height={210}
            series={[{ key: "exchange_mwh", name: "exchange purchase", color: "var(--s6)", type: "bar" }]} />
        </Card>
        {hasRE ? (
          <Card title="Renewable generation" info="RE"
            sub="Daily solar and wind scheduled generation — the DSM exposure an RE developer in this state carries.">
            <TimeSeries data={series} xKey="t" yLabel="MWh/day" height={210}
              series={[
                { key: "solar_mwh", name: "solar", color: "var(--s4)" },
                { key: "wind_mwh", name: "wind", color: "var(--s5)" },
              ]} />
          </Card>
        ) : (
          <Card title="Renewable generation" sub="Plant-level generation history is still backfilling for this state.">
            <div className="note">Solar/wind daily series appears once the
              MERIT generation backfill covers {profile.name}.</div>
          </Card>
        )}
      </div>

      <h2 className="section-title">
        How the forecast is built for {profile.name}
        <InfoTip text="A pooled (global) model is trained across ALL modelled states at once with state identity and scale as features, so every state benefits from far more training rows than it owns. This is the standard result from the M4/M5 forecasting competitions for short, related series. What is served is an equal-weight blend of that model and a seasonal-naive baseline." />
      </h2>
      <Card sub={energy.approach}>
        <div className="grid3">
          <Stat label="Training rows" value={(energy.n_train_rows || 0).toLocaleString("en-IN")}
            hint={`${energy.n_states} states pooled · ${energy.history_from} → ${energy.history_to}`} />
          <Stat label="All-states test sMAPE" value={energy.overall_smape_pct != null ? `${energy.overall_smape_pct}%` : "—"}
            hint={`vs ${energy.naive_smape_pct}% seasonal-naive (same weekday last week)`} />
          <Stat label="States beating naive" value={energy.states_beating_naive || "—"}
            hint={`on a held-out ${energy.test_days}-day window`} />
        </div>
        <div className="note info" style={{ marginTop: 12 }}>
          <b>Why pooled, not 23 separate models:</b> no state except Delhi publishes
          enough history to train a strong standalone model. One global learner with
          state identity as a feature lets short series borrow strength from long
          ones — the finding that decided the M4/M5 competitions.
          <br /><br />
          <b>Read the resolution difference, it matters.</b> Everything on this page
          is a <b>daily</b> forecast: one number per state per day, which is what
          MERIT publishes. Delhi separately runs a <b>15-minute intraday</b> model
          (4.33% MAPE) because it is the only state publishing 5 years of 5-minute
          SLDC load. A daily state number cannot drive a 15-minute market — it sizes
          and screens, it does not trade. That gap is the honest limit of the
          multi-state layer today.
        </div>
      </Card>

      <h2 className="section-title">All states — forecast accuracy leaderboard</h2>
      <Card sub="Served error on the untouched test window, beside the seasonal-naive baseline (same weekday last week). 'What we serve' is an equal-weight combination of the two, not a choice between them — per-state selection was measured worse than either pure strategy. Sorted best first; a good average is never allowed to hide a bad state.">
        <HBar height={Math.max(260, ranked.length * 19)} valLabel="Served sMAPE %"
          data={[...ranked].reverse().map((s) => ({ name: s.name, value: s.smape_pct }))} />
        <div className="scroll-x">
          <table className="data">
            <thead><tr>
              <th>State</th><th className="num">Served</th><th className="num">Pooled model</th>
              <th className="num">Seasonal naive</th><th>What we serve</th>
              <th className="num">vs naive</th><th className="num">Mean MWh/day</th>
            </tr></thead>
            <tbody>
              {ranked.map((s) => {
                const better = s.naive_smape_pct - s.smape_pct;
                return (
                  <tr key={s.code}
                    style={s.code === profile.code ? { background: "var(--band)" } : undefined}>
                    <td><b>{s.name}</b></td>
                    <td className="num"><b>{s.smape_pct}%</b></td>
                    <td className="num" style={{ color: "var(--muted)" }}>{s.model_only_smape_pct}%</td>
                    <td className="num" style={{ color: "var(--muted)" }}>{s.naive_smape_pct}%</td>
                    <td><span className={`pill ${String(s.champion).startsWith("blend") ? "verified"
                      : s.champion === "model" ? "verified" : "hold"}`}>{s.champion}</span></td>
                    <td className="num" style={{ color: better > 0 ? "var(--delta-good)" : "var(--critical)" }}>
                      {better > 0 ? "−" : "+"}{Math.abs(better).toFixed(1)} pp
                    </td>
                    <td className="num">{s.mean_mwh?.toLocaleString("en-IN")}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {energy.underperforming?.length > 0 && (
          <div className="note crit" style={{ marginTop: 12 }}>
            <b>Where we currently lose to the trivial baseline:</b>{" "}
            {energy.underperforming.map((u) => `${u.name} (${u.served_smape_pct}% vs ${u.naive_smape_pct}%)`).join(", ")}.
            These are states whose weekly rhythm is so regular that "same weekday
            last week" is genuinely hard to beat — Punjab's baseline scores 0.53%.
            Blending still costs a little there, and we take that trade knowingly:
            the same rule nearly halves the error on the worst state. We show the
            rows rather than drop them.
          </div>
        )}
      </Card>

      {exch.overall_smape_pct != null && (
        <>
          <h2 className="section-title">Exchange-purchase forecast</h2>
          <Card sub="A second pooled model predicts how much each state will buy on the power exchange tomorrow — the number a DISCOM trader and our own Forecast-as-a-Service customers care about most.">
            <div className="grid3">
              <Stat label="All-states test sMAPE" value={`${exch.overall_smape_pct}%`}
                hint={`vs ${exch.naive_smape_pct}% seasonal-naive`} />
              <Stat label={`${profile.name} sMAPE`}
                value={perStateX ? `${perStateX.smape_pct}%` : "—"}
                hint={perStateX ? `vs ${perStateX.naive_smape_pct}% seasonal-naive` : "—"} />
              <Stat label="States beating naive" value={exch.states_beating_naive || "—"} />
            </div>
            <div className="note" style={{ marginTop: 10 }}>
              Exchange volumes are far spikier than total demand, so errors are
              larger — we report them rather than quietly dropping the harder target.
            </div>
          </Card>
        </>
      )}

      {sf?.tiers && (
        <>
          <h2 className="section-title">
            Coverage tiers — what each state can actually support
            <InfoTip text="Data availability differs sharply by state. Rather than claim uniform 23-state forecasting, each state sits in the tier its published data supports — and the tier is shown on screen." />
          </h2>
          <Card sub="Surveyed by probing MERIT's historical endpoint on spread-out dates per state. A state is only promoted a tier when its data actually supports it.">
            {Object.entries(sf.tiers).map(([key, t]) => (
              <div className="rev-row" key={key}>
                <span className="rev-name" style={{ minWidth: 180 }}>
                  {key.replace(/_/g, " ")}
                  <br />
                  <span className={`pill ${key.startsWith("live") ? "identified" : "verified"}`}>
                    {t.states.length} state{t.states.length === 1 ? "" : "s"}
                  </span>
                </span>
                <span className="rev-desc">
                  <b>{t.resolution}</b>
                  {t.accuracy ? <> · <b style={{ color: "var(--delta-good)" }}>{t.accuracy}</b></> : null}
                  <br />
                  {t.basis}. <span style={{ color: "var(--muted)" }}>{t.note}</span>
                  <br />
                  <span className="mono" style={{ fontSize: 11.5 }}>{t.states.join(" · ")}</span>
                </span>
              </div>
            ))}
          </Card>
        </>
      )}

      {regRow && (
        <Card style={{ marginTop: 14 }} title={`Data sources for ${profile.name}`}
          sub={`Live coverage status: ${regRow.status}`}>
          <div style={{ fontSize: 13, color: "var(--ink-2)" }}>{regRow.notes}</div>
        </Card>
      )}
    </div>
  );
}
