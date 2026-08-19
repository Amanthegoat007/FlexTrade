/* Rolling-origin validation, rendered beside the forecast it validates.

   These statistics used to live only on the Methodology page — the one page a
   sceptical reader reaches last, if at all. A coverage claim belongs next to
   the band it describes and a Diebold-Mariano result next to the model it
   beats, so this is shared by the Forecast Lab and the Trading Desk rather
   than duplicated into each. */
import { Card, Stat } from "./ui";

const pct = (v, d = 1) => (v == null ? "—" : `${Number(v).toFixed(d)}%`);

export const wfModel = (meta, match) =>
  meta?.metrics?.walkforward?.models?.find((m) =>
    m.model.toLowerCase().includes(match));

export default function WalkForward({ meta, match, title, children }) {
  const m = wfModel(meta, match);
  if (!m) return null;
  const dm = m.vs_benchmark;
  return (
    <Card title={title || "Rolling-origin validation"}
      sub={`${m.origins_run} non-overlapping origins × ${m.test_days} days, ${m.window}, ${m.blocks?.toLocaleString("en-IN")} scored points. The model is refitted at every origin, so no test window sits inside its own training data.`}>
      <div className="grid cols-3">
        <Stat label="WAPE across origins" value={pct(m.wape_pct?.mean, 2)}
          hint={`worst origin ${pct(m.wape_pct?.worst, 2)} · sd ${Number(m.wape_pct?.std ?? 0).toFixed(2)}`} />
        <Stat label="MAE across origins"
          value={m.mae?.mean != null ? Math.round(m.mae.mean).toLocaleString("en-IN") : "—"}
          unit={m.unit}
          hint={`worst origin ${Math.round(m.mae?.worst ?? 0).toLocaleString("en-IN")} ${m.unit}`} />
        {dm ? (
          <Stat label="Diebold-Mariano vs baseline"
            value={`t = ${Number(dm.stat).toFixed(2)}`}
            hint={dm.p_value === 0 ? "p < 0.001 — the win is not luck" : `p = ${dm.p_value}`} />
        ) : (
          <Stat label="Interval score"
            value={m.interval_score_mean != null ? Math.round(m.interval_score_mean).toLocaleString("en-IN") : "—"}
            hint="proper rule: width + miss penalty, lower is better" />
        )}
      </div>
      {m.bias?.mean != null && Math.abs(m.bias.mean) > 1 && (
        <div className="note" style={{ marginTop: 12 }}>
          <b>Signed bias {m.bias.mean > 0 ? "+" : ""}{Math.round(m.bias.mean).toLocaleString("en-IN")} {m.unit}</b>
          {" "}across origins (worst {Math.round(m.bias.worst).toLocaleString("en-IN")}).
          {m.bias.mean < 0
            ? " The model under-forecasts. For a battery that is the direction that costs money — a discharge opportunity valued too cheaply is bid too cheaply — so it is stated rather than averaged into the headline."
            : " The model over-forecasts on average."}
        </div>
      )}
      {children}
    </Card>
  );
}
