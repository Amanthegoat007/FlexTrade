import { useState } from "react";

import { STATIC_MODE, ago, useLiveLink } from "../lib/api";
import { gloss } from "../lib/glossary";

/* Page hero — a consistent title block at the top of every page, so the
   app reads like one product instead of eight loosely-related screens. */
export function PageHeader({ eyebrow, title, lead, children }) {
  return (
    <div className="page-head">
      <div className="page-head-main">
        {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
        <h1>{title}</h1>
        {lead ? <p className="lead">{lead}</p> : null}
      </div>
      {children ? <div className="page-head-aside">{children}</div> : null}
    </div>
  );
}

/* In-page tabbed sub-navigation. `tabs` = [{ id, label, hint }]. Controlled
   internally; renders the active panel via the `children` render-prop. */
export function Tabs({ tabs, initial, children }) {
  const [active, setActive] = useState(initial || tabs[0]?.id);
  return (
    <>
      <div className="tabbar" role="tablist">
        {tabs.map((t) => (
          <button key={t.id} role="tab" aria-selected={active === t.id}
            className={`tab ${active === t.id ? "active" : ""}`}
            onClick={() => setActive(t.id)}>
            {t.icon ? <span className="tab-icon">{t.icon}</span> : null}
            <span>{t.label}</span>
            {t.hint ? <span className="tab-hint">{t.hint}</span> : null}
          </button>
        ))}
      </div>
      <div className="tab-panel" role="tabpanel">{children(active)}</div>
    </>
  );
}

/* ⓘ info button: hover/focus shows full forms + definitions.
   `terms` = comma-separated glossary keys ("DAM, MCP"); `text` = free text. */
export function InfoTip({ terms, text }) {
  const body = [terms ? gloss(terms) : null, text].filter(Boolean).join("\n\n");
  if (!body) return null;
  return (
    <span className="infotip" tabIndex={0} aria-label={body}>
      <svg width="13" height="13" viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth="1.4" />
        <rect x="7.2" y="6.8" width="1.6" height="5" rx="0.8" fill="currentColor" />
        <circle cx="8" cy="4.4" r="1" fill="currentColor" />
      </svg>
      <span className="tipbox" role="tooltip">
        {body.split("\n\n").map((p, i) => {
          const dash = p.indexOf(" — ");
          return dash > 0 && dash < 14
            ? <p key={i}><b>{p.slice(0, dash)}</b>{p.slice(dash)}</p>
            : <p key={i}>{p}</p>;
        })}
      </span>
    </span>
  );
}

export function Badge({ live, label, asof }) {
  return (
    <span className={`badge ${live ? "live" : "cached"}`} title={asof ? `as of ${asof}` : ""}>
      <span className="dot" />
      {label}
      {!live && asof ? ` · ${ago(asof)}` : ""}
    </span>
  );
}

/* Which mode the page is in, always visible in the topbar. On a
   server-hosted build there is only one mode, so it renders nothing rather
   than stating the obvious. */
export function LinkBadge() {
  const { live, base, staticBuild } = useLiveLink();
  if (!staticBuild) return null;
  if (live === null) {
    return (
      <span className="badge cached" title="checking whether the live backend is reachable">
        <span className="dot" /> connecting…
      </span>
    );
  }
  return live ? (
    <span className="badge live" title={`live backend: ${base}`}>
      <span className="dot" /> LIVE backend
    </span>
  ) : (
    <span className="badge cached"
      title={base
        ? `backend ${base} is not reachable — showing the bundled snapshot`
        : "no backend configured — showing the bundled snapshot"}>
      <span className="dot" /> SNAPSHOT
    </span>
  );
}

export function Stat({ label, value, unit, hint, delta, deltaDir, info, infoText }) {
  return (
    <div className="card stat">
      <div className="label">{label}{(info || infoText) ? <InfoTip terms={info} text={infoText} /> : null}</div>
      <div className="value">
        {value}
        {unit ? <small> {unit}</small> : null}
      </div>
      {delta ? <div className={`delta ${deltaDir || ""}`}>{delta}</div> : null}
      {hint ? <div className="hint">{hint}</div> : null}
    </div>
  );
}

export function Card({ title, sub, children, style, info, infoText }) {
  return (
    <div className="card" style={style}>
      {title ? <h3>{title}{(info || infoText) ? <InfoTip terms={info} text={infoText} /> : null}</h3> : null}
      {sub ? <div className="sub">{sub}</div> : null}
      {children}
    </div>
  );
}

export function Loading({ error }) {
  return <div className="skeleton">{error ? `⚠ ${error}` : "loading…"}</div>;
}

/* Pipeline health banner. Silence was the real bug behind the 27-28 Jul
   outage: the daily run died on a DNS race and nothing on screen said so
   for four days. This makes a failed or stale run impossible to miss. */
export function HealthBanner({ health }) {
  const { live } = useLiveLink();
  if (!health) return null;
  const stale = health.plan_is_current === false;
  const failed = health.ok === false;
  const old = (health.age_hours ?? 0) > 36;

  /* A statically-published copy is a snapshot: it cannot re-run the pipeline,
     so it ages by design. Saying "pipeline attention" there would report a
     failure that has not happened. It still must not pass age off as live, so
     the snapshot gets its own honest, non-alarming notice.

     Only when the live backend is NOT answering, though — the same published
     page goes live the moment the laptop's tunnel is reachable, and then the
     normal pipeline-health rules apply again. */
  if (STATIC_MODE && live === false) {
    const age = health.age_hours;
    return (
      <div className="note health-banner" style={{ borderColor: "var(--s1)" }}>
        <b>📸 Published snapshot</b> — this is a static copy of the dashboard,
        exported {age != null ? `${Math.round(age)} h ago` : "at build time"}
        {health.last_run_at ? <> (<span className="mono">{health.last_run_at}</span>)</> : null}.
        Every number is real and was produced by the live pipeline, but this
        copy cannot refresh itself, so nothing here updates until it is
        rebuilt and re-uploaded.
        {failed ? <> The pipeline run behind this snapshot failed at stage{" "}
          <b>{health.stage}</b>, so model-derived panels may be incomplete.</> : null}
      </div>
    );
  }

  if (!stale && !failed && !old) return null;

  const bits = [];
  if (failed) bits.push(`last pipeline run failed at stage “${health.stage}”`);
  if (stale) {
    bits.push(health.plan_days_stale > 0
      ? `the delivery plan is ${health.plan_days_stale} day(s) behind (showing ${health.plan_delivery_day})`
      : `the delivery plan is not for tomorrow (showing ${health.plan_delivery_day})`);
  }
  if (old && !failed) bits.push(`no successful run for ${health.age_hours} h`);

  return (
    <div className="note crit health-banner">
      <b>⚠ Data pipeline attention</b> — {bits.join("; ")}.
      {health.detail ? <> <span className="mono">{health.detail}</span></> : null}
      <div style={{ marginTop: 4, fontSize: 12, color: "var(--muted)" }}>
        Live market and grid panels below still show real fetched data; only
        model-derived plans may lag. Re-run <span className="mono">python run_pipeline.py</span> to refresh.
      </div>
    </div>
  );
}
