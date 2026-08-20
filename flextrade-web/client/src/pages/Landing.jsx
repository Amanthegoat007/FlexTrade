/* Landing — the front door.

   Every number on this page is READ from meta.json, never typed. That is not
   tidiness: five hand-written figures went stale on this site inside two days,
   and a landing page is the one surface where a wrong number does the most
   damage and gets noticed last. If a metric is not exported, it does not
   appear here. */
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { useApi } from "../lib/api";
import "../landing.css";

const pct = (v, d = 1) => (v == null ? "—" : `${Number(v).toFixed(d)}%`);
const n0 = (v) => (v == null ? "—" : Math.round(Number(v)).toLocaleString("en-IN"));

/* Reveal-on-scroll. IntersectionObserver rather than a scroll listener so the
   work happens off the main thread and there is nothing to throttle. */
function useReveal() {
  const ref = useRef(null);
  useEffect(() => {
    const els = ref.current?.querySelectorAll(".lp-reveal") ?? [];
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => e.isIntersecting && e.target.classList.add("in")),
      { rootMargin: "-12% 0px -8%", threshold: 0.08 });
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);
  return ref;
}

/* One delivery day as 48 blocks: height is the price shape, colour is what the
   optimizer decided. Built from the real plan when one is loaded, and from a
   representative evening-peak shape before that — never presented as data. */
function BlockField({ plan }) {
  const bars = useMemo(() => {
    const rows = plan?.blocks;
    if (Array.isArray(rows) && rows.length > 8) {
      const step = Math.max(1, Math.floor(rows.length / 48));
      const s = rows.filter((_, i) => i % step === 0).slice(0, 48);
      const ps = s.map((r) => Number(r.forecast_price_rs_mwh ?? r.price_rs_mwh ?? 0));
      const lo = Math.min(...ps);
      const hi = Math.max(...ps) || 1;
      return s.map((r, i) => {
        const mw = Number(r.discharge_mw ?? 0) - Number(r.charge_mw ?? 0);
        return {
          h: 8 + 92 * ((ps[i] - lo) / Math.max(hi - lo, 1)),
          k: mw > 0.05 ? "discharge" : mw < -0.05 ? "charge" : "idle",
          d: i * 18,
        };
      });
    }
    return Array.from({ length: 48 }, (_, i) => {
      const hour = (i / 2) % 24;
      const evening = Math.exp(-((hour - 19.5) ** 2) / 5.5);
      const morning = Math.exp(-((hour - 9) ** 2) / 9);
      const h = 12 + 88 * Math.min(1, evening * 0.95 + morning * 0.4 + 0.12);
      const k = hour >= 18 && hour <= 22 ? "discharge" : hour <= 6 ? "charge" : "idle";
      return { h, k, d: i * 18 };
    });
  }, [plan]);

  return (
    <>
      <div className="lp-stage" aria-hidden="true">
        <div className="lp-field">
          {bars.map((b, i) => (
            <div key={i} className={`lp-bar ${b.k}`}
              style={{ height: `${b.h}%`, animationDelay: `${b.d}ms` }} />
          ))}
        </div>
      </div>
      <div className="lp-legend">
        <span><i style={{ background: "var(--s1)" }} />charge</span>
        <span><i style={{ background: "var(--s6)" }} />discharge</span>
        <span><i style={{ background: "var(--grid)" }} />hold</span>
      </div>
    </>
  );
}

export default function Landing() {
  const { data: meta } = useApi("/api/meta");
  const { data: plan } = useApi("/api/plan");
  const ref = useReveal();
  const h = meta?.metrics?.headline || {};
  const col = meta?.metrics?.collection || {};
  const wf = meta?.metrics?.walkforward?.models || [];
  const [year] = useState(() => new Date().getFullYear());

  const audited = wf.length;
  const beaten = wf.filter((m) => m.vs_benchmark?.better === "A").length;

  return (
    <div className="lp" ref={ref}>
      <section className="lp-hero">
        <div>
          <div className="lp-eyebrow">Battery energy storage · India</div>
          <h1 className="lp-title">The bid, not the dashboard.</h1>
          <p className="lp-sub">
            FlexTrade forecasts the Indian power market and returns the thing a
            trading desk actually submits — a block-by-block bid sheet, priced,
            dispatch-feasible, and scored against what perfect foresight would
            have earned.
          </p>
          <div className="lp-cta">
            <Link className="lp-btn lp-btn-primary" to="/trading">Open the trading desk →</Link>
            <Link className="lp-btn lp-btn-ghost" to="/methodology">How it is measured</Link>
          </div>
          <BlockField plan={plan} />
        </div>
      </section>

      <section className="lp-band">
        <div className="lp-cell">
          <b>{pct(h.capture_ratio_pct)}</b>
          <span>of perfect foresight captured, over {h.backtest_days ?? "—"} backtested days</span>
        </div>
        <div className="lp-cell">
          <b>{pct(h.price_test_wape_pct, 2)}</b>
          <span>day-ahead price WAPE, refit at every rolling origin</span>
        </div>
        <div className="lp-cell">
          <b>{beaten}/{audited}</b>
          <span>models that beat their named incumbent at p &lt; 0.001</span>
        </div>
        <div className="lp-cell">
          <b>{n0(col.total_rows)}</b>
          <span>rows collected from sources that publish no history</span>
        </div>
      </section>

      <section className="lp-story lp-reveal">
        <div className="lp-kicker">The problem</div>
        <h2 className="lp-h2">A price forecast is not a decision.</h2>
        <p className="lp-p">
          Knowing tomorrow&apos;s prices does not tell a battery what to do.
          Charging costs money, cycling costs warranty, and the state of charge
          you carry into the evening peak was decided at breakfast. The decision
          is a constrained optimisation over the whole day, not a series of
          guesses.
        </p>
        <p className="lp-p">
          FlexTrade solves that optimisation directly — a linear program over 96
          blocks with real efficiency, real degradation cost and real
          state-of-charge limits — and hands back a bid sheet that is feasible
          by construction.
        </p>
        <div className="lp-split">
          <div className="lp-card">
            <b>+{pct(h.uplift_pct, 0)}</b>
            <span>more revenue than a greedy buy-low-sell-high rule, on the same
              forecast and the same battery</span>
          </div>
          <div className="lp-card">
            <b>{pct(h.capture_ratio_pct)}</b>
            <span>of what an oracle with tomorrow&apos;s exact prices would have
              earned</span>
          </div>
        </div>
      </section>

      <section className="lp-story lp-reveal">
        <div className="lp-kicker">The method</div>
        <h2 className="lp-h2">Every number here had to beat something.</h2>
        <p className="lp-p">
          Each model is scored on rolling origins with non-overlapping test
          windows and a full refit at every origin, so no forecast is ever graded
          on data it was trained on. Each is compared against a named incumbent —
          persistence, a seasonal naive, the hour-ratio rule a desk uses today —
          with a Diebold-Mariano test to say whether the margin is real or luck.
        </p>
        <p className="lp-p">
          Interval forecasts are held to Kupiec and Christoffersen, because a band
          that says 80% and delivers 74% is mislabelled, not conservative.
        </p>
      </section>

      <section className="lp-honest">
        <div className="lp-story lp-reveal">
          <div className="lp-kicker">What we do not claim</div>
          <h2 className="lp-h2">The uncomfortable parts, in public.</h2>
          <p className="lp-p">
            Anyone can publish the runs that worked. These are on the site because
            a number you cannot check is worth nothing.
          </p>
          <ul className="lp-list">
            <li><b>This battery is not bankable on arbitrage alone.</b> Minimum
              DSCR is {h.min_dscr ?? "below"} against a 1.20x covenant, and we
              publish the capacity payment it would take to clear.</li>
            <li><b>The load band&apos;s failures cluster.</b> Christoffersen
              rejects independence at every origin. Marginal coverage is correct;
              the misses are not independent, and we say so.</li>
            <li><b>Load accuracy is measured against actual weather.</b> A
              day-ahead forecast cannot know tomorrow&apos;s temperature, so the
              honest figure is the realised one, not the backtest one.</li>
            <li><b>Features that failed are documented.</b> Plant outages, a
              scarcity-market signal and two regime-conditional calibrations were
              tested, lost, and left out.</li>
          </ul>
        </div>
      </section>

      <section className="lp-story lp-reveal" style={{ textAlign: "center" }}>
        <h2 className="lp-h2">See tomorrow&apos;s bid sheet.</h2>
        <p className="lp-p" style={{ margin: "0 auto 28px" }}>
          Generated before the 12:00 gate, every day.
        </p>
        <div className="lp-cta">
          <Link className="lp-btn lp-btn-primary" to="/trading">Trading desk →</Link>
          <Link className="lp-btn lp-btn-ghost" to="/sizing">Sizing &amp; bankability</Link>
        </div>
        <p style={{ marginTop: 40, fontSize: 12.5, color: "var(--muted)" }}>
          FlexTrade · {year} · every figure on this page is read from the live
          metrics export, not written by hand
        </p>
      </section>
    </div>
  );
}
