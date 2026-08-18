import { useEffect, useState } from "react";

/* ---------------------------------------------------------------- static ---
   The app normally talks to the Express server. Built with VITE_STATIC=1 it
   instead reads the exported JSON straight from ./data/, which turns the whole
   dashboard into plain files that any free static host will serve — no Node
   process, no SQLite, nothing to keep running. That is how teammates get a URL.

   The two SQLite-backed endpoints are pre-rendered into the same folder by
   `export_web.py --static`, so a static build is missing no panel. What it
   cannot do is refresh itself: the numbers are frozen at export time, and the
   Overview page's LIVE badges will read as of that moment. Staleness is
   therefore shown, not hidden — meta.json carries the export timestamp and the
   health banner keys off it exactly as it does on the live server. */
export const STATIC_MODE = import.meta.env.VITE_STATIC === "1";

const STATIC_FILES = {
  "/api/meta": "meta",
  "/api/plan": "plan",
  "/api/backtest": "backtest",
  "/api/dsm": "dsm",
  "/api/states": "states",
  "/api/bess": "bess",
  "/api/modules": "modules",
  "/api/state-forecast": "state_forecast",
  "/api/forecasts": "forecasts",
  "/api/trade-book": "trade_book",
  "/api/bankability": "bankability",
  "/api/re-state": "re_state",
  "/api/stress": "stress",
  "/api/dsm-state": "dsm_state",
  "/api/tras": "tras",
  "/api/ists-rule": "ists_rule",
  "/api/live": "live",
  "/api/bess/history": "bess_history",
  "/api/load/recent": "load_recent",
};

function snapshotUrl(path) {
  const name = STATIC_FILES[path.split("?")[0]];
  // BASE_URL keeps this correct when the site is served from a sub-path
  return name ? `${import.meta.env.BASE_URL}data/${name}.json` : null;
}

/* -------------------------------------------------------------- live link ---
   A published build is not required to stay frozen. If the laptop running the
   Express server is online and reachable over a tunnel, the same page will use
   it and everything goes live; when that laptop is closed it silently falls
   back to the bundled snapshot. One deployment, two modes, decided at runtime.

   The API base is resolved in this order, most explicit first:
     1. ?api=https://...   in the URL — and remembered, so a link can carry it
     2. whatever was remembered last (localStorage)
     3. VITE_API_BASE baked in at build time, if a stable tunnel URL exists
     4. same-origin, which is the case when Express itself serves the page

   `?api=off` forces snapshot mode and clears the memory — useful when the
   tunnel is up but you want to demo the offline behaviour. */
const API_KEY = "ft-api-base";

function initialBase() {
  if (typeof window === "undefined") return "";
  const q = new URLSearchParams(window.location.search).get("api");
  if (q === "off") {
    try { localStorage.removeItem(API_KEY); } catch { /* private mode */ }
    return "";
  }
  if (q) {
    const clean = q.replace(/\/+$/, "");
    try { localStorage.setItem(API_KEY, clean); } catch { /* private mode */ }
    return clean;
  }
  try {
    const saved = localStorage.getItem(API_KEY);
    if (saved) return saved;
  } catch { /* private mode */ }
  return (import.meta.env.VITE_API_BASE || "").replace(/\/+$/, "");
}

export const API_BASE = initialBase();

/* Whether the live backend answered. null = not probed yet. In a server-hosted
   build there is nothing to probe: it IS the server. */
let liveOk = STATIC_MODE ? null : true;
const liveListeners = new Set();

function setLive(v) {
  if (liveOk === v) return;
  liveOk = v;
  liveListeners.forEach((fn) => fn(v));
}

export function useLiveLink() {
  const [state, setState] = useState(liveOk);
  useEffect(() => {
    liveListeners.add(setState);
    return () => liveListeners.delete(setState);
  }, []);
  return { live: state, base: API_BASE, staticBuild: STATIC_MODE };
}

/* ngrok's free tier shows a browser interstitial: a request carrying a normal
   browser User-Agent gets an HTML "You are about to visit..." page instead of
   the API response. curl never sees it, which makes it a nasty one to diagnose
   — the tunnel tests fine from a terminal and the dashboard still falls back to
   the snapshot, because fetch() receives HTML and .json() throws. This header
   opts out of it. Harmless on every other host. */
const API_HEADERS = { "ngrok-skip-browser-warning": "true" };

/** Probe the backend once at startup. Short timeout: a closed laptop must not
 *  hold the whole dashboard on a spinner while a TCP connect times out. */
export async function probeLive(timeoutMs = 3500) {
  if (!STATIC_MODE) return true;
  if (!API_BASE) { setLive(false); return false; }
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeoutMs);
    const r = await fetch(`${API_BASE}/api/health`, {
      signal: ctl.signal, headers: API_HEADERS,
    });
    clearTimeout(t);
    setLive(r.ok);
    return r.ok;
  } catch {
    setLive(false);
    return false;
  }
}

async function fetchJson(url, timeoutMs, live = false) {
  const ctl = new AbortController();
  const t = timeoutMs ? setTimeout(() => ctl.abort(), timeoutMs) : null;
  try {
    const r = await fetch(url, {
      signal: ctl.signal,
      ...(live ? { headers: API_HEADERS } : {}),
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    const ct = r.headers.get("content-type") || "";
    if (!ct.includes("json")) {
      // an interstitial or a captive portal — treat as unreachable, not as data
      throw new Error(`expected JSON, got ${ct.split(";")[0] || "unknown"}`);
    }
    return await r.json();
  } finally {
    if (t) clearTimeout(t);
  }
}

/** Live first, snapshot second. Returns [data, isLive]. */
async function load(path) {
  const snap = snapshotUrl(path);
  if (!STATIC_MODE) return [await fetchJson(path), true];

  if (API_BASE && liveOk !== false) {
    try {
      const data = await fetchJson(`${API_BASE}${path}`, 6000, true);
      setLive(true);
      return [data, true];
    } catch {
      // the tunnel died mid-session: drop to the snapshot rather than error
      setLive(false);
    }
  }
  if (!snap) throw new Error("no snapshot bundled for this endpoint");
  return [await fetchJson(snap), false];
}

const cache = new Map();

export function useApi(path, { refreshMs = 0 } = {}) {
  const [state, setState] = useState(() => ({
    data: cache.get(path) || null,
    loading: !cache.has(path),
    error: null,
  }));

  useEffect(() => {
    let alive = true;
    async function run() {
      try {
        const [data, isLive] = await load(path);
        cache.set(path, data);
        if (alive) setState({ data, loading: false, error: null, live: isLive });
      } catch (e) {
        if (alive) setState((s) => ({ ...s, loading: false, error: String(e) }));
      }
    }
    run();
    // Only poll when a live backend is actually answering — re-reading a
    // static file on a timer just burns battery for the same bytes.
    const id = refreshMs && (!STATIC_MODE || API_BASE)
      ? setInterval(run, refreshMs) : null;
    return () => { alive = false; if (id) clearInterval(id); };
  }, [path, refreshMs]);

  return state;
}

export const fmtINR = (v, opts = {}) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const abs = Math.abs(v);
  if (opts.compact) {
    if (abs >= 1e7) return `₹${(v / 1e7).toFixed(2)} Cr`;
    if (abs >= 1e5) return `₹${(v / 1e5).toFixed(1)} L`;
  }
  return `₹${Math.round(v).toLocaleString("en-IN")}`;
};

export const fmtMW = (v) =>
  v === null || v === undefined ? "—" : `${Math.round(v).toLocaleString("en-IN")} MW`;

export const fmtTs = (s) => {
  if (!s) return "—";
  const d = new Date(String(s).replace(" ", "T"));
  if (Number.isNaN(d.getTime())) return String(s);
  return d.toLocaleString("en-IN", {
    day: "2-digit", month: "short",
    hour: "2-digit", minute: "2-digit", hour12: false,
  });
};

export const fmtDate = (s) => {
  if (!s) return "—";
  const d = new Date(String(s).replace(" ", "T"));
  if (Number.isNaN(d.getTime())) return String(s);
  return d.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" });
};

export const ago = (s) => {
  if (!s) return "never";
  const d = new Date(String(s).replace(" ", "T"));
  const mins = Math.round((Date.now() - d.getTime()) / 60000);
  if (Number.isNaN(mins)) return String(s);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  if (mins < 60 * 36) return `${Math.round(mins / 60)} h ago`;
  return `${Math.round(mins / 1440)} d ago`;
};
