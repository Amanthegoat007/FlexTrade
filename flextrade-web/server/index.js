/**
 * FlexTrade API server.
 *
 * Design rule: this server never re-implements a scraper or a model. The
 * Python quant stack (../..:/flextrade) owns every fetch, model and
 * settlement calculation; it exports JSON artifacts to output/web/ and
 * keeps SQLite (data/flextrade.db) current. This server:
 *   - serves those JSON artifacts as /api/*
 *   - queries SQLite read-only for time-series endpoints
 *   - triggers `python export_web.py --live` (rate-limited) so the
 *     Overview page is genuinely live, single implementation in Python
 *   - serves the built React client
 */
import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import Database from "better-sqlite3";
import express from "express";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FLEXTRADE = path.resolve(__dirname, "..", "..", "flextrade");
const WEB_JSON = path.join(FLEXTRADE, "output", "web");
const DB_PATH = path.join(FLEXTRADE, "data", "flextrade.db");
const CLIENT_DIST = path.resolve(__dirname, "..", "client", "dist");
const PORT = process.env.PORT || 8090;

const app = express();
const db = new Database(DB_PATH, { readonly: true, fileMustExist: true });

/* ---------- CORS ----------
   The published static copy (Netlify/Cloudflare Pages) fetches this API from a
   different origin whenever the laptop is online, so the browser needs explicit
   permission. Read-only GETs of already-public grid data, so this is not
   sensitive — but it is still restricted rather than opened to the world.

   ALLOWED_ORIGINS is a comma-separated env var; localhost is always allowed so
   local development needs no configuration:

     $env:ALLOWED_ORIGINS = "https://flextrade.netlify.app"; node server/index.js

   Set it to "*" only if you knowingly want any site to be able to read it. */
// A browser's Origin header is always scheme://host[:port] with NO trailing
// slash, so "https://site.netlify.app/" pasted from the address bar would never
// match and every request would be silently refused. Normalise both sides.
const normOrigin = (s) => s.trim().replace(/\/+$/, "").toLowerCase();
const ALLOWED = (process.env.ALLOWED_ORIGINS || "")
  .split(",").map(normOrigin).filter(Boolean);

if (ALLOWED.length) console.log("CORS allowed origins:", ALLOWED.join(", "));
else console.log("CORS: no ALLOWED_ORIGINS set — only localhost may call this API");

app.use((req, res, next) => {
  const origin = req.headers.origin;
  if (origin) {
    const local = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin);
    if (ALLOWED.includes("*")) res.setHeader("Access-Control-Allow-Origin", "*");
    else if (local || ALLOWED.includes(normOrigin(origin))) {
      res.setHeader("Access-Control-Allow-Origin", origin);
      res.setHeader("Vary", "Origin");
    } else {
      // loud on the server side so a blocked teammate is diagnosable from here
      console.warn(`CORS refused origin: ${origin}`);
    }
  }
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers",
    "Content-Type, ngrok-skip-browser-warning");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

// ---------- helpers ----------
const sendJsonFile = (name) => (req, res) => {
  const f = path.join(WEB_JSON, `${name}.json`);
  if (!fs.existsSync(f)) return res.status(404).json({ error: `${name}.json not exported yet — run export_web.py` });
  res.setHeader("X-Artifact-Modified", fs.statSync(f).mtime.toISOString());
  res.sendFile(f);
};

// ---------- live refresh (rate-limited, single-flight) ----------
let refreshing = null;
let lastRefresh = 0;
const REFRESH_MIN_MS = 90_000; // don't hammer the upstream sites

function refreshLive() {
  if (refreshing) return refreshing;
  if (Date.now() - lastRefresh < REFRESH_MIN_MS) return Promise.resolve("cached");
  refreshing = new Promise((resolve) => {
    execFile(
      "python", ["export_web.py", "--live"],
      { cwd: FLEXTRADE, timeout: 45_000, env: { ...process.env, PYTHONIOENCODING: "utf-8" } },
      (err) => {
        lastRefresh = Date.now();
        refreshing = null;
        resolve(err ? `error: ${err.message}` : "refreshed");
      }
    );
  });
  return refreshing;
}

// ---------- artifact endpoints ----------
app.get("/api/meta", sendJsonFile("meta"));
app.get("/api/plan", sendJsonFile("plan"));
app.get("/api/backtest", sendJsonFile("backtest"));
app.get("/api/dsm", sendJsonFile("dsm"));
app.get("/api/states", sendJsonFile("states"));
app.get("/api/bess", sendJsonFile("bess"));
app.get("/api/modules", sendJsonFile("modules"));
app.get("/api/state-forecast", sendJsonFile("state_forecast"));
app.get("/api/forecasts", sendJsonFile("forecasts"));

app.get("/api/live", async (req, res) => {
  const f = path.join(WEB_JSON, "live.json");
  const stale = !fs.existsSync(f) || Date.now() - fs.statSync(f).mtimeMs > REFRESH_MIN_MS;
  if (stale) await refreshLive();
  sendJsonFile("live")(req, res);
});

app.post("/api/refresh", async (_req, res) => {
  res.json({ status: await refreshLive() });
});

// ---------- SQLite time-series endpoints ----------
app.get("/api/prices/recent", (req, res) => {
  const market = { dam: "dam_price", rtm: "rtm_price", gdam: "gdam_price" }[req.query.market || "dam"];
  if (!market) return res.status(400).json({ error: "market must be dam|rtm|gdam" });
  const days = Math.min(parseInt(req.query.days || "7", 10), 60);
  const rows = db.prepare(`
    SELECT ts, mcp_rs_mwh FROM ${market}
    WHERE ts >= datetime((SELECT MAX(ts) FROM ${market}), '-${days} days')
    ORDER BY ts`).all();
  res.json({ market: req.query.market || "dam", days, rows });
});

app.get("/api/load/recent", (req, res) => {
  const days = Math.min(parseInt(req.query.days || "7", 10), 90);
  // 15-min mean of the 5-min feed, done in SQL so the payload stays small
  const rows = db.prepare(`
    SELECT strftime('%Y-%m-%d %H:', ts) ||
           printf('%02d', (CAST(strftime('%M', ts) AS INT) / 15) * 15) || ':00' AS block,
           ROUND(AVG(delhi), 1) AS delhi_mw
    FROM load_5min
    WHERE ts >= datetime((SELECT MAX(ts) FROM load_5min), '-${days} days')
    GROUP BY block ORDER BY block`).all();
  res.json({ days, rows });
});

app.get("/api/frequency/recent", (_req, res) => {
  const rows = db.prepare(
    "SELECT ts, frequency_hz FROM frequency ORDER BY ts DESC LIMIT 600").all().reverse();
  res.json({ rows });
});

app.get("/api/bess/history", (req, res) => {
  const hours = Math.min(parseInt(req.query.hours || "48", 10), 24 * 14);
  const rows = db.prepare(`
    SELECT ts, discharge_mw, soc_pct FROM bess_telemetry
    WHERE ts >= datetime((SELECT MAX(ts) FROM bess_telemetry), '-${hours} hours')
    ORDER BY ts`).all();
  res.json({ hours, rows });
});

app.get("/api/health", (_req, res) => {
  const meta = path.join(WEB_JSON, "meta.json");
  res.json({
    ok: true,
    now: new Date().toISOString(),
    db: fs.existsSync(DB_PATH),
    lastExport: fs.existsSync(meta) ? fs.statSync(meta).mtime.toISOString() : null,
  });
});

// ---------- client ----------
app.use(express.static(CLIENT_DIST));
app.get(/^(?!\/api\/).*/, (_req, res) => {
  const idx = path.join(CLIENT_DIST, "index.html");
  if (fs.existsSync(idx)) return res.sendFile(idx);
  res.status(503).send("client not built yet — run `npm run build` in ../client");
});

const server = app.listen(PORT, () => {
  console.log(`FlexTrade server on http://localhost:${PORT}`);
  console.log(`  quant stack: ${FLEXTRADE}`);
  console.log(`  db: ${DB_PATH}`);
});

// "address already in use" is by far the most common way starting this fails,
// and Node's default is an unhandled 'error' event that buries the one useful
// word under twenty lines of stack. Say what happened and how to fix it.
server.on("error", (err) => {
  if (err.code === "EADDRINUSE") {
    console.error(`\nPort ${PORT} is already in use — a FlexTrade server is ` +
      `almost certainly still running from an earlier session.\n`);
    console.error("Free it, then start again:");
    console.error(`  PowerShell:  Get-NetTCPConnection -LocalPort ${PORT} ` +
      `-State Listen | Select-Object -Expand OwningProcess -Unique | ` +
      `ForEach-Object { Stop-Process -Id $_ -Force }`);
    console.error(`\nOr just use a different port for this run:`);
    console.error(`  $env:PORT = "8091"; node server\\index.js`);
    console.error(`  (then tunnel 8091 instead: ngrok http 8091 --url=...)\n`);
    process.exit(1);
  }
  throw err;
});
