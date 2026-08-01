# Sharing FlexTrade with remote teammates

**Yes — a plain Netlify upload is frozen at build time.** The folder Netlify
serves has no Python, no database and no scrapers behind it, so nothing can
update.

So the app supports what you described: **one deployment, two modes, decided in
the browser at page load.**

```
teammate opens https://your-site.netlify.app
        │
        ├─ probes your laptop's tunnel  (GET /api/health, 3.5 s timeout)
        │
        ├─ laptop ON  → every panel reads the LIVE API   → badge: "LIVE backend"
        └─ laptop OFF → every panel reads bundled JSON   → badge: "SNAPSHOT"
```

Nobody has to know which mode they are in, and nothing breaks either way. If the
tunnel dies mid-session, individual requests fall back to the snapshot rather
than erroring.

---

## Setup

### 1. Get a public URL for your laptop

The frontend is on HTTPS, so the API must be too — a plain `http://your-ip`
would be blocked as mixed content. A tunnel handles that.

```powershell
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8090
```

It prints something like `https://brave-forest-1234.trycloudflare.com`.

**The catch:** a quick tunnel's URL is random and changes every restart. Two
ways around it, both free:

- **ngrok** — the free plan includes **one permanent domain**. Claim it at
  <https://dashboard.ngrok.com/domains>, then
  `ngrok http 8090 --url=your-name.ngrok-free.app`. Same URL forever. This is
  the one to use if you want to set it up once and forget it.
- **Cloudflare named tunnel** — permanent, but needs a domain you already have
  on Cloudflare.

If you would rather not bother: build without a baked URL and paste the tunnel
URL into the address bar when you need it — see *Changing the backend without
rebuilding* below.

### 2. Build the site with that URL baked in

```powershell
cd flextrade
python build_static_site.py --api=https://your-domain.ngrok-free.dev
```

Use the domain **ngrok assigned to your account** (Dashboard → Domains), not an
invented one — a free account may only serve the one it was given. Include
`https://`; a bare host is treated by the browser as a relative path and every
call 404s into the snapshot silently. The build now adds the scheme for you and
says so.

Prints `mode: HYBRID` on success. Output is `flextrade-web/client/dist-static/`,
about **1.3 MB**.

### 3. Publish it

| Host | How |
|---|---|
| **Netlify Drop** | drag `dist-static` onto <https://app.netlify.com/drop> |
| **Cloudflare Pages** | <https://pages.cloudflare.com> → Direct Upload |
| **GitHub Pages** | push `dist-static` to a `gh-pages` branch |

### 4. Let the deployed site call your laptop

Browsers block cross-origin requests unless the server allows them, so start the
server with your site's origin:

```powershell
cd flextrade-web
$env:ALLOWED_ORIGINS = "https://your-site.netlify.app"
node server\index.js
```

On start it prints `CORS allowed origins: ...` — check that line matches your
site. A trailing slash is tolerated. Any refused origin is logged as
`CORS refused origin: ...`, so a blocked teammate is diagnosable from here.

`localhost` is always allowed, so local development needs no configuration. Any
origin not on the list gets no CORS header and is refused by the browser.

**That's it.** Laptop on and tunnel running → teammates see live data. Laptop
shut → they see the snapshot, clearly labelled.

---

## Changing the backend without rebuilding

The API base is resolved at runtime, most explicit first:

1. **`?api=` in the URL** — `https://your-site.netlify.app/?api=https://abc.trycloudflare.com`
   The value is remembered in `localStorage`, so it survives navigation. This is
   how you use a random quick-tunnel URL: send the link with `?api=` on it.
2. Whatever was remembered last.
3. `VITE_API_BASE`, baked in by `--api=` at build time.
4. Same origin — which is the case when Express serves the page itself.

`?api=off` forces snapshot mode and forgets the saved URL. Useful for showing
the offline behaviour deliberately, or if a stale tunnel URL is stuck.

---

## What each mode can and cannot do

| | Live (laptop on) | Snapshot (laptop off) |
|---|---|---|
| All 11 pages, every chart | ✅ | ✅ |
| Forecast Lab, State Workspace | ✅ | ✅ |
| Numbers update as data arrives | ✅ | ❌ frozen at build time |
| Auto-refresh timers | ✅ | off — no point re-reading a file |
| LIVE / CACHED source badges | real | as of export |
| Banner | normal pipeline health | 📸 "Published snapshot" + age |

The snapshot's numbers are all **real** — produced by the live pipeline — they
are simply frozen. The banner says so rather than letting an old export read as
current. Refresh it any time with another `python build_static_site.py` and
re-upload.

---

## Snapshot-only build (no laptop involved)

```powershell
python build_static_site.py
```

Same 1.3 MB folder, no backend probing. Good as a demo-day fallback: if the
venue network dies, open the published URL on a phone hotspot.

---

## Notes

- The build sets `VITE_STATIC=1`, which switches the API layer to `./data/*.json`
  and the router to **hash mode** (`/#/forecasts`). Hash routing needs no server
  rewrite rules, so deep links and refreshes work on every host with zero config.
- `--base ./` makes asset paths relative, so the folder also works from a
  sub-path or a plain `file://` open — you can zip and email it.
- `robots.txt` disallows crawling, so a published snapshot will not turn up in
  search results before the demo. Delete it if you want the site indexed.
- `.nojekyll` is included so GitHub Pages does not mangle asset paths.
- The two endpoints the server answers from SQLite (recent load, BESS telemetry)
  are pre-rendered by `export_web.py --static`, so no panel is empty offline.

### Why not host the Node server on a free tier?

Considered and rejected on specifics: the free tiers that run a process
(Render, Railway, Fly) sleep after inactivity, expire, or want a card — and this
server additionally needs Python, LightGBM and an **88 MB SQLite file** just to
answer two chart endpoints. Your laptop already has all of that installed and
working. Tunnelling to it costs nothing and keeps one source of truth.

### Security

The tunnel exposes a **read-only** API of already-public grid data (SLDC, IEX,
MERIT are all public sources) — but it is still a public URL while running:

- CORS is restricted to the origins you list; it is not open to the world.
- The SQLite handle is opened `readonly`.
- `POST /api/refresh` triggers a scrape on your machine, so do not leave a
  tunnel running unattended for long periods.
- Take the tunnel down when you are done: `Ctrl-C`.
