"""Build a self-contained static copy of the FlexTrade dashboard.

    python build_static_site.py

Produces `flextrade-web/client/dist-static/` — plain HTML, JS, CSS and JSON
with no server, no Python and no database behind it. Drop that folder on any
free static host and remote teammates get the whole dashboard on a URL.

Why static rather than hosting the Node server: the free tiers that run a
process (Render, Fly, Railway) all sleep, expire, or need a card, and ours
would additionally need Python, LightGBM and an 88 MB SQLite file just to
answer two chart endpoints. Those two endpoints are pre-rendered here instead,
which drops the whole site to ~1.5 MB of files that any CDN serves free
forever.

The honest trade: a static copy is a SNAPSHOT. It cannot re-fetch SLDC or IEX,
so every number is frozen at build time. That is why meta.json's export
timestamp is baked in and the existing health banner keys off it unchanged —
a stale copy announces itself rather than quietly showing old numbers as live.
Re-run this script and re-upload whenever you want the snapshot refreshed.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIENT = HERE.parent / "flextrade-web" / "client"
WEB_JSON = HERE / "output" / "web"
OUT_DIR = CLIENT / "dist-static"

NPM = "npm.cmd" if sys.platform == "win32" else "npm"


def _normalise_api(url: str) -> str:
    """Make sure --api is an ABSOLUTE url, or refuse to build.

    Passing a bare host ("abc.ngrok-free.dev") is the easy mistake, and it
    fails in the worst possible way: the browser treats it as a RELATIVE path,
    so every call becomes /abc.ngrok-free.dev/api/meta on the static host, 404s,
    and the page quietly serves the snapshot forever. The fallback hides the
    error instead of surfacing it, so it has to be caught here.
    """
    if not url:
        return ""
    url = url.strip().rstrip("/")
    if url.startswith("http://") or url.startswith("https://"):
        if url.startswith("http://") and "localhost" not in url \
                and "127.0.0.1" not in url:
            print(f"  WARNING: {url} is plain http. A site published over "
                  "https cannot call it —\n"
                  "           browsers block mixed content. Use https.")
        return url
    if "://" in url:
        raise SystemExit(f"--api must be http(s), got: {url}")
    fixed = f"https://{url}"
    print(f"  note: --api had no scheme, using {fixed}")
    return fixed


NETLIFY = "netlify.cmd" if sys.platform == "win32" else "netlify"


def deploy(folder: Path) -> None:
    """Push the built folder straight to Netlify — no drag and drop.

    `netlify deploy --prod` uploads only the files whose hashes changed, so a
    re-publish after a pipeline run moves a few KB of JSON rather than the
    whole bundle, and the URL never changes.

    Site identity comes from `.netlify/state.json`, written the first time you
    run `netlify link` (or `netlify deploy` and pick a site). It is committed
    nowhere and holds no secret beyond the site id.
    """
    import os
    if not shutil.which(NETLIFY) and not shutil.which("netlify"):
        raise SystemExit(
            "netlify CLI not found. Install it once:\n"
            "    npm install -g netlify-cli\n"
            "    netlify login\n"
            f"    cd {folder.parent} ; netlify link      # pick your site\n"
            "then re-run with --deploy.")

    linked = (folder.parent / ".netlify" / "state.json").exists() \
        or (HERE.parent / ".netlify" / "state.json").exists() \
        or bool(os.environ.get("NETLIFY_SITE_ID"))
    if not linked:
        print("  note: no linked Netlify site found; the CLI will ask which "
              "site to deploy to.")

    print("4/4  deploying to Netlify")
    cmd = [NETLIFY if shutil.which(NETLIFY) else "netlify",
           "deploy", "--prod", "--dir", str(folder)]
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=folder.parent, text=True)
    if r.returncode != 0:
        raise SystemExit("netlify deploy failed (see output above)")
    print("\ndeployed. Same URL as before — teammates just refresh.")


def run(cmd: list[str], cwd: Path) -> None:
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + "\n" + r.stderr[-3000:] + "\n")
        raise SystemExit(f"failed: {' '.join(cmd)}")


def main() -> None:
    import os
    api_base = ""
    for i, a in enumerate(sys.argv[1:]):
        if a.startswith("--api="):
            api_base = a.split("=", 1)[1].rstrip("/")
        elif a == "--api" and i + 2 <= len(sys.argv) - 1:
            api_base = sys.argv[i + 2].rstrip("/")
    api_base = api_base or os.environ.get("FLEXTRADE_API_BASE", "").rstrip("/")
    api_base = _normalise_api(api_base)

    print("1/3  exporting fresh JSON artifacts")
    run([sys.executable, "export_web.py", "--static"], HERE)

    print("2/3  building the client in static mode")
    # VITE_STATIC=1 switches the API layer to ./data/*.json and the router to
    # hash mode; base=./ makes the bundle work from any sub-path, including a
    # plain file:// open. VITE_API_BASE, when given, is the tunnel URL the
    # published page tries FIRST — it goes live whenever that laptop is on and
    # falls back to the bundled snapshot when it is not.
    env_build = [NPM, "run", "build", "--",
                 "--outDir", "dist-static", "--base", "./"]
    env = {**os.environ, "VITE_STATIC": "1"}
    if api_base:
        env["VITE_API_BASE"] = api_base
        print(f"  live backend baked in: {api_base}")
    else:
        print("  no --api=<url> given: snapshot-only build "
              "(a live URL can still be added at view time with ?api=...)")
    print(f"  $ {' '.join(env_build)}   (VITE_STATIC=1)")
    r = subprocess.run(env_build, cwd=CLIENT, text=True, capture_output=True,
                       env=env)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + "\n" + r.stderr[-3000:] + "\n")
        raise SystemExit("client build failed")

    print("3/3  copying JSON artifacts into the bundle")
    data_dir = OUT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(WEB_JSON.glob("*.json")):
        shutil.copy2(f, data_dir / f.name)
        n += 1

    # keep the snapshot out of search results until the team decides otherwise
    (OUT_DIR / "robots.txt").write_text("User-agent: *\nDisallow: /\n")

    # A .nojekyll file stops GitHub Pages from swallowing paths that begin with
    # an underscore; harmless everywhere else.
    (OUT_DIR / ".nojekyll").write_text("")

    size_mb = sum(p.stat().st_size for p in OUT_DIR.rglob("*") if p.is_file()) / 1e6
    print()
    print(f"built {OUT_DIR}")
    print(f"  {n} JSON artifacts · {size_mb:.1f} MB total")
    print(f"  snapshot taken {datetime.now():%Y-%m-%d %H:%M}")
    mode = (f"HYBRID - live when {api_base} is up, snapshot when not"
            if api_base else "SNAPSHOT ONLY")
    print(f"  mode: {mode}")

    if "--deploy" in sys.argv:
        print()
        deploy(OUT_DIR)
        return
    print()
    print("check it locally first:")
    print(f"  cd {OUT_DIR} && python -m http.server 8000   ->  localhost:8000")
    print()
    print("then publish (any ONE of these, all free):")
    print("  netlify:    drag the dist-static folder onto https://app.netlify.com/drop")
    print("  cloudflare: https://pages.cloudflare.com -> Direct Upload")
    print("  gh pages:   push dist-static to a gh-pages branch, enable Pages")
    if api_base:
        print()
        print("remember to let the deployed site call your laptop:")
        print("  $env:ALLOWED_ORIGINS = \"https://<your-site>\"; node server/index.js")


if __name__ == "__main__":
    main()
