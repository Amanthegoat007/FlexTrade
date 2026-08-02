"""Karnataka SLDC (KPTCL) — third state with a first-party feed.

Karnataka matters for a specific reason: it carries India's largest solar
fleet, so its net-load shape is the one most distorted by RE, and that is
exactly the shape a battery is paid to smooth.

Unlike UP (clean JSON) this is a plain HTML page and the values have to be
pulled out of the markup. That is a lower-trust route, so the numbers are
CROSS-CHECKED against MERIT — an entirely independent source for the same
state — before anything is stored.

That check is not ceremony. Four state SLDC portals were scraped the same
way on 2 Aug 2026 and only this one survived:

    state   scraped     MERIT      verdict
    KA      10,132 MW   10,023 MW  1.1% apart  -> accepted
    MH         400 MW   24,211 MW  nonsense    -> rejected
    PB      40,109 MW   15,707 MW  ~2.6x out   -> rejected
    WB       2,621 MW   10,248 MW  ~4x out     -> rejected

Every one of those three would have looked like a plausible number sitting
alone in a dashboard cell. Scraping a label off an HTML page finds A number,
not necessarily THE number, and the only defence is a second source.

What KPTCL adds beyond MERIT: inter-state DRAWAL and grid FREQUENCY, neither
of which MERIT publishes per state.
"""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import store  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://kptclsldc.in/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}

# Karnataka system limits: peak ~17 GW, overnight trough ~5 GW
DEMAND_MIN_MW, DEMAND_MAX_MW = 3_000.0, 20_000.0
# how far the scrape may sit from MERIT before we refuse to store it
CROSSCHECK_TOL_PCT = 12.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ka_live (
    fetched_at TEXT PRIMARY KEY,
    demand_mw REAL, drawal_mw REAL, frequency_hz REAL,
    merit_demand_mw REAL, crosscheck_pct REAL);
"""


def _plain(html: str) -> str:
    t = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    t = re.sub(r"<style.*?</style>", " ", t, flags=re.S | re.I)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " | ", t))


def _near(text: str, label: str) -> float | None:
    m = re.search(re.escape(label) + r"[^0-9\-]{0,60}(-?[0-9][0-9,]{1,8}(?:\.[0-9]+)?)",
                  text, re.I)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _merit_demand() -> float | None:
    """Latest MERIT reading for Karnataka — the independent check."""
    try:
        with store.connect() as con:
            row = con.execute(
                "SELECT demand_mw FROM state_live WHERE code='KA' "
                "ORDER BY fetched_at DESC LIMIT 1").fetchone()
        return float(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def fetch(retries: int = 3, backoff_s: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(URL, headers=UA, timeout=30, verify=False)
            r.raise_for_status()
            text = _plain(r.text)
            break
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(backoff_s * (2 ** attempt))
    else:
        raise last

    out = {
        "demand_mw": _near(text, "Demand"),
        "drawal_mw": _near(text, "Drawal"),
        "frequency_hz": _near(text, "Frequency"),
    }

    d = out["demand_mw"]
    if d is None or not (DEMAND_MIN_MW <= d <= DEMAND_MAX_MW):
        raise ValueError(
            f"KA demand {d} MW outside the plausible {DEMAND_MIN_MW:,.0f}-"
            f"{DEMAND_MAX_MW:,.0f} MW band — treating as a parse failure")

    f = out["frequency_hz"]
    if f is not None and not (47.0 <= f <= 53.0):
        out["frequency_hz"] = None       # picked up something that wasn't frequency

    # the part that actually earns trust
    merit = _merit_demand()
    out["merit_demand_mw"] = merit
    if merit:
        gap = abs(d - merit) / merit * 100
        out["crosscheck_pct"] = round(gap, 2)
        if gap > CROSSCHECK_TOL_PCT:
            raise ValueError(
                f"KA scrape {d:,.0f} MW disagrees with MERIT {merit:,.0f} MW by "
                f"{gap:.1f}% (tolerance {CROSSCHECK_TOL_PCT}%) — refusing to store. "
                "The page layout has probably changed and the regex is now reading "
                "a different cell.")
    else:
        out["crosscheck_pct"] = None
    return out


def poll() -> dict:
    snap = fetch()
    with store.connect() as con:
        con.executescript(_SCHEMA)
        con.execute(
            "INSERT OR REPLACE INTO ka_live (fetched_at, demand_mw, drawal_mw, "
            "frequency_hz, merit_demand_mw, crosscheck_pct) VALUES (?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), snap["demand_mw"],
             snap["drawal_mw"], snap["frequency_hz"], snap["merit_demand_mw"],
             snap["crosscheck_pct"]))
    store.log_fetch("kptcl", True,
                    f"demand {snap['demand_mw']:.0f} MW, "
                    f"crosscheck {snap['crosscheck_pct']}%")
    return snap


if __name__ == "__main__":
    s = poll()
    print("Karnataka SLDC (KPTCL) live snapshot")
    for k, v in s.items():
        print(f"  {k:18s} {v}")
    with store.connect() as con:
        n = con.execute("SELECT COUNT(*) FROM ka_live").fetchone()[0]
    print(f"\nstored — ka_live now holds {n} snapshots")
