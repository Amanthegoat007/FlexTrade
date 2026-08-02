"""Uttar Pradesh SLDC — the second state with a real first-party feed.

Everything outside Delhi has been MERIT until now: one instantaneous demand
number per state, polled every 15 minutes. UP's own load despatch centre
publishes considerably more, as clean JSON, with no scraping at all:

    https://www.upsldc.org/assets/dataset/dynamic-data.json

    Demand Met                 24,295 MW
    Schedule (Inter State)     10,807 MW
    Drawl (Inter State)        11,463 MW
    Deviation (MW)                657 MW
    Total Intra State Generation 12,831 MW
    UPRVUNL Generation          4,232 MW
    Maximum / Minimum Demand Met  with the clock time they occurred
    Last Updated

Why this matters more than "one more state":

  * UP is the LARGEST load in India — ~31 GW peak, roughly 5x Delhi. It is the
    single biggest addressable book in the country.
  * It publishes SCHEDULE, DRAWAL and DEVIATION. That is the DSM triplet, and
    it is the thing we have been unable to observe anywhere except Delhi.
    models/dsm_forecast.py had to price a hypothetical generator because no
    real schedule-vs-actual series existed; this is one.
  * It publishes intra-state generation separately from inter-state drawal, so
    the state's own supply stack is visible rather than inferred.

Same discipline as every other fetcher here: a plausibility guard, because a
wrong number is worse than a crash. UP's all-time peak is ~31 GW and its
overnight trough rarely goes below ~9 GW, so anything outside 5-40 GW is a
parse failure wearing a number's clothes.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import store  # noqa: E402

URL = "https://www.upsldc.org/assets/dataset/dynamic-data.json"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120",
      "Referer": "https://www.upsldc.org/"}

# UP system limits — peak ~31 GW, overnight trough ~9 GW
DEMAND_MIN_MW, DEMAND_MAX_MW = 5_000.0, 40_000.0

FIELDS = {
    "Demand Met": "demand_met_mw",
    "Schedule (Inter State)": "schedule_mw",
    "Drawl (Inter State )": "drawal_mw",
    "Deviation (MW)": "deviation_mw",
    "Total Intra State Generation": "intra_gen_mw",
    "UPRVUNL Generation": "uprvunl_gen_mw",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS up_live (
    fetched_at TEXT PRIMARY KEY,
    demand_met_mw REAL, schedule_mw REAL, drawal_mw REAL, deviation_mw REAL,
    intra_gen_mw REAL, uprvunl_gen_mw REAL,
    max_demand_mw REAL, max_demand_at TEXT,
    min_demand_mw REAL, min_demand_at TEXT,
    source_updated TEXT, deviation_signed_mw REAL);
"""


def _num(s) -> float | None:
    """First number in a string like '31202 at 23:57' or '24,295'."""
    if s is None:
        return None
    txt = str(s).replace(",", "").strip()
    out, seen = [], False
    for ch in txt:
        if ch.isdigit() or (ch == "." and not seen and out):
            out.append(ch)
            seen = seen or ch == "."
        elif out:
            break
    try:
        return float("".join(out)) if out else None
    except ValueError:
        return None


def _at(s) -> str | None:
    """The 'at HH:MM' half of 'Maximum Demand Met' style values."""
    txt = str(s or "")
    return txt.split("at", 1)[1].strip() if "at" in txt else None


def fetch(retries: int = 3, backoff_s: int = 4) -> dict:
    """Live UP snapshot. Raises on an implausible reading rather than storing it."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(URL, headers=UA, timeout=25)
            r.raise_for_status()
            rows = r.json()
            break
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(backoff_s * (2 ** attempt))
    else:
        raise last

    by_name = {str(x.get("DISPLAY_NAME", "")).strip(): x.get("POINT_VAL")
               for x in rows if isinstance(x, dict)}
    out: dict = {}
    for label, key in FIELDS.items():
        # the feed's own key for drawal carries a stray space; match loosely
        val = by_name.get(label)
        if val is None:
            hit = next((v for k, v in by_name.items()
                        if k.replace(" ", "") == label.replace(" ", "")), None)
            val = hit
        out[key] = _num(val)

    for label, vkey, tkey in (("Maximum Demand Met", "max_demand_mw", "max_demand_at"),
                              ("Minimum Demand Met", "min_demand_mw", "min_demand_at")):
        raw = by_name.get(label)
        out[vkey], out[tkey] = _num(raw), _at(raw)

    out["source_updated"] = next(
        (str(v) for k, v in by_name.items() if "update" in k.lower()), None)

    d = out.get("demand_met_mw")
    if d is None or not (DEMAND_MIN_MW <= d <= DEMAND_MAX_MW):
        raise ValueError(
            f"UP demand {d} MW outside the plausible {DEMAND_MIN_MW:,.0f}-"
            f"{DEMAND_MAX_MW:,.0f} MW band — treating as a parse failure")

    # The published "Deviation (MW)" is an UNSIGNED MAGNITUDE. Confirmed on the
    # first live pull: drawal 10,616 - schedule 10,807 = -191, and the feed
    # reported +190. Sign is exactly what matters for DSM — over-drawal and
    # under-drawal are charged differently — so the signed value is computed
    # here and the published figure is kept only as a cross-check.
    s, dr = out.get("schedule_mw"), out.get("drawal_mw")
    if None not in (s, dr):
        out["deviation_signed_mw"] = dr - s
        out["over_drawing"] = bool(dr > s)
        pub = out.get("deviation_mw")
        if pub is not None and abs(abs(dr - s) - abs(pub)) > max(50.0, 0.05 * abs(pub or 1)):
            # magnitudes disagree too — the feed has changed shape under us
            out["deviation_check"] = (
                f"|drawal-schedule|={abs(dr - s):.0f} but feed says {abs(pub):.0f}")
    return out


def poll() -> dict:
    """Fetch and persist one snapshot."""
    snap = fetch()
    with store.connect() as con:
        con.executescript(_SCHEMA)
        cols = [c for c in (
            "demand_met_mw", "schedule_mw", "drawal_mw", "deviation_mw",
            "intra_gen_mw", "uprvunl_gen_mw", "max_demand_mw", "max_demand_at",
            "min_demand_mw", "min_demand_at", "source_updated",
            "deviation_signed_mw") if c in snap]
        con.execute(
            f"INSERT OR REPLACE INTO up_live (fetched_at,{','.join(cols)}) "
            f"VALUES ({','.join(['?'] * (len(cols) + 1))})",
            [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + [snap.get(c) for c in cols])
    store.log_fetch("upsldc", True, f"demand {snap['demand_met_mw']:.0f} MW")
    return snap


if __name__ == "__main__":
    s = poll()
    print("UP SLDC live snapshot")
    for k, v in s.items():
        print(f"  {k:20s} {v}")
    with store.connect() as con:
        n = con.execute("SELECT COUNT(*) FROM up_live").fetchone()[0]
    print(f"\nstored — up_live now holds {n} snapshots")
