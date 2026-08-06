"""Collector for CI — appends snapshots to CSV instead of SQLite.

The laptop collector (poll_states.py) writes to SQLite, which cannot be shared
through a git repository without constant binary conflicts. This writes the
same snapshots as one CSV per source per UTC month, which merges cleanly, is
diffable, and can be pulled back into the database by ingest/merge_collected.py.

It exists because the laptop has a structural collection hole — hours 05:00 to
07:00 have never once been captured in 14 days — and those blocks are gone for
good. Everything here is snapshot-only upstream, so the only fix is to collect
from somewhere that does not sleep.

Deliberately dependency-light: requests and pandas, nothing from the model
stack, so a CI runner needs no LightGBM, no SQLite file and no secrets.
"""
from __future__ import annotations

import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "collected"
STATUS = ROOT / "data" / "_collector_status.json"   # outside the committed dir
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}
TIMEOUT = 45


def _write(source: str, rows: list[dict]) -> int:
    """Append rows to data/collected/<source>-YYYY-MM.csv."""
    if not rows:
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc)
    path = OUT / f"{source}-{stamp:%Y-%m}.csv"
    cols = list(rows[0].keys())
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------- sources --

# MERIT's own state codes, mirrored from ingest/states.py so this file stays
# importable on a bare CI runner with no model stack installed
MERIT_CODES = {
    "DL": "DL", "HR": "HRN", "PB": "PNB", "RJ": "RJ", "UP": "UP", "UK": "UTK",
    "HP": "HP", "CH": "CHG", "MH": "MHA", "GJ": "GJT", "MP": "MPD", "CT": "CTG",
    "GA": "GOA", "TN": "TND", "KA": "KRT", "TG": "TLG", "AP": "AP", "KL": "KRL",
    "WB": "BGL", "BR": "BHR", "OD": "ODI", "JH": "JHK", "AS": "ASM",
}


def merit() -> int:
    """23-state demand / own generation / import, Ministry of Power.

    One POST per state — MERIT has no all-states endpoint that carries the
    import breakdown, and import is the field the whole State Stress index
    rests on. A single state failing must not lose the other 22.
    """
    import urllib3
    urllib3.disable_warnings()
    ts, rows = _now(), []
    for code, merit_code in MERIT_CODES.items():
        try:
            r = requests.post(
                "https://meritindia.in/StateWiseDetails/BindCurrentStateStatus",
                json={"StateCode": merit_code},
                headers={**UA, "Content-Type": "application/json; charset=utf-8"},
                timeout=TIMEOUT, verify=False)
            r.raise_for_status()
            row = (r.json() or [{}])[0]
            rows.append({
                "fetched_at_utc": ts, "code": code,
                "demand_mw": row.get("Demand"),
                "own_gen_mw": row.get("ISGS"),
                "import_mw": row.get("ImportData"),
            })
        except Exception:
            continue
    return _write("merit_state", rows)


def upsldc() -> int:
    """UP: schedule, drawal and deviation — the DSM triplet."""
    r = requests.get("https://www.upsldc.org/assets/dataset/dynamic-data.json",
                     headers={**UA, "Referer": "https://www.upsldc.org/"},
                     timeout=TIMEOUT)
    r.raise_for_status()
    by = {str(x.get("DISPLAY_NAME", "")).strip(): x.get("POINT_VAL")
          for x in r.json() if isinstance(x, dict)}

    def num(label):
        v = by.get(label)
        if v is None:
            v = next((y for k, y in by.items()
                      if k.replace(" ", "") == label.replace(" ", "")), None)
        if v is None:
            return None
        txt, out, seen = str(v).replace(",", "").strip(), [], False
        for ch in txt:
            if ch.isdigit() or (ch == "." and not seen and out):
                out.append(ch)
                seen = seen or ch == "."
            elif out:
                break
        return float("".join(out)) if out else None

    sched, drawal = num("Schedule (Inter State)"), num("Drawl (Inter State )")
    return _write("upsldc", [{
        "fetched_at_utc": _now(),
        "demand_met_mw": num("Demand Met"),
        "schedule_mw": sched, "drawal_mw": drawal,
        # the published "Deviation" is an unsigned magnitude; sign is what DSM
        # charges on, so it is computed here rather than trusted
        "deviation_signed_mw": (drawal - sched) if (sched is not None
                                                    and drawal is not None) else None,
        "intra_gen_mw": num("Total Intra State Generation"),
        "source_updated": next((str(v) for k, v in by.items()
                                if "update" in k.lower()), None),
    }])


def area_price() -> int:
    """35-area clearing price — the only state-level price signal available."""
    s = requests.Session()
    s.headers.update({**UA, "Referer": "https://vidyutpravah.in/",
                      "X-Requested-With": "XMLHttpRequest"})
    s.get("https://vidyutpravah.in/", timeout=TIMEOUT)
    rows = s.post("https://vidyutpravah.in/PXDashboard/BindStatePricesFromJS",
                  data={}, timeout=TIMEOUT).json()
    blk = s.post("https://vidyutpravah.in/PXDashboard/BindCurrentDateTimeForJson",
                 data={}, timeout=TIMEOUT).json()
    b = (blk or [{}])[0]
    ts = _now()
    return _write("area_price", [{
        "fetched_at_utc": ts, "block_from": b.get("FromTime"),
        "block_date": b.get("CurrentDate"),
        "area": str(x.get("StateCode")), "acp_rs_mwh": x.get("ACP"),
    } for x in rows if x.get("StateCode") is not None])


SOURCES = [("merit", merit), ("upsldc", upsldc), ("area_price", area_price)]

HOSTS = ["meritindia.in", "www.upsldc.org", "vidyutpravah.in"]


def probe() -> None:
    """Report whether each upstream is even reachable from THIS machine.

    Exists because the collector ran green on GitHub for four consecutive
    scheduled runs while capturing nothing, and there was no way to tell a dead
    upstream from a runner that cannot reach Indian infrastructure at all.
    These are government and utility sites; several are known to treat foreign
    datacentre IPs differently from Indian residential ones, and the laptop
    (which works) and the runner (which does not) differ in exactly that way.

    Printed on every run so the log answers the question by itself.
    """
    import socket
    for host in HOSTS:
        try:
            ip = socket.gethostbyname(host)
        except Exception as e:
            print(f"  probe {host:20s} DNS FAIL  {type(e).__name__}")
            continue
        try:
            r = requests.get(f"https://{host}/", headers=UA, timeout=20,
                             verify=False, allow_redirects=True)
            print(f"  probe {host:20s} {ip:15s} HTTP {r.status_code} "
                  f"{len(r.content):,}B")
        except Exception as e:
            print(f"  probe {host:20s} {ip:15s} UNREACHABLE "
                  f"{type(e).__name__}: {str(e)[:70]}")


def main() -> int:
    import urllib3
    urllib3.disable_warnings()
    print(f"{_now()}  reachability from this runner:")
    probe()
    ok, rows_total, detail = 0, 0, {}
    for name, fn in SOURCES:
        try:
            n = fn()
            print(f"{_now()}  {name:11s} OK   {n} rows")
            detail[name] = n
            rows_total += n
            ok += 1
        except Exception as e:
            # one dead source must never cost us the others
            print(f"{_now()}  {name:11s} FAIL {type(e).__name__}: {str(e)[:200]}")
            detail[name] = f"{type(e).__name__}: {str(e)[:200]}"
    print(f"{_now()}  ---> {ok}/{len(SOURCES)} sources captured, "
          f"{rows_total} rows written")

    # Machine-readable outcome so the workflow can distinguish "upstream quiet"
    # from "collector produced nothing", which a green checkmark cannot.
    #
    # Deliberately NOT inside data/collected: the workflow stages that whole
    # directory, and a status file carrying a timestamp changes on every tick —
    # so it would always look like new data, always commit, and defeat the very
    # empty-tick check it exists to feed.
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "at_utc": _now(), "sources_ok": ok, "sources_total": len(SOURCES),
        "rows": rows_total, "detail": detail}, indent=1), encoding="utf-8")

    if ok == 0:
        print("FATAL: every source failed. That is not a quiet upstream — it is "
              "a broken collector or a runner that cannot reach these hosts. "
              "Failing loudly so this does not sit green and empty.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
