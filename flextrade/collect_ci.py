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


def _write_new(source: str, rows: list[dict], keys: tuple[str, ...]) -> int:
    """Append only rows whose key tuple is not already stored this month.

    NPP serves a rolling ~4.1 hour window on every request (measured 2026-08-18:
    64 timestamps, median gap 4.0 min). Polling every 15 minutes therefore
    returns most of the same 4-minute blocks around sixteen times over. Writing
    them all would put ~5,700 rows a day into a file that should hold 360, so
    this reads back what is already stored and keeps only what is genuinely new.

    At a month boundary the new file is empty and up to one window may be
    rewritten into it. ingest/merge_collected.py dedupes on the same keys, so
    that costs a few duplicate rows and loses nothing.
    """
    if not rows:
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{source}-{datetime.now(timezone.utc):%Y-%m}.csv"
    seen = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            for old in csv.DictReader(fh):
                seen.add(tuple(str(old.get(k, "")) for k in keys))
    fresh = [r for r in rows
             if tuple(str(r.get(k, "")) for k in keys) not in seen]
    return _write(source, fresh)


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


def _upsldc_legacy() -> int:
    """Fallback UP parser: display labels, string-matched, units walked off."""
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


def upsldc() -> int:
    """UP: schedule, drawal, deviation, generation split and frequency.

    Uses assets/dataset/real-time-summary.json, which returns typed numeric
    fields: SCHEDULE_MW, DRAWL_MW, OD_UD, DEMAND_MW, TOTAL_SSGS_MW, the
    generation split (UP thermal / IPP thermal / hydro / cogen / solar),
    FREQUENC_HZ and DEVIATION_RATE_PAISE_PER_UNIT.

    The older dynamic-data.json carries schedule and drawal behind display
    labels that have to be string-matched — one of them is literally
    "Drawl (Inter State )", stray space included — with values that then have
    to be character-walked to strip their units. Every one of those is a
    silent-breakage surface the typed endpoint simply does not have, so the
    legacy parser is kept only as a fallback for when this one is down.

    DEVIATION_RATE_PAISE_PER_UNIT is the published DSM rate. It reads null
    outside deviation events, so it is stored as-is rather than defaulted.
    """
    try:
        r = requests.get("https://upsldc.org/assets/dataset/real-time-summary.json",
                         headers={**UA, "Referer": "https://www.upsldc.org/"},
                         timeout=TIMEOUT, verify=False)
        r.raise_for_status()
        d = (r.json() or [{}])[0]
        if not d:
            raise ValueError("empty payload")
    except Exception as e:
        print(f"    upsldc typed endpoint failed ({type(e).__name__}), "
              f"falling back to the legacy parser")
        return _upsldc_legacy()

    sched, drawal = d.get("SCHEDULE_MW"), d.get("DRAWL_MW")
    return _write("upsldc", [{
        "fetched_at_utc": _now(),
        "demand_met_mw": d.get("DEMAND_MW"),
        "schedule_mw": sched, "drawal_mw": drawal,
        # the published OD_UD is already signed; recompute anyway so the two
        # can be cross-checked and a sign-convention change cannot pass silently
        "deviation_signed_mw": (drawal - sched) if (sched is not None
                                                    and drawal is not None) else None,
        "deviation_published_mw": d.get("OD_UD"),
        "intra_gen_mw": d.get("TOTAL_SSGS_MW"),
        "up_thermal_mw": d.get("UP_THERMAL_GENERATION_MW"),
        "ipp_thermal_mw": d.get("IPP_THERMAL_GENERATION_MW"),
        "up_hydro_mw": d.get("UP_HYDRO_GENERATION_MW"),
        "cogen_cpp_mw": d.get("COGEN_CPP_GENERATION_MW"),
        "re_solar_mw": d.get("RE_SOLAR_GENERATION_MW"),
        "frequency_hz": d.get("FREQUENC_RAW") or d.get("FREQUENC_HZ"),
        "dsm_rate_paise_kwh": d.get("DEVIATION_RATE_PAISE_PER_UNIT"),
        "source_updated": None,
    }])


def pstcl() -> int:
    """Punjab: frequency, schedule, drawal and deviation — typed JSON.

    Returns HTTP 500 intermittently (observed twice inside ten minutes on
    2026-08-18, recovering on the next attempt), so a single failure means
    nothing and is retried rather than reported as a dead source.
    """
    last = None
    for _ in range(3):
        try:
            r = requests.get("https://sldcapi.pstcl.org/wsDataService.asmx/dynamicData",
                             headers={**UA, "Referer": "https://www.pstcl.org/"},
                             timeout=TIMEOUT, verify=False)
            r.raise_for_status()
            d = r.json()
            sched, drawal = d.get("scheduleMW"), d.get("drawalMW")
            return _write("pstcl", [{
                "fetched_at_utc": _now(),
                "source_updated": d.get("updateDate"),
                "frequency_hz": d.get("frequencyHz"),
                "demand_met_mw": d.get("loadMW"),
                "schedule_mw": sched, "drawal_mw": drawal,
                "deviation_signed_mw": (drawal - sched) if (sched is not None
                                                            and drawal is not None) else None,
                "deviation_published_mw": d.get("odUD"),
            }])
        except Exception as e:
            last = e
    raise last


NPP_DASH = "https://npp.gov.in/dashBoard/"


def npp_national() -> int:
    """All-India demand met and the six-fuel generation mix, 4-minute blocks.

    This is the highest-resolution national signal any source in this project
    publishes, and the fuel mix is the supply-side driver the price model does
    not yet see: solar collapsing out of the stack into the evening ramp is the
    mechanism that pins DAM blocks at the Rs 10,000 cap. Coal days-of-stock,
    the one supply-side feature already in the model, correlates -0.457 with
    cap share; this is the same class of signal at 4-minute resolution.

    Both endpoints serve a ROLLING ~4.1 hour window and neither has a history
    parameter. Anything older than that window is gone permanently, for us and
    for anyone else, so this cannot be backfilled later — only collected now.
    """
    out = 0
    for endpoint, source in (("demandmet1chartdata", "npp_demand"),
                             ("demandmet2chartdata", "npp_fuelmix")):
        # This host drops TCP connections intermittently, from everywhere.
        # Measured 2026-08-18 from an Indian residential line: four attempts
        # back to back gave 21.07s timeout, 21.05s, 21.05s, then 0.52s and
        # 7,201 bytes. It either answers at once or never, so a long timeout
        # only waits longer for the same failure — short and repeated wins.
        r = None
        for _ in range(4):
            try:
                r = requests.get(NPP_DASH + endpoint, headers=UA,
                                 timeout=6, verify=False)
                r.raise_for_status()
                break
            except Exception:
                r = None
        if r is None:
            raise RuntimeError(f"npp {endpoint} unreachable after 4 attempts")
        rows = []
        for x in r.json():
            stamp = x.get("updated_on")
            if not stamp:
                continue
            rows.append({
                "ts_utc": datetime.fromtimestamp(stamp / 1000, timezone.utc)
                                  .strftime("%Y-%m-%d %H:%M:%S"),
                "series": x.get("name_of_data"),
                "value_mw": x.get("value_of_data"),
                "fetched_at_utc": _now(),
            })
        # keyed on (ts, series): demandmet2 carries six fuels per timestamp and
        # their stamps drift a few seconds apart, so a max-timestamp filter
        # would drop stragglers
        out += _write_new(source, rows, ("ts_utc", "series"))
    return out


SOURCES = [
    ("merit", merit),
    ("npp_national", npp_national),   # 4-min national demand + fuel mix
    ("upsldc", upsldc),
    ("pstcl", pstcl),
    ("area_price", area_price),
]

HOSTS = ["meritindia.in", "npp.gov.in", "upsldc.org",
         "sldcapi.pstcl.org", "vidyutpravah.in"]


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
