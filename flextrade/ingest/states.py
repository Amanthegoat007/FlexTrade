"""Multi-state expansion layer.

Delhi is one instance of a general pattern, not a hardcoded destination.
This module has two tiers of state coverage, deliberately kept separate
because they have very different reliability:

TIER 1 — Northern Region snapshot (zero extra scraping cost)
--------------------------------------------------------------
Delhi SLDC's own real-time page already publishes a "States Drawl" table
covering the whole Northern Regional grid: Chandigarh, Haryana, Himachal
Pradesh, Jammu & Kashmir, Punjab, Rajasthan, Uttarakhand and Uttar
Pradesh — Schedule / Drawl / OD-UD / Load in MW, refreshed on the same
page we already fetch for Delhi's own numbers. This gives 8 more states
for free, at snapshot (not 5-min-history) resolution. Uttar Pradesh alone
carries ~29 GW of load right now — roughly 4x Delhi's — which is a far
bigger addressable market than Delhi on its own.

TIER 2 — MERIT national layer (verified 2026-07-24, all-India)
--------------------------------------------------------------
meritindia.in (Ministry of Power's MERIT portal) exposes a clean JSON
endpoint, POST /StateWiseDetails/BindCurrentStateStatus {"StateCode":..},
returning current Demand Met / Own Generation / Import per state, plus
/Dashboard/BindAllIndiaMap with the national position (demand met and
the full fuel-mix generation split). 23 state codes were discovered by
crawling each /state-data/<slug> page for its hidden StateCode field and
verified with live calls; values cross-check against our independent
feeds (Delhi ~6.4 GW vs our SLDC number, Gujarat 16,101 vs sldcguj.com's
16,077 fetched a minute apart). ~30 s upstream refresh.

TIER 3 — Per-state SLDC adapters (deep data, added one at a time)
--------------------------------------------------------------
The state's own SLDC gives depth MERIT can't (frequency, DSM rate,
generation mix, RE plant telemetry). Verified so far: Delhi (reference,
full history) and Gujarat (sldcguj.com homepage server-renders live
frequency + "Gujarat Catered" demand + DAM rate — regex-scraped, no
login). Rajasthan's endpoints are fully mapped (read-sftp JSON: tag
03046004=freq, 03046001=DSM rate paise/unit, 03046008=load,
03046002/7=NR schedule/drawal, 03046006=OD/UD, 03046009=generation —
currently returns 500 upstream, their own homepage widget is equally
broken; plus /view-realtime-data/show, a working DataTables JSON of
~151 RE plants' MW by substation whose upstream feed shows stale
timestamps). Maharashtra publishes its public SCADA overview only as a
JPEG image (mahasldc.in/scada/reports/report2), real data behind an HO
login; WRLDC's index.aspx/GetLiveData webmethod answers but returns
null without a browser session; SRLDC's indexPageDataInEvery5min 500s;
vidyutpravah.in timed out repeatedly. All of that is recorded honestly
in the registry notes below — a state is "verified" only when a live
fetch returned believable numbers in this codebase.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import requests
import urllib3

from . import sldc, store

# some state-government sites (Gujarat, Rajasthan) present certificate
# chains requests can't verify; we fetch them with verify=False and
# silence only that specific warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

NORTHERN_REGION_URL = "https://www.delhisldc.org/Redirect.aspx?Loc=0804"


def fetch_northern_region_snapshot() -> pd.DataFrame:
    """Live snapshot of the Northern Grid neighbour states, parsed from
    the same page sldc.fetch_realtime() reads for Delhi. Returns one row
    per state: schedule_mw, drawl_mw, od_ud_mw, load_mw."""
    import io
    html = requests.get(NORTHERN_REGION_URL, headers=UA, timeout=25).text
    tables = pd.read_html(io.StringIO(html))
    t = next((tb for tb in tables if len(tb) and str(tb.iloc[0, 0]).strip() == "STATE"),
             None)
    if t is None:
        raise ValueError("States Drawl table not found on Delhi SLDC page")
    t.columns = ["state", "schedule_mw", "drawl_mw", "od_ud_mw", "load_mw"]
    t = t.iloc[1:].reset_index(drop=True)
    for c in ["schedule_mw", "drawl_mw", "od_ud_mw", "load_mw"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["fetched_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    return t


def get_northern_region_snapshot():
    """Live fetch with cache fallback. Returns (df, meta)."""
    try:
        df = fetch_northern_region_snapshot()
        with store.connect() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS northern_region_snapshot
                (fetched_at TEXT, state TEXT, schedule_mw REAL, drawl_mw REAL,
                 od_ud_mw REAL, load_mw REAL)""")
            df.to_sql("northern_region_snapshot", con, if_exists="append", index=False)
        store.log_fetch("northern_region", True)
        return df, {"live": True, "asof": pd.Timestamp.now()}
    except Exception as e:
        store.log_fetch("northern_region", False, str(e))
        with store.connect() as con:
            try:
                last_ts = pd.read_sql(
                    "SELECT MAX(fetched_at) t FROM northern_region_snapshot", con
                ).iloc[0, 0]
                cached = pd.read_sql(
                    "SELECT * FROM northern_region_snapshot WHERE fetched_at=?",
                    con, params=(last_ts,)) if last_ts else pd.DataFrame()
            except Exception:
                cached = pd.DataFrame()
        return cached, {"live": False, "asof": store.last_good_fetch("northern_region"),
                        "error": str(e)}


# =========================================================================
# Tier 2 — MERIT national layer (meritindia.in, Ministry of Power)
# =========================================================================

MERIT_BASE = "https://meritindia.in"

# state code -> (merit StateCode, display name, grid region)
# every code below was discovered from the state's own MERIT page and
# verified with a live BindCurrentStateStatus call on 2026-07-24
MERIT_CODES = {
    "DL": ("DL", "Delhi", "Northern"),
    "HR": ("HRN", "Haryana", "Northern"),
    "PB": ("PNB", "Punjab", "Northern"),
    "RJ": ("RJ", "Rajasthan", "Northern"),
    "UP": ("UP", "Uttar Pradesh", "Northern"),
    "UK": ("UTK", "Uttarakhand", "Northern"),
    "HP": ("HP", "Himachal Pradesh", "Northern"),
    "CH": ("CHG", "Chandigarh", "Northern"),
    "MH": ("MHA", "Maharashtra", "Western"),
    "GJ": ("GJT", "Gujarat", "Western"),
    "MP": ("MPD", "Madhya Pradesh", "Western"),
    "CT": ("CTG", "Chhattisgarh", "Western"),
    "GA": ("GOA", "Goa", "Western"),
    "TN": ("TND", "Tamil Nadu", "Southern"),
    "KA": ("KRT", "Karnataka", "Southern"),
    "TG": ("TLG", "Telangana", "Southern"),
    "AP": ("AP", "Andhra Pradesh", "Southern"),
    "KL": ("KRL", "Kerala", "Southern"),
    "WB": ("BGL", "West Bengal", "Eastern"),
    "BR": ("BHR", "Bihar", "Eastern"),
    "OD": ("ODI", "Odisha", "Eastern"),
    "JH": ("JHK", "Jharkhand", "Eastern"),
    "AS": ("ASM", "Assam", "North-Eastern"),
}


def _num(s) -> float | None:
    """'27,825' -> 27825.0; '-'/None -> None."""
    if s is None:
        return None
    s = str(s).replace(",", "").strip()
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_merit_state(code: str) -> dict:
    """Current Demand Met / Own Generation / Import for one state (MW)."""
    merit_code, name, region = MERIT_CODES[code]
    r = requests.post(f"{MERIT_BASE}/StateWiseDetails/BindCurrentStateStatus",
                      json={"StateCode": merit_code},
                      headers={**UA, "Content-Type": "application/json; charset=utf-8"},
                      timeout=25, verify=False)
    r.raise_for_status()
    row = r.json()[0]
    return {"code": code, "name": name, "grid_region": region,
            "demand_mw": _num(row.get("Demand")),
            "own_gen_mw": _num(row.get("ISGS")),
            "import_mw": _num(row.get("ImportData"))}


def fetch_merit_states(codes: list[str] | None = None) -> pd.DataFrame:
    """All (or selected) states from MERIT, fetched concurrently (the
    upstream answers each call in ~1 s; 8 workers keeps the whole batch
    under ~4 s for the live path). Skips states that fail rather than
    failing the batch; per-state errors are in df.attrs['errors']."""
    from concurrent.futures import ThreadPoolExecutor
    codes = codes or list(MERIT_CODES)
    rows, errors = [], {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for c, fut in [(c, ex.submit(fetch_merit_state, c)) for c in codes]:
            try:
                rows.append(fut.result())
            except Exception as e:
                errors[c] = str(e)[:80]
    order = {c: i for i, c in enumerate(codes)}
    rows.sort(key=lambda r: order[r["code"]])
    df = pd.DataFrame(rows)
    df.attrs["errors"] = errors
    return df


_MERIT_MAP_FIELDS = [
    ("DEMAND", "demand_met_mw"), ("THERMAL", "thermal_mw"), ("GAS", "gas_mw"),
    ("NUCLEAR", "nuclear_mw"), ("HYDRO", "hydro_mw"),
    ("RENEWABLE", "renewable_mw"), ("STORAGE", "storage_mw"),
    ("OTHER", "other_mw"), ("TRANS NATIONAL", "transnational_mw"),
]


def fetch_merit_national() -> dict:
    """All-India current position: demand met + generation by fuel (MW).
    Parsed from the BindAllIndiaMap HTML fragment (labelled value tiles)."""
    r = requests.get(f"{MERIT_BASE}/Dashboard/BindAllIndiaMap",
                     headers=UA, timeout=25, verify=False)
    r.raise_for_status()
    html = r.text
    out = {}
    for label, key in _MERIT_MAP_FIELDS:
        # anchor on the tile's title element so words like "storage" inside
        # tooltip prose can't hijack the match
        pat = (r'gen_title_sec"?[^>]*>\s*'
               + label.replace(" ", r"(?:\s|<br\s*/?>)+")
               + r"(?:\s|<br\s*/?>)*[A-Z]*.*?counter\">\s*([\d,\-\s]+)</span>")
        m = re.search(pat, html, re.S | re.I)
        out[key] = _num(m.group(1)) if m else None
    if out.get("demand_met_mw") is None:
        raise ValueError("MERIT national fragment did not parse")
    return out


def get_india_snapshot():
    """Live MERIT fetch (national + all states) with cache fallback.
    Returns ({'national': dict, 'states': DataFrame}, meta)."""
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        nat = fetch_merit_national()
        df = fetch_merit_states()
        if not len(df):
            raise ValueError(f"no states parsed; errors={df.attrs.get('errors')}")
        df["fetched_at"] = ts
        with store.connect() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS state_live
                (fetched_at TEXT, code TEXT, name TEXT, grid_region TEXT,
                 demand_mw REAL, own_gen_mw REAL, import_mw REAL)""")
            df[["fetched_at", "code", "name", "grid_region",
                "demand_mw", "own_gen_mw", "import_mw"]].to_sql(
                "state_live", con, if_exists="append", index=False)
            con.execute("""CREATE TABLE IF NOT EXISTS national_snapshot
                (fetched_at TEXT, demand_met_mw REAL, thermal_mw REAL,
                 gas_mw REAL, nuclear_mw REAL, hydro_mw REAL, renewable_mw REAL,
                 storage_mw REAL, other_mw REAL, transnational_mw REAL)""")
            con.execute("""INSERT INTO national_snapshot VALUES
                (:ts,:demand_met_mw,:thermal_mw,:gas_mw,:nuclear_mw,:hydro_mw,
                 :renewable_mw,:storage_mw,:other_mw,:transnational_mw)""",
                        {"ts": ts, **nat})
        store.log_fetch("merit_india", True)
        return {"national": nat, "states": df}, {"live": True, "asof": pd.Timestamp.now(),
                                                 "errors": df.attrs.get("errors", {})}
    except Exception as e:
        store.log_fetch("merit_india", False, str(e))
        with store.connect() as con:
            try:
                last = pd.read_sql("SELECT MAX(fetched_at) t FROM state_live", con).iloc[0, 0]
                df = pd.read_sql("SELECT * FROM state_live WHERE fetched_at=?",
                                 con, params=(last,)) if last else pd.DataFrame()
                nat = pd.read_sql(
                    "SELECT * FROM national_snapshot ORDER BY fetched_at DESC LIMIT 1",
                    con)
                nat = nat.iloc[0].to_dict() if len(nat) else {}
            except Exception:
                df, nat = pd.DataFrame(), {}
        return {"national": nat, "states": df}, {
            "live": False, "asof": store.last_good_fetch("merit_india"),
            "error": str(e)}


# =========================================================================
# Tier 3 — deep per-state SLDC adapters
# =========================================================================

def fetch_gujarat_realtime() -> dict:
    """Gujarat SLDC homepage server-renders live frequency, state demand
    ('Gujarat Catered') and the DAM rate — public, no login. Verified
    live 2026-07-24 (demand ticked 16,043 -> 16,077 MW between calls)."""
    r = requests.get("https://www.sldcguj.com/", headers=UA, timeout=30,
                     verify=False)
    r.raise_for_status()
    html = r.text

    def grab(label):
        m = re.search(label + r".*?text-dark\">\s*([\d.\-]+)", html, re.S)
        return float(m.group(1)) if m else None

    out = {"frequency_hz": grab("Grid Frequency"),
           "demand_mw": grab("Gujarat Catered"),
           "dam_rate_rs_unit": grab("DAM Rate")}
    if out["demand_mw"] is None:
        raise ValueError("Gujarat Catered value not found — page layout changed?")
    return out


# Rajasthan SLDC read-sftp tag map, taken verbatim from the page's own JS
# (sldc.rajasthan.gov.in/rrvpnl, fetchDynamicData()). The endpoint currently
# answers 500 upstream — their homepage widget is equally broken — so this
# adapter reports its own health honestly instead of pretending.
RAJ_SFTP_TAGS = {
    "03046004": "frequency_hz", "03046001": "dsm_rate_paise_unit",
    "03046008": "load_mw", "03046002": "nr_schedule_mw",
    "03046007": "nr_drawal_mw", "03046006": "od_ud_mw",
    "03046009": "generation_mw",
}


def fetch_rajasthan_overview() -> dict:
    """Rajasthan SLDC dynamic-data JSON (frequency, DSM rate, load,
    schedule/drawal, generation). Raises while the upstream endpoint is
    down; the MERIT layer still covers Rajasthan's demand meanwhile."""
    r = requests.get("https://sldc.rajasthan.gov.in/rrvpnl/read-sftp",
                     params={"type": "overview", "home": "sftp"},
                     headers={**UA, "X-Requested-With": "XMLHttpRequest"},
                     timeout=30, verify=False)
    r.raise_for_status()
    data = r.json()["data"]
    out = {name: _num(data[tag][0]["Average2"])
           for tag, name in RAJ_SFTP_TAGS.items() if tag in data}
    out["asof"] = r.json().get("date")
    return out


def fetch_rajasthan_re_plants(length: int = 200) -> pd.DataFrame:
    """Plant-level solar/wind injection from Rajasthan SLDC's realtime
    DataTable (verified working JSON; upstream feed timestamps have been
    observed stale, so always surface date_modified to the caller)."""
    r = requests.get("https://sldc.rajasthan.gov.in/rrvpnl/view-realtime-data/show",
                     params={"draw": 1, "start": 0, "length": length},
                     headers={**UA, "X-Requested-With": "XMLHttpRequest"},
                     timeout=30, verify=False)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"])
    if not len(df):
        return df
    df["SW_injection"] = pd.to_numeric(df["SW_injection"], errors="coerce")
    return df[["reference_id", "QCA_name", "GSS_name", "SW_injection",
               "date_modified"]].rename(columns={"SW_injection": "injection_mw"})


# =========================================================================
# StateAdapter pattern
# =========================================================================

@dataclass(frozen=True)
class StateAdapter:
    """One state's data-source configuration. `code` is the short key
    used everywhere else in the platform (dashboard selector, DB table
    prefixes, forecast model cache keys)."""
    code: str
    name: str
    lat: float
    lon: float
    grid_region: str
    status: str          # "verified" | "identified" — see module docstring
    peak_load_gw: float | None = None  # rough scale, for the state picker
    notes: str = ""
    fetch_realtime: callable | None = None  # -> dict, or None if unverified


def _delhi_realtime():
    snap, meta = sldc.get_realtime()
    return {**snap, "_live": meta["live"]}


REGISTRY: dict[str, StateAdapter] = {
    "DL": StateAdapter(
        code="DL", name="Delhi", lat=28.6448, lon=77.2167,
        grid_region="Northern", status="verified", peak_load_gw=8.7,
        notes="Full adapter: 5-min load history (2021-present), realtime "
              "snapshot, frequency curve, BRPL Kilokari BESS telemetry. "
              "The reference implementation every other adapter follows.",
        fetch_realtime=_delhi_realtime,
    ),
    "HR": StateAdapter(
        code="HR", name="Haryana", lat=29.0588, lon=76.0856,
        grid_region="Northern", status="verified", peak_load_gw=14.4,
        notes="Snapshot only (schedule/drawl/OD-UD/load), via the "
              "Northern Region table on Delhi's page. No 5-min history "
              "or frequency curve until a dedicated adapter is built.",
    ),
    "PB": StateAdapter(
        code="PB", name="Punjab", lat=30.7333, lon=76.7794,
        grid_region="Northern", status="verified", peak_load_gw=15.3,
        notes="Snapshot only, same source as Haryana above.",
    ),
    "RJ": StateAdapter(
        code="RJ", name="Rajasthan", lat=26.9124, lon=75.7873,
        grid_region="Northern", status="verified", peak_load_gw=16.8,
        notes="Three sources: NR table on Delhi's page (schedule/drawal/"
              "load), MERIT live demand, and Rajasthan SLDC's own JSON "
              "endpoints now fully mapped — read-sftp (freq, DSM rate, "
              "load, generation; upstream currently 500s, their own "
              "widget is broken too, adapter retries and reports) and a "
              "working ~151-plant RE injection DataTable (upstream "
              "timestamps observed stale; date_modified surfaced). "
              "India's largest RE base — first target for multi-state "
              "RE forecasting.",
    ),
    "UP": StateAdapter(
        code="UP", name="Uttar Pradesh", lat=26.8467, lon=80.9462,
        grid_region="Northern", status="verified", peak_load_gw=29.4,
        notes="Snapshot only. Largest load of any state we currently "
              "touch — bigger than Delhi by a wide margin.",
    ),
    "UK": StateAdapter(
        code="UK", name="Uttarakhand", lat=30.0668, lon=79.0193,
        grid_region="Northern", status="verified", peak_load_gw=2.4,
    ),
    "HP": StateAdapter(
        code="HP", name="Himachal Pradesh", lat=31.1048, lon=77.1734,
        grid_region="Northern", status="verified", peak_load_gw=1.5,
        notes="Net exporter on hydro most of the year — negative "
              "schedule/drawl in the snapshot is expected, not an error.",
    ),
    "CH": StateAdapter(
        code="CH", name="Chandigarh", lat=30.7333, lon=76.7794,
        grid_region="Northern", status="verified", peak_load_gw=0.4,
    ),
    "MH": StateAdapter(
        code="MH", name="Maharashtra", lat=19.0760, lon=72.8777,
        grid_region="Western", status="verified", peak_load_gw=28.0,
        notes="Live demand/own-gen/import via MERIT (verified 20.3 GW on "
              "first fetch). Own SLDC (mahasldc.in) publishes its public "
              "SCADA overview only as a JPEG image; numeric feeds sit "
              "behind an 'Authorized HO Users' login — the partnership "
              "route, not a scrape.",
    ),
    "GJ": StateAdapter(
        code="GJ", name="Gujarat", lat=23.0225, lon=72.5714,
        grid_region="Western", status="verified", peak_load_gw=25.0,
        notes="Two independent live sources: MERIT, and sldcguj.com's own "
              "homepage (server-rendered frequency + 'Gujarat Catered' "
              "demand + DAM rate, fetch_gujarat_realtime). Cross-checked "
              "within 0.2% of each other on verification day.",
    ),
    "MP": StateAdapter(
        code="MP", name="Madhya Pradesh", lat=23.2599, lon=77.4126,
        grid_region="Western", status="verified", peak_load_gw=17.0,
        notes="MERIT live demand/own-gen/import.",
    ),
    "CT": StateAdapter(
        code="CT", name="Chhattisgarh", lat=21.2514, lon=81.6296,
        grid_region="Western", status="verified", peak_load_gw=5.5,
        notes="MERIT live demand/own-gen/import.",
    ),
    "GA": StateAdapter(
        code="GA", name="Goa", lat=15.4909, lon=73.8278,
        grid_region="Western", status="verified", peak_load_gw=0.8,
        notes="MERIT live demand (own-gen published as null — imports "
              "nearly everything).",
    ),
    "TN": StateAdapter(
        code="TN", name="Tamil Nadu", lat=13.0827, lon=80.2707,
        grid_region="Southern", status="verified", peak_load_gw=20.8,
        notes="MERIT live. Second-largest wind base in India.",
    ),
    "KA": StateAdapter(
        code="KA", name="Karnataka", lat=12.9716, lon=77.5946,
        grid_region="Southern", status="verified", peak_load_gw=17.2,
        notes="MERIT live. Largest solar state in the south.",
    ),
    "TG": StateAdapter(
        code="TG", name="Telangana", lat=17.3850, lon=78.4867,
        grid_region="Southern", status="verified", peak_load_gw=15.6,
        notes="MERIT live demand/own-gen/import.",
    ),
    "AP": StateAdapter(
        code="AP", name="Andhra Pradesh", lat=16.5062, lon=80.6480,
        grid_region="Southern", status="verified", peak_load_gw=13.4,
        notes="MERIT live demand/own-gen/import.",
    ),
    "KL": StateAdapter(
        code="KL", name="Kerala", lat=8.5241, lon=76.9366,
        grid_region="Southern", status="verified", peak_load_gw=5.5,
        notes="MERIT live. Own SLDC (sldckerala.com) is a PHP site with "
              "system-statistics pages — identified, not yet adapted.",
    ),
    "WB": StateAdapter(
        code="WB", name="West Bengal", lat=22.5726, lon=88.3639,
        grid_region="Eastern", status="verified", peak_load_gw=10.5,
        notes="MERIT live demand/own-gen/import.",
    ),
    "BR": StateAdapter(
        code="BR", name="Bihar", lat=25.5941, lon=85.1376,
        grid_region="Eastern", status="verified", peak_load_gw=8.5,
        notes="MERIT live demand/own-gen/import.",
    ),
    "OD": StateAdapter(
        code="OD", name="Odisha", lat=20.2961, lon=85.8245,
        grid_region="Eastern", status="verified", peak_load_gw=6.5,
        notes="MERIT live demand/own-gen/import.",
    ),
    "JH": StateAdapter(
        code="JH", name="Jharkhand", lat=23.3441, lon=85.3096,
        grid_region="Eastern", status="verified", peak_load_gw=2.5,
        notes="MERIT live demand/own-gen/import.",
    ),
    "AS": StateAdapter(
        code="AS", name="Assam", lat=26.1445, lon=91.7362,
        grid_region="North-Eastern", status="verified", peak_load_gw=2.6,
        notes="MERIT live demand/own-gen/import.",
    ),
}


def list_states(status: str | None = None) -> list[StateAdapter]:
    vals = list(REGISTRY.values())
    return [s for s in vals if status is None or s.status == status]


def get_state(code: str) -> StateAdapter:
    if code not in REGISTRY:
        raise KeyError(f"unknown state code {code!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[code]


if __name__ == "__main__":
    snap, meta = get_india_snapshot()
    nat, df = snap["national"], snap["states"]
    print(f"MERIT India snapshot (live={meta['live']}, asof={meta['asof']}):")
    if nat:
        print(f"  ALL INDIA demand met {nat.get('demand_met_mw', 0):,.0f} MW | "
              f"thermal {nat.get('thermal_mw', 0):,.0f} | hydro {nat.get('hydro_mw', 0):,.0f} | "
              f"RE {nat.get('renewable_mw', 0):,.0f} | storage {nat.get('storage_mw', 0):,.0f}")
    if len(df):
        print(df[["code", "name", "grid_region", "demand_mw", "own_gen_mw",
                  "import_mw"]].to_string(index=False))
    print()
    try:
        g = fetch_gujarat_realtime()
        print(f"Gujarat SLDC direct: {g['demand_mw']:,.0f} MW @ {g['frequency_hz']} Hz "
              f"| DAM Rs {g['dam_rate_rs_unit']}/unit")
    except Exception as e:
        print(f"Gujarat SLDC direct: FAILED ({e})")
    try:
        r = fetch_rajasthan_overview()
        print(f"Rajasthan SLDC overview: {r}")
    except Exception as e:
        print(f"Rajasthan SLDC overview: down as documented ({str(e)[:60]})")
    print()
    print(f"Registry ({sum(1 for s in REGISTRY.values() if s.status == 'verified')} "
          f"verified of {len(REGISTRY)}):")
    for s in REGISTRY.values():
        gw = f"{s.peak_load_gw:>5.1f} GW" if s.peak_load_gw else "   n/a"
        print(f"  {s.code:3s} {s.name:18s} {s.grid_region:13s} {gw}  [{s.status}]")
