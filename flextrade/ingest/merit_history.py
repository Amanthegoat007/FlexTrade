"""Historical per-state data from MERIT (Ministry of Power) — the unlock
that makes 23-state forecasting possible.

The problem this solves: Delhi has ~5 years of 5-minute SLDC load history,
which is why its model reaches 4.33% MAPE. No other state publishes
anything like that, and our own 15-minute MERIT poller only started on
24 Jul — far too little to train on before the demo.

What we found (verified 28 Jul 2026): the MERIT state pages are backed by
two POST endpoints that accept a DATE and answer for any day going back
2+ years, at ~95% coverage:

  StateWiseDetails/GetStateWiseDetailsForPiChart {StateCode, date}
      -> daily energy (MWh) by procurement source:
         State Generation | Central ISGS | Other ISGS | Bilateral |
         Power Exchange
      Their sum is the state's energy met for the day, and the
      "Power Exchange" leg is literally our addressable market.

  StateWiseDetails/GetPowerStationData {StateCode, date}
      -> plant-level scheduled generation for the day, tagged by
         TypeOfGeneration (Renewable / Hydro / Thermal / ...), including
         explicit SOLAR and WIND rows — a daily RE series per state.

IMPORTANT: the date must be formatted %m/%d/%Y. The %d/%m/%Y form the
site's own UI appears to use returns the sentinel "-3" (no data), which
is easy to mistake for "history is unavailable". That single detail is
the difference between 23 states of history and none.

Backfill is concurrent, resumable and idempotent: already-stored
(state, date) pairs are skipped, so it can be run repeatedly and topped
up daily by the pipeline.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import states, store  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://meritindia.in"
PIE = f"{BASE}/StateWiseDetails/GetStateWiseDetailsForPiChart"
PSD = f"{BASE}/StateWiseDetails/GetPowerStationData"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0",
      "Content-Type": "application/json; charset=utf-8"}

# the sentinel MERIT returns for "no data for that date"
NO_DATA = "-3"
DATE_FMT = "%m/%d/%Y"   # NOT %d/%m/%Y — see module docstring

# Coverage is STATE-SPECIFIC, surveyed 28 Jul 2026 by probing 5 spread-out
# dates per state. MERIT's historical energy series exists only for states
# that report into its merit-order dispatch process — notably NOT Delhi,
# UP, Karnataka or Bihar, which return well-formed responses with every
# EnergyValue null. Backfilling all 23 wastes ~half the requests on states
# that have nothing, so the default scope is the states that do.
#
#   full     5/5 probe days with data
#   partial  1-2/5 — usable but gappy; the model sees fewer rows for them
#   none     0/5 — live monitoring only (see the registry's tier notes)
HISTORY_FULL = ["RJ", "HP", "MH", "GJ", "MP", "WB", "KL", "TN"]
HISTORY_PARTIAL = ["HR", "PB", "AP", "TG"]
HISTORY_NONE = ["DL", "UP", "UK", "CH", "CT", "GA", "KA", "BR", "OD", "JH", "AS"]
HISTORY_STATES = HISTORY_FULL + HISTORY_PARTIAL

SOURCE_COLS = {
    "State Generation": "state_gen_mwh",
    "Central ISGS": "central_isgs_mwh",
    "Other ISGS": "other_isgs_mwh",
    "Bilateral": "bilateral_mwh",
    "Power Exchange": "exchange_mwh",
}


def _post(url: str, payload: dict, timeout: int = 25):
    r = requests.post(url, json=payload, headers=UA, timeout=timeout, verify=False)
    r.raise_for_status()
    txt = r.text.strip()
    if not txt or txt.strip('"') == NO_DATA:
        return None
    return r.json()


def _num(v):
    try:
        f = float(str(v).replace(",", ""))
        return f
    except (TypeError, ValueError):
        return None


def fetch_state_day(code: str, d: date) -> dict | None:
    """Daily energy mix for one state/day. None when MERIT has no data."""
    merit_code = states.MERIT_CODES[code][0]
    js = _post(PIE, {"StateCode": merit_code, "date": d.strftime(DATE_FMT)})
    if not js:
        return None
    row = {"code": code, "day": d.isoformat()}
    for item in js:
        col = SOURCE_COLS.get(str(item.get("TypeOfEnergy", "")).strip())
        if col:
            row[col] = _num(item.get("EnergyValue"))
    vals = [row.get(c) for c in SOURCE_COLS.values()]
    if all(v is None for v in vals):
        return None                      # present but empty -> treat as no data

    # PARTIAL DAYS ARE POISON. MERIT sometimes publishes only some of the
    # five procurement legs for a day. Summing whatever is present makes an
    # artificial cliff: Rajasthan days with 2 of 5 legs averaged 34,438 MWh
    # against 318,843 MWh on complete days — a 10x drop that is a reporting
    # gap, not a demand collapse. Day-over-day volatility was 2.7% when the
    # leg count held steady and 13.0% when it changed, so these rows wreck
    # both training and evaluation. We still STORE them (they are real
    # observations of what MERIT published) but mark completeness so the
    # modelling layer can exclude them; see read_energy(complete_only=True).
    row["n_components"] = sum(v is not None for v in vals)
    row["energy_met_mwh"] = sum(v for v in vals if v is not None)
    return row


def fetch_state_generation(code: str, d: date) -> dict | None:
    """Daily generation by fuel type + explicit solar/wind for one state."""
    merit_code = states.MERIT_CODES[code][0]
    js = _post(PSD, {"StateCode": merit_code, "date": d.strftime(DATE_FMT)})
    if not js:
        return None
    row = {"code": code, "day": d.isoformat(),
           "renewable_mwh": 0.0, "hydro_mwh": 0.0, "thermal_mwh": 0.0,
           "other_gen_mwh": 0.0, "solar_mwh": 0.0, "wind_mwh": 0.0,
           "n_stations": 0}
    any_val = False
    for st in js:
        sched = _num(st.get("Schedule")) or 0.0
        nonsched = _num(st.get("NonSchedule")) or 0.0
        total = sched + nonsched
        if total:
            any_val = True
        row["n_stations"] += 1
        kind = str(st.get("TypeOfGeneration", "")).strip().lower()
        if kind.startswith("renew"):
            row["renewable_mwh"] += total
        elif kind.startswith("hydro"):
            row["hydro_mwh"] += total
        elif kind.startswith("therm"):
            row["thermal_mwh"] += total
        else:
            row["other_gen_mwh"] += total
        name = str(st.get("PowerStationName", "")).upper()
        # MERIT publishes solar already halved ("SCALED TO 1/2"); undo it
        # so the series is the real energy, and say so in the column docs
        if "SOLAR" in name:
            row["solar_mwh"] += total * (2.0 if "1/2" in name else 1.0)
        elif "WIND" in name:
            row["wind_mwh"] += total
    return row if any_val else None


# ------------------------------------------------------------------ store --

def _ensure_tables():
    with store.connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS state_energy_daily (
            code TEXT, day TEXT,
            state_gen_mwh REAL, central_isgs_mwh REAL, other_isgs_mwh REAL,
            bilateral_mwh REAL, exchange_mwh REAL, energy_met_mwh REAL,
            fetched_at TEXT, PRIMARY KEY (code, day))""")
        con.execute("""CREATE TABLE IF NOT EXISTS state_generation_daily (
            code TEXT, day TEXT,
            renewable_mwh REAL, hydro_mwh REAL, thermal_mwh REAL,
            other_gen_mwh REAL, solar_mwh REAL, wind_mwh REAL,
            n_stations INTEGER, fetched_at TEXT, PRIMARY KEY (code, day))""")


def _existing(table: str) -> set[tuple[str, str]]:
    with store.connect() as con:
        try:
            df = pd.read_sql(f"SELECT code, day FROM {table}", con)
        except Exception:
            return set()
    return set(zip(df["code"], df["day"]))


def _save(table: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    df["fetched_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with store.connect() as con:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        df = df[[c for c in cols if c in df.columns]]
        df.to_sql("_tmp_bf", con, if_exists="replace", index=False)
        con.execute(f"INSERT OR REPLACE INTO {table} "
                    f"({','.join(df.columns)}) SELECT {','.join(df.columns)} "
                    f"FROM _tmp_bf")
        con.execute("DROP TABLE _tmp_bf")
    return len(df)


def backfill(days: int = 180, codes: list[str] | None = None,
             workers: int = 5, with_generation: bool = True,
             chunk: int = 60) -> dict:
    """Pull `days` of history for each state. Resumable and idempotent.

    Concurrency is deliberately modest. At 12 workers MERIT's response
    time degraded from ~1 s to ~8 s — we were the load. Five workers
    keeps us fast without hammering a public government service, and
    saving every `chunk` rows means an interrupted run keeps everything
    it already earned (re-running skips what is stored).
    """
    return _backfill(days, codes, workers, with_generation, chunk)


def _backfill(days: int, codes, workers: int, with_generation: bool,
              chunk: int) -> dict:
    """Pull `days` of history for each state. Resumable and idempotent."""
    _ensure_tables()
    codes = codes or HISTORY_STATES   # skip states MERIT has nothing for
    today = date.today()
    wanted = [(c, today - timedelta(days=n)) for c in codes
              for n in range(1, days + 1)]

    jobs = [("energy", "state_energy_daily", fetch_state_day)]
    if with_generation:
        jobs.append(("generation", "state_generation_daily", fetch_state_generation))

    summary = {}
    for label, table, fn in jobs:
        have = _existing(table)
        todo = [(c, d) for c, d in wanted if (c, d.isoformat()) not in have]
        print(f"{label}: {len(wanted) - len(todo):,} already stored, "
              f"{len(todo):,} to fetch", flush=True)
        got, empty, failed, buf = 0, 0, 0, []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, c, d): (c, d) for c, d in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    row = fut.result()
                    if row:
                        buf.append(row)
                        got += 1
                    else:
                        empty += 1
                except Exception:
                    failed += 1
                if len(buf) >= chunk:
                    _save(table, buf)
                    buf = []
                if i % 100 == 0:
                    print(f"  {label}: {i:,}/{len(todo):,} "
                          f"(ok {got:,} · empty {empty:,} · failed {failed:,})",
                          flush=True)
        _save(table, buf)
        summary[label] = {"fetched": got, "empty": empty, "failed": failed}
        print(f"  {label} done: {got:,} stored, {empty:,} empty, {failed:,} failed")
    return summary


def read_energy(codes: list[str] | None = None,
                complete_only: bool = True) -> pd.DataFrame:
    """Daily state energy. `complete_only` keeps days where MERIT published
    all five procurement legs — the only rows whose total is trustworthy
    (see the note in fetch_state_day). Completeness is derived at read time
    so it applies to rows stored before the check existed."""
    with store.connect() as con:
        try:
            df = pd.read_sql("SELECT * FROM state_energy_daily", con,
                             parse_dates=["day"])
        except Exception:
            return pd.DataFrame()
    if codes:
        df = df[df["code"].isin(codes)]
    legs = list(SOURCE_COLS.values())

    # Completeness is judged PER STATE against the legs that state actually
    # reports. Himachal, Haryana and Punjab never publish a bilateral or
    # power-exchange leg at all (0% of days), so demanding all five would
    # throw away their entire history as "incomplete". A leg present on at
    # least half a state's days is treated as expected; a day is complete
    # when every expected leg is there.
    df["n_components"] = df[legs].notna().sum(axis=1)
    df["is_complete"] = False
    for code, idx in df.groupby("code").groups.items():
        g = df.loc[idx]
        expected = [c for c in legs if g[c].notna().mean() >= 0.5]
        if not expected:
            continue
        df.loc[idx, "expected_legs"] = len(expected)
        df.loc[idx, "is_complete"] = g[expected].notna().all(axis=1)
    if complete_only:
        df = df[df["is_complete"]]
    return df.sort_values(["code", "day"])


def reports_exchange(code: str) -> bool:
    """Does this state publish a power-exchange leg at all? (Only those can
    be used for the exchange-purchase forecast.)"""
    g = read_energy([code], complete_only=False)
    return bool(len(g)) and g["exchange_mwh"].notna().mean() >= 0.5


def validate_scale() -> pd.DataFrame:
    """Physical plausibility check on each state's daily-energy series.

    energy_met_mwh / 24 is the state's average MW for the day, which cannot
    exceed its peak demand. Checking that caught a real defect: Madhya
    Pradesh's `State Generation` leg averages ~1.36 million MWh/day, i.e.
    ~56 GW from state plant alone against a ~17 GW state peak — impossible.
    MP's OTHER legs sum to ~13,000 MW average, which matches its live
    demand almost exactly, so the State Generation field is either in
    different units or is a different quantity entirely. Either way the
    total is unusable and must not be modelled or displayed as demand.

    Verdicts:
      ok             average MW is a believable fraction of state peak
      implausible    exceeds the state's peak — excluded from modelling
      partial_legs   believable but low because the state does not publish
                     every procurement leg; the series is a SUBSET of
                     demand, fine to forecast, wrong to call "total demand"
    """
    from ingest import states as _states
    e = read_energy()
    if not len(e):
        return pd.DataFrame()
    legs = list(SOURCE_COLS.values())
    rows = []
    for code, g in e.groupby("code"):
        implied = float(g["energy_met_mwh"].mean()) / 24.0
        adapter = _states.REGISTRY.get(code)
        peak_mw = (adapter.peak_load_gw * 1000.0
                   if adapter and adapter.peak_load_gw else None)
        n_legs = int(g[legs].notna().any().sum())
        share = implied / peak_mw if peak_mw else None
        if peak_mw and implied > peak_mw:
            verdict = "implausible"
        elif n_legs < len(legs):
            verdict = "partial_legs"
        else:
            verdict = "ok"
        rows.append({"code": code, "days": len(g),
                     "implied_avg_mw": round(implied, 0),
                     "state_peak_mw": peak_mw,
                     "pct_of_peak": round(100 * share, 1) if share else None,
                     "legs_reported": n_legs, "verdict": verdict})
    return pd.DataFrame(rows).sort_values("pct_of_peak", ascending=False)


def modelable_states() -> list[str]:
    """States whose energy series is physically believable enough to model."""
    v = validate_scale()
    if not len(v):
        return []
    return v.loc[v["verdict"] != "implausible", "code"].tolist()


def completeness() -> pd.DataFrame:
    """How much of each state's stored history is usable for modelling."""
    allrows = read_energy(complete_only=False)
    if not len(allrows):
        return pd.DataFrame()
    g = allrows.groupby("code")
    out = pd.DataFrame({
        "days_stored": g.size(),
        "days_complete": g["is_complete"].sum(),
    })
    out["complete_pct"] = (100 * out["days_complete"] / out["days_stored"]).round(1)
    return out.reset_index().sort_values("days_complete", ascending=False)


def read_generation(codes: list[str] | None = None) -> pd.DataFrame:
    with store.connect() as con:
        try:
            df = pd.read_sql("SELECT * FROM state_generation_daily", con,
                             parse_dates=["day"])
        except Exception:
            return pd.DataFrame()
    if codes:
        df = df[df["code"].isin(codes)]
    return df.sort_values(["code", "day"])


def coverage() -> pd.DataFrame:
    """Per-state history depth — what the forecaster actually has."""
    e = read_energy()
    if not len(e):
        return pd.DataFrame()
    g = read_generation()
    rows = []
    for code, grp in e.groupby("code"):
        gg = g[g["code"] == code] if len(g) else pd.DataFrame()
        rows.append({
            "code": code, "name": states.MERIT_CODES.get(code, (None, code))[1],
            "days": len(grp),
            "from": str(grp["day"].min().date()), "to": str(grp["day"].max().date()),
            "mean_energy_mwh": round(float(grp["energy_met_mwh"].mean()), 0),
            "exchange_share_pct": round(float(
                100 * grp["exchange_mwh"].sum() / grp["energy_met_mwh"].sum()), 2)
            if grp["energy_met_mwh"].sum() else None,
            "gen_days": len(gg),
        })
    return pd.DataFrame(rows).sort_values("days", ascending=False)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    gen = "--no-gen" not in sys.argv
    backfill(days=n, with_generation=gen)
    print()
    print(coverage().to_string(index=False))
