"""Delhi SLDC live scrapers.

- fetch_realtime():  current Delhi load / schedule / drawl / frequency from
  the real-time monitoring page (updates every few seconds).
- fetch_day_curve(): full 5-min load curve for any past date from
  Loaddata.aspx — same table the historical hackathon dataset came from,
  which lets us backfill right up to yesterday.
"""
import io
import re
from datetime import date, timedelta

import pandas as pd
import requests

from . import store

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
RT_URL = "https://www.delhisldc.org/Redirect.aspx?Loc=0804"
DAY_URL = "https://www.delhisldc.org/Loaddata.aspx?mode={d:%d/%m/%Y}"


def _first_number(v):
    """'6334' -> 6334.0;  \"7256 at 00:00:42\" -> 7256.0;  junk -> None."""
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


def fetch_realtime() -> dict:
    """Current Delhi snapshot from the SLDC realtime page.

    Parsed as TABLES, not as "label followed by a number". In late
    Jul 2026 SLDC restyled this page into header-row / value-row tables
    ("DELHI LOAD | SCHEDULE | DRAWL" above "6334 | 5944 | 5717") and
    uppercased the labels, which broke the old inline regex — the
    snapshot then silently served stale cache. Reading header->value
    pairs survives both the case change and the relayout.
    """
    html = requests.get(RT_URL, headers=UA, timeout=25).text

    # Read header->value per table and keep them SEPARATE. A flat merge is
    # wrong here: "SCHEDULE", "DRAWL" and "OD/UD" also head the generation
    # and per-DISCOM tables, so a global lookup silently returns a DISCOM's
    # numbers as Delhi's (270 MW instead of 5,944 MW when first written).
    blocks: list[dict[str, str]] = []
    try:
        for t in pd.read_html(io.StringIO(html)):
            if t.shape[0] < 2 or t.shape[1] < 2:
                continue
            heads = [str(h).strip().upper() for h in t.iloc[0]]
            vals = [str(v).strip() for v in t.iloc[1]]
            d = {h: v for h, v in zip(heads, vals) if h and h != "NAN"}
            if d:
                blocks.append(d)
    except ValueError:
        blocks = []  # no tables parsed; falls through to the guard below

    def block_with(anchor: str) -> dict[str, str]:
        return next((b for b in blocks if anchor in b), {})

    summary = block_with("DELHI LOAD")   # DELHI LOAD | SCHEDULE | DRAWL | ...
    sysinfo = block_with("FREQUENCY")    # FREQUENCY | OD/UD | DELHI GENERATION

    def pick(src: dict, *names):
        for n in names:
            if n in src:
                got = _first_number(src[n])
                if got is not None:
                    return got
        return None

    snap = {
        "delhi_load": pick(summary, "DELHI LOAD"),
        "schedule": pick(summary, "SCHEDULE"),
        "drawl": pick(summary, "DRAWL", "DRAWAL"),
        "frequency": pick(sysinfo, "FREQUENCY"),
        "od_ud": pick(sysinfo, "OD/UD", "OD / UD"),
    }
    if snap["delhi_load"] is None:
        # legacy inline layout, kept as a fallback so an SLDC rollback
        # doesn't break us again
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
        m = re.search(r"Delhi\s*Load\s*[:\-]?\s*(-?[\d.]+)", text, re.IGNORECASE)
        if m:
            snap["delhi_load"] = float(m.group(1))
    if snap["delhi_load"] is None:
        raise ValueError("could not parse Delhi Load from realtime page")

    # Plausibility guard. The failure mode we actually hit was not a crash
    # but a WRONG number (a DISCOM's 270 MW read as Delhi's system load),
    # which no exception would have caught. Delhi's system load has never
    # been below ~2 GW nor above its ~8.6 GW record.
    load = snap["delhi_load"]
    if not (2000 <= load <= 9500):
        raise ValueError(
            f"Delhi load {load} MW is outside the plausible 2,000-9,500 MW "
            "band — the SLDC page layout probably changed again")
    if snap["frequency"] is not None and not (47 <= snap["frequency"] <= 53):
        snap["frequency"] = None  # implausible: report missing, never wrong
    return snap


def fetch_day_curve(d: date) -> pd.DataFrame:
    """5-min load curve (DELHI + discoms, MW) for one date."""
    html = requests.get(DAY_URL.format(d=d), headers=UA, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    big = max(tables, key=len)
    big.columns = [str(c).strip().lower() for c in big.iloc[0]]
    big = big.iloc[1:]
    big = big.rename(columns={"timeslot": "slot"})
    big["ts"] = pd.to_datetime(f"{d:%Y-%m-%d} " + big["slot"], errors="coerce")
    big = big.dropna(subset=["ts"]).set_index("ts").drop(columns="slot")
    big = big.apply(pd.to_numeric, errors="coerce")
    big.columns = [c if c != "tpddl" else "ndpl" for c in big.columns]  # renamed discom
    return big[["delhi", "brpl", "bypl", "ndpl", "ndmc", "mes"]]


FREQ_URL = "https://www.delhisldc.org/Freqcurve.aspx"


def fetch_frequency_curve() -> pd.Series:
    """Today's 5-min system frequency (Hz) so far.

    SLDC renders the curve as a server-side chart image, but the
    accompanying HTML image-map carries every point as a tooltip:
        title="Time Slot: 14:40\\nFreq Bawana: (49.89)"
    so the series is recoverable exactly rather than read off a picture.
    Frequency drives the DSM charge rate (see models/dsm.py).

    IMPORTANT: the page accepts a `mode=DD/MM/YYYY` query parameter but
    **ignores it** — it always renders the current day. Verified by
    requesting two different dates and getting byte-identical values. So
    this function deliberately takes no date argument: history can only be
    accumulated by sampling daily (see run_pipeline), never backfilled.
    Labelling this data with a past date would fabricate history.
    """
    html = requests.get(FREQ_URL, headers=UA, timeout=30).text
    pts = re.findall(r"Time Slot:\s*(\d{1,2}:\d{2})[^\d]*?Freq[^(]*\(([\d.]+)\)", html)
    if not pts:
        raise ValueError("no frequency points in image map")
    today = date.today()
    rows = {}
    for slot, hz in pts:
        ts = pd.to_datetime(f"{today:%Y-%m-%d} {slot}", errors="coerce")
        if pd.notna(ts):
            rows[ts] = float(hz)
    s = pd.Series(rows, name="frequency_hz").sort_index()
    s = s[(s > 47) & (s < 53)]  # drop obvious telemetry garbage
    # guard: this endpoint only ever describes today. If anything ever
    # writes a different date into this series, that is fabricated
    # history and must not reach the database.
    stamped = set(s.index.date)
    if stamped - {today}:
        raise ValueError(f"frequency series carries non-today dates: {stamped}")
    return s


def get_frequency() -> tuple[pd.Series, dict]:
    """Today's frequency curve, appended to the `frequency` table.

    Returns everything stored (today plus whatever earlier days were
    sampled on previous runs), so callers can look up any day we have
    genuinely observed.
    """
    try:
        s = fetch_frequency_curve()
        store.upsert("frequency", s.to_frame())
        store.log_fetch("sldc_freq", True)
        meta = {"live": True, "asof": pd.Timestamp.now()}
    except Exception as e:
        store.log_fetch("sldc_freq", False, str(e))
        meta = {"live": False, "asof": store.last_good_fetch("sldc_freq"),
                "error": str(e)}
    stored = store.read("frequency")
    series = (stored["frequency_hz"] if len(stored) else pd.Series(dtype=float))
    return series, meta


def frequency_for_day(d: date) -> pd.Series:
    """Observed frequency for a day, or an empty series if never sampled."""
    stored = store.read("frequency")
    if not len(stored):
        return pd.Series(dtype=float)
    return stored[stored.index.date == d]["frequency_hz"]


def get_realtime():
    """Live fetch with cached fallback. Returns (snap_dict, meta)."""
    try:
        snap = fetch_realtime()
        row = pd.DataFrame([snap])
        row.insert(0, "fetched_at", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        with store.connect() as con:
            row.to_sql("rt_snapshot", con, if_exists="append", index=False)
        store.log_fetch("sldc_rt", True)
        return snap, {"live": True, "asof": pd.Timestamp.now()}
    except Exception as e:
        store.log_fetch("sldc_rt", False, str(e))
        with store.connect() as con:
            df = pd.read_sql(
                "SELECT * FROM rt_snapshot ORDER BY fetched_at DESC LIMIT 1", con)
        snap = df.iloc[0].to_dict() if len(df) else {}
        return snap, {"live": False, "asof": snap.get("fetched_at"), "error": str(e)}


def backfill_load(start: date, end: date | None = None, verbose: bool = True,
                  retries: int = 2) -> tuple[int, list[date]]:
    """Pull day curves for [start, end] into the DB.

    Returns (rows_written, failed_days). A day that yields fewer than 200
    of the expected 288 5-min slots is treated as a failure and retried —
    SLDC occasionally serves a partial table under load."""
    end = end or (date.today() - timedelta(days=1))
    total, failed, d = 0, [], start
    while d <= end:
        for attempt in range(retries + 1):
            try:
                df = fetch_day_curve(d)
                if len(df) < 200:
                    raise ValueError(f"partial table ({len(df)} rows)")
                total += store.upsert("load_5min", df)
                if verbose:
                    print(f"  sldc {d}: {len(df)} rows")
                break
            except Exception as e:
                if attempt == retries:
                    failed.append(d)
                    print(f"  sldc {d}: FAILED after {retries + 1} tries ({e})")
        d += timedelta(days=1)
    ok = not failed
    store.log_fetch("sldc_hist", ok,
                    f"backfilled to {end}" if ok else f"{len(failed)} days failed")
    return total, failed


def ensure_load_current(verbose: bool = True) -> list[date]:
    """Self-heal: backfill whatever is missing between the last stored day
    and yesterday. Returns the list of days that could not be fetched."""
    stored = store.read("load_5min")
    yesterday = date.today() - timedelta(days=1)
    if not len(stored):
        return backfill_load(yesterday - timedelta(days=30), yesterday, verbose)[1]
    last = stored.index.max().date()
    if last >= yesterday:
        if verbose:
            print(f"  load_5min current through {last}")
        return []
    gap = (yesterday - last).days
    if verbose:
        print(f"  load_5min stale by {gap} day(s) — backfilling "
              f"{last + timedelta(days=1)} → {yesterday}")
    return backfill_load(last + timedelta(days=1), yesterday, verbose)[1]
