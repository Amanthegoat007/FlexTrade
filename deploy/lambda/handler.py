"""Collector for AWS Lambda — the 24/7 half of the pipeline.

WHY THIS EXISTS, AND WHAT IT DOES NOT REPLACE

The sources in this project split into two kinds, and only one of them needs
a machine that never sleeps:

  windowed        NPP demand + fuel mix serve a rolling ~4.1 hour history on
                  every call (measured 2026-08-18: 64 timestamps, 4.0 min
                  median gap). Poll once inside that window and nothing is
                  lost, so GitHub Actions at its throttled ~12 runs a day
                  already covers them completely.

  instantaneous   MERIT, Vidyut PRAVAH area price, UPSLDC and PSTCL publish
                  "now" and nothing else. Every poll missed is a block gone
                  permanently. 12 samples a day against the 96 we want is the
                  hole this function exists to close.

It collects both anyway. Redundancy across two independent runners costs
nothing here and means neither one being down loses a block.

STDLIB ONLY, DELIBERATELY

Nothing outside the Python standard library plus boto3, which the Lambda
runtime already ships. No requests, no pandas, no layer, no container image —
the deployment is one zipped file. That is worth more than the convenience of
requests, because a dependency layer is one more thing to get wrong at 3am
when nobody is watching and the data is not being written.
"""
from __future__ import annotations

import csv
import http.cookiejar
import io
import json
import os
import ssl
import urllib.request
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

BUCKET = os.environ["BUCKET"]
PREFIX = os.environ.get("PREFIX", "collected")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"
TIMEOUT = 25

# Several of these are state utility sites with expired or mis-chained
# certificates. The laptop collector already runs with verification off for
# the same reason; this is public, unauthenticated, read-only data and there
# is no secret to protect on the wire.
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

s3 = boto3.client("s3")


# One cookie-aware opener shared by every source. Vidyut PRAVAH rejects a cold
# POST to its dashboard endpoints, so the landing page has to be fetched first
# and the session cookie carried forward — which urlopen() alone will not do.
OPENER = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
    urllib.request.HTTPSHandler(context=SSL_CTX))


def _open(req: urllib.request.Request) -> bytes:
    with OPENER.open(req, timeout=TIMEOUT) as r:
        return r.read()


def _headers(referer: str | None, extra: dict | None = None) -> dict:
    h = {"User-Agent": UA}
    if referer:
        h["Referer"] = referer
    return {**h, **(extra or {})}


def _get(url: str, referer: str | None = None) -> bytes:
    return _open(urllib.request.Request(url, headers=_headers(referer)))


def _post(url: str, payload: dict, referer: str | None = None) -> bytes:
    """JSON POST — MERIT's per-state endpoint."""
    return _open(urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers=_headers(referer, {"Content-Type": "application/json; charset=utf-8"})))


def _post_form(url: str) -> bytes:
    """Empty form POST — the Vidyut PRAVAH dashboard endpoints.

    They are ASP.NET AJAX handlers and answer only to an XHR-shaped request
    with a form content type, so a JSON body gets a 500 rather than data.
    """
    return _open(urllib.request.Request(
        url, data=b"",
        headers=_headers("https://vidyutpravah.in/", {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest"})))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _key(source: str) -> str:
    return f"{PREFIX}/{source}/{datetime.now(timezone.utc):%Y-%m-%d}.csv"


def _store(source: str, rows: list[dict], dedupe: tuple[str, ...] = ()) -> int:
    """Read the day's object, append only genuinely new rows, write it back.

    One object per source per UTC day keeps each read-modify-write to a few
    hundred rows, so the whole cycle is well inside a Lambda invocation and
    the object stays small enough that a partial write cannot corrupt much.

    `dedupe` names the columns that identify a row. NPP needs it because its
    rolling window re-serves the same 4-minute blocks on every poll; the
    instantaneous sources pass nothing, since every fetch is by definition new.
    """
    if not rows:
        return 0
    key, existing = _key(source), ""
    try:
        existing = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode()
    except ClientError as e:
        # The first write of any UTC day is a genuine miss. S3 reports that as
        # NoSuchKey only to callers who hold s3:ListBucket — without it the
        # same miss comes back as 403 AccessDenied, because S3 refuses to
        # confirm or deny a key's existence to someone who cannot list. The
        # policy in deploy/README.md grants ListBucket so the two stay
        # distinguishable, but a genuine permission failure still surfaces
        # loudly one line later when put_object is refused.
        if e.response["Error"]["Code"] not in ("NoSuchKey", "404",
                                               "AccessDenied", "403"):
            raise

    seen = set()
    if existing and dedupe:
        for old in csv.DictReader(io.StringIO(existing)):
            seen.add(tuple(str(old.get(c, "")) for c in dedupe))
    fresh = [r for r in rows
             if not dedupe or tuple(str(r.get(c, "")) for c in dedupe) not in seen]
    if not fresh:
        return 0

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    if not existing:
        w.writeheader()
    for r in fresh:
        w.writerow(r)
    s3.put_object(Bucket=BUCKET, Key=key,
                  Body=(existing + buf.getvalue()).encode(),
                  ContentType="text/csv")
    return len(fresh)


# --------------------------------------------------------------- sources --

# MERIT's own state codes, mirrored from ingest/states.py so this file stays
# standalone — it must zip and run with nothing else from the repo beside it.
MERIT_CODES = {
    "DL": "DL", "HR": "HRN", "PB": "PNB", "RJ": "RJ", "UP": "UP", "UK": "UTK",
    "HP": "HP", "CH": "CHG", "MH": "MHA", "GJ": "GJT", "MP": "MPD", "CT": "CTG",
    "GA": "GOA", "TN": "TND", "KA": "KRT", "TG": "TLG", "AP": "AP", "KL": "KRL",
    "WB": "BGL", "BR": "BHR", "OD": "ODI", "JH": "JHK", "AS": "ASM",
}


def merit() -> int:
    """23-state demand / own generation / import, Ministry of Power.

    One POST per state: MERIT has no all-states endpoint carrying the import
    breakdown, and import is the field the State Stress index rests on. One
    state failing must never cost us the other 22.
    """
    ts, rows = _now(), []
    for code, merit_code in MERIT_CODES.items():
        try:
            raw = _post("https://meritindia.in/StateWiseDetails/BindCurrentStateStatus",
                        {"StateCode": merit_code})
            row = (json.loads(raw) or [{}])[0]
            rows.append({"fetched_at_utc": ts, "code": code,
                         "demand_mw": row.get("Demand"),
                         "own_gen_mw": row.get("ISGS"),
                         "import_mw": row.get("ImportData")})
        except Exception:
            continue
    return _store("merit_state", rows)


def upsldc() -> int:
    """UP: the DSM triplet plus generation split, frequency and the DSM rate."""
    d = (json.loads(_get("https://upsldc.org/assets/dataset/real-time-summary.json",
                         "https://www.upsldc.org/")) or [{}])[0]
    sched, drawal = d.get("SCHEDULE_MW"), d.get("DRAWL_MW")
    return _store("upsldc", [{
        "fetched_at_utc": _now(),
        "demand_met_mw": d.get("DEMAND_MW"),
        "schedule_mw": sched, "drawal_mw": drawal,
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
    }])


def pstcl() -> int:
    """Punjab: frequency, schedule, drawal, deviation.

    Answers 500 intermittently — twice inside ten minutes on 2026-08-18, each
    time recovering on the next attempt — so one failure means nothing.
    """
    last = None
    for _ in range(3):
        try:
            d = json.loads(_get("https://sldcapi.pstcl.org/wsDataService.asmx/dynamicData",
                                "https://www.pstcl.org/"))
            sched, drawal = d.get("scheduleMW"), d.get("drawalMW")
            return _store("pstcl", [{
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


def area_price() -> int:
    """35-area clearing price — the only state-level price signal published.

    Needs a session cookie: the dashboard endpoints reject a cold POST, so the
    landing page is fetched first through the shared cookie-aware opener.
    """
    _get("https://vidyutpravah.in/")
    rows = json.loads(_post_form("https://vidyutpravah.in/PXDashboard/BindStatePricesFromJS"))
    blk = json.loads(_post_form("https://vidyutpravah.in/PXDashboard/BindCurrentDateTimeForJson"))
    b = (blk or [{}])[0]
    ts = _now()
    return _store("area_price", [{
        "fetched_at_utc": ts, "block_from": b.get("FromTime"),
        "block_date": b.get("CurrentDate"),
        "area": str(x.get("StateCode")), "acp_rs_mwh": x.get("ACP"),
    } for x in rows if x.get("StateCode") is not None])


def npp_national() -> int:
    """All-India demand met and the six-fuel mix at 4-minute resolution.

    Rolling ~4.1 hour window, no date parameter, no history anywhere. Keyed on
    (ts, series) rather than a max timestamp because the six fuels carry stamps
    a few seconds apart and a max filter would drop stragglers.
    """
    out = 0
    for endpoint, source in (("demandmet1chartdata", "npp_demand"),
                             ("demandmet2chartdata", "npp_fuelmix")):
        rows = []
        for x in json.loads(_get("https://npp.gov.in/dashBoard/" + endpoint)):
            stamp = x.get("updated_on")
            if not stamp:
                continue
            rows.append({
                "ts_utc": datetime.fromtimestamp(stamp / 1000, timezone.utc)
                                  .strftime("%Y-%m-%d %H:%M:%S"),
                "series": x.get("name_of_data"),
                "value_mw": x.get("value_of_data"),
                "fetched_at_utc": _now()})
        out += _store(source, rows, dedupe=("ts_utc", "series"))
    return out


SOURCES = [("merit", merit), ("npp_national", npp_national), ("upsldc", upsldc),
           ("pstcl", pstcl), ("area_price", area_price)]


def lambda_handler(event, context):
    """Collect every source. One dead upstream must not cost us the others.

    Raises only when EVERY source failed, which is not a quiet upstream — it is
    a broken collector or a runner that cannot reach Indian infrastructure. A
    green invocation that wrote nothing is the exact failure mode that hid a
    total outage for eleven days on the CI collector, so it is made loud here.
    """
    ok, total, detail = 0, 0, {}
    for name, fn in SOURCES:
        try:
            n = fn()
            detail[name], total, ok = n, total + n, ok + 1
        except Exception as e:
            detail[name] = f"{type(e).__name__}: {str(e)[:200]}"
    print(json.dumps({"at": _now(), "ok": f"{ok}/{len(SOURCES)}",
                      "rows": total, "detail": detail}))
    if ok == 0:
        raise RuntimeError(f"every source failed: {detail}")
    return {"sources_ok": ok, "rows": total, "detail": detail}
