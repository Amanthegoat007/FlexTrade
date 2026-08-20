"""Maharashtra RE deviation settlement — what forecast error actually costs.

MSLDC publishes a weekly RE-DSM bill under MERC (Forecasting, Scheduling and
Deviation Settlement for Solar and Wind Generation) Regulations 2018. Each bill
names every QCA, every pooling sub-station, and the rupees payable to or
receivable from the deviation pool for that week.

This is the only source in the project that prices forecast error in SETTLED
MONEY rather than in error metrics. One QCA in one week of April 2026 owed
Rs 1.69 crore across 81 sub-stations. That is the size of the problem a better
RE forecast solves, quoted by the regulator rather than estimated by us.

DISCOVERY

The report pages are JavaScript-rendered and load jsencrypt, so the file list
cannot be scraped — but the filenames follow a pattern and the asset directory
serves anything you can name. Probing MSLDC_TECH_REDSM_Bill_{year}_{week}.pdf
found 39 weeks in 2025 and 42 in 2026. Gaps are real (2025 skips weeks 5-10),
not scan failures, so a miss is recorded rather than retried forever.

Two sibling series enumerate the same way: mr10_{MM}{YYYY}.pdf (monthly
curtailment) and RCR1_{MM}{YYYY}.pdf (monthly wind/solar common registry).

WHAT THIS ARCHIVE IS, AND WHAT IT IS NOT

Each published PDF is the bill addressed to ONE QCA — verified, no file in the
archive carries more than one. The URL does not encode which, so the public
archive is a rotating sample rather than a census: 66 weeks recovered, but
spread across 12 QCAs.

    MH_MANIKARAN       21 weeks   82 sub-stations   Rs 77.14 cr
    MH_RATNAGIRI       13 weeks    1 sub-station    Rs  0.02 cr
    MH_TPREL            4 weeks    6 sub-stations   Rs  0.04 cr
    ...nine more, one to four weeks each

So a row total across the table is NOT Maharashtra's RE deviation bill and must
never be quoted as one. It sums different companies in different weeks. The
only defensible series is per-QCA, and only MH_MANIKARAN has enough weeks to be
one: 21 weeks, median Rs 1.43 crore, min 0.40, max 50.87.

That maximum is real, not a parse error. In 2025 week 29 three INTER-state
sub-stations carried it — Nigade Inter 220kV alone was Rs 44.3 crore payable
against Rs 14.4 crore receivable. Intra-state charges that week were zero. A
single bad forecast week at an inter-state pooling station costs an order of
magnitude more than a normal one, which is the whole commercial argument and
also the reason a mean is the wrong summary here.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import store  # noqa: E402

BASE = "https://mahasldc.in/assets/shared/reports/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS re_dsm_bill (
    year INTEGER NOT NULL, week INTEGER NOT NULL,
    qca TEXT NOT NULL, pss TEXT NOT NULL,
    period_from TEXT, period_to TEXT,
    intra_payable_rs REAL, inter_payable_rs REAL, inter_receivable_rs REAL,
    net_rs REAL, fetched_at TEXT,
    PRIMARY KEY (year, week, qca, pss));
"""

_NUM = re.compile(r"^-?[\d,]+\.?\d*$")


def _f(tok: str):
    tok = (tok or "").replace(",", "").strip()
    return float(tok) if _NUM.match(tok) else None


def bill_url(year: int, week: int) -> str:
    return f"{BASE}MSLDC_TECH_REDSM_Bill_{year}_{week}.pdf"


def parse_bill(pdf_bytes: bytes, year: int, week: int) -> pd.DataFrame:
    """Pull the per-sub-station rows out of one weekly bill.

    The table is not a clean grid — pdfplumber finds a table object but the
    header spans three wrapped lines, so rows are read off the TEXT instead.
    A data row is "<n> <name...> <five numbers>"; a QCA header is "<n> <NAME>"
    with no numbers and sets the owner for the rows beneath it.
    """
    import pdfplumber

    rows, qca, pfrom, pto = [], None, None, None
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            m = re.search(r"Period\s*:?\s*(\d{2}-\d{2}-\d{4})\s*to\s*(\d{2}-\d{2}-\d{4})",
                          text)
            if m and not pfrom:
                pfrom, pto = m.group(1), m.group(2)
            for line in text.split("\n"):
                parts = line.split()
                if len(parts) < 2 or not parts[0].isdigit():
                    continue
                nums = [p for p in parts if _NUM.match(p.replace(",", ""))][1:]
                if len(nums) >= 5:
                    name = " ".join(parts[1:len(parts) - 5]).strip()
                    v = [_f(x) for x in nums[-5:]]
                    if name and qca and None not in v[:1]:
                        rows.append({
                            "year": year, "week": week, "qca": qca, "pss": name,
                            "period_from": pfrom, "period_to": pto,
                            "intra_payable_rs": v[0], "inter_payable_rs": v[3],
                            "inter_receivable_rs": v[4],
                            "net_rs": (v[0] or 0) + (v[3] or 0) - (v[4] or 0)})
                elif len(parts) == 2 and re.match(r"^[A-Z][A-Z_]+$", parts[1]):
                    qca = parts[1]
    return pd.DataFrame(rows)


def backfill(years=(2025, 2026), weeks=range(1, 53), verbose: bool = True) -> dict:
    got, missing, stored = [], [], 0
    with store.connect() as con:
        con.executescript(_SCHEMA)
    for year in years:
        for week in weeks:
            url = bill_url(year, week)
            try:
                r = requests.get(url, headers=UA, timeout=90, verify=False)
                if r.status_code != 200 or not r.content.startswith(b"%PDF"):
                    missing.append((year, week))
                    continue
                df = parse_bill(r.content, year, week)
                if not len(df):
                    missing.append((year, week))
                    continue
                df["fetched_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                cols = list(df.columns)
                with store.connect() as con:
                    con.executemany(
                        f"INSERT OR REPLACE INTO re_dsm_bill ({','.join(cols)}) "
                        f"VALUES ({','.join('?' * len(cols))})",
                        df[cols].where(pd.notna(df[cols]), None)
                        .itertuples(index=False, name=None))
                got.append((year, week))
                stored += len(df)
                if verbose:
                    print(f"  {year} wk{week:02d}: {len(df):3d} rows  "
                          f"net Rs {df.net_rs.sum():,.0f}", flush=True)
            except Exception as e:
                missing.append((year, week))
                if verbose:
                    print(f"  {year} wk{week:02d}: {type(e).__name__}", flush=True)
    return {"weeks": len(got), "rows": stored, "missing": len(missing)}
