"""Resolve thermal plants to the state they physically sit in.

CEA's coal report groups its first column by OWNER rather than location: the
three largest "states" in it are IPP (73 GW), NTPC (55 GW) and NTPC JV (8 GW).
That is fine for a fuel-management report and useless for a state-level view —
126 of 198 plants land in an owner bucket with no state at all, which is why
the State Grid Stress index could only show coal for a minority of states.

CEA publishes no plant-to-state mapping. But it publishes the SAME plant names
in the daily maintenance report (ingest/outages.py), and THAT one is grouped by
state. So the mapping can be derived from data we already hold, by joining the
two reports on plant name.

Matching is deliberately conservative, in three passes of decreasing certainty,
and every resolution records which pass produced it:

  exact   normalised name identical              (highest confidence)
  prefix  one name is a prefix of the other, and the shared part is long
          enough to be distinctive — catches "SIPAT STPS" vs "SIPAT STPS-I"
  token   all significant tokens of the shorter name appear in the longer,
          with a distinctive anchor token in common

Anything unresolved stays unresolved and is reported as such. A wrong state
attribution is worse than a missing one: it would move megawatts of coal and
outage from one state's risk picture to another's, silently.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import store  # noqa: E402

# first-column values in the coal report that are owners, not places
OWNER_BUCKETS = {"ipp", "ntpc", "ntpc jv", "ntpcjv", "nlc", "dvc", "nhpc",
                 "neepco", "sjvnl", "thdc", "npcil", "central sector",
                 "private sector", "state sector"}

# words that carry no location information
NOISE = {"TPS", "TPP", "STPS", "STPP", "UMTPP", "CTPS", "TPC", "POWER",
         "PLANT", "STATION", "THERMAL", "LIMITED", "LTD", "PVT", "PRIVATE",
         "CORPORATION", "COMPANY", "ENERGY", "GENERATION", "PROJECT", "UNIT",
         "PH", "PHASE", "EXT", "EXTN", "EXTENSION", "STAGE", "I", "II", "III",
         "IV", "V", "VI", "A", "B", "C", "D", "NEW", "OLD", "SUPER", "CRITICAL"}


def _norm(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _tokens(s) -> set[str]:
    raw = re.split(r"[^A-Za-z0-9]+", str(s).upper())
    return {t for t in raw if t and t not in NOISE and not t.isdigit() and len(t) > 2}


def is_owner_bucket(state) -> bool:
    return str(state).strip().lower() in OWNER_BUCKETS


def build_lookup() -> pd.DataFrame:
    """plant name -> state, derived by joining the coal and outage reports."""
    with store.connect() as con:
        coal = pd.read_sql("SELECT DISTINCT plant, state FROM coal_stock", con)
        try:
            out = pd.read_sql("SELECT DISTINCT station, state FROM unit_outage", con)
        except Exception:
            out = pd.DataFrame(columns=["station", "state"])

    # every (name, state) pair we can trust, from BOTH reports
    known: dict[str, str] = {}
    for _, r in out.iterrows():
        st = str(r["state"]).strip()
        if st and st.lower() not in ("nan", "") and not is_owner_bucket(st):
            known.setdefault(_norm(r["station"]), st)
    for _, r in coal.iterrows():
        st = str(r["state"]).strip()
        if st and st.lower() not in ("nan", "") and not is_owner_bucket(st):
            known.setdefault(_norm(r["plant"]), st)

    known_tokens = {k: _tokens(k) for k in known}
    # rebuild tokens from the ORIGINAL names, not the squashed keys
    orig_tokens: dict[str, set[str]] = {}
    for _, r in out.iterrows():
        orig_tokens.setdefault(_norm(r["station"]), _tokens(r["station"]))
    for _, r in coal.iterrows():
        orig_tokens.setdefault(_norm(r["plant"]), _tokens(r["plant"]))

    rows = []
    for _, r in coal.iterrows():
        name, listed = r["plant"], str(r["state"]).strip()
        key = _norm(name)
        if not is_owner_bucket(listed) and listed.lower() not in ("nan", ""):
            rows.append({"plant": name, "state": listed, "method": "listed"})
            continue

        # 1. exact
        if key in known:
            rows.append({"plant": name, "state": known[key], "method": "exact"})
            continue

        # 2. prefix, with enough shared characters to be distinctive
        hit = None
        for k, st in known.items():
            if len(k) >= 8 and (key.startswith(k) or k.startswith(key)):
                hit = (st, "prefix")
                break
        # 3. token containment with a distinctive anchor
        if hit is None:
            mine = _tokens(name)
            if mine:
                for k, st in known.items():
                    theirs = orig_tokens.get(k, set())
                    if not theirs:
                        continue
                    shared = mine & theirs
                    if shared and (mine <= theirs or theirs <= mine):
                        if max(len(t) for t in shared) >= 5:
                            hit = (st, "token")
                            break
        if hit:
            rows.append({"plant": name, "state": hit[0], "method": hit[1]})
        else:
            rows.append({"plant": name, "state": None, "method": "unresolved"})

    return pd.DataFrame(rows)


def resolved_map() -> dict[str, str]:
    df = build_lookup()
    ok = df[df["state"].notna()]
    return dict(zip(ok["plant"], ok["state"]))


def coverage() -> dict:
    df = build_lookup()
    by = df["method"].value_counts().to_dict()
    resolved = int(df["state"].notna().sum())
    return {"plants": int(len(df)), "resolved": resolved,
            "resolved_pct": round(resolved / max(len(df), 1) * 100, 1),
            "by_method": by,
            "unresolved": sorted(df.loc[df["state"].isna(), "plant"].tolist())[:25]}


if __name__ == "__main__":
    c = coverage()
    print(f"plant -> state resolution: {c['resolved']}/{c['plants']} "
          f"({c['resolved_pct']}%)")
    for m, n in sorted(c["by_method"].items(), key=lambda x: -x[1]):
        print(f"   {m:11s} {n}")
    if c["unresolved"]:
        print(f"\n  still unresolved ({len(c['unresolved'])} shown):")
        for p in c["unresolved"][:12]:
            print(f"    {p}")
