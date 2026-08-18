# Upstream live sources — verified 2026-08-18

Every URL below was probed from this machine on 2026-08-18. Status is what the
server actually returned, not what a vendor page claims. The list was recovered
from energymap.in/developers/provenance and then checked one by one; two of
their published sources are dead, which is noted.

All of these are public government / utility endpoints. None require a key.

## Live and NOT yet collected by FlexTrade

| Endpoint | Verified | Payload | Why it matters |
|---|---|---|---|
| `npp.gov.in/dashBoard/demandmet2chartdata` | 200 JSON | 6 fuels x 4-min: THERMAL, HYDRO, WIND, SOLAR, NUCLEAR, GAS | National fuel mix at 4-min. Rolling 4.1h window ONLY — unrecoverable if not polled. |
| `npp.gov.in/dashBoard/demandmet1chartdata` | 200 JSON | All-India DEMAND MET, 4-min | National demand at 4-min. Same 4.1h window. |
| `upsldc.org/assets/dataset/real-time-summary.json` | 200 JSON | SCHEDULE_MW, DRAWL_MW, OD_UD, DEMAND_MW, TOTAL_SSGS_MW, UP_THERMAL, IPP_THERMAL, UP_HYDRO, COGEN_CPP, RE_SOLAR, FREQUENC_HZ, FREQUENC_RAW, DEVIATION_RATE_PAISE_PER_UNIT | Typed replacement for the fragile dynamic-data.json string parse. Carries the DSM deviation rate directly. |
| `sldcapi.pstcl.org/wsDataService.asmx/dynamicData` | 200 JSON | updateDate, frequencyHz, drawalMW, scheduleMW | Punjab. New state, clean JSON, DSM triplet. |
| `sldckerala.com/index.php?id=9` | 200 HTML | system statistics | Kerala. Needs parsing. |
| `hpaldc.org/` | 200 HTML | HP load despatch | Himachal. Needs parsing. |
| `cea.nic.in/api/psp_peak.php` | TIMEOUT | power supply position, peak | Documented; no response from this host. Retry. |
| `cea.nic.in/api/psp_energy.php` | TIMEOUT | power supply position, energy | Documented; no response from this host. Retry. |
| `cea.nic.in/api/power_generation.php` | TIMEOUT | generation | Documented; no response from this host. Retry. |

## Live and ALREADY collected by FlexTrade

| Endpoint | Module |
|---|---|
| `meritindia.in/StateWiseDetails/BindCurrentStateStatus` | ingest/states.py, collect_ci.py |
| `vidyutpravah.in/PXDashboard/BindStatePricesFromJS` | ingest/vidyutpravah.py |
| `iexindia.com/market-data/{day-ahead,real-time,green-day-ahead}-market/market-snapshot` | ingest/iex.py |
| `delhisldc.org/Loaddata.aspx`, `Freqcurve.aspx`, `bess.aspx` | ingest/sldc.py, ingest/bess.py |
| `kptclsldc.in/` | ingest/kptcl.py |
| `upsldc.org/assets/dataset/dynamic-data.json` | ingest/upsldc.py (SUPERSEDED — see above) |
| `sldc.rajasthan.gov.in/rrvpnl/view-realtime-data/show` | ingest/sldc.py |
| `sldcguj.com/` | ingest/sldc.py |
| `npp.gov.in/public-reports/cea/daily/dgr/` | ingest/plant_state.py |
| `npp.gov.in/public-reports/cea/daily/fuel/` | ingest/coal.py |
| `api.open-meteo.com/v1/forecast`, `archive-api.open-meteo.com/v1/archive` | ingest/weather.py |

## Published by energymap.in but DEAD as of 2026-08-18

| Endpoint | Status |
|---|---|
| `rtservice.aptransco.co.in/rtuPointDataJson` | 404 |
| `webapi.grid-india.in/api/v1/communique` | 404 |

## Non-endpoint sources they also use

- `apdcl.org/website/docs/documents/daily_power.pdf` — Assam, daily PDF
- `upcl.uk.gov.in/document-category/power-outage/` — Uttarakhand outages
- `download.geofabrik.de/asia/india-latest.osm.pbf` — OSM transmission geometry
- `cea.nic.in/cdm-co2-baseline-database/` — CO2 emission factors
- `posoco.in/reports/daily-reports/` — POSOCO daily reports
- `grid-india.in/operations/atc-revisions` — ATC corridor revisions
