@echo off
cd /d "C:\Users\AmanPratapSingh\Desktop\HACKATHON\flextrade"
set PYTHONIOENCODING=utf-8
"C:\Users\AmanPratapSingh\AppData\Local\Programs\Python\Python310\python.exe" -c "from ingest import bess; r,m=bess.poll_once(); print(r['ts'], r['soc_pct'], r['discharge_mw'], 'live' if m['live'] else 'CACHED')" >> "C:\Users\AmanPratapSingh\Desktop\HACKATHON\flextrade\logs\poller.log" 2>&1
