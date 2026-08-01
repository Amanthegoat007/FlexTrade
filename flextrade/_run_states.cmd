@echo off
cd /d "C:\Users\AmanPratapSingh\Desktop\HACKATHON\flextrade"
set PYTHONIOENCODING=utf-8
"C:\Users\AmanPratapSingh\AppData\Local\Programs\Python\Python310\python.exe" -c "from ingest import states; snap,m=states.get_india_snapshot(); n=snap['national']; print(m['asof'], 'live' if m['live'] else 'CACHED', len(snap['states']), 'states,', n.get('demand_met_mw'), 'MW national')" >> "C:\Users\AmanPratapSingh\Desktop\HACKATHON\flextrade\logs\states.log" 2>&1
