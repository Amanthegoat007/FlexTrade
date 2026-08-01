@echo off
cd /d "C:\Users\AmanPratapSingh\Desktop\HACKATHON\flextrade"
set PYTHONIOENCODING=utf-8
"C:\Users\AmanPratapSingh\AppData\Local\Programs\Python\Python310\python.exe" poll_bess.py 120 >> "C:\Users\AmanPratapSingh\Desktop\HACKATHON\flextrade\logs\poller.log" 2>&1
