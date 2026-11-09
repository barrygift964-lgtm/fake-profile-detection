@echo off
REM One-click start for Windows: creates the environment, installs packages,
REM creates the admin account, trains on the dataset and starts the server.
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt
if not exist instance\fpd.sqlite3 (
    python seed.py --email admin@fpd.local --password Admin12345 --dataset data\sample_profiles_DEMO.csv
)
start "" http://127.0.0.1:5000
python app.py
