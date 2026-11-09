#!/usr/bin/env bash
# One-click start for Linux / macOS.
set -e
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
[ -f instance/fpd.sqlite3 ] || python seed.py --email admin@fpd.local --password Admin12345 --dataset data/sample_profiles_DEMO.csv
python app.py
