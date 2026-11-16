"""Download the three public data sources used in this project into data/.

- PhiUSIIL Phishing URL Dataset (UCI, CC BY 4.0)
- OpenPhish community feed (a fresh snapshot of live phishing URLs, saved with today's date)
- Tranco top-1M list (research-grade ranking of popular domains)

Usage:  python src/download_data.py
"""
import datetime
import io
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
UA = {"User-Agent": "phishing-url-detector research script"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


(DATA / "raw").mkdir(parents=True, exist_ok=True)
(DATA / "external").mkdir(parents=True, exist_ok=True)

print("PhiUSIIL ...")
zipfile.ZipFile(io.BytesIO(get("https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"))).extractall(DATA / "raw")

print("OpenPhish feed ...")
today = datetime.date.today().isoformat()
(DATA / "external" / f"openphish_feed_{today}.txt").write_bytes(get("https://openphish.com/feed.txt"))

print("Tranco top-1M ...")
zipfile.ZipFile(io.BytesIO(get("https://tranco-list.eu/top-1m.csv.zip"))).extractall(DATA / "external")
print("Done.")
