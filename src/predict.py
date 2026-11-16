"""Score one or more URLs with the trained model. The URLs are never visited.

Usage:
    python src/predict.py "http://paypal-login-verify.example.web.app/" "https://www.wikipedia.org"
    python src/predict.py --file urls.txt
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from urlfeatures import features, to_host  # noqa: E402

MODEL = Path(__file__).resolve().parents[1] / "models" / "phishing_url_model.joblib"


def score(urls, bundle):
    hosts = [to_host(u) for u in urls]
    X = pd.DataFrame([features(h) for h in hosts], columns=bundle["feature_names"])
    p = (bundle["hgb"].predict_proba(X)[:, 1] + bundle["chars"].predict_proba(hosts)[:, 1]) / 2
    return hosts, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*")
    ap.add_argument("--file", help="text file with one URL per line")
    args = ap.parse_args()
    urls = list(args.urls)
    if args.file:
        urls += [l.strip() for l in open(args.file, encoding="utf-8") if l.strip()]
    if not urls:
        ap.error("give at least one URL")
    bundle = joblib.load(MODEL)
    hosts, probs = score(urls, bundle)
    for url, host, p in zip(urls, hosts, probs):
        verdict = "PHISHING" if p >= bundle["threshold"] else "looks ok"
        print(f"{p:6.3f}  {verdict:9s} {host}")


if __name__ == "__main__":
    main()
