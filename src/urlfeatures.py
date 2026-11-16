"""Turn a URL into a hostname and a set of hand-crafted lexical features.

Everything here looks only at the text of the URL. Nothing is fetched from the
network, so the model can score a link without visiting it.
"""
import math
import re
from collections import Counter
from urllib.parse import urlsplit

# Two-level public suffixes that matter for the data used here (not exhaustive).
TWO_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.nz", "co.za", "org.za",
    "com.ng", "org.ng", "gov.ng", "edu.ng", "co.ke", "co.in", "net.in", "org.in", "com.br", "com.cn",
    "com.mx", "co.jp", "ne.jp", "or.jp", "com.tr", "com.sg", "com.my", "co.id", "com.ar", "com.co",
    "com.pk", "com.ph", "com.vn", "com.ua", "co.kr", "com.hk", "com.tw", "com.eg", "com.sa",
}

# Platforms where anyone can host a page on a subdomain. Attackers use them a lot.
FREE_HOSTING = (
    "firebaseapp.com", "web.app", "godaddysites.com", "weebly.com", "wixsite.com", "000webhostapp.com",
    "blogspot.com", "github.io", "netlify.app", "vercel.app", "pages.dev", "workers.dev", "glitch.me",
    "ipfs", "herokuapp.com", "azurewebsites.net", "appspot.com", "r2.dev", "repl.co", "square.site",
    "webflow.io", "framer.app", "duckdns.org", "ngrok", "sites.google.com", "carrd.co", "mystrikingly.com",
)

BRANDS = (
    "paypal", "apple", "icloud", "microsoft", "office", "outlook", "netflix", "amazon", "facebook",
    "instagram", "whatsapp", "google", "metamask", "coinbase", "binance", "trustwallet", "ledger",
    "dhl", "fedex", "usps", "chase", "wellsfargo", "bankofamerica", "att", "adobe", "docusign", "steam",
)

LURE_WORDS = (
    "login", "signin", "secure", "verify", "verification", "account", "update", "wallet", "support",
    "auth", "bank", "confirm", "recover", "unlock", "billing", "invoice", "webmail", "service", "help",
)

VOWELS = set("aeiou")


def to_host(url: str) -> str:
    """Lower-case hostname without scheme, port, path or a leading 'www.'."""
    url = url.strip()
    if "://" not in url:
        url = "http://" + url
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def registered_domain(host: str) -> str:
    """Approximate 'example.co.uk' from 'a.b.example.co.uk'."""
    labels = host.split(".")
    if re.fullmatch(r"[\d.]+", host):
        return host
    n = 3 if len(labels) >= 3 and ".".join(labels[-2:]) in TWO_LEVEL_SUFFIXES else 2
    return ".".join(labels[-n:])


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def _longest_run(s: str, chars) -> int:
    best = cur = 0
    for ch in s:
        cur = cur + 1 if ch in chars else 0
        best = max(best, cur)
    return best


def features(host: str) -> dict:
    reg = registered_domain(host)
    labels = host.split(".")
    letters = [c for c in host if c.isalpha()]
    consonants = set("bcdfghjklmnpqrstvwxyz")
    tld = labels[-1] if labels else ""
    return {
        "host_length": len(host),
        "num_labels": len(labels),
        "num_subdomain_labels": max(0, len(labels) - len(reg.split("."))),
        "longest_label": max((len(l) for l in labels), default=0),
        "num_hyphens": host.count("-"),
        "num_digits": sum(c.isdigit() for c in host),
        "digit_ratio": sum(c.isdigit() for c in host) / max(1, len(host)),
        "vowel_ratio": sum(c in VOWELS for c in letters) / max(1, len(letters)),
        "longest_consonant_run": _longest_run(host, consonants),
        "longest_digit_run": _longest_run(host, set("0123456789")),
        "entropy": _entropy(host.replace(".", "")),
        "is_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))),
        "has_punycode": int("xn--" in host),
        "tld_length": len(tld),
        "tld_is_generic": int(tld in {"com", "org", "net", "edu", "gov"}),
        "free_hosting": int(any(p in host for p in FREE_HOSTING)),
        "brand_in_subdomain": int(any(b in host[: max(0, len(host) - len(reg))] for b in BRANDS)),
        "brand_anywhere": int(any(b in host for b in BRANDS)),
        "lure_word_count": sum(w in host for w in LURE_WORDS),
    }


FEATURE_NAMES = list(features("example.com").keys())
