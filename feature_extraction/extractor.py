"""
Layer 2 - Feature Extraction.

Takes a raw URL string (from the client layer) and turns it into a fixed-size
vector of numeric features that a classifier can consume. Every feature here
is computed purely from the URL text itself - no network calls, no WHOIS
lookups, no page content - so this layer stays fast and works offline.

Both model training (model/train_model.py) and the live API (app.py) import
FEATURE_NAMES and url_to_features from this one module, so the training-time
feature order and the inference-time feature order can never drift apart.
"""
import math
import re
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click",
    "loan", "men", "gdn", "kim", "country", "science", "date", "faith",
}

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "cutt.ly", "rebrand.ly", "tiny.cc",
}

FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "digit_ratio",
    "num_special_chars",
    "num_subdomains",
    "has_ip_host",
    "has_at_symbol",
    "has_double_slash_redirect",
    "is_https",
    "has_suspicious_tld",
    "is_shortened",
    "brand_token_in_hostname",
    "url_entropy",
]

_IPV4_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
_SPECIAL_CHARS_RE = re.compile(r"[!$^*()+{}\[\]|\\;:'\",<>]")


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def _normalize(url: str) -> str:
    url = url.strip()
    if "://" not in url:
        url = "http://" + url
    return url


def url_to_features(raw_url: str) -> list[float]:
    url = _normalize(raw_url)
    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    digits = sum(ch.isdigit() for ch in url)

    tld = hostname.rsplit(".", 1)[-1].lower() if "." in hostname else ""
    registrable_labels = hostname.split(".") if hostname else []
    # "www.paypal.com" -> 1 subdomain-ish label before the registrable domain;
    # a bare "example.com" has 0.
    num_subdomains = max(len(registrable_labels) - 2, 0)

    # classic phishing trick: stuff "https"/a brand-shaped token into a
    # subdomain so the real domain gets pushed further along, e.g.
    # "paypal.com.verify-login.ru" - the registrable domain is verify-login.ru
    # but "paypal" sits right there in the hostname looking legitimate.
    brand_hint = 1.0 if re.search(r"(secure|login|verify|account|update|signin)", hostname) else 0.0

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": digits,
        "digit_ratio": digits / len(url) if url else 0.0,
        "num_special_chars": len(_SPECIAL_CHARS_RE.findall(url)),
        "num_subdomains": num_subdomains,
        "has_ip_host": 1.0 if _IPV4_RE.match(hostname) else 0.0,
        "has_at_symbol": 1.0 if "@" in url else 0.0,
        "has_double_slash_redirect": 1.0 if url.find("//", url.find("://") + 3) != -1 else 0.0,
        "is_https": 1.0 if parsed.scheme == "https" else 0.0,
        "has_suspicious_tld": 1.0 if tld in SUSPICIOUS_TLDS else 0.0,
        "is_shortened": 1.0 if hostname in SHORTENER_DOMAINS else 0.0,
        "brand_token_in_hostname": brand_hint,
        "url_entropy": _shannon_entropy(url),
    }

    return [features[name] for name in FEATURE_NAMES]
