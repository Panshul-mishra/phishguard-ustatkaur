"""
Layer 4 - Response.

Turns a raw model score into what a client actually wants: a 0-1 score, a
human-readable risk level, and a short, plain-English explanation of what
drove the score - built from which risky signals fired, weighted by how much
the model actually relies on each feature (its global feature importance)
rather than a fixed if/else priority list.
"""

# (feature_name, condition-on-raw-value, message) - checked in order of the
# model's own feature_importances_, not the order written here.
_REASON_RULES = {
    "has_ip_host": (lambda v: v == 1.0, "the link uses a raw IP address instead of a domain name"),
    "has_at_symbol": (lambda v: v == 1.0, "the URL contains an '@' symbol, often used to hide the real destination"),
    "has_suspicious_tld": (lambda v: v == 1.0, "the domain uses a top-level domain commonly abused for phishing"),
    "is_shortened": (lambda v: v == 1.0, "the URL was created with a link-shortening service, hiding the real destination"),
    "has_double_slash_redirect": (lambda v: v == 1.0, "the URL path contains a redirect pattern ('//') after the domain"),
    "brand_token_in_hostname": (lambda v: v == 1.0, "the hostname contains words like 'secure' / 'login' / 'verify', often used to look trustworthy"),
    "is_https": (lambda v: v == 0.0, "the connection is not encrypted (no HTTPS)"),
    "num_hyphens": (lambda v: v >= 3, "the domain contains an unusually high number of hyphens"),
    "num_subdomains": (lambda v: v >= 3, "the URL has an unusually deep chain of subdomains"),
    "digit_ratio": (lambda v: v >= 0.25, "the URL text is made up of an unusually large proportion of digits"),
    "url_length": (lambda v: v >= 75, "the URL is unusually long"),
    "url_entropy": (lambda v: v >= 4.5, "the URL text looks randomly generated rather than human-chosen"),
}

_SAFE_REASON_RULES = {
    "is_https": (lambda v: v == 1.0, "the connection is encrypted (HTTPS)"),
    "url_length": (lambda v: v < 30, "the URL is short and simple"),
    "num_subdomains": (lambda v: v == 0, "the domain has no unusual subdomain nesting"),
}


def _score_to_level(score: float) -> str:
    if score < 0.2:
        return "Safe"
    if score < 0.4:
        return "Low Risk"
    if score < 0.6:
        return "Medium Risk"
    if score < 0.8:
        return "High Risk"
    return "Phishing"


def _explain(features: dict, feature_importance: dict, score: float, max_reasons: int = 4) -> list[str]:
    ranked = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)
    rules = _REASON_RULES if score >= 0.5 else _SAFE_REASON_RULES

    reasons = []
    for name, _importance in ranked:
        if name not in rules:
            continue
        condition, message = rules[name]
        if condition(features.get(name, 0)):
            reasons.append(message)
        if len(reasons) >= max_reasons:
            break

    if not reasons:
        reasons = (
            ["multiple minor risk signals combined to raise the score"]
            if score >= 0.5
            else ["no strong risk signals were detected in the URL structure"]
        )
    return reasons


def build_response(url: str, features: dict, feature_importance: dict, score: float, latency_ms: float) -> dict:
    return {
        "url": url,
        "score": round(score, 4),
        "level": _score_to_level(score),
        "explanation": _explain(features, feature_importance, score),
        "latency_ms": round(latency_ms, 2),
    }
