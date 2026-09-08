"""
Builds a clean, balanced train/test CSV from the raw PhiUSIIL dataset.

Three things this fixes vs. using the raw file directly:

1. Size - the raw file has ~236k rows. That's overkill for a class project
   and slow to featurize in pure Python, so this samples a balanced 30k/30k
   subset.
2. Path bias - every single legitimate URL in the raw dataset is a bare
   homepage (e.g. "https://www.example.com"), while a good share of the
   phishing URLs have a path. Train on that directly and the model just
   learns "URL has a path -> phishing", which falls apart on a totally
   normal link like "https://en.wikipedia.org/wiki/Phishing". To stop that
   shortcut from working, legitimate rows get a synthetic but
   realistic-looking (randomized, not templated) path appended.
3. "www." bias - checked separately and it's even more extreme than the path
   bias: in this dataset *every single* legitimate URL starts with a
   subdomain (almost always "www."), while a meaningful chunk of phishing
   URLs are bare domains. Trained as-is, the model learns "no www. -> almost
   certainly phishing" - which flags huge numbers of real, modern legitimate
   sites that skip www (github.com, stackoverflow.com, openai.com, ...).
   Fixed by stripping "www." from a big chunk of the legitimate sample so
   bare-domain legitimate examples actually exist to learn from.

Run: python data/prepare_dataset.py
"""
import os
import random

import pandas as pd

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "PhiUSIIL_Phishing_URL_Dataset.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed_urls.csv")

SAMPLE_PER_CLASS = 30_000
# probability a legitimate row gets a synthetic path appended - applied to
# (almost) every row rather than a fixed subset, with the path shape itself
# randomized, so the model can't just memorize a handful of fixed path
# lengths/shapes as "the safe ones" and flag any other real-world path.
PATH_AUGMENT_PROB = 0.85
# fraction of legitimate rows to strip "www." from, so the training set has
# plenty of bare-domain legitimate examples (see bias #3 above)
STRIP_WWW_PROB = 0.55

WORDS = [
    "about", "contact", "products", "product", "blog", "article", "news",
    "help", "support", "faq", "login", "signup", "account", "settings",
    "search", "category", "docs", "guide", "profile", "cart", "checkout",
    "wiki", "team", "careers", "pricing", "download", "events", "gallery",
    "review", "story", "topic", "tag", "user", "post", "page", "index",
    "watch", "video", "photo", "map", "store", "shop", "item", "order",
]


def random_path() -> str:
    depth = random.choice([1, 1, 2, 2, 3])
    segments = []
    for _ in range(depth):
        word = random.choice(WORDS)
        if random.random() < 0.35:
            word += str(random.randint(1, 999999))
        if random.random() < 0.15:
            word = word + "-" + random.choice(WORDS)
        segments.append(word)
    path = "/".join(segments)
    if random.random() < 0.2:
        path += "?" + random.choice(["id", "ref", "q", "page"]) + "=" + str(random.randint(1, 9999))
    return path


def add_fake_path(url: str) -> str:
    if random.random() >= PATH_AUGMENT_PROB:
        return url  # keep some legitimate rows as bare homepages too
    return url.rstrip("/") + "/" + random_path()


def maybe_strip_www(url: str) -> str:
    if random.random() >= STRIP_WWW_PROB:
        return url
    return url.replace("://www.", "://", 1)


def main():
    print(f"reading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH, usecols=["URL", "label"])

    # raw dataset: label 1 = legitimate, label 0 = phishing.
    # this project's convention (matches the API response): 1 = phishing.
    df["is_phishing"] = 1 - df["label"]

    legit = df[df["is_phishing"] == 0].sample(n=SAMPLE_PER_CLASS, random_state=42).copy()
    phish = df[df["is_phishing"] == 1].sample(n=SAMPLE_PER_CLASS, random_state=42).copy()

    legit["URL"] = legit["URL"].apply(maybe_strip_www)
    legit["URL"] = legit["URL"].apply(add_fake_path)

    out = pd.concat([legit, phish], ignore_index=True)[["URL", "is_phishing"]]
    out = out.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(out)} rows to {OUT_PATH}")
    print(out["is_phishing"].value_counts())


if __name__ == "__main__":
    main()
