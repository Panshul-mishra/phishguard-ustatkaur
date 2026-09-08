# PhishGuard

Ustat Kaur · BCA · Enrollment No. A1004824040

A cloud-hosted phishing URL classifier built around the required 4-layer design: **client -> feature extraction -> model inference -> response**. A user (or another program) submits a URL, it gets converted into numeric features, a random forest classifier scores it, and the result comes back as a risk score, a risk level, and a plain-English explanation of what drove the score.

## Architecture

```
Client Layer          web dashboard (client/) - submits a URL to the API
      |
Feature Extraction    feature_extraction/extractor.py -> 20 numeric features
      |
Model Inference       model_inference/classifier.py -> random forest (scikit-learn)
      |
Response Layer        response/builder.py -> { score, level, explanation }
```

- `feature_extraction/extractor.py` - turns a raw URL string into 20 numeric features: length-based (URL/hostname/path/query length), character counts (dots, hyphens, digits, slashes, special chars), structural flags (IP-as-host, `@` symbol, HTTPS, suspicious TLD, URL shortener, subdomain depth), and a Shannon-entropy score of the URL text.
- `model_inference/classifier.py` - loads the trained `RandomForestClassifier` once at startup and exposes a single `predict(features) -> probability` method, so the API layer never touches scikit-learn directly.
- `response/builder.py` - converts the raw probability into a 5-level risk label (Safe / Low / Medium / High / Phishing) and builds a short explanation by checking which of the model's most important features actually fired as "risky" for that specific URL - not a fixed priority list, so the explanation adapts if the model is retrained on different data.
- `app.py` - Flask app that wires the four layers together and serves the dashboard.
- `model/prepare_dataset.py` (see `data/prepare_dataset.py`) - builds the training set from the raw dataset.
- `model/train_model.py` - trains the classifier and saves it + metrics to `model/artifacts/`.

## Dataset

[PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (UCI, 2024, ~235k labeled URLs). Only the `URL` and `label` columns are used - all of the dataset's own 50+ pre-computed columns are ignored, since the point of this project is to do that feature engineering ourselves.

**Two dataset biases found and fixed while building this** (worth mentioning in a viva - this is exactly the kind of thing that's invisible in the accuracy number but breaks the model on real URLs):

1. **Path bias.** Every legitimate URL in the raw dataset is a bare homepage (`https://www.example.com`), while a good share of phishing URLs have a path. Trained as-is, the model just learns "URL has a path -> phishing", which fails on any normal link like `https://en.wikipedia.org/wiki/Phishing`. Fixed in `data/prepare_dataset.py` by appending a randomly-generated (not templated) path to most legitimate URLs, so path length/depth stops being a free shortcut.
2. **`www.` bias - the bigger one.** Checked the raw data directly: **100% of legitimate URLs start with a subdomain (almost always `www.`)**, while a meaningful share of phishing URLs are bare domains. Trained as-is, the model learns "no `www.` -> phishing" almost perfectly on the training distribution - and then flags huge numbers of real, modern legitimate sites that skip `www.` entirely (`github.com`, `stackoverflow.com`, `openai.com`, ...) as phishing with 95%+ confidence. Caught this by testing the trained model on real-world URLs instead of trusting the held-out accuracy score, then fixed it in `data/prepare_dataset.py` by stripping `www.` from ~55% of the legitimate sample.

To rebuild the dataset:
```
venv/Scripts/python.exe data/prepare_dataset.py
```

## Setup

```
python -m venv venv
venv/Scripts/pip install -r requirements.txt
venv/Scripts/python.exe data/prepare_dataset.py
venv/Scripts/python.exe -m model.train_model
venv/Scripts/python.exe app.py
```
Then open `http://localhost:5000`.

## Evaluation

On a held-out 20% split (12,000 URLs): **97.1% accuracy**, 0.997 precision, 0.945 recall, 0.993 ROC-AUC (see `model/artifacts/metrics.json`). Precision is kept deliberately high relative to recall - a false "Phishing" verdict on a legitimate site is more damaging to user trust in a tool like this than occasionally missing a phishing URL.

Recall dropped a bit (from ~99% to ~94%) after fixing the two biases above versus an earlier version trained without the bias fixes - which is expected and correct: some of that lost "recall" was the model exploiting `www.`/path shortcuts rather than real phishing signals. The honest number on a debiased dataset is lower but generalizes far better, which matters more than a inflated held-out score for a tool meant to run on arbitrary real-world URLs.

**Honest limitation:** URLs that don't fit common shapes at all - e.g. a deep link with a URL fragment like `https://mail.google.com/mail/u/0/#inbox` - can still get an inflated score, since the training data has relatively few examples of that specific shape on the legitimate side. A production system would combine this URL-only score with domain-age/certificate data and page content analysis rather than relying on URL text alone.

## API

```
POST /api/predict
{ "url": "https://paypal-secure.account-update.info/signin" }

->

{
  "url": "https://paypal-secure.account-update.info/signin",
  "score": 0.6494,
  "level": "High Risk",
  "explanation": [
    "the hostname contains words like 'secure' / 'login' / 'verify', often used to look trustworthy"
  ],
  "latency_ms": 59.36
}
```

## Deployment

Runs as a small Flask app behind gunicorn (see `Dockerfile`), so it works on any container host - Render, Railway, Fly.io, Google Cloud Run, or just `python app.py` locally for a viva demo. **No GitHub account is required** - see the options below.

### Option A - run it locally (simplest, no cloud needed)
Everything above (`Setup`) runs entirely on a laptop. For a viva/demo this is often enough - open `http://localhost:5000` and show it live, no deployment required at all.

### Option B - deploy without your own GitHub repo (PythonAnywhere)
[PythonAnywhere](https://www.pythonanywhere.com) has a free tier that lets you upload files directly through its web dashboard (or `git clone` a *public* URL if you ever do use GitHub) - no GitHub account needed. Upload the project folder, create a virtualenv in their console, `pip install -r requirements.txt`, and configure a Flask web app pointing at `app.py`.

### Option C - deploy with a CLI, no web-based GitHub connection (Railway / Fly.io)
Both `railway up` (Railway) and `flyctl deploy` (Fly.io) deploy straight from the project folder on your machine to their cloud - you only need an account with them (email signup is enough), not a GitHub account, and no repository is involved at all.

### Option D - deploy via Render like the reference example
Render's standard flow does need a Git repository, but it doesn't have to be *her* GitHub - it can be pushed from **any** GitHub account (e.g. yours), since Render only needs read access to wherever the code lives. The live URL and the deployed app itself don't reveal whose account it was pushed from.

The same Docker image (`Dockerfile`) works for B, C, and D without changes.

## Browser extension

Not included in this build (kept the client layer to the web dashboard only) - the 4-layer requirement is satisfied by the dashboard alone, since it's just one more caller of the same `/api/predict` endpoint. A Manifest V3 extension could be added later by pointing its `fetch()` call at the deployed API.
