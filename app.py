"""
Entry point that wires the 4 layers together and exposes them over HTTP.

GET  /                  -> client layer: the web dashboard
GET  /api/health         -> liveness check
POST /api/predict        -> url in -> {score, level, explanation} out

Run locally: python app.py
"""
import os
import time

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from feature_extraction import FEATURE_NAMES, url_to_features
from model_inference import PhishingClassifier
from response import build_response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "artifacts", "random_forest.joblib")
CLIENT_DIR = os.path.join(BASE_DIR, "client")

app = Flask(__name__, static_folder=CLIENT_DIR, static_url_path="")
CORS(app)  # allows a browser extension / a different-origin client to call this API

classifier = PhishingClassifier(MODEL_PATH)


@app.get("/")
def index():
    return send_from_directory(CLIENT_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    if not url:
        return jsonify({"error": "'url' is required"}), 400
    if len(url) > 2048:
        return jsonify({"error": "url is too long"}), 400

    start = time.time()

    # Layer 2: feature extraction
    vector = url_to_features(url)
    feature_dict = dict(zip(FEATURE_NAMES, vector))

    # Layer 3: model inference (cloud-hosted random forest)
    score = classifier.predict(vector)

    # Layer 4: response (score, level, explanation)
    latency_ms = (time.time() - start) * 1000
    response = build_response(url, feature_dict, classifier.feature_importance, score, latency_ms)

    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
