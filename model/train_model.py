"""
Trains the cloud classifier (RandomForestClassifier) used by the model
inference layer and saves it + evaluation metrics to model/artifacts/.

Run: python -m model.train_model
"""
import json
import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from feature_extraction import FEATURE_NAMES, url_to_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_urls.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model", "artifacts")


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print(f"loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    print(f"extracting features for {len(df)} URLs ...")
    t0 = time.time()
    X = pd.DataFrame([url_to_features(u) for u in df["URL"]], columns=FEATURE_NAMES)
    y = df["is_phishing"]
    print(f"done in {time.time() - t0:.1f}s")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("training RandomForestClassifier ...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_importance": dict(sorted(
            zip(FEATURE_NAMES, model.feature_importances_.tolist()),
            key=lambda kv: kv[1], reverse=True,
        )),
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "feature_importance"}, indent=2))

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "random_forest.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"saved model + metrics to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
