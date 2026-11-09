"""Machine learning engine: pre-processing, training, evaluation and detection.

The Random Forest model is trained on twelve profile and behavioural features
of X (Twitter) accounts and classifies an account as Fake (1) or Genuine (0).
"""
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split  # noqa: E402

FEATURES = [
    "statuses_count", "followers_count", "friends_count", "favourites_count",
    "listed_count", "FFratio", "URLratio", "average_mentions",
    "average_hashtags", "average_favorites", "average_retweet", "average_reply",
]
INT_FEATURES = FEATURES[:5]

# Existing system used for comparison (Sheikhi, 2020), as reported in the study.
EXISTING_SYSTEM = {"name": "Existing system (Sheikhi, 2020)", "accuracy": 86.57}


class DatasetError(ValueError):
    """Raised when an uploaded dataset is not usable."""


def preprocess(csv_path):
    """Load a CSV file, clean it and return (X, y)."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # unreadable or not a CSV
        raise DatasetError(f"Could not read the file as CSV ({exc}).") from exc
    df = df.drop_duplicates()
    if "FFratio" not in df.columns and {"followers_count", "friends_count"} <= set(df.columns):
        df["FFratio"] = df["followers_count"] / df["friends_count"].clip(lower=1)
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise DatasetError("Missing columns: " + ", ".join(missing))
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])]
    if df["label"].nunique() < 2 or len(df) < 20:
        raise DatasetError("The dataset must contain at least 20 rows with both "
                           "fake (1) and genuine (0) accounts.")
    return df[FEATURES], df["label"].astype(int)


def count_classes(csv_path):
    _, y = preprocess(csv_path)
    return len(y), int((y == 0).sum()), int((y == 1).sum())


def train_and_test(csv_path, model_path, chart_dir, n_trees=100):
    """Train Random Forest on 80% of the data, test on 20%, save model and charts."""
    X, y = preprocess(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=n_trees, criterion="gini",
                                   max_features="sqrt", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    acc = accuracy_score(y_test, y_pred) * 100
    metrics = {
        "train_size": int(len(X_train)), "test_size": int(len(X_test)),
        "accuracy": round(acc, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "error_rate": round(100 - acc, 2),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_trees": n_trees,
    }
    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: t[1])
    os.makedirs(chart_dir, exist_ok=True)
    _comparison_chart(metrics["accuracy"], os.path.join(chart_dir, "comparison.png"))
    _importance_chart(importances, os.path.join(chart_dir, "importance.png"))
    metrics["importances"] = [(f, round(float(v) * 100, 2)) for f, v in reversed(importances)]
    return metrics


def _comparison_chart(accuracy, path):
    labels = ["Accuracy", "Error rate"]
    proposed = [accuracy, round(100 - accuracy, 2)]
    existing = [EXISTING_SYSTEM["accuracy"], round(100 - EXISTING_SYSTEM["accuracy"], 2)]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    xs = range(len(labels))
    b1 = ax.bar([i - 0.2 for i in xs], proposed, 0.4, color="#111111", label="Proposed (Random Forest)")
    b2 = ax.bar([i + 0.2 for i in xs], existing, 0.4, color="#A0A0A0", label="Existing system")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.2f}%",
                    ha="center", fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percentage (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _importance_chart(importances, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh([f for f, _ in importances], [v * 100 for _, v in importances], color="#111111")
    ax.set_xlabel("Importance (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


_model_cache = {}


def load_model(model_path):
    mtime = os.path.getmtime(model_path)
    cached = _model_cache.get(model_path)
    if cached is None or cached[0] != mtime:
        _model_cache[model_path] = (mtime, joblib.load(model_path))
    return _model_cache[model_path][1]


def detect(values, model_path):
    """Classify one profile. Returns (label, confidence %, votes for label, total trees)."""
    model = load_model(model_path)
    x = pd.DataFrame([[float(values[f]) for f in FEATURES]], columns=FEATURES)
    votes_fake = sum(int(tree.predict(x.values)[0]) for tree in model.estimators_)
    n = len(model.estimators_)
    label = 1 if votes_fake * 2 > n else 0
    votes = votes_fake if label == 1 else n - votes_fake
    return ("Fake" if label == 1 else "Genuine"), round(votes / n * 100, 2), votes, n
