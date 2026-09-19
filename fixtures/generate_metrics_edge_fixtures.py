"""Differential fixtures for metrics on degenerate and random label sets.

Each case records what sklearn returns, or that it raises, so the Go tests can
check both the values and the error behavior on edge cases (one class, one
sample, constant clusterings, labels that do not include the positive class).
"""
import json
import os
import warnings

import numpy as np
from sklearn import metrics as skm

warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "metrics", "testdata", "metrics_edge_fixtures.json")

rng = np.random.default_rng(2024)


def run(fn):
    """Return a JSON-safe value, or {"error": true} if sklearn raises."""
    try:
        v = fn()
    except Exception:
        return {"error": True}
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    return v


def label_sets():
    hand = [
        ([0], [0]),
        ([1], [1]),
        ([0], [1]),
        ([0, 0, 0], [0, 0, 0]),
        ([1, 1, 1], [1, 1, 1]),
        ([5, 5], [5, 5]),
        ([0, 1, 0], [0, 0, 0]),
        ([0, 0, 0], [0, 1, 0]),
        ([0, 1], [1, 0]),
        ([0, 1, 2], [0, 1, 2]),
        ([0, 1, 2], [2, 1, 0]),
        ([0, 0, 1, 1], [0, 0, 0, 0]),
        ([0, 0, 0, 0], [0, 1, 2, 3]),
        ([0, 1, 2, 3], [0, 0, 0, 0]),
        ([2, 3, 2], [2, 3, 3]),
        ([-1, 1, -1, 1], [1, 1, -1, -1]),
        ([0.5, 1.5, 0.5], [0.5, 0.5, 1.5]),
    ]
    rand = []
    for _ in range(300):
        n = int(rng.integers(1, 14))
        k = int(rng.integers(1, 5))
        rand.append((rng.integers(0, k, n).tolist(), rng.integers(0, k, n).tolist()))
    return hand + rand


cases = []
for yt, yp in label_sets():
    yt_f, yp_f = [float(v) for v in yt], [float(v) for v in yp]
    case = {
        "y_true": yt_f,
        "y_pred": yp_f,
        "accuracy": run(lambda: skm.accuracy_score(yt, yp)),
        "confusion": run(lambda: skm.confusion_matrix(yt, yp).tolist()),
        "precision": run(lambda: skm.precision_score(yt, yp, zero_division=0)),
        "recall": run(lambda: skm.recall_score(yt, yp, zero_division=0)),
        "f1": run(lambda: skm.f1_score(yt, yp, zero_division=0)),
    }
    for avg in ("macro", "weighted"):
        case[avg] = run(
            lambda: [
                float(x)
                for x in skm.precision_recall_fscore_support(
                    yt, yp, average=avg, zero_division=0
                )[:3]
            ]
        )
    # Clustering metrics treat the two vectors as label assignments.
    case["ari"] = run(lambda: skm.adjusted_rand_score(yt, yp))
    case["homogeneity"] = run(lambda: skm.homogeneity_score(yt, yp))
    case["completeness"] = run(lambda: skm.completeness_score(yt, yp))
    case["v_measure"] = run(lambda: skm.v_measure_score(yt, yp))
    case["ami"] = run(lambda: skm.adjusted_mutual_info_score(yt, yp))
    cases.append(case)

# Regression metrics on random vectors, including constant targets.
reg = []
for _ in range(60):
    n = int(rng.integers(1, 12))
    yt = rng.normal(size=n) * float(rng.choice([0.0, 1.0, 5.0]))
    yp = yt + rng.normal(size=n) * float(rng.choice([0.0, 0.1, 1.0]))
    if rng.random() < 0.2:
        yt = np.full(n, 3.0)
    reg.append(
        {
            "y_true": yt.tolist(),
            "y_pred": yp.tolist(),
            "mse": run(lambda: skm.mean_squared_error(yt, yp)),
            "rmse": run(lambda: skm.root_mean_squared_error(yt, yp)),
            "mae": run(lambda: skm.mean_absolute_error(yt, yp)),
            "r2": run(lambda: skm.r2_score(yt, yp)),
            "explained_variance": run(lambda: skm.explained_variance_score(yt, yp)),
            "max_error": run(lambda: skm.max_error(yt, yp)),
        }
    )

# Silhouette needs points and a labeling with 2 <= n_labels <= n_samples - 1.
sil = []
for _ in range(80):
    n = int(rng.integers(1, 15))
    d = int(rng.integers(1, 4))
    X = rng.normal(size=(n, d))
    if rng.random() < 0.15:
        X = np.round(X)  # duplicate points -> zero distances
    labels = rng.integers(0, int(rng.integers(1, 5)), n)
    sil.append(
        {
            "X": X.tolist(),
            "labels": [float(v) for v in labels],
            "silhouette": run(lambda: skm.silhouette_score(X, labels)),
        }
    )

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump({"labels": cases, "regression": reg, "silhouette": sil}, f)

print("metrics edge fixtures written:", len(cases), "label cases,", len(reg), "regression,", len(sil), "silhouette")
