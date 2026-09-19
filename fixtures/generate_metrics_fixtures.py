import json
import os

import numpy as np
from sklearn import metrics as skm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "metrics", "testdata", "metrics_fixtures.json")

rng = np.random.default_rng(7)

fixtures = {}

# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
y_true_reg = rng.normal(size=60)
y_pred_reg = y_true_reg * 0.85 + rng.normal(size=60) * 0.4

fixtures["regression"] = {
    "y_true": y_true_reg.tolist(),
    "y_pred": y_pred_reg.tolist(),
    "mean_squared_error": float(skm.mean_squared_error(y_true_reg, y_pred_reg)),
    "root_mean_squared_error": float(
        skm.root_mean_squared_error(y_true_reg, y_pred_reg)
    ),
    "mean_absolute_error": float(skm.mean_absolute_error(y_true_reg, y_pred_reg)),
    "r2_score": float(skm.r2_score(y_true_reg, y_pred_reg)),
    "explained_variance_score": float(
        skm.explained_variance_score(y_true_reg, y_pred_reg)
    ),
    "max_error": float(skm.max_error(y_true_reg, y_pred_reg)),
}

# Constant-target regression: exercises the R2 / EVS denominator edge cases.
y_const = np.full(10, 5.0)
fixtures["constant_target"] = {
    "y_true": y_const.tolist(),
    "y_pred": y_const.tolist(),
    "mean_squared_error": float(skm.mean_squared_error(y_const, y_const)),
    "root_mean_squared_error": float(skm.root_mean_squared_error(y_const, y_const)),
    "mean_absolute_error": float(skm.mean_absolute_error(y_const, y_const)),
    "r2_score": float(skm.r2_score(y_const, y_const)),
    "explained_variance_score": float(skm.explained_variance_score(y_const, y_const)),
    "max_error": float(skm.max_error(y_const, y_const)),
}

# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------
y_true_bin = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1])
y_pred_bin = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0])

fixtures["binary_classification"] = {
    "y_true": y_true_bin.tolist(),
    "y_pred": y_pred_bin.tolist(),
    "confusion_matrix": skm.confusion_matrix(y_true_bin, y_pred_bin).tolist(),
    "accuracy": float(skm.accuracy_score(y_true_bin, y_pred_bin)),
    "precision": float(skm.precision_score(y_true_bin, y_pred_bin)),
    "recall": float(skm.recall_score(y_true_bin, y_pred_bin)),
    "f1": float(skm.f1_score(y_true_bin, y_pred_bin)),
    # pos_label=0 variant
    "precision_pos0": float(skm.precision_score(y_true_bin, y_pred_bin, pos_label=0)),
    "recall_pos0": float(skm.recall_score(y_true_bin, y_pred_bin, pos_label=0)),
    "f1_pos0": float(skm.f1_score(y_true_bin, y_pred_bin, pos_label=0)),
}

# ---------------------------------------------------------------------------
# Multiclass classification
# ---------------------------------------------------------------------------
y_true_mc = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred_mc = np.array([0, 2, 2, 0, 1, 2, 1, 1, 2, 0, 2, 1, 0, 1, 2, 2, 1, 2])

p, r, f1, sup = skm.precision_recall_fscore_support(y_true_mc, y_pred_mc)

fixtures["multiclass_classification"] = {
    "y_true": y_true_mc.tolist(),
    "y_pred": y_pred_mc.tolist(),
    "confusion_matrix": skm.confusion_matrix(y_true_mc, y_pred_mc).tolist(),
    "accuracy": float(skm.accuracy_score(y_true_mc, y_pred_mc)),
    "macro_precision": float(skm.precision_score(y_true_mc, y_pred_mc, average="macro")),
    "macro_recall": float(skm.recall_score(y_true_mc, y_pred_mc, average="macro")),
    "macro_f1": float(skm.f1_score(y_true_mc, y_pred_mc, average="macro")),
    "weighted_precision": float(
        skm.precision_score(y_true_mc, y_pred_mc, average="weighted")
    ),
    "weighted_recall": float(skm.recall_score(y_true_mc, y_pred_mc, average="weighted")),
    "weighted_f1": float(skm.f1_score(y_true_mc, y_pred_mc, average="weighted")),
    "classes": [
        {
            "label": float(l),
            "precision": float(pi),
            "recall": float(ri),
            "f1": float(f),
            "support": int(s),
        }
        for l, pi, ri, f, s in zip(np.unique(y_true_mc), p, r, f1, sup)
    ],
}

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
labels_pred = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 0])

# Perfect agreement special case.
labels_same = np.array([0, 0, 0, 1, 1, 1])

# Silhouette: two clearly separated Gaussian blobs.
X_sil = rng.normal(size=(30, 2))
X_sil[:15] += 4.0
labels_sil = np.array([0] * 15 + [1] * 15)

fixtures["clustering"] = {
    "labels_true": labels_true.tolist(),
    "labels_pred": labels_pred.tolist(),
    "adjusted_rand": float(skm.adjusted_rand_score(labels_true, labels_pred)),
    "homogeneity": float(skm.homogeneity_score(labels_true, labels_pred)),
    "completeness": float(skm.completeness_score(labels_true, labels_pred)),
    "v_measure": float(skm.v_measure_score(labels_true, labels_pred)),
    "adjusted_mutual_info": float(
        skm.adjusted_mutual_info_score(labels_true, labels_pred)
    ),
    "X_silhouette": X_sil.tolist(),
    "labels_silhouette": labels_sil.tolist(),
    "silhouette": float(skm.silhouette_score(X_sil, labels_sil)),
}

fixtures["clustering_perfect"] = {
    "labels_true": labels_same.tolist(),
    "labels_pred": labels_same.tolist(),
    "adjusted_rand": float(skm.adjusted_rand_score(labels_same, labels_same)),
    "homogeneity": float(skm.homogeneity_score(labels_same, labels_same)),
    "completeness": float(skm.completeness_score(labels_same, labels_same)),
    "v_measure": float(skm.v_measure_score(labels_same, labels_same)),
    "adjusted_mutual_info": float(
        skm.adjusted_mutual_info_score(labels_same, labels_same)
    ),
}

with open(OUT, "w") as f:
    json.dump(fixtures, f, indent=2)

for name in fixtures:
    print(f"{name} written")
