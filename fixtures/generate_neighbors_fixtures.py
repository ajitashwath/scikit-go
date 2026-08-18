import json
import os

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "neighbors_fixtures.json")

rng = np.random.default_rng(21)

fixtures = {}


def to_list(a):
    return a.tolist()


def base(key, X, y, X_test):
    fixtures[key] = {"X": to_list(X), "y": to_list(y), "X_test": to_list(X_test)}


# ---------------------------------------------------------------------------
# Regression data
# ---------------------------------------------------------------------------
X_reg, y_reg = make_regression(
    n_samples=60, n_features=3, n_informative=3, noise=0.05, random_state=1
)
X_reg_test = rng.normal(size=(20, 3)) * 2.0
base("knn_reg_k3_uniform", X_reg, y_reg, X_reg_test)
base("knn_reg_k5_uniform", X_reg, y_reg, X_reg_test)
base("knn_reg_k5_distance", X_reg, y_reg, X_reg_test)
base("knn_reg_k7_manhattan", X_reg, y_reg, X_reg_test)

# A synthetic tie case: duplicate rows guarantee exact distance ties.
X_tie = np.vstack([X_reg[:6], X_reg[:6], X_reg[6:16]])
y_tie = np.concatenate([y_reg[:6], y_reg[:6], y_reg[6:16]])
base("knn_reg_tie", X_tie, y_tie, X_reg[:10])

# ---------------------------------------------------------------------------
# Classification data
# ---------------------------------------------------------------------------
X_cls, y_cls = make_classification(
    n_samples=80,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.02,
    random_state=2,
)
y_cls = y_cls.astype(float)
X_cls_test = rng.normal(size=(20, 3)) * 2.0
base("knn_cls_k3_uniform", X_cls, y_cls, X_cls_test)
base("knn_cls_k5_uniform", X_cls, y_cls, X_cls_test)
base("knn_cls_k5_distance", X_cls, y_cls, X_cls_test)
base("knn_cls_k7_manhattan", X_cls, y_cls, X_cls_test)

X_tiec = np.vstack([X_cls[:6], X_cls[:6], X_cls[6:16]])
y_tiec = np.concatenate([y_cls[:6], y_cls[:6], y_cls[6:16]])
base("knn_cls_tie", X_tiec, y_tiec, X_cls[:10])

# ---------------------------------------------------------------------------
# Fit each configuration
# ---------------------------------------------------------------------------
for key, k, weights, p in [
    ("knn_reg_k3_uniform", 3, "uniform", 2),
    ("knn_reg_k5_uniform", 5, "uniform", 2),
    ("knn_reg_k5_distance", 5, "distance", 2),
    ("knn_reg_k7_manhattan", 7, "uniform", 1),
    ("knn_reg_tie", 3, "uniform", 2),
]:
    d = fixtures[key]
    model = KNeighborsRegressor(n_neighbors=k, weights=weights, p=p)
    model.fit(d["X"], d["y"])
    d["pred_train"] = to_list(model.predict(d["X"]))
    d["pred_test"] = to_list(model.predict(d["X_test"]))

for key, k, weights, p in [
    ("knn_cls_k3_uniform", 3, "uniform", 2),
    ("knn_cls_k5_uniform", 5, "uniform", 2),
    ("knn_cls_k5_distance", 5, "distance", 2),
    ("knn_cls_k7_manhattan", 7, "uniform", 1),
    ("knn_cls_tie", 3, "uniform", 2),
]:
    d = fixtures[key]
    model = KNeighborsClassifier(n_neighbors=k, weights=weights, p=p)
    model.fit(d["X"], d["y"])
    d["pred_train"] = to_list(model.predict(d["X"]))
    d["pred_test"] = to_list(model.predict(d["X_test"]))
    d["proba_test"] = to_list(model.predict_proba(d["X_test"]))
    d["classes"] = to_list(model.classes_)

with open(OUT, "w") as f:
    json.dump(fixtures, f, indent=2)

for name in fixtures:
    print(f"{name} written")