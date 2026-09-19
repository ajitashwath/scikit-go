import json
import os

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "feature_selection", "testdata", "feature_selection_fixtures.json")

fixtures = {}


def to_list(a):
    return np.asarray(a).tolist()


# ---------------------------------------------------------------------------
# VarianceThreshold: constant, near-constant and informative columns.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(5)
X_var = np.column_stack(
    [
        np.full(30, 3.0),  # constant
        rng.normal(scale=0.1, size=30),  # variance ~0.01
        rng.normal(scale=1.0, size=30),  # variance ~1
        rng.normal(scale=3.0, size=30),  # variance ~9
        (rng.random(30) > 0.5).astype(float),  # Bernoulli, variance ~0.25
    ]
)
fixtures["variance_threshold"] = {"X": to_list(X_var), "cases": {}}
for name, thr in [("zero", 0.0), ("half", 0.5), ("one", 1.0)]:
    vt = VarianceThreshold(threshold=thr).fit(X_var)
    fixtures["variance_threshold"]["cases"][name] = {
        "threshold": thr,
        "variances": to_list(vt.variances_),
        "support": to_list(vt.get_support()),
        "transformed": to_list(vt.transform(X_var)),
    }

# ---------------------------------------------------------------------------
# SelectKBest with f_classif and f_regression.
# ---------------------------------------------------------------------------
Xc, yc = make_classification(
    n_samples=90,
    n_features=8,
    n_informative=3,
    n_redundant=1,
    n_classes=3,
    random_state=6,
)
yc = yc.astype(float)
scores, pvalues = f_classif(Xc, yc)
sel = SelectKBest(f_classif, k=3).fit(Xc, yc)
fixtures["f_classif"] = {
    "X": to_list(Xc),
    "y": to_list(yc),
    "scores": to_list(scores),
    "pvalues": to_list(pvalues),
    "k3_support": to_list(sel.get_support()),
    "k3_transformed": to_list(sel.transform(Xc)),
}

Xr, yr = make_regression(
    n_samples=70, n_features=6, n_informative=3, noise=10.0, random_state=7
)
scores, pvalues = f_regression(Xr, yr)
sel = SelectKBest(f_regression, k=2).fit(Xr, yr)
fixtures["f_regression"] = {
    "X": to_list(Xr),
    "y": to_list(yr),
    "scores": to_list(scores),
    "pvalues": to_list(pvalues),
    "k2_support": to_list(sel.get_support()),
    "k2_transformed": to_list(sel.transform(Xr)),
}

# ---------------------------------------------------------------------------
# RFE with a linear model and with a depth-limited tree.
# ---------------------------------------------------------------------------
Xe, ye = make_regression(
    n_samples=80, n_features=8, n_informative=3, noise=0.5, random_state=8
)
fixtures["rfe"] = {"X": to_list(Xe), "y": to_list(ye), "cases": {}}
for name, est, n_sel, step in [
    ("linear_step1", LinearRegression(), 3, 1),
    ("linear_step2", LinearRegression(), 3, 2),
    ("linear_default", LinearRegression(), None, 1),
    ("tree_step1", DecisionTreeRegressor(max_depth=3, random_state=0), 3, 1),
]:
    rfe = RFE(est, n_features_to_select=n_sel, step=step).fit(Xe, ye)
    fixtures["rfe"]["cases"][name] = {
        "n_features_to_select": 0 if n_sel is None else n_sel,
        "step": step,
        "support": to_list(rfe.support_),
        "ranking": to_list(rfe.ranking_),
        "transformed": to_list(rfe.transform(Xe)),
        "predictions": to_list(rfe.predict(Xe)),
    }

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(fixtures, f, indent=2)

print("feature_selection fixtures written to", OUT)
