import copy
import json
import os

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tree_fixtures.json")

rng = np.random.default_rng(11)

fixtures = {}


def to_list(a):
    return a.tolist()


def base(key, X, y, X_test=None):
    d = {"X": to_list(X), "y": to_list(y)}
    if X_test is not None:
        d["X_test"] = to_list(X_test)
    fixtures[key] = d


# ---------------------------------------------------------------------------
# Regression data
# ---------------------------------------------------------------------------
X_reg, y_reg = make_regression(
    n_samples=60, n_features=3, n_informative=3, noise=0.05, random_state=1
)
X_reg_test = rng.normal(size=(20, 3)) * 2.0
for k in ("reg_mse", "reg_mae", "reg_mse_depth3"):
    base(k, X_reg, y_reg, X_reg_test)

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
for k in ("cls_gini", "cls_entropy", "cls_gini_depth2"):
    base(k, X_cls, y_cls, X_cls_test)

X_xor = rng.uniform(-1, 1, size=(80, 2))
y_xor = np.where((X_xor[:, 0] > 0) != (X_xor[:, 1] > 0), 1.0, -1.0)
for k in ("cls_xor_gini", "cls_xor_entropy"):
    base(k, X_xor, y_xor)

# ---------------------------------------------------------------------------
# Fit each configuration and record predictions + importances
# ---------------------------------------------------------------------------
for key, criterion, max_depth in [
    ("reg_mse", "squared_error", None),
    ("reg_mae", "absolute_error", None),
    ("reg_mse_depth3", "squared_error", 3),
]:
    model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, random_state=0)
    model.fit(fixtures[key]["X"], fixtures[key]["y"])
    fixtures[key]["pred_train"] = to_list(model.predict(fixtures[key]["X"]))
    fixtures[key]["pred_test"] = to_list(model.predict(fixtures[key]["X_test"]))
    fixtures[key]["feature_importances"] = to_list(model.feature_importances_)

cls_specs = [
    ("cls_gini", "gini", None),
    ("cls_entropy", "entropy", None),
    ("cls_xor_gini", "gini", None),
    ("cls_xor_entropy", "entropy", None),
    ("cls_gini_depth2", "gini", 2),
]
for key, criterion, max_depth in cls_specs:
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=0)
    model.fit(fixtures[key]["X"], fixtures[key]["y"])
    fixtures[key]["pred_train"] = to_list(model.predict(fixtures[key]["X"]))
    if "X_test" in fixtures[key]:
        fixtures[key]["pred_test"] = to_list(model.predict(fixtures[key]["X_test"]))
        fixtures[key]["proba_test"] = to_list(model.predict_proba(fixtures[key]["X_test"]))
    fixtures[key]["classes"] = to_list(model.classes_)
    fixtures[key]["feature_importances"] = to_list(model.feature_importances_)

with open(OUT, "w") as f:
    json.dump(fixtures, f, indent=2)

for name in fixtures:
    print(f"{name} written")