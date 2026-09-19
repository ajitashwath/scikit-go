import json
import os

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "ensemble", "testdata", "ensemble_fixtures.json")

fixtures = {}


def to_list(a):
    return a.tolist()


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
X, y = make_regression(
    n_samples=120, n_features=5, n_informative=3, noise=5.0, random_state=3
)
X_train, X_test, y_train, y_test = X[:80], X[80:], y[:80], y[80:]
reg = {
    "X": to_list(X_train),
    "y": to_list(y_train),
    "X_test": to_list(X_test),
    "y_test": to_list(y_test),
}

# Deterministic configuration: every tree sees all rows and all features, so
# the forest is reproducible independent of the bootstrap. Trees are fully grown,
# which exercises the tie-breaking in tiny nodes.
model = RandomForestRegressor(
    n_estimators=5, bootstrap=False, max_features=None, random_state=0
)
model.fit(X_train, y_train)
reg["nobootstrap"] = {
    "pred_test": to_list(model.predict(X_test)),
    "pred_train": to_list(model.predict(X_train)),
    "importances": to_list(model.feature_importances_),
}

# Stochastic configuration: only the held-out R^2 is comparable.
model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=0)
model.fit(X_train, y_train)
reg["bootstrap"] = {
    "r2_test": float(model.score(X_test, y_test)),
    "importances": to_list(model.feature_importances_),
}
fixtures["regression"] = reg

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
X, y = make_classification(
    n_samples=150,
    n_features=6,
    n_informative=3,
    n_redundant=1,
    n_classes=3,
    random_state=4,
)
y = y.astype(float)
X_train, X_test, y_train, y_test = X[:100], X[100:], y[:100], y[100:]
cls = {
    "X": to_list(X_train),
    "y": to_list(y_train),
    "X_test": to_list(X_test),
    "y_test": to_list(y_test),
}

model = RandomForestClassifier(
    n_estimators=5, bootstrap=False, max_features=None, random_state=0
)
model.fit(X_train, y_train)
cls["nobootstrap"] = {
    "classes": to_list(model.classes_),
    "pred_test": to_list(model.predict(X_test)),
    "proba_test": to_list(model.predict_proba(X_test)),
    "importances": to_list(model.feature_importances_),
}

model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)
model.fit(X_train, y_train)
cls["bootstrap"] = {
    "accuracy_test": float(model.score(X_test, y_test)),
    "importances": to_list(model.feature_importances_),
}
fixtures["classification"] = cls

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(fixtures, f, indent=2)

print("ensemble fixtures written to", OUT)
print("reg R2 (bootstrap):", fixtures["regression"]["bootstrap"]["r2_test"])
print("cls acc (bootstrap):", fixtures["classification"]["bootstrap"]["accuracy_test"])
