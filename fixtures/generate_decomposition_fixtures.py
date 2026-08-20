import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_regression

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "decomposition_fixtures.json")

rng = np.random.default_rng(41)

fixtures = {}


def to_list(a):
    return a.tolist()


# ---------------------------------------------------------------------------
# Data: full-rank Gaussian data with clear covariance structure.
# ---------------------------------------------------------------------------
X2, _ = make_regression(n_samples=80, n_features=4, n_informative=4, noise=0.2, random_state=1)
X3, _ = make_regression(n_samples=120, n_features=6, n_informative=6, noise=0.5, random_state=2)
X_test = rng.normal(size=(30, 4)) * 2.0

fixtures["pca_k2"] = {"X": to_list(X2), "X_test": to_list(X_test)}
fixtures["pca_k4"] = {"X": to_list(X2), "X_test": to_list(X_test)}
fixtures["pca_k3_n6"] = {"X": to_list(X3)}

# ---------------------------------------------------------------------------
# Fit each configuration
# ---------------------------------------------------------------------------
for key, n_components in [("pca_k2", 2), ("pca_k4", 4), ("pca_k3_n6", 3)]:
    d = fixtures[key]
    model = PCA(n_components=n_components)
    model.fit(d["X"])
    d["components"] = to_list(model.components_)
    d["explained_variance"] = to_list(model.explained_variance_)
    d["explained_variance_ratio"] = to_list(model.explained_variance_ratio_)
    d["singular_values"] = to_list(model.singular_values_)
    d["mean"] = to_list(model.mean_)
    d["transformed"] = to_list(model.transform(d["X"]))
    if "X_test" in d:
        d["transformed_test"] = to_list(model.transform(d["X_test"]))
        d["inverse"] = to_list(model.inverse_transform(model.transform(d["X_test"])))

with open(OUT, "w") as f:
    json.dump(fixtures, f, indent=2)

for name in fixtures:
    print(f"{name} written")