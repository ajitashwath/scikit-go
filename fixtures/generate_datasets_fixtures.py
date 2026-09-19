import json
import os

import numpy as np
from sklearn.datasets import load_diabetes, load_iris, make_moons

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(HERE, "..", "datasets", "testdata")

os.makedirs(TESTDATA, exist_ok=True)


def write_csv(path, values):
    with open(path, "w") as f:
        for row in values:
            f.write(",".join(repr(float(v)) for v in row) + "\n")


# ---------------------------------------------------------------------------
# Bundled datasets (written verbatim from sklearn's bundled copies)
# ---------------------------------------------------------------------------
diabetes = load_diabetes()
write_csv(os.path.join(TESTDATA, "diabetes_X.csv"), diabetes.data)
write_csv(os.path.join(TESTDATA, "diabetes_y.csv"), diabetes.target.reshape(-1, 1))

iris = load_iris()
write_csv(os.path.join(TESTDATA, "iris_X.csv"), iris.data)
write_csv(os.path.join(TESTDATA, "iris_y.csv"), iris.target.reshape(-1, 1))

# Verification stats for the Go Load* tests.
datasets_fixtures = {
    "diabetes": {
        "n_samples": int(diabetes.data.shape[0]),
        "n_features": int(diabetes.data.shape[1]),
        "feature_means": [float(v) for v in diabetes.data.mean(axis=0)],
        "y_mean": float(diabetes.target.mean()),
        "y_min": float(diabetes.target.min()),
        "y_max": float(diabetes.target.max()),
    },
    "iris": {
        "n_samples": int(iris.data.shape[0]),
        "n_features": int(iris.data.shape[1]),
        "feature_means": [float(v) for v in iris.data.mean(axis=0)],
        "y_mean": float(iris.target.mean()),
        "y_min": float(iris.target.min()),
        "y_max": float(iris.target.max()),
        "class_counts": [int((iris.target == c).sum()) for c in np.unique(iris.target)],
    },
}

with open(os.path.join(TESTDATA, "datasets_fixtures.json"), "w") as f:
    json.dump(datasets_fixtures, f, indent=2)

# ---------------------------------------------------------------------------
# make_moons with noise=0 is fully deterministic and must match Go exactly.
# ---------------------------------------------------------------------------
X_moons, y_moons = make_moons(n_samples=100, noise=0.0, random_state=0)
moons_fixtures = {
    "X": X_moons.tolist(),
    "y": y_moons.tolist(),
}
with open(os.path.join(TESTDATA, "moons_fixtures.json"), "w") as f:
    json.dump(moons_fixtures, f, indent=2)

print("diabetes/iris CSVs written to", TESTDATA)
print("datasets_fixtures.json and moons_fixtures.json written to", TESTDATA)
print("diabetes shape:", diabetes.data.shape)
print("iris shape:", iris.data.shape)