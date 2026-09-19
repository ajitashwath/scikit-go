import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "cluster", "testdata", "cluster_fixtures.json")

rng = np.random.default_rng(31)

fixtures = {}


def to_list(a):
    return a.tolist()


# ---------------------------------------------------------------------------
# Blobs data: well-separated clusters so any reasonable seed converges to the
# global optimum, making Go-vs-sklearn comparison robust to seed differences.
# ---------------------------------------------------------------------------
X3, y3 = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=1.0, random_state=11)
X5, y5 = make_blobs(n_samples=200, n_features=3, centers=5, cluster_std=1.2, random_state=12)
X_test = rng.normal(size=(40, 2)) * 3.0

fixtures["kmeans_blobs3"] = {"X": to_list(X3), "X_test": to_list(X_test)}
fixtures["kmeans_blobs5"] = {"X": to_list(X5)}
fixtures["kmeans_blobs5_k2"] = {"X": to_list(X5)}

# ---------------------------------------------------------------------------
# Fit each configuration
# ---------------------------------------------------------------------------
model = KMeans(n_clusters=3, n_init=10, random_state=0)
model.fit(X3)
d = fixtures["kmeans_blobs3"]
d["labels"] = to_list(model.labels_)
d["centers"] = to_list(model.cluster_centers_)
d["inertia"] = float(model.inertia_)
d["n_iter"] = int(model.n_iter_)
d["pred"] = to_list(model.predict(X_test))

model = KMeans(n_clusters=5, n_init=10, random_state=0)
model.fit(X5)
d = fixtures["kmeans_blobs5"]
d["labels"] = to_list(model.labels_)
d["centers"] = to_list(model.cluster_centers_)
d["inertia"] = float(model.inertia_)
d["n_iter"] = int(model.n_iter_)

model = KMeans(n_clusters=2, n_init=10, random_state=0)
model.fit(X5)
d = fixtures["kmeans_blobs5_k2"]
d["labels"] = to_list(model.labels_)
d["centers"] = to_list(model.cluster_centers_)
d["inertia"] = float(model.inertia_)
d["n_iter"] = int(model.n_iter_)

with open(OUT, "w") as f:
    json.dump(fixtures, f, indent=2)

for name in fixtures:
    print(f"{name} written")