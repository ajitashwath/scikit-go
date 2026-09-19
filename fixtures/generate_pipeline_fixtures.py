import json
import os

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "pipeline", "testdata", "pipeline_fixtures.json")

fixtures = {}


def to_list(a):
    return np.asarray(a).tolist()


def split(X, y, n_train):
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


# ---------------------------------------------------------------------------
# StandardScaler -> LinearRegression
# ---------------------------------------------------------------------------
Xr, yr = make_regression(n_samples=80, n_features=4, n_informative=3, noise=3.0, random_state=1)
Xr[:, 0] = Xr[:, 0] * 50 + 100  # very different feature scales
X_train, X_test, y_train, y_test = split(Xr, yr, 60)
pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]).fit(X_train, y_train)
fixtures["scaler_linear"] = {
    "X": to_list(X_train),
    "y": to_list(y_train),
    "X_test": to_list(X_test),
    "y_test": to_list(y_test),
    "predict_test": to_list(pipe.predict(X_test)),
    "score_test": float(pipe.score(X_test, y_test)),
    "transform_test": to_list(pipe[:-1].transform(X_test)),
}

# ---------------------------------------------------------------------------
# StandardScaler -> PCA -> KNeighborsClassifier
# ---------------------------------------------------------------------------
Xc, yc = make_classification(
    n_samples=120,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=2,
)
yc = yc.astype(float)
X_train, X_test, y_train, y_test = split(Xc, yc, 90)
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
).fit(X_train, y_train)
fixtures["scaler_pca_knn"] = {
    "X": to_list(X_train),
    "y": to_list(y_train),
    "X_test": to_list(X_test),
    "y_test": to_list(y_test),
    "predict_test": to_list(pipe.predict(X_test)),
    "proba_test": to_list(pipe.predict_proba(X_test)),
    "score_test": float(pipe.score(X_test, y_test)),
}

# ---------------------------------------------------------------------------
# StandardScaler -> SelectKBest(f_regression) -> LinearRegression: a supervised
# selector in the middle of the chain must receive y.
# ---------------------------------------------------------------------------
Xs, ys = make_regression(n_samples=90, n_features=8, n_informative=3, noise=2.0, random_state=3)
X_train, X_test, y_train, y_test = split(Xs, ys, 70)
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_regression, k=3)),
        ("lr", LinearRegression()),
    ]
).fit(X_train, y_train)
fixtures["scaler_kbest_linear"] = {
    "X": to_list(X_train),
    "y": to_list(y_train),
    "X_test": to_list(X_test),
    "y_test": to_list(y_test),
    "predict_test": to_list(pipe.predict(X_test)),
    "score_test": float(pipe.score(X_test, y_test)),
    "support": to_list(pipe.named_steps["select"].get_support()),
}

# ---------------------------------------------------------------------------
# VarianceThreshold -> StandardScaler -> SVC
# ---------------------------------------------------------------------------
Xv, yv = make_classification(n_samples=100, n_features=4, n_informative=3, n_redundant=0, random_state=4)
Xv = np.column_stack([np.full(len(Xv), 7.0), Xv])  # a constant column to drop
yv = yv.astype(float)
X_train, X_test, y_train, y_test = split(Xv, yv, 75)
pipe = Pipeline(
    [("vt", VarianceThreshold()), ("scaler", StandardScaler()), ("svc", SVC(C=2.0))]
).fit(X_train, y_train)
fixtures["variance_scaler_svc"] = {
    "X": to_list(X_train),
    "y": to_list(y_train),
    "X_test": to_list(X_test),
    "y_test": to_list(y_test),
    "predict_test": to_list(pipe.predict(X_test)),
    "decision_test": to_list(pipe.decision_function(X_test)),
    "score_test": float(pipe.score(X_test, y_test)),
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(fixtures, f, indent=2)

print("pipeline fixtures written to", OUT)
