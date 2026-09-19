import json
import os
import warnings

import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore", category=FutureWarning)  # SVC(probability=True) is deprecated in sklearn 1.9

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "svm", "testdata", "svm_fixtures.json")

rng = np.random.default_rng(31)
fixtures = {}


def to_list(a):
    return np.asarray(a).tolist()


def svc_case(key, X, y, X_test, **kw):
    prob = kw.pop("probability", False)
    model = SVC(probability=prob, random_state=0, **kw).fit(X, y)
    case = {
        "X": to_list(X),
        "y": to_list(y),
        "X_test": to_list(X_test),
        "params": {k: v for k, v in kw.items()},
        "classes": to_list(model.classes_),
        "support": to_list(model.support_),
        "n_support": to_list(model.n_support_),
        "dual_coef": to_list(model.dual_coef_),
        "intercept": to_list(model.intercept_),
        "gamma": float(model._gamma),
        "predict_test": to_list(model.predict(X_test)),
        "decision_test": to_list(model.decision_function(X_test)),
        "score_train": float(model.score(X, y)),
    }
    if prob:
        case["probability"] = True
        case["proba_test"] = to_list(model.predict_proba(X_test))
    fixtures[key] = case


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------
Xb, yb = make_blobs(n_samples=80, centers=2, cluster_std=3.0, random_state=1)
Xb_test = rng.normal(size=(25, 2)) * 4 + Xb.mean(axis=0)
yb = yb.astype(float)
svc_case("binary_linear", Xb, yb, Xb_test, kernel="linear", C=1.0)
svc_case("binary_linear_C10", Xb, yb, Xb_test, kernel="linear", C=10.0)
svc_case("binary_rbf", Xb, yb, Xb_test, kernel="rbf", C=1.0)
svc_case("binary_rbf_gamma", Xb, yb, Xb_test, kernel="rbf", C=5.0, gamma=0.2)
svc_case("binary_poly", Xb, yb, Xb_test, kernel="poly", C=1.0, degree=3, gamma=0.1, coef0=1.0)
svc_case("binary_sigmoid", Xb, yb, Xb_test, kernel="sigmoid", C=1.0, gamma=0.05, coef0=0.0)

# Non-linearly separable (rings) to exercise the rbf kernel and bounded alphas.
theta = rng.uniform(0, 2 * np.pi, 100)
radius = np.where(np.arange(100) < 50, 1.0, 2.5) + rng.normal(scale=0.2, size=100)
Xr = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
yr = (np.arange(100) >= 50).astype(float)
Xr_test = rng.normal(size=(30, 2)) * 1.8
svc_case("rings_rbf", Xr, yr, Xr_test, kernel="rbf", C=2.0, gamma=1.0)

# ---------------------------------------------------------------------------
# Multiclass (one-vs-one) with non-contiguous labels
# ---------------------------------------------------------------------------
Xm, ym = make_classification(
    n_samples=120,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=1.2,
    random_state=2,
)
ym = np.array([2.0, 5.0, 9.0])[ym]
Xm_test = rng.normal(size=(30, 4)) * 1.5
svc_case("multi_rbf", Xm, ym, Xm_test, kernel="rbf", C=1.0)
svc_case("multi_linear", Xm, ym, Xm_test, kernel="linear", C=1.0)
svc_case("multi_poly", Xm, ym, Xm_test, kernel="poly", C=1.0, degree=2, gamma="scale", coef0=0.5)

# ---------------------------------------------------------------------------
# Probability estimates (cross-validated internally, so only loosely comparable)
# ---------------------------------------------------------------------------
svc_case("proba_binary", Xb, yb, Xb_test, kernel="rbf", C=1.0, probability=True)
svc_case("proba_multi", Xm, ym, Xm_test, kernel="rbf", C=1.0, probability=True)

# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
xs = np.sort(rng.uniform(-4, 4, 70))
ys = np.sin(xs) + rng.normal(scale=0.1, size=70)
Xs = xs.reshape(-1, 1)
Xs_test = np.linspace(-4.5, 4.5, 30).reshape(-1, 1)

Xm3 = rng.normal(size=(60, 3))
ym3 = 2.0 * Xm3[:, 0] - Xm3[:, 1] + 0.5 * Xm3[:, 2] + rng.normal(scale=0.1, size=60)
Xm3_test = rng.normal(size=(20, 3))


def svr_case(key, X, y, X_test, **kw):
    model = SVR(**kw).fit(X, y)
    fixtures[key] = {
        "X": to_list(X),
        "y": to_list(y),
        "X_test": to_list(X_test),
        "params": kw,
        "support": to_list(model.support_),
        "dual_coef": to_list(model.dual_coef_[0]),
        "intercept": float(model.intercept_[0]),
        "gamma": float(model._gamma),
        "predict_test": to_list(model.predict(X_test)),
        "predict_train": to_list(model.predict(X)),
        "score_train": float(model.score(X, y)),
    }


svr_case("svr_rbf", Xs, ys, Xs_test, kernel="rbf", C=10.0, epsilon=0.1)
svr_case("svr_rbf_eps", Xs, ys, Xs_test, kernel="rbf", C=1.0, epsilon=0.3, gamma=0.5)
svr_case("svr_linear", Xm3, ym3, Xm3_test, kernel="linear", C=1.0, epsilon=0.05)
svr_case("svr_poly", Xm3, ym3, Xm3_test, kernel="poly", C=1.0, degree=2, gamma=0.3, coef0=1.0, epsilon=0.1)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(fixtures, f, indent=2)

print("svm fixtures written to", OUT)
for k, v in fixtures.items():
    print(k, "n_sv=", len(v["support"]), "score_train=", round(v["score_train"], 3))
