"""Golden fixtures for linear.Ridge, linear.Lasso, linear.ElasticNet and linear.LogisticRegression.

Ridge has a closed form, and Lasso/ElasticNet use the same cyclic coordinate descent as the Go port, so
those are compared at sklearn's own default tolerance. LogisticRegression is solved with different
optimizers (sklearn's L-BFGS vs gonum's), so its fixtures use a very tight tolerance: both then land on
the unique optimum of the strictly convex L2 problem.
"""
import json
import os

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "linear", "testdata", "linear_models_fixtures.json")


def to_list(a):
    return np.asarray(a).tolist()


def regression_data(n, p, n_informative, noise, seed, offset=0.0, n_test=20):
    X, y = make_regression(n_samples=n + n_test, n_features=p, n_informative=n_informative, noise=noise,
                           random_state=seed)
    X = X + offset
    return X[:n], y[:n], X[n:], y[n:]


def regression_case(model, Xtr, ytr, Xte, yte, params):
    model.fit(Xtr, ytr)
    case = {
        "params": params,
        "X": to_list(Xtr), "y": to_list(ytr), "X_test": to_list(Xte), "y_test": to_list(yte),
        "coef": to_list(model.coef_), "intercept": float(model.intercept_),
        "predict_test": to_list(model.predict(Xte)), "score_test": float(model.score(Xte, yte)),
    }
    if getattr(model, "n_iter_", None) is not None:
        case["n_iter"] = int(np.max(model.n_iter_))
    return case


fixtures = {"ridge": {}, "lasso": {}, "elastic_net": {}, "logistic": {}}

# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------
Xtr, ytr, Xte, yte = regression_data(60, 5, 4, 5.0, 0)
fixtures["ridge"]["basic"] = regression_case(Ridge(alpha=1.0), Xtr, ytr, Xte, yte, {"alpha": 1.0, "fit_intercept": True})
fixtures["ridge"]["large_alpha"] = regression_case(Ridge(alpha=100.0), Xtr, ytr, Xte, yte, {"alpha": 100.0, "fit_intercept": True})
fixtures["ridge"]["tiny_alpha"] = regression_case(Ridge(alpha=1e-4), Xtr, ytr, Xte, yte, {"alpha": 1e-4, "fit_intercept": True})

Xo, yo, Xot, yot = regression_data(50, 4, 3, 3.0, 3, offset=10.0)
fixtures["ridge"]["no_intercept"] = regression_case(
    Ridge(alpha=0.5, fit_intercept=False), Xo, yo, Xot, yot, {"alpha": 0.5, "fit_intercept": False})

Xw, yw, Xwt, ywt = regression_data(15, 40, 5, 2.0, 5)
fixtures["ridge"]["wide"] = regression_case(Ridge(alpha=1.0), Xw, yw, Xwt, ywt, {"alpha": 1.0, "fit_intercept": True})

Xc, yc, Xct, yct = regression_data(40, 4, 3, 1.0, 7)
Xc = np.hstack([Xc, Xc[:, :1]])  # an exact duplicate column
Xct = np.hstack([Xct, Xct[:, :1]])
fixtures["ridge"]["collinear"] = regression_case(Ridge(alpha=0.1), Xc, yc, Xct, yct, {"alpha": 0.1, "fit_intercept": True})

# ---------------------------------------------------------------------------
# Lasso and ElasticNet (default tol=1e-4, max_iter=1000 unless noted)
# ---------------------------------------------------------------------------
Xl, yl, Xlt, ylt = regression_data(80, 10, 3, 4.0, 11)


def lasso(name, alpha, X=Xl, y=yl, Xt=Xlt, yt=ylt, **kw):
    params = {"alpha": alpha, "fit_intercept": kw.get("fit_intercept", True),
              "max_iter": kw.get("max_iter", 1000), "tol": kw.get("tol", 1e-4)}
    fixtures["lasso"][name] = regression_case(Lasso(alpha=alpha, **kw), X, y, Xt, yt, params)


lasso("basic", 0.1)
lasso("sparse", 2.0)
lasso("all_zero", 1e4)
lasso("no_intercept", 0.5, X=Xo, y=yo, Xt=Xot, yt=yot, fit_intercept=False)
lasso("tight_tol", 0.05, tol=1e-12, max_iter=100000)
lasso("wide", 0.3, X=Xw, y=yw, Xt=Xwt, yt=ywt)
lasso("collinear", 0.2, X=Xc, y=yc, Xt=Xct, yt=yct)


def enet(name, alpha, l1_ratio, X=Xl, y=yl, Xt=Xlt, yt=ylt, **kw):
    params = {"alpha": alpha, "l1_ratio": l1_ratio, "fit_intercept": kw.get("fit_intercept", True),
              "max_iter": kw.get("max_iter", 1000), "tol": kw.get("tol", 1e-4)}
    fixtures["elastic_net"][name] = regression_case(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kw), X, y, Xt, yt, params)


enet("half", 0.1, 0.5)
enet("mostly_l2", 0.5, 0.2)
enet("mostly_l1_no_intercept", 0.05, 0.9, X=Xo, y=yo, Xt=Xot, yt=yot, fit_intercept=False)
enet("pure_l2", 0.1, 0.0, tol=1e-12, max_iter=100000)
enet("all_zero", 1e4, 0.5)
enet("collinear", 0.2, 0.5, X=Xc, y=yc, Xt=Xct, yt=yct)


# ---------------------------------------------------------------------------
# LogisticRegression (tight tolerance: compare optima, not solver paths)
# ---------------------------------------------------------------------------
def logistic_case(name, n, p, n_informative, n_classes, C, seed, fit_intercept=True, labels=None):
    X, y = make_classification(n_samples=n + 30, n_features=p, n_informative=n_informative, n_redundant=0,
                               n_classes=n_classes, n_clusters_per_class=1, class_sep=0.8, random_state=seed)
    if labels is not None:
        y = np.asarray(labels, dtype=float)[y]
    Xtr, ytr, Xte = X[:n], y[:n], X[n:]
    m = LogisticRegression(C=C, fit_intercept=fit_intercept, tol=1e-12, max_iter=100000).fit(Xtr, ytr)
    fixtures["logistic"][name] = {
        "params": {"C": C, "fit_intercept": fit_intercept},
        "X": to_list(Xtr), "y": to_list(ytr), "X_test": to_list(Xte),
        "classes": to_list(m.classes_), "coef": to_list(m.coef_), "intercept": to_list(m.intercept_),
        "predict_test": to_list(m.predict(Xte)), "proba_test": to_list(m.predict_proba(Xte)),
        "decision_test": to_list(m.decision_function(Xte)),
        "score_train": float(m.score(Xtr, ytr)),
    }


logistic_case("binary", 100, 4, 3, 2, 1.0, 1)
logistic_case("binary_strong_reg", 100, 4, 3, 2, 0.01, 2)
logistic_case("binary_weak_reg", 100, 4, 3, 2, 100.0, 3)
logistic_case("binary_no_intercept", 100, 4, 3, 2, 1.0, 4, fit_intercept=False)
logistic_case("binary_odd_labels", 100, 4, 3, 2, 1.0, 5, labels=[-2.0, 7.0])
logistic_case("multiclass", 150, 5, 4, 3, 1.0, 6)
logistic_case("multiclass_strong_reg", 150, 5, 4, 3, 0.05, 7)
logistic_case("multiclass_odd_labels", 150, 5, 4, 3, 1.0, 8, labels=[10.0, 2.0, 5.0])
logistic_case("four_class_no_intercept", 200, 6, 5, 4, 1.0, 9, fit_intercept=False)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(fixtures, f)
print("wrote", os.path.relpath(OUT, HERE))
