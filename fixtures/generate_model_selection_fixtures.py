"""Golden fixtures for model_selection: KFold, StratifiedKFold, cross_val_score, GridSearchCV.

Only unshuffled splitters are fixed here. Shuffled splits depend on numpy's RNG, which the
Go port does not reproduce, so those are covered by property tests instead.
"""
import json
import os
import warnings

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "model_selection", "testdata", "model_selection_fixtures.json")


def to_list(a):
    return np.asarray(a).tolist()


def splits_of(cv, X, y=None):
    return [{"train": to_list(tr), "test": to_list(te)} for tr, te in cv.split(X, y)]


fixtures = {}

# ---------------------------------------------------------------------------
# KFold (no shuffle)
# ---------------------------------------------------------------------------
fixtures["kfold"] = []
for n, k in [(10, 3), (10, 5), (7, 7), (11, 4), (100, 5), (5, 2)]:
    X = np.zeros((n, 1))
    fixtures["kfold"].append({"n": n, "k": k, "splits": splits_of(KFold(n_splits=k), X)})

# ---------------------------------------------------------------------------
# StratifiedKFold (no shuffle). Includes an unsorted label order, class labels that
# are not 0..K-1, an imbalanced set, and a class with fewer members than n_splits
# (sklearn only warns about that case).
# ---------------------------------------------------------------------------
strat_cases = {
    "sorted_binary": ([0] * 6 + [1] * 4, 2),
    "interleaved_three_class": ([2, 0, 1, 0, 1, 2, 2, 0, 1, 1, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2], 3),
    "first_appearance_order": ([3, 1, 3, 1, 2, 2, 3, 1, 2, 3, 1, 2, 3], 3),
    "imbalanced": ([0] * 12 + [1] * 3, 3),
    "small_class": ([0] * 8 + [1] * 8 + [2] * 2, 4),
    "negative_labels": ([-1, 2, -1, 2, 2, -1, 0, 0, 2, -1, 0, 2], 3),
}
fixtures["stratified"] = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, (y, k) in strat_cases.items():
        y = np.asarray(y, dtype=float)
        X = np.zeros((len(y), 1))
        fixtures["stratified"].append(
            {"name": name, "y": to_list(y), "k": k, "splits": splits_of(StratifiedKFold(n_splits=k), X, y)}
        )

# ---------------------------------------------------------------------------
# cross_val_score with estimators the Go library also implements
# ---------------------------------------------------------------------------
Xc, yc = make_classification(n_samples=90, n_features=5, n_informative=3, n_redundant=0, n_classes=3,
                             n_clusters_per_class=1, class_sep=1.0, random_state=4)
Xr, yr = make_regression(n_samples=80, n_features=4, n_informative=3, noise=5.0, random_state=2)

fixtures["cross_val"] = {
    "knn_kfold": {
        "X": to_list(Xc), "y": to_list(yc), "cv": {"kind": "kfold", "k": 5},
        "scores": to_list(cross_val_score(KNeighborsClassifier(), Xc, yc, cv=KFold(5))),
    },
    "knn_stratified": {
        "X": to_list(Xc), "y": to_list(yc), "cv": {"kind": "stratified", "k": 4},
        "scores": to_list(cross_val_score(KNeighborsClassifier(), Xc, yc, cv=StratifiedKFold(4))),
    },
    "linear_kfold": {
        "X": to_list(Xr), "y": to_list(yr), "cv": {"kind": "kfold", "k": 4},
        "scores": to_list(cross_val_score(LinearRegression(), Xr, yr, cv=KFold(4))),
    },
}


# ---------------------------------------------------------------------------
# GridSearchCV
# ---------------------------------------------------------------------------
def grid_case(estimator, grid, X, y, cv, cv_desc):
    gs = GridSearchCV(estimator, grid, cv=cv).fit(X, y)
    res = gs.cv_results_
    fold_scores = np.array([res[f"split{i}_test_score"] for i in range(cv.get_n_splits())]).T
    return {
        "X": to_list(X), "y": to_list(y), "cv": cv_desc,
        "grid": {k: list(v) for k, v in grid.items()},
        "params": [{k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()} for p in res["params"]],
        "fold_scores": to_list(fold_scores),
        "mean_test_score": to_list(res["mean_test_score"]),
        "std_test_score": to_list(res["std_test_score"]),
        "rank_test_score": to_list(res["rank_test_score"]),
        "best_index": int(gs.best_index_),
        "best_score": float(gs.best_score_),
        "predict": to_list(gs.predict(X[:15])),
        "predict_X": to_list(X[:15]),
    }


fixtures["grid"] = {
    "knn_classifier": grid_case(
        KNeighborsClassifier(),
        {"n_neighbors": [1, 3, 5, 7], "weights": ["uniform", "distance"]},
        Xc, yc, StratifiedKFold(3), {"kind": "stratified", "k": 3},
    ),
    "knn_regressor": grid_case(
        KNeighborsRegressor(),
        {"n_neighbors": [2, 4, 8]},
        Xr, yr, KFold(4), {"kind": "kfold", "k": 4},
    ),
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(fixtures, f)
print("wrote", os.path.relpath(OUT, HERE))
