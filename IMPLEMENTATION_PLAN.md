# Implementation Plan: Remaining Module Directories

> **Status:** every module below is implemented (`metrics`, `datasets`, `utils`, `tree`,
> `neighbors`, `cluster`, `decomposition`, `ensemble`, `feature_selection`, `svm`,
> `pipeline`). The plan is kept as the design record; deviations from it are noted inline
> under "Implemented as".

This plan covers the eleven module directories that are currently empty scaffolding:

`cluster/`, `datasets/`, `decomposition/`, `ensemble/`, `feature_selection/`,
`metrics/`, `neighbors/`, `pipeline/`, `svm/`, `tree/`, `utils/`.

It defines scope, ordering, conventions, and acceptance criteria so the project can grow from
its two existing estimators (`linear.LinearRegression`, `preprocessing.StandardScaler`) into a
coherent scikit-learn sibling without reworking the architecture.

---

## 1. Current state and principles

The repository is an early-stage sklearn clone. Existing, working code:

- `core/interfaces.go` — `Estimator`, `Predictor`, `Transformer`, `Classifier`, `Clusterer`, `Saver`.
- `internal/matutil` — input validation (`ValidateXy`, `ValidateX`, finite/NaN/Inf checks),
  `ToDense`, `DenseToSlice`, and shared sentinel errors (`ErrEmptyInput`, `ErrRaggedInput`, ...).
- `linear` — OLS via gonum QR on an augmented design matrix; versioned gob `Save`/`Load`.
- `preprocessing` — `StandardScaler`; `FitTransform`, `InverseTransform`; gob `Save`/`Load`.
- `fixtures/` — Python generators producing JSON parity fixtures, mirrored into package `testdata/`.

Every new module **must** follow the established patterns:

1. **Constructors** — `New<Type>() *<Type>` returning an unfitted value.
2. **Input validation** — every `Fit`/`Transform`/`Predict` calls `matutil.ValidateXy`/`ValidateX`
   first and wraps errors as `fmt.Errorf("<Pkg>.<Method>: %w", err)`.
3. **Fit state** — unexported `nFeatures`, `fitted bool`; guard predict/transform/save with
   `matutil.ErrNotFitted`.
4. **Interface conformance** — compile-time checks (`var _ core.Transformer = (*Type)(nil)`) and
   implement the `core` interface that matches the estimator's role.
5. **Save/Load** — versioned gob payload struct (`<type>Gob{Version, ...}` + format-version const),
   `Save(path string) error`, `Load<Type>(path string) (*<Type>, error)`. Version mismatch returns
   a descriptive error; loading is only supported for the current version.
6. **Sklearn parity** — every estimator gets a fixture generator in `fixtures/`, the JSON output
   copied into `<pkg>/testdata/`, and a `Test<Type>_AgainstSklearn` test comparing within tolerance.
7. **Tests + benchmarks** — edge-case tests (before-fit, empty, ragged, NaN/Inf, wrong feature
   count, save/load round trip, save-before-fit, nonexistent file) mirroring
   `linear/linear_regression_test.go`; benchmarks mirroring
   `linear/linear_regression_bench_test.go` (Small/Medium/Large).
8. **No new external deps** — use gonum (`mat`, `stat`, `stat/distuv`) and the Go stdlib only.

---

## 2. Dependency order and milestones

Dependencies flow bottom-up. Implement in this order; each phase is independently shippable
(compile + test green) before starting the next.

```
Phase 0  Foundations        metrics, datasets, utils
Phase 1  Base estimators    tree, neighbors, cluster, decomposition
Phase 2  Composites         ensemble (needs tree), feature_selection (needs metrics/linear),
                            pipeline (needs core interfaces)
Phase 3  Advanced           svm
```

### Phase 0 — Foundations

| Module | Depends on | Deliverable |
|--------|-----------|-------------|
| `metrics` | `internal/matutil` | Regression/classification/clustering score functions |
| `datasets` | — | Synthetic data generators + bundled real datasets |
| `utils`   | `internal/matutil` | `TrainTestSplit`, `Shuffle`, parameter helpers |

### Phase 1 — Base estimators

| Module | Depends on | Deliverable |
|--------|-----------|-------------|
| `tree` | `core`, `internal/matutil` | `DecisionTreeRegressor`, `DecisionTreeClassifier` |
| `neighbors` | `core`, `internal/matutil`, `metrics` | `KNeighborsRegressor`, `KNeighborsClassifier` |
| `cluster` | `core`, `internal/matutil`, `utils` | `KMeans` |
| `decomposition` | `core`, `internal/matutil` | `PCA` |

### Phase 2 — Composites

| Module | Depends on | Deliverable |
|--------|-----------|-------------|
| `ensemble` | `tree`, `core` | `RandomForestRegressor`, `RandomForestClassifier` |
| `feature_selection` | `metrics`, `linear`, `internal/matutil` | `VarianceThreshold`, `SelectKBest`, `RFE` |
| `pipeline` | `core` | `Pipeline`, `MakePipeline` |

### Phase 3 — Advanced

| Module | Depends on | Deliverable |
|--------|-----------|-------------|
| `svm` | `core`, `internal/matutil` | `SVC`, `SVR` (SMO dual solver) |

---

## 3. Module plans

### 3.1 `metrics`

Pure-function score metrics used by every estimator's `Score` and by `feature_selection`.

**Regression**: `MeanSquaredError`, `RootMeanSquaredError`, `MeanAbsoluteError`, `R2Score`,
`ExplainedVarianceScore`, `MaxError`.
**Classification**: `AccuracyScore`, `PrecisionScore`, `RecallScore`, `F1Score`,
`ConfusionMatrix`, `ClassificationReport`.
**Clustering**: `AdjustedRandIndex`, `AdjustedMutualInfo`, `SilhouetteScore`,
`HomogeneityScore`, `CompletenessScore`, `VMeasure`.

- Signature convention: `func R2Score(yTrue, yPred []float64) (float64, error)`; validate
  equal non-empty lengths (reuse `matutil` style sentinel `ErrLengthMismatch`).
- Adopt the existing `R2` logic from `linear.LinearRegression.Score` (`linear/linear_regression.go:93`)
  including the `ssTot == 0` edge cases; refactor `LinearRegression.Score` to delegate to
  `metrics.R2Score` (update its tests to keep coverage).
- Classification functions operate on `[]float64` labels (matching `Estimator.Fit`'s `y []float64`),
  with a documented `[]int`-friendly helper if needed.
- Fixtures: `fixtures/generate_metrics_fixtures.py` comparing against `sklearn.metrics` on
  deterministic arrays; tests verify each function.

**Implemented as:** `LinearRegression` centers the data and returns sklearn's minimum-norm
least-squares solution via SVD (with sklearn's `tol=1e-6` rank cutoff), so constant or collinear
columns and `n_features > n_samples` fit instead of failing. `SilhouetteScore` raises unless
`2 <= n_labels <= n_samples-1`, `R2Score` is NaN below two samples, and precision/recall/F1 return 0
when only one label is present, all as in sklearn (found by the differential tests).

### 3.2 `datasets`

Data for examples, tests, and benchmarks. Two sub-features:

**Synthetic generators** (reproducible via seed, mirroring the benchmark helpers):
- `MakeRegression(nSamples, nFeatures, noise, seed)` — sklearn `make_regression` parity.
- `MakeClassification(nSamples, nFeatures, nClasses, seed)`.
- `MakeBlobs(nSamples, nFeatures, nCenters, seed)` — for clustering.
- `MakeMoons(nSamples, noise, seed)` — for KNN/tree demos.

**Bundled datasets**: `LoadDiabetes()`, `LoadIris()` (already used in
`fixtures/generate_fixtures.py`); embed as gzip-compressed CSV via `go:embed`, exposed as
`(X [][]float64, y []float64, err error)`.

- Generator signatures mirror sklearn's names and parameter defaults where practical.
- Fixtures: `generate_datasets_fixtures.py` writes arrays from sklearn's own generators with a
  fixed seed so Go reproduces them bit-for-bit; `LoadIris`/`LoadDiabetes` fixtures are the exact
  sklearn copies.
- Benchmark helpers (`randomDataset`/`randomMatrix` in existing bench files) should move here
  eventually and be shared via `datasets.MakeRegression`-style calls.

### 3.3 `utils`

Generic helpers not tied to a single estimator.

- `TrainTestSplit(X, y, testSize, seed)` → `(XTr, XTe, yTr, yTe)` matching sklearn semantics
  (shuffle + deterministic split, `test_size` as fraction).
- `Shuffle(X, y, seed)` — in-place permutation with a seeded `rand.Rand`.
- `CheckConsistentLength(...)` / reshape helpers that wrap `matutil` errors.
- Cross-validation and hyperparameter search live in the separate `model_selection` package
  (see below), not here.

### 3.3.1 `model_selection`

`KFold`, `StratifiedKFold`, `CrossValScore` and `GridSearchCV`, added after v0.1 groundwork once
the estimators were stable enough to be worth tuning (this reverses the earlier "keep it small"
decision).

- Splitters return `[]Split{Train, Test []int}` eagerly. Without shuffling they reproduce sklearn's
  folds exactly (checked against golden fixtures, including `StratifiedKFold`'s first-appearance
  class ordering). With `Shuffle` the fold sizes and stratification match but the indices do not,
  because the permutation comes from Go's seeded RNG, the same deviation as `utils.TrainTestSplit`.
- Estimators are mutable and have no clone method, so `CrossValScore` takes a `ModelFactory`
  (`func() (Model, error)`) and `GridSearchCV` a `ModelBuilder` (`func(params) (Model, error)`).
  Pipelines fit in through `Pipeline.SetParams` and its `<step>__<Field>` names.
- Ranking treats mean scores within a relative 1e-12 as tied (the earliest combination wins),
  unlike sklearn's exact float comparison, so that summing identical fold scores in a different
  order cannot change the winner.
- Fold data is deep-copied so a model that edits its inputs cannot leak into other folds.
- `cv == nil` means a 5-fold `KFold`; unlike sklearn there is no automatic switch to stratified
  folds for classifiers (no `is_classifier` check), so pass `StratifiedKFold` explicitly.
- Known gaps: runs on one goroutine, no `RandomizedSearchCV`, no multi-metric scoring, no
  `error_score` (any failing fit or score aborts the search), `GridSearchCV` is not serializable
  (save `BestModel()` instead).

### 3.3.2 `linear`: regularized models

`Ridge`, `Lasso`, `ElasticNet` and `LogisticRegression` next to `LinearRegression`. All follow the
usual pattern (`New*` constructor with sklearn's defaults, `Fit`/`Predict`/`Score`, versioned gob
`Save`/`Load*`, registered in `pipeline` and, for the coefficient models, usable by `RFE`).

- `Ridge` solves the penalized least squares from the SVD of the centered data, so it is stable for
  collinear columns, works when p > n and at `Alpha = 0` (where tiny singular values are dropped).
- `Lasso` and `ElasticNet` port sklearn 1.9's cyclic coordinate descent, including its duality-gap
  stopping rule (the ridge-only and no-penalty gap variants, and the gap check before the first pass),
  so coefficients and even `n_iter_` match sklearn on the fixtures. `Fit` does not error at `MaxIter`;
  `Converged()` reports it.
- `LogisticRegression` minimizes the same strictly convex L2 objective as sklearn (one coefficient row
  for two classes, softmax with one row per class otherwise) with a small in-package L-BFGS
  (`lbfgs.go`, strong-Wolfe line search). gonum's `optimize` package was rejected because it pulls
  `golang.org/x/tools` into `go.mod`, and gonum is meant to stay the only dependency. Defaults are
  tighter than sklearn's (`Tol` 1e-6, `MaxIter` 1000) because the optimizers differ; the fixtures solve to
  1e-10 and agree to about 1e-7.
- Persisted models carry a `Kind` tag: gob matches fields by name, so without it a Lasso file would load
  as an ElasticNet.
- Known gaps: no L1/elastic-net penalty for logistic regression, no class or sample weights, no
  `positive`/`selection="random"` options, dense input only.

### 3.4 `tree`

CART decision trees — the first "real" learner and the backbone of `ensemble`.

- `DecisionTreeRegressor{Criterion: "mse"|"mae", MaxDepth, MinSamplesSplit, MinSamplesLeaf, MaxFeatures, Seed}`.
- `DecisionTreeClassifier{Criterion: "gini"|"entropy", ...}`.
- Public API: `Fit(X, y) error`, `Predict(X)`, `PredictProba(X)` (classifier), `FeatureImportances()`,
  `Save`/`Load`, `Score`.
- Algorithm: recursive greedy splitting; sample-weighted impurity (MSE / mean absolute error for
  regression; Gini / entropy for classification); stop on depth / min samples / leaf constraints;
  handle ties and constant columns deterministically.
- Internal node representation in a single slice (preorder) for cache-friendliness; store
  `feature`, `threshold`, `left`/`right` child indices, `value` (leaf mean / class distribution).
- Predictions: iterative traversal (no recursion) to avoid stack overflow on deep trees.
- Fixtures: `generate_tree_fixtures.py` with sklearn `DecisionTreeRegressor/Classifier` (fixed
  `random_state`) on small deterministic datasets; compare predictions, not just node internals.
- Tests: sklearn parity, criterion correctness on trivial data, depth/leaf constraints,
  feature-importance sanity, save/load round trip, all edge cases from §1.
- Benchmarks: fit/predict at Small/Medium/Large (n_samples × n_features).

### 3.5 `neighbors`

Brute-force k-nearest-neighbors with optional uniform/distance weighting.

- `KNeighborsRegressor{K, Weights: "uniform"|"distance", P (Minkowski exponent)}` (brute-force search only; there is no `Algorithm` option).
- `KNeighborsClassifier{...}` (+ `PredictProba`).
- Public API mirrors `linear.LinearRegression`: `Fit`, `Predict`, `Score`, `Save`/`Load`.
- Implementation: precompute pairwise distances (`matutil` + gonum `stat` or a dedicated
  distance function supporting Euclidean/Manhattan/Minkowski); on `Predict`, find the k nearest
  training rows (linear scan or partial sort), aggregate labels/values with optional
  distance-weighted voting; tie-breaking matches sklearn (`argsort` semantics).
- Store training data at fit time (lazy storage: keep the slices, do not re-copy unless mutating).
- `kd-tree`/`ball-tree` are explicitly out of scope for v1; note as a future optimization.
- Fixtures: `generate_neighbors_fixtures.py` vs sklearn KNN on small datasets (also cover
  `weights="distance"` and a tie case).
- Benchmarks: fit/predict at Small/Medium/Large.

### 3.6 `cluster`

- `KMeans{NClusters, MaxIter, NInit, Tol, Seed}` — k-means++ seeding, Lloyd iterations,
  empty-cluster re-seeding (assign to farthest point), `n_init` restarts keeping best inertia.
- Public API: `Fit(X) error`, `Predict(X)` (nearest centroid), `FitPredict(X)`, `Labels()`,
  `ClusterCenters()`, `Inertia()`, `Score` (negative inertia for sklearn parity), `Save`/`Load`.
- Implement directly rather than via `gonum/stat/clustering/kmeans` (its API does not match
  sklearn semantics); reuse gonum only for vector math if convenient.
- Fixtures: `generate_cluster_fixtures.py` with sklearn `KMeans` (fixed seed, `n_init`, `tol`)
  on blobs data; compare `labels_`, `cluster_centers_`, `inertia_` within a looser tolerance
  (centroid labeling is order-sensitive — normalize label permutation before comparing).
- Benchmarks: fit at Small/Medium/Large.

### 3.7 `decomposition`

- `PCA{NComponents, ...}` — full SVD via `gonum/mat.SVD` on the centered matrix.
- Public API: `Fit`, `Transform`, `FitTransform`, `InverseTransform`, `Components()`,
  `ExplainedVariance()`, `ExplainedVarianceRatio()`, `Mean()`, `Save`/`Load`.
- Match sklearn semantics: center by column means, decompose, keep top `NComponents`
  (or `min(n, p)` when unset), sign convention as in sklearn's `svd_flip` (deterministic
  sign fixing so components match sklearn's fixtures).
- Fixtures: `generate_decomposition_fixtures.py` vs sklearn `PCA`; compare `components_`,
  `explained_variance_ratio_`, transformed output.
- Benchmarks: fit/transform at Small/Medium/Large.

### 3.8 `ensemble`

**Implemented as:** trees are seeded and bootstrapped through a port of numpy's legacy
`RandomState` (`internal/nprandom`), so with `Bootstrap=false` and all features the fully
grown forest matches sklearn exactly. Bootstrap forests duplicate rows where sklearn uses
sample weights, so they are checked by held-out score, not bit-for-bit.

The tree builder mirrors sklearn's in-place sample array: samples are sorted with a port of
sklearn's introsort (`tree/sort.go`, validated against `_py_simultaneous_sort`, including
tie order) and partitioned with sklearn's swap scheme, so child nodes inherit sklearn's
sample order. That order decides split ties that differ only in float rounding of
`sum_total`; without it a 1e-15 perturbation of one target could change a tree.

- `RandomForestRegressor{NTrees, MaxDepth, MaxFeatures, MinSamplesSplit, Seed}`.
- `RandomForestClassifier{...}` (+ `PredictProba` averaging class probabilities).
- Public API: `Fit`, `Predict`, `PredictProba`, `FeatureImportances`, `Score`, `Save`/`Load`.
- Implementation: bootstrap sample per tree (with replacement), `max_features` random feature
  subset per split, aggregate by averaging (regressor) / majority or probability mean (classifier);
  deterministic across runs (per-tree seeded RNG derived from a master seed).
- Save/load the constituent trees through the same gob format as `tree`.
- Fixtures: `generate_ensemble_fixtures.py` vs sklearn `RandomForest*` (fixed `random_state`,
  small `n_estimators` for speed) on deterministic data; compare within tolerance
  (forests are sensitive to RNG details — use loose tolerance and small trees).
- Benchmarks: fit/predict at Small/Medium/Large; optionally a parallel-vs-serial sanity check.

### 3.9 `feature_selection`

- `VarianceThreshold{Threshold}` — `Fit`, `Transform`, `FitTransform`, `Support()`,
  `GetSupportMask()`; sklearn parity is trivial.
- `SelectKBest{K, ScoreFunc}` — uses `metrics` scorers (`FRegression`,
  `FClassif` from `stat/distuv`-based F-tests or a small ANOVA implementation) to rank features.
- `RFE{Estimator, NFeaturesToSelect, Step}` — recursive elimination wrapping a fitted estimator.
- Public API: `Fit`, `Transform`, `FitTransform`, `Support()`; pipeline-friendly.
- Fixtures: `generate_feature_selection_fixtures.py` vs sklearn on a dataset with known
  informative features (regression: correlated/noise columns; classification: `make_classification`).
- Benchmarks: transform at Small/Medium/Large.

### 3.10 `pipeline`

Composition over `core` interfaces — no new math.

- `Pipeline{Steps []Step}` where `Step` is `{Name string, Estimator core.Estimator}` with a
  runtime type switch: all but the last step must satisfy `core.Transformer`, the last must
  satisfy `core.Estimator` (+ optional `core.Predictor`).
- Public API: `Fit(X, y) error`, `Predict(X)`, `FitTransform(X)` (when last step is a
  transformer), `SetParams`, `GetParams`, `Save`/`Load` (serialize each step with its own
  gob format into one file; a top-level version header).
- `MakePipeline(steps ...core.Estimator) (*Pipeline, error)` convenience constructor.
- Validation at construction time (reject a non-transformer step before the last).
- Fixtures: `generate_pipeline_fixtures.py` — `StandardScaler → LinearRegression` and
  `StandardScaler → PCA → KNeighborsClassifier` parity vs sklearn's `Pipeline`; also verify
  `predict` passes through transforms.
- Tests: construction errors, fit/predict through steps, save/load round trip across step types.

### 3.11 `svm`

**Implemented as:** a libsvm-faithful SMO (second-order working-set selection, no
shrinking), one-vs-one `SVC` with libsvm's support-vector layout, `SVR`, linear/poly/rbf/
sigmoid kernels, and Platt-scaled probabilities. `SVC` matches sklearn on support vectors,
dual coefficients, intercepts and decision values; nonlinear `SVR` matches on predictions
but individual dual coefficients differ within solver tolerance on near-singular kernels.
`PredictProba` uses seeded cross-validation folds, so it agrees only approximately.

Highest difficulty — a proper quadratic-programming dual solver.

- `SVC{C, Kernel: "linear"|"rbf"|"poly", Gamma, Degree, Coef0, Tol, MaxIter}`.
- `SVR{...}` (epsilon-insensitive loss).
- Public API: `Fit`, `Predict`, `PredictProba` (via Platt scaling, only for SVC),
  `SupportVectors()`, `DecisionFunction()`, `Save`/`Load`, `Score`.
- Algorithm: implement SMO (Sequential Minimal Optimization) for the dual, mirroring libsvm's
  solve; kernel evaluations computed on demand. This is the riskiest component — plan a
  dedicated spike (see §6) before committing the full implementation.
- Fixtures: `generate_svm_fixtures.py` vs sklearn SVC/SVR on small deterministic datasets
  (linear + RBF), loose tolerance (SMO tolerances differ from libsvm).
- Benchmarks: fit/predict at Small/Medium (the `100000` Large case is unrealistic for SMO;
  cap at a size that stays under ~30s).

---

## 4. Cross-cutting concerns

### 4.1 `core` interface growth

Add interfaces incrementally, only when an estimator needs them:

```go
type Classifier interface {
    PredictProba(X [][]float64) ([][]float64, error)
}

type Clusterer interface {
    FitPredict(X [][]float64) ([]float64, error)
    Labels() []float64
}
```

Keep compile-time checks in each module. Do not add a `Scorer` interface — `Score` stays a
method on concrete types (matches existing `LinearRegression.Score`).

### 4.2 Fixture pipeline

**Implemented as:** each package's `testdata/` holds the only copy of its JSON, and every
generator in `fixtures/` writes straight into it, so there is nothing to sync or to drift.
`fixtures/generate_all.py` regenerates everything in one pass. Every generator is
deterministic (`np.random.seed` / fixed sklearn `random_state`).

### 4.3 gob versioning

Reuse the exact pattern from `linearRegressionGob`/`standardScalerGob`: exported-field payload
struct + `Version` int + format const + explicit version check on load. Bump the version and
keep a compatibility note whenever a payload changes.

### 4.4 Determinism and RNG

All randomness is explicitly seeded (never the global `rand`). Each estimator holds a `Seed`
field; derived per-subcomponent seeds use the pattern fixed in `linear`/`preprocessing` benches
(`rand.New(rand.NewSource(seed))`). Parallel components (random forest trees, `n_init` restarts)
derive unique sub-seeds from the master seed so results are reproducible under `-race`.

---

## 5. Definition of done per module

A module is *done* when:

- `go build ./...` and `go vet ./...` pass; `gofmt` clean (mind the repo's CRLF-on-Windows
  caveat — check against LF-normalized copies).
- `go test ./...` and `go test -race ./...` pass, including a `Test<Type>_AgainstSklearn`
  parity test with committed fixtures.
- Edge-case tests exist for every public method (§1 item 7).
- `Save`/`Load` round trip is tested; unfitted save and wrong-version load return errors.
- Benchmarks exist and run without panic at `-benchtime=1x`.
- Doc comments describe each exported type/method and its sklearn counterpart.

---

## 6. Risks and open questions

- **SVM (SMO)**: numerical behavior and kernel caching are hard to match to libsvm; treat as a
  spike first. Fallback: document `tolerance` expectations in fixtures rather than chasing
  exact sklearn values.
- **RandomForest parity**: sklearn uses a specific feature-sampling and bootstrap RNG scheme;
  fixture tolerances must be loose or fixtures must use tiny trees. Prefer correctness +
  reproducibility over bit-parity with sklearn internals.
- **KMeans label permutation**: centroid order is arbitrary; fixture comparisons must align
  labels before asserting.
- **PCA sign convention**: apply sklearn's `svd_flip` deterministic sign rule so `components_`
  match; otherwise compare `abs()` and the transformed variance instead.
- **`go:embed` datasets**: keeps binaries self-contained but adds binary size; acceptable for
  the small bundled sets (diabetes, iris).
- **Ordering guarantee**: metrics/datasets/utils must land first because everything else imports
  them; resist starting `svm` before `pipeline`/`ensemble` since the foundational modules unblock
  more value sooner.

## 7. Suggested commit sequence

1. `metrics` + tests + fixtures.
2. `utils` (split/shuffle) + `datasets` (generators + diabetes/iris) + refactor benches to use them.
3. `tree` (regressor then classifier).
4. `neighbors` → `cluster` → `decomposition`.
5. `ensemble` (on top of tree).
6. `feature_selection` (on top of metrics/linear) → `pipeline` (on top of core).
7. `svm` spike, then full `SVC`/`SVR`.

Each step lands independently green (build, vet, test, race, gofmt, benchmarks).
