# Scikit-Go

A scikit-learn-style machine learning library in Go, built on [gonum](https://gonum.org).
Estimators follow sklearn's shape (`Fit`, `Predict`, `Transform`, `Score`) and are checked
against sklearn with committed golden fixtures.

[![CI](https://github.com/ajitashwath/scikit-go/actions/workflows/ci.yml/badge.svg)](https://github.com/ajitashwath/scikit-go/actions/workflows/ci.yml)

```
go get github.com/ajitashwath/scikit-go@latest
```

Requires Go 1.24 or newer (the floor set by gonum v0.17.0, the only dependency).

| Package | Contents |
|---|---|
| `linear` | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression` |
| `preprocessing` | `StandardScaler` |
| `decomposition` | `PCA` |
| `cluster` | `KMeans` |
| `neighbors` | `KNeighborsClassifier`, `KNeighborsRegressor` |
| `tree` | `DecisionTreeClassifier`, `DecisionTreeRegressor` |
| `ensemble` | `RandomForestClassifier`, `RandomForestRegressor` |
| `svm` | `SVC`, `SVR` |
| `feature_selection` | `VarianceThreshold`, `SelectKBest` (`FClassif`, `FRegression`), `RFE` |
| `pipeline` | `Pipeline`, `MakePipeline` |
| `metrics` | regression, classification and clustering scores |
| `datasets` | `MakeRegression`/`MakeClassification`/`MakeBlobs`/`MakeMoons`, `LoadIris`, `LoadDiabetes` |
| `model_selection` | `KFold`, `StratifiedKFold`, `CrossValScore`, `GridSearchCV` |
| `utils` | `TrainTestSplit`, `Shuffle` |

Every estimator has a versioned gob `Save` / `Load<Type>`. Loaders validate the file's shape and return
an error for a truncated or edited payload rather than producing a model that panics on first use.

```go
p, _ := pipeline.MakePipeline(preprocessing.NewStandardScaler(), svm.NewSVC())
_ = p.Fit(Xtrain, ytrain)
pred, _ := p.Predict(Xtest)
```

## Scope

This is a v0.x library, so the surface above is deliberately narrow. Not implemented yet:

- **Most of model selection.** `model_selection` has `KFold`, `StratifiedKFold`, `CrossValScore` and
  `GridSearchCV`, and that is all. There is no `RandomizedSearchCV`, `ShuffleSplit`, `GroupKFold`,
  `cross_val_predict`, learning curves or multi-metric scoring. Unlike sklearn, cross-validation does
  not switch to stratified folds for classifiers on its own; pass `StratifiedKFold` yourself.
- **The rest of the linear models.** `linear` has least squares, `Ridge`, `Lasso`, `ElasticNet` and
  L2-regularized `LogisticRegression`. There is no L1 or elastic-net logistic regression, no class or
  sample weights, no `SGDClassifier`, and no `RidgeCV`/`LassoCV` (use `GridSearchCV` over `Alpha`).
  Lasso and ElasticNet use cyclic coordinate descent only.
- **Boosting.** There is no gradient boosting; `ensemble` has random forests only.
- **Sparse input, `partial_fit`, and cancellation via `context.Context`.** Everything takes dense
  `[][]float64` and fits in one call.

Anything in the list above is a gap, not a design decision; none of it is ruled out for later releases.

## Performance expectations

- **Only `ensemble` is parallel.** Random-forest trees are grown on up to `NJobs` goroutines
  (`0` means `GOMAXPROCS`). Every other estimator, including `SVC`/`SVR`, the decision trees,
  `KMeans`, `PCA` and the k-nearest-neighbors models, runs on a single goroutine, so expect them
  to be slow on large datasets. Cross-validation and `GridSearchCV` run their folds and
  parameter combinations one after another too, so a large grid multiplies that cost.
- **Iterative linear models stop on a tolerance, not an error.** `Lasso`, `ElasticNet` and
  `LogisticRegression` run until `Tol` or `MaxIter`, like sklearn, and `Fit` does not fail if it runs
  out of iterations: check `Converged()`. `LogisticRegression` uses its own small L-BFGS (no
  dependency beyond gonum), so it matches sklearn to the solver tolerance rather than bit for bit.
- **k-nearest-neighbors is brute force.** Each prediction computes the distance to every training
  sample. There is no kd-tree or ball-tree, so prediction cost grows linearly with the
  training-set size.
- The library favors matching sklearn's results over raw speed; see `IMPLEMENTATION_PLAN.md`.

## Stability and versioning

- The module follows [semantic versioning](https://semver.org). While the major version is 0,
  minor releases (`v0.x.0`) may contain breaking API changes; patch releases (`v0.x.y`) will not.
  From `v1.0.0` on, breaking changes need a new major version.
- **Saved models.** Every estimator's `Save` writes a versioned gob payload, and every `Load<Type>`
  checks that version and returns an error for a file it does not support, so a mismatched file
  fails loudly instead of loading wrong. The format version does not change in a patch release.
  A format change is a breaking change: it ships in a minor release (before 1.0) or a major
  release (after), and is called out in the release notes.
- **Loaders read only the current format.** A model saved by one format version cannot be read by a
  release that has moved to a newer one. Until that changes, keep the training data (or a re-fit
  script) alongside a persisted model, and re-save after upgrading across a format bump.

## How it is checked

- **Golden fixtures** from scikit-learn for every estimator (`<pkg>/testdata/`).
- **Differential tests** that run hundreds of small random and degenerate inputs through both
  sklearn and this library (`metrics/edge_test.go`), so behavior on edge cases such as a single
  class or a constant clustering matches sklearn, not just the happy path.
- **A robustness sweep** (`robustness/`) that fits every estimator on degenerate data (one sample,
  constant columns, more features than samples, extreme magnitudes) and fails on any panic or hang.
- Sorting, seeding and tree-growing routines are ported from sklearn/numpy and validated against them
  (`internal/nprandom`, `tree/sort.go`).

## Try it

[`examples/`](examples) has runnable programs for classification, regression, regularized linear
models, logistic regression, hyperparameter tuning, clustering, dimensionality reduction, feature
selection, nonlinear models and saving/loading models, each importing the library by its public
path. `examples/quickstart` fits a model from each estimator package on the bundled iris and
diabetes datasets, prints a score for each, and exits non-zero if any score falls below its floor:

```
go run ./examples/quickstart
go run ./examples/iris_classification
go run ./examples/regularized_regression
go run ./examples/logistic_regression
go run ./examples/hyperparameter_tuning
```

Cross-validation and tuning take a function that builds a fresh model, since estimators are
mutable and every fold needs its own. Pipelines are tuned through their `<step>__<Field>` names:

```go
build := func(params map[string]any) (model_selection.Model, error) {
	p, err := pipeline.MakePipeline(preprocessing.NewStandardScaler(), svm.NewSVC())
	if err != nil {
		return nil, err
	}
	return p, p.SetParams(params)
}
search := model_selection.NewGridSearchCV(build, model_selection.ParamGrid{
	"svc__C":      {0.1, 1.0, 10.0},
	"svc__Kernel": {"linear", "rbf"},
})
search.CV = model_selection.NewStratifiedKFold(5)
_ = search.Fit(Xtrain, ytrain)          // cross-validates all 6 combinations, refits the best
fmt.Println(search.BestParams(), search.BestScore())
pred, _ := search.Predict(Xtest)
```

## Testing

```
go test ./...
go test -race ./...
go test -bench . -benchtime=1x ./...
```

Golden fixtures live in each package's `testdata/`. To regenerate them (needs Python with
numpy, scikit-learn and scipy):

```
python fixtures/generate_all.py
```

Regenerated numbers can differ from the committed ones in the last float digits (numpy/sklearn
versions); the tests are written to tolerate that, including the decision-tree split ties that
depend on float rounding.

See `IMPLEMENTATION_PLAN.md` for design notes and known deviations from sklearn.

## License

[MIT](LICENSE).
