# Scikit-Go

A scikit-learn-style machine learning library in Go, built on [gonum](https://gonum.org).
Estimators follow sklearn's shape (`Fit`, `Predict`, `Transform`, `Score`) and are checked
against sklearn with committed golden fixtures.

| Package | Contents |
|---|---|
| `linear` | `LinearRegression` |
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
| `utils` | `TrainTestSplit`, `Shuffle` |

Every estimator has a versioned gob `Save` / `Load<Type>`. Loaders validate the file's shape and return
an error for a truncated or edited payload rather than producing a model that panics on first use.

```go
p, _ := pipeline.MakePipeline(preprocessing.NewStandardScaler(), svm.NewSVC())
_ = p.Fit(Xtrain, ytrain)
pred, _ := p.Predict(Xtest)
```

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

`examples/quickstart` fits a model from every package on the bundled iris and diabetes
datasets, prints a score for each, and exits non-zero if any score falls below its floor:

```
go run ./examples/quickstart
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
