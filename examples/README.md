# Examples

Runnable programs that use scikit-go the way an application would. Each one is a
small `package main` that imports the library by its public path,
`github.com/ajitashwath/scikit-go/...`, and prints what it finds.

| Example | What it shows | Packages |
|---|---|---|
| [`quickstart`](quickstart) | One model from every package, with a score floor for each; a smoke test of the whole library | all |
| [`iris_classification`](iris_classification) | Four classifiers compared on iris, then a confusion matrix and per-class report for the winner | `neighbors`, `tree`, `ensemble`, `svm`, `pipeline`, `metrics` |
| [`diabetes_regression`](diabetes_regression) | Four regressors compared with R², RMSE and MAE; which features drive the prediction | `linear`, `neighbors`, `ensemble`, `svm`, `pipeline`, `metrics` |
| [`customer_segmentation`](customer_segmentation) | Unsupervised segmentation: scale, choose k by silhouette, read segment profiles in original units | `cluster`, `preprocessing`, `metrics` |
| [`pca_dimensionality`](pca_dimensionality) | Variance per component, reconstruction error, and what a classifier loses in 2 dimensions | `decomposition`, `preprocessing`, `neighbors` |
| [`feature_selection`](feature_selection) | `VarianceThreshold`, `SelectKBest` and `RFE` finding known-informative columns among noise | `feature_selection`, `pipeline`, `linear` |
| [`moons_nonlinear`](moons_nonlinear) | Why nonlinear models matter: linear vs RBF SVM on two moons, with a text plot of the decision regions | `svm`, `neighbors`, `ensemble`, `datasets` |
| [`regularized_regression`](regularized_regression) | OLS vs Ridge, Lasso and ElasticNet: shrinking coefficients, Lasso dropping features at no cost in R², and picking `alpha` by cross-validation | `linear`, `model_selection`, `pipeline` |
| [`logistic_regression`](logistic_regression) | Class probabilities on iris, the least-confident predictions, and how the regularization strength `C` changes accuracy and confidence | `linear`, `model_selection`, `pipeline` |
| [`hyperparameter_tuning`](hyperparameter_tuning) | Compare models with stratified cross-validation, tune an SVM with a grid search, then score once on a held-out test set | `model_selection`, `pipeline`, `svm`, `metrics` |
| [`model_persistence`](model_persistence) | Train, `Save`, `Load` in a "serving" step, and what happens with a damaged file | `pipeline`, `ensemble` |

Run any of them from the repository root:

```
go run ./examples/iris_classification
```

Once a version is tagged you can also run one without cloning the repository:

```
go run github.com/ajitashwath/scikit-go/examples/iris_classification@latest
```

## Using the library in your own module

```
go mod init example.com/myapp
go get github.com/ajitashwath/scikit-go@latest
```

```go
package main

import (
	"fmt"

	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/pipeline"
	"github.com/ajitashwath/scikit-go/preprocessing"
	"github.com/ajitashwath/scikit-go/svm"
	"github.com/ajitashwath/scikit-go/utils"
)

func main() {
	X, y, _ := datasets.LoadIris()
	Xtr, Xte, ytr, yte, _ := utils.TrainTestSplit(X, y, 0.3, 42)

	model, _ := pipeline.MakePipeline(preprocessing.NewStandardScaler(), svm.NewSVC())
	_ = model.Fit(Xtr, ytr)
	acc, _ := model.Score(Xte, yte)
	fmt.Printf("accuracy %.3f\n", acc)
}
```

(Errors are ignored here for brevity; the examples above handle them.)

## Tests

Every example has a `main_test.go` that runs it and checks the numbers it produces
(scores above a floor, the right features selected, a saved model loading back
identically), so `go test ./...` fails if an example stops working. The programs are
deterministic: every random step takes a seed.
