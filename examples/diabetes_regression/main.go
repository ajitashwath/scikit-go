// Command diabetes_regression predicts diabetes disease progression from ten
// baseline measurements, comparing four regressors and then asking the linear
// model and the forest which features matter.
//
//	go run ./examples/diabetes_regression
package main

import (
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/ensemble"
	"github.com/ajitashwath/scikit-go/linear"
	"github.com/ajitashwath/scikit-go/metrics"
	"github.com/ajitashwath/scikit-go/neighbors"
	"github.com/ajitashwath/scikit-go/pipeline"
	"github.com/ajitashwath/scikit-go/preprocessing"
	"github.com/ajitashwath/scikit-go/svm"
	"github.com/ajitashwath/scikit-go/utils"
)

// Column names of the bundled diabetes data: age, sex, body mass index, mean
// blood pressure, then six blood serum measurements.
var featureNames = []string{"age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"}

type regressor interface {
	core.Estimator
	core.Predictor
}

func main() {
	if _, err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// run returns each model's held-out R^2, keyed by model name.
func run(w io.Writer) (map[string]float64, error) {
	X, y, err := datasets.LoadDiabetes()
	if err != nil {
		return nil, err
	}
	Xtr, Xte, ytr, yte, err := utils.TrainTestSplit(X, y, 0.3, 42)
	if err != nil {
		return nil, err
	}

	lr := linear.NewLinearRegression()

	forest := ensemble.NewRandomForestRegressor()
	forest.NTrees = 200
	forest.Seed = 1

	scaledKNN, err := pipeline.MakePipeline(preprocessing.NewStandardScaler(), neighbors.NewKNeighborsRegressor())
	if err != nil {
		return nil, err
	}
	// Pipeline hyperparameters are addressed as "<step name>__<field>".
	scaledSVR, err := pipeline.MakePipeline(preprocessing.NewStandardScaler(), svm.NewSVR())
	if err != nil {
		return nil, err
	}
	if err := scaledSVR.SetParams(map[string]any{"svr__C": 100.0, "svr__Kernel": "linear"}); err != nil {
		return nil, err
	}

	models := []struct {
		name  string
		model regressor
	}{
		{"LinearRegression", lr},
		{"StandardScaler + KNN", scaledKNN},
		{"RandomForestRegressor", forest},
		{"StandardScaler + SVR", scaledSVR},
	}

	fmt.Fprintf(w, "diabetes: %d training / %d test samples\n\n", len(Xtr), len(Xte))
	fmt.Fprintf(w, "  %-24s %8s %8s %8s\n", "model", "R^2", "RMSE", "MAE")
	scores := make(map[string]float64, len(models))
	for _, m := range models {
		if err := m.model.Fit(Xtr, ytr); err != nil {
			return nil, fmt.Errorf("%s: %w", m.name, err)
		}
		pred, err := m.model.Predict(Xte)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", m.name, err)
		}
		r2, err := metrics.R2Score(yte, pred)
		if err != nil {
			return nil, err
		}
		rmse, err := metrics.RootMeanSquaredError(yte, pred)
		if err != nil {
			return nil, err
		}
		mae, err := metrics.MeanAbsoluteError(yte, pred)
		if err != nil {
			return nil, err
		}
		scores[m.name] = r2
		fmt.Fprintf(w, "  %-24s %8.4f %8.2f %8.2f\n", m.name, r2, rmse, mae)
	}

	fmt.Fprintln(w, "\nlargest linear coefficients (by magnitude):")
	for _, i := range topK(lr.Coef, 4, math.Abs) {
		fmt.Fprintf(w, "  %-4s %+9.2f\n", featureNames[i], lr.Coef[i])
	}
	fmt.Fprintln(w, "\nrandom forest feature importances:")
	imp := forest.FeatureImportances()
	for _, i := range topK(imp, 4, nil) {
		fmt.Fprintf(w, "  %-4s %9.3f\n", featureNames[i], imp[i])
	}
	return scores, nil
}

// topK returns the indices of the k largest values of v, ranked by key(v[i])
// (or by v[i] itself when key is nil).
func topK(v []float64, k int, key func(float64) float64) []int {
	if key == nil {
		key = func(x float64) float64 { return x }
	}
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return key(v[idx[a]]) > key(v[idx[b]]) })
	return idx[:min(k, len(idx))]
}
