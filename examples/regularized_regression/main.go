// Command regularized_regression compares ordinary least squares with Ridge, Lasso
// and ElasticNet on the diabetes dataset. Regularization shrinks the coefficients;
// the L1 penalty also sets some of them to exactly zero, giving a smaller model that
// predicts about as well. The second half picks each model's strength (alpha) by
// cross-validation.
//
//	go run ./examples/regularized_regression
package main

import (
	"fmt"
	"io"
	"math"
	"os"

	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/linear"
	"github.com/ajitashwath/scikit-go/model_selection"
	"github.com/ajitashwath/scikit-go/pipeline"
	"github.com/ajitashwath/scikit-go/preprocessing"
	"github.com/ajitashwath/scikit-go/utils"
)

var featureNames = []string{"age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"}

func main() {
	if _, err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// newPipeline scales the features and then applies the given model. The penalties
// treat every coefficient alike, so the features must be on a common scale first, and
// putting the scaler inside the pipeline means cross-validation refits it per fold.
func newPipeline(model any) (*pipeline.Pipeline, error) {
	return pipeline.MakePipeline(preprocessing.NewStandardScaler(), model)
}

// run returns "<model>_r2" (held-out R^2) for ols, ridge, lasso and elastic_net,
// "lasso_zeros" (coefficients that are exactly 0), the "ridge_shrink" ratio of the ridge
// coefficient norm to the OLS one, and "tuned_<model>_alpha" / "tuned_<model>_r2".
func run(w io.Writer) (map[string]float64, error) {
	X, y, err := datasets.LoadDiabetes()
	if err != nil {
		return nil, err
	}
	Xtr, Xte, ytr, yte, err := utils.TrainTestSplit(X, y, 0.3, 42)
	if err != nil {
		return nil, err
	}
	scores := map[string]float64{}

	// --- 1. the same data, four ways of fitting a line -------------------------
	ridge := linear.NewRidge()
	ridge.Alpha = 10
	lasso := linear.NewLasso()
	lasso.Alpha = 2
	enet := linear.NewElasticNet()
	enet.Alpha, enet.L1Ratio = 2, 0.5
	fixed := []struct {
		key, label string
		step       string // the pipeline step name, which is the lower-cased type name
		model      any
	}{
		{"ols", "OLS", "linearregression", linear.NewLinearRegression()},
		{"ridge", "Ridge (alpha 10)", "ridge", ridge},
		{"lasso", "Lasso (alpha 2)", "lasso", lasso},
		{"elastic_net", "ElasticNet (alpha 2, l1 0.5)", "elasticnet", enet},
	}

	fmt.Fprintf(w, "diabetes: %d training / %d test samples, features standardized\n\n", len(Xtr), len(Xte))
	coefs := make([][]float64, len(fixed))
	for i, m := range fixed {
		p, err := newPipeline(m.model)
		if err != nil {
			return nil, err
		}
		if err := p.Fit(Xtr, ytr); err != nil {
			return nil, fmt.Errorf("%s: %w", m.label, err)
		}
		r2, err := p.Score(Xte, yte)
		if err != nil {
			return nil, err
		}
		scores[m.key+"_r2"] = r2
		est, _ := p.Named(m.step)
		coefs[i] = coefficients(est)
	}

	fmt.Fprintf(w, "coefficients (per standard deviation of each feature):\n  %-6s", "")
	for _, m := range fixed {
		fmt.Fprintf(w, " %12s", m.key)
	}
	fmt.Fprintln(w)
	for j, name := range featureNames {
		fmt.Fprintf(w, "  %-6s", name)
		for i := range fixed {
			if coefs[i][j] == 0 {
				fmt.Fprintf(w, " %12s", ".")
			} else {
				fmt.Fprintf(w, " %12.2f", coefs[i][j])
			}
		}
		fmt.Fprintln(w)
	}
	fmt.Fprintf(w, "\n  %-30s %8s %11s\n", "model", "test R^2", "zero coefs")
	for i, m := range fixed {
		zeros := 0
		for _, c := range coefs[i] {
			if c == 0 {
				zeros++
			}
		}
		if m.key == "lasso" {
			scores["lasso_zeros"] = float64(zeros)
		}
		fmt.Fprintf(w, "  %-30s %8.4f %11d\n", m.label, scores[m.key+"_r2"], zeros)
	}
	scores["ridge_shrink"] = norm(coefs[1]) / norm(coefs[0])
	fmt.Fprintf(w, "\nRidge shrinks the coefficient vector to %.0f%% of the OLS length; Lasso zeroes %d features outright.\n",
		100*scores["ridge_shrink"], int(scores["lasso_zeros"]))

	// --- 2. choose alpha by cross-validation ---------------------------------
	alphas := []any{0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0}
	tuned := []struct {
		key, label, step string
		build            func() any
	}{
		{"ridge", "Ridge", "ridge", func() any { return linear.NewRidge() }},
		{"lasso", "Lasso", "lasso", func() any { return linear.NewLasso() }},
		{"elastic_net", "ElasticNet", "elasticnet", func() any { return linear.NewElasticNet() }},
	}
	fmt.Fprintf(w, "\nchoosing alpha by 5-fold cross-validation over %d values:\n", len(alphas))
	fmt.Fprintf(w, "  %-12s %11s %10s %10s\n", "model", "best alpha", "CV R^2", "test R^2")
	for _, m := range tuned {
		m := m
		build := func(params map[string]any) (model_selection.Model, error) {
			p, err := newPipeline(m.build())
			if err != nil {
				return nil, err
			}
			return p, p.SetParams(params)
		}
		search := model_selection.NewGridSearchCV(build, model_selection.ParamGrid{m.step + "__Alpha": alphas})
		search.CV = model_selection.NewKFold(5)
		if err := search.Fit(Xtr, ytr); err != nil {
			return nil, fmt.Errorf("%s: %w", m.label, err)
		}
		alpha := search.BestParams()[m.step+"__Alpha"].(float64)
		r2, err := search.Score(Xte, yte)
		if err != nil {
			return nil, err
		}
		scores["tuned_"+m.key+"_alpha"], scores["tuned_"+m.key+"_r2"] = alpha, r2
		fmt.Fprintf(w, "  %-12s %11v %10.4f %10.4f\n", m.label, alpha, search.BestScore(), r2)
	}
	return scores, nil
}

// coefficients returns the fitted coefficient vector of a linear model.
func coefficients(est any) []float64 {
	switch m := est.(type) {
	case *linear.LinearRegression:
		return m.Coef
	case *linear.Ridge:
		return m.Coef
	case *linear.Lasso:
		return m.Coef
	case *linear.ElasticNet:
		return m.Coef
	}
	return nil
}

func norm(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}
