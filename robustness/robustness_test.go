// Package robustness holds a cross-package test that throws degenerate but valid
// data at every estimator. An estimator may return an error for such input (a
// single class, too few samples for its hyperparameters) but must never panic,
// and whatever it fits must survive predict and a save/load round trip.
package robustness

import (
	"fmt"
	"math"
	"path/filepath"
	"runtime/debug"
	"strings"
	"testing"
	"time"

	"github.com/ajitashwath/scikit-go/cluster"
	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/decomposition"
	"github.com/ajitashwath/scikit-go/ensemble"
	"github.com/ajitashwath/scikit-go/feature_selection"
	"github.com/ajitashwath/scikit-go/linear"
	"github.com/ajitashwath/scikit-go/neighbors"
	"github.com/ajitashwath/scikit-go/pipeline"
	"github.com/ajitashwath/scikit-go/preprocessing"
	"github.com/ajitashwath/scikit-go/svm"
	"github.com/ajitashwath/scikit-go/tree"
)

type dataset struct {
	name string
	X    [][]float64
	y    []float64
}

func repeat(v float64, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func datasets() []dataset {
	wide := make([][]float64, 3)
	for i := range wide {
		wide[i] = make([]float64, 40)
		for j := range wide[i] {
			wide[i][j] = float64((i+1)*(j+2)%7) + 0.5*float64(j%3)
		}
	}
	constX := [][]float64{{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}}
	return []dataset{
		{"one sample one feature", [][]float64{{1}}, []float64{1}},
		{"one sample many features", [][]float64{{1, 2, 3}}, []float64{0}},
		{"two samples one feature", [][]float64{{1}, {2}}, []float64{0, 1}},
		{"two samples two classes", [][]float64{{0, 0}, {1, 1}}, []float64{0, 1}},
		{"constant features", constX, []float64{0, 1, 0, 1, 0, 1}},
		{"constant features and target", constX, repeat(3, 6)},
		{"single class", [][]float64{{0, 1}, {1, 0}, {2, 2}, {3, 1}, {4, 4}, {5, 0}}, repeat(1, 6)},
		{"all-zero X", [][]float64{{0, 0}, {0, 0}, {0, 0}, {0, 0}}, []float64{0, 1, 0, 1}},
		{"duplicate rows", [][]float64{{1, 2}, {1, 2}, {1, 2}, {3, 4}, {3, 4}, {3, 4}}, []float64{0, 1, 0, 1, 0, 1}},
		{"more features than samples", wide, []float64{0, 1, 0}},
		{"negative and non-integer labels", [][]float64{{0}, {1}, {2}, {3}, {4}, {5}}, []float64{-2.5, -2.5, 0.25, 0.25, 7, 7}},
		{"huge magnitudes", [][]float64{{1e150, 1}, {-1e150, 2}, {2e150, 3}, {-2e150, 4}, {0, 5}, {1e150, 6}}, []float64{0, 1, 0, 1, 0, 1}},
		{"tiny magnitudes", [][]float64{{1e-150, 1e-160}, {2e-150, 3e-160}, {3e-150, 2e-160}, {4e-150, 5e-160}, {5e-150, 4e-160}, {6e-150, 6e-160}}, []float64{0, 1, 0, 1, 0, 1}},
		{"three classes minimal", [][]float64{{0, 0}, {5, 5}, {10, 0}}, []float64{0, 1, 2}},
		{"imbalanced classes", [][]float64{{0, 0}, {0.1, 0}, {0, 0.1}, {0.1, 0.1}, {9, 9}}, []float64{0, 0, 0, 0, 1}},
	}
}

// subject is one estimator configuration under test.
type subject struct {
	name string
	make func() any
}

func subjects() []subject {
	return []subject{
		{"LinearRegression", func() any { return linear.NewLinearRegression() }},
		{"Ridge", func() any { return linear.NewRidge() }},
		{"Ridge alpha=0", func() any { r := linear.NewRidge(); r.Alpha = 0; return r }},
		{"Ridge no intercept", func() any { r := linear.NewRidge(); r.FitIntercept = false; return r }},
		{"Lasso", func() any { l := linear.NewLasso(); l.Alpha = 0.1; return l }},
		{"Lasso alpha=0", func() any { l := linear.NewLasso(); l.Alpha = 0; return l }},
		{"ElasticNet", func() any { e := linear.NewElasticNet(); e.Alpha = 0.1; return e }},
		{"ElasticNet pure L2", func() any { e := linear.NewElasticNet(); e.Alpha, e.L1Ratio = 0.1, 0; return e }},
		{"LogisticRegression", func() any { return linear.NewLogisticRegression() }},
		{"LogisticRegression no intercept", func() any { m := linear.NewLogisticRegression(); m.FitIntercept = false; return m }},
		{"LogisticRegression C=inf", func() any { m := linear.NewLogisticRegression(); m.C = math.Inf(1); m.MaxIter = 50; return m }},
		{"StandardScaler", func() any { return preprocessing.NewStandardScaler() }},
		{"PCA default", func() any { return decomposition.NewPCA() }},
		{"PCA 1 component", func() any { p := decomposition.NewPCA(); p.NComponents = 1; return p }},
		{"KMeans k=1", func() any { k := cluster.NewKMeans(); k.NClusters = 1; return k }},
		{"KMeans k=3", func() any { k := cluster.NewKMeans(); k.NClusters = 3; return k }},
		{"KNeighborsClassifier k=1", func() any { k := neighbors.NewKNeighborsClassifier(); k.NNeighbors = 1; return k }},
		{"KNeighborsClassifier default", func() any { return neighbors.NewKNeighborsClassifier() }},
		{"KNeighborsRegressor k=1 distance", func() any {
			k := neighbors.NewKNeighborsRegressor()
			k.NNeighbors, k.Weights = 1, "distance"
			return k
		}},
		{"DecisionTreeClassifier", func() any { return tree.NewDecisionTreeClassifier() }},
		{"DecisionTreeRegressor", func() any { return tree.NewDecisionTreeRegressor() }},
		{"DecisionTreeRegressor mae", func() any { t := tree.NewDecisionTreeRegressor(); t.Criterion = "mae"; return t }},
		{"RandomForestClassifier", func() any { f := ensemble.NewRandomForestClassifier(); f.NTrees = 5; return f }},
		{"RandomForestRegressor", func() any { f := ensemble.NewRandomForestRegressor(); f.NTrees = 5; return f }},
		{"SVC rbf", func() any { return svm.NewSVC() }},
		{"SVC linear probability", func() any { s := svm.NewSVC(); s.Kernel, s.Probability = "linear", true; return s }},
		{"SVR rbf", func() any { return svm.NewSVR() }},
		{"SVR poly", func() any { s := svm.NewSVR(); s.Kernel = "poly"; return s }},
		{"VarianceThreshold", func() any { return feature_selection.NewVarianceThreshold() }},
		{"SelectKBest f_classif k=1", func() any { s := feature_selection.NewSelectKBest(); s.K = 1; return s }},
		{"SelectKBest f_regression k=1", func() any {
			s := feature_selection.NewSelectKBest()
			s.K, s.Score = 1, feature_selection.ScoreFRegression
			return s
		}},
		{"RFE linear", func() any { return feature_selection.NewRFE(linear.NewLinearRegression()) }},
		{"RFE lasso", func() any { l := linear.NewLasso(); l.Alpha = 0.1; return feature_selection.NewRFE(l) }},
		{"RFE logistic", func() any { return feature_selection.NewRFE(linear.NewLogisticRegression()) }},
		{"Pipeline scaler+logistic", func() any {
			p, _ := pipeline.MakePipeline(preprocessing.NewStandardScaler(), linear.NewLogisticRegression())
			return p
		}},
		{"Pipeline scaler+linear", func() any {
			p, _ := pipeline.MakePipeline(preprocessing.NewStandardScaler(), linear.NewLinearRegression())
			return p
		}},
		{"Pipeline scaler+pca+svc", func() any {
			p, _ := pipeline.MakePipeline(preprocessing.NewStandardScaler(), decomposition.NewPCA(), svm.NewSVC())
			return p
		}},
	}
}

// caseTimeout bounds one Fit or predict call. Every case here is a few samples,
// so anything near this long is an infinite loop rather than slow work.
const caseTimeout = 20 * time.Second

// guard runs f and converts a panic into a returned description with a stack. A
// call that does not return within caseTimeout is reported as a hang; its goroutine
// is abandoned, which is acceptable in a test that is already failing.
func guard(f func()) string {
	done := make(chan string, 1)
	go func() { done <- guardPanic(f) }()
	select {
	case msg := <-done:
		return msg
	case <-time.After(caseTimeout):
		return "did not return within " + caseTimeout.String() + " (infinite loop?)"
	}
}

func guardPanic(f func()) (panicked string) {
	defer func() {
		if r := recover(); r != nil {
			stack := strings.Split(string(debug.Stack()), "\n")
			var frames []string
			for _, line := range stack {
				if strings.Contains(line, "github.com/ajitashwath/scikit-go/") && !strings.Contains(line, "robustness") {
					frames = append(frames, strings.TrimSpace(line))
					if len(frames) == 3 {
						break
					}
				}
			}
			panicked = fmt.Sprintf("%v [%s]", r, strings.Join(frames, " <- "))
		}
	}()
	f()
	return ""
}

func TestEstimatorsNeverPanicOnDegenerateData(t *testing.T) {
	tmp := t.TempDir()
	var failures []string

	for _, ds := range datasets() {
		for _, sub := range subjects() {
			id := fmt.Sprintf("%s on %q", sub.name, ds.name)
			est := sub.make()

			var fitErr error
			if msg := guard(func() { fitErr = fit(est, ds.X, ds.y) }); msg != "" {
				failures = append(failures, id+": Fit panicked: "+msg)
				continue
			}
			if fitErr != nil {
				continue // rejecting the input with an error is acceptable
			}

			if msg := guard(func() { use(est, ds.X) }); msg != "" {
				failures = append(failures, id+": predict/transform panicked: "+msg)
			}
			if saver, ok := est.(core.Saver); ok {
				path := filepath.Join(tmp, "m.gob")
				var saveErr error
				if msg := guard(func() { saveErr = saver.Save(path) }); msg != "" {
					failures = append(failures, id+": Save panicked: "+msg)
					continue
				}
				if saveErr != nil {
					failures = append(failures, fmt.Sprintf("%s: Save of a fitted model failed: %v", id, saveErr))
				}
			}
		}
	}
	if len(failures) > 0 {
		t.Errorf("%d problems:\n  %s", len(failures), strings.Join(failures, "\n  "))
	}
}

// fit calls whichever Fit shape the estimator has.
func fit(est any, X [][]float64, y []float64) error {
	switch e := est.(type) {
	case interface {
		Fit([][]float64, []float64) error
	}:
		return e.Fit(X, y)
	case interface{ Fit([][]float64) error }:
		return e.Fit(X)
	}
	return fmt.Errorf("no Fit method on %T", est)
}

// use exercises every prediction-side method the estimator has.
func use(est any, X [][]float64) {
	if p, ok := est.(interface {
		Predict([][]float64) ([]float64, error)
	}); ok {
		p.Predict(X)
	}
	if p, ok := est.(interface {
		PredictProba([][]float64) ([][]float64, error)
	}); ok {
		p.PredictProba(X)
	}
	if p, ok := est.(interface {
		Transform([][]float64) ([][]float64, error)
	}); ok {
		p.Transform(X)
	}
	if p, ok := est.(interface {
		DecisionFunction([][]float64) ([][]float64, error)
	}); ok {
		p.DecisionFunction(X)
	}
}

// Non-finite output for finite input is a bug even when nothing panics: a fitted
// model should not silently emit NaN or Inf for ordinary data.
func TestPredictionsAreFiniteOnOrdinaryData(t *testing.T) {
	X := [][]float64{{0, 1}, {1, 0}, {2, 2}, {3, 1}, {4, 4}, {5, 0}, {6, 3}, {7, 2}}
	y := []float64{0, 1, 0, 1, 0, 1, 0, 1}
	var problems []string
	for _, sub := range subjects() {
		est := sub.make()
		if fit(est, X, y) != nil {
			continue
		}
		if p, ok := est.(interface {
			Predict([][]float64) ([]float64, error)
		}); ok {
			out, err := p.Predict(X)
			if err != nil {
				continue
			}
			for i, v := range out {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					problems = append(problems, fmt.Sprintf("%s: prediction %d is %v", sub.name, i, v))
					break
				}
			}
		}
		if p, ok := est.(interface {
			Transform([][]float64) ([][]float64, error)
		}); ok {
			out, err := p.Transform(X)
			if err != nil {
				continue
			}
			for i := range out {
				for _, v := range out[i] {
					if math.IsNaN(v) || math.IsInf(v, 0) {
						problems = append(problems, fmt.Sprintf("%s: transform row %d has %v", sub.name, i, v))
						break
					}
				}
			}
		}
	}
	if len(problems) > 0 {
		t.Errorf("non-finite output:\n  %s", strings.Join(problems, "\n  "))
	}
}
