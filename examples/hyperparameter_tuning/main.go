// Command hyperparameter_tuning shows the standard model-selection workflow:
// hold out a test set, compare candidate models and tune hyperparameters with
// cross-validation on the training data only, then score the winner once on the
// test set.
//
//	go run ./examples/hyperparameter_tuning
package main

import (
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/ensemble"
	"github.com/ajitashwath/scikit-go/metrics"
	"github.com/ajitashwath/scikit-go/model_selection"
	"github.com/ajitashwath/scikit-go/neighbors"
	"github.com/ajitashwath/scikit-go/pipeline"
	"github.com/ajitashwath/scikit-go/preprocessing"
	"github.com/ajitashwath/scikit-go/svm"
	"github.com/ajitashwath/scikit-go/tree"
	"github.com/ajitashwath/scikit-go/utils"
)

func main() {
	if _, err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// newScaledSVC builds StandardScaler -> SVC and applies params such as
// {"svc__C": 10.0, "svc__Kernel": "rbf"}. Pipeline parameters are addressed as
// "<step name>__<field>".
func newScaledSVC(params map[string]any) (model_selection.Model, error) {
	p, err := pipeline.MakePipeline(preprocessing.NewStandardScaler(), svm.NewSVC())
	if err != nil {
		return nil, err
	}
	if err := p.SetParams(params); err != nil {
		return nil, err
	}
	return p, nil
}

// run returns "cv_<model>" for each compared model, "grid_best_cv" (the best grid
// combination's mean CV accuracy), "grid_combinations" and "test_accuracy".
func run(w io.Writer) (map[string]float64, error) {
	X, y, err := datasets.LoadIris()
	if err != nil {
		return nil, err
	}
	// The test set is set aside first and not touched until the very end.
	Xtr, Xte, ytr, yte, err := utils.TrainTestSplit(X, y, 0.3, 42)
	if err != nil {
		return nil, err
	}
	scores := map[string]float64{}

	// --- 1. compare models with stratified 5-fold cross-validation ------------
	// A ModelFactory returns a fresh unfitted model, because every fold needs its own.
	cv := model_selection.NewStratifiedKFold(5)
	candidates := []struct {
		key, name string
		factory   model_selection.ModelFactory
	}{
		{"knn", "KNeighborsClassifier", func() (model_selection.Model, error) {
			return neighbors.NewKNeighborsClassifier(), nil
		}},
		{"tree", "DecisionTreeClassifier", func() (model_selection.Model, error) {
			return tree.NewDecisionTreeClassifier(), nil
		}},
		{"forest", "RandomForestClassifier", func() (model_selection.Model, error) {
			f := ensemble.NewRandomForestClassifier()
			f.NTrees, f.Seed = 100, 1
			return f, nil
		}},
		{"svc", "StandardScaler + SVC", func() (model_selection.Model, error) {
			return newScaledSVC(nil)
		}},
	}
	fmt.Fprintf(w, "iris: %d training samples, %d held out for the final test\n\n", len(Xtr), len(Xte))
	fmt.Fprintln(w, "5-fold stratified cross-validation accuracy on the training data:")
	for _, c := range candidates {
		foldScores, err := model_selection.CrossValScore(c.factory, Xtr, ytr, cv, nil)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", c.name, err)
		}
		mean, std := meanStd(foldScores)
		scores["cv_"+c.key] = mean
		fmt.Fprintf(w, "  %-24s %.4f +/- %.4f\n", c.name, mean, std)
	}

	// --- 2. tune the SVM with a grid search -----------------------------------
	grid := model_selection.ParamGrid{
		"svc__C":      {0.1, 1.0, 10.0, 100.0},
		"svc__Kernel": {"linear", "rbf"},
	}
	search := model_selection.NewGridSearchCV(newScaledSVC, grid)
	search.CV = cv
	if err := search.Fit(Xtr, ytr); err != nil {
		return nil, err
	}

	results := search.Results()
	sort.SliceStable(results, func(i, j int) bool { return results[i].Rank < results[j].Rank })
	fmt.Fprintf(w, "\ngrid search over %d combinations (best first):\n", len(results))
	fmt.Fprintf(w, "  %4s  %-6s %-8s %10s %8s\n", "rank", "C", "kernel", "mean acc", "std")
	for _, r := range results {
		fmt.Fprintf(w, "  %4d  %-6v %-8v %10.4f %8.4f\n", r.Rank, r.Params["svc__C"], r.Params["svc__Kernel"], r.MeanScore, r.StdScore)
	}
	best := search.BestParams()
	fmt.Fprintf(w, "\nbest: C=%v kernel=%v (cross-validated accuracy %.4f)\n", best["svc__C"], best["svc__Kernel"], search.BestScore())
	scores["grid_best_cv"] = search.BestScore()
	scores["grid_combinations"] = float64(len(results))

	// --- 3. the one look at the test set --------------------------------------
	// Fit already refit the best combination on all of the training data.
	pred, err := search.Predict(Xte)
	if err != nil {
		return nil, err
	}
	acc, err := metrics.AccuracyScore(yte, pred)
	if err != nil {
		return nil, err
	}
	scores["test_accuracy"] = acc
	fmt.Fprintf(w, "held-out test accuracy of the tuned model: %.4f\n", acc)
	return scores, nil
}

func meanStd(v []float64) (mean, std float64) {
	for _, x := range v {
		mean += x
	}
	mean /= float64(len(v))
	for _, x := range v {
		std += (x - mean) * (x - mean)
	}
	return mean, math.Sqrt(std / float64(len(v)))
}
