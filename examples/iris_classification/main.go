// Command iris_classification compares four classifiers on the iris dataset,
// then digs into the winner with a confusion matrix and a per-class report.
//
//	go run ./examples/iris_classification
package main

import (
	"fmt"
	"io"
	"os"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/ensemble"
	"github.com/ajitashwath/scikit-go/metrics"
	"github.com/ajitashwath/scikit-go/neighbors"
	"github.com/ajitashwath/scikit-go/pipeline"
	"github.com/ajitashwath/scikit-go/preprocessing"
	"github.com/ajitashwath/scikit-go/svm"
	"github.com/ajitashwath/scikit-go/tree"
	"github.com/ajitashwath/scikit-go/utils"
)

var classNames = []string{"setosa", "versicolor", "virginica"}

// classifier is anything that can be fit and then asked for predictions. Every
// estimator in scikit-go, and every Pipeline, satisfies it.
type classifier interface {
	core.Estimator
	core.Predictor
}

func main() {
	if _, err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// run returns each model's held-out accuracy, keyed by model name.
func run(w io.Writer) (map[string]float64, error) {
	X, y, err := datasets.LoadIris()
	if err != nil {
		return nil, err
	}
	// Hold out 30% of the samples for scoring. The seed makes the split repeatable.
	Xtr, Xte, ytr, yte, err := utils.TrainTestSplit(X, y, 0.3, 42)
	if err != nil {
		return nil, err
	}

	forest := ensemble.NewRandomForestClassifier()
	forest.NTrees = 200
	forest.Seed = 1

	// SVMs are sensitive to feature scale, so put a scaler in front of one.
	scaledSVC, err := pipeline.MakePipeline(preprocessing.NewStandardScaler(), svm.NewSVC())
	if err != nil {
		return nil, err
	}

	models := []struct {
		name  string
		model classifier
	}{
		{"KNeighborsClassifier", neighbors.NewKNeighborsClassifier()},
		{"DecisionTreeClassifier", tree.NewDecisionTreeClassifier()},
		{"RandomForestClassifier", forest},
		{"StandardScaler + SVC", scaledSVC},
	}

	fmt.Fprintf(w, "iris: %d training / %d test samples\n\n", len(Xtr), len(Xte))
	scores := make(map[string]float64, len(models))
	bestName, bestAcc := "", -1.0
	var bestPred []float64
	for _, m := range models {
		if err := m.model.Fit(Xtr, ytr); err != nil {
			return nil, fmt.Errorf("%s: %w", m.name, err)
		}
		pred, err := m.model.Predict(Xte)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", m.name, err)
		}
		acc, err := metrics.AccuracyScore(yte, pred)
		if err != nil {
			return nil, err
		}
		scores[m.name] = acc
		fmt.Fprintf(w, "  %-24s accuracy %.4f\n", m.name, acc)
		if acc > bestAcc {
			bestName, bestAcc, bestPred = m.name, acc, pred
		}
	}

	fmt.Fprintf(w, "\nbest model: %s\n\nconfusion matrix (rows = true class, columns = predicted):\n", bestName)
	cm, err := metrics.ConfusionMatrix(yte, bestPred)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(w, "  %-12s", "")
	for _, n := range classNames {
		fmt.Fprintf(w, "%12s", n)
	}
	fmt.Fprintln(w)
	for i, row := range cm {
		fmt.Fprintf(w, "  %-12s", classNames[i])
		for _, v := range row {
			fmt.Fprintf(w, "%12d", v)
		}
		fmt.Fprintln(w)
	}

	rep, err := metrics.ClassificationReport(yte, bestPred)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(w, "\n  %-12s %10s %10s %10s %10s\n", "", "precision", "recall", "f1", "support")
	for _, c := range rep.Classes {
		fmt.Fprintf(w, "  %-12s %10.3f %10.3f %10.3f %10d\n", classNames[int(c.Label)], c.Precision, c.Recall, c.F1, c.Support)
	}
	fmt.Fprintf(w, "  %-12s %10.3f %10.3f %10.3f\n", "macro avg", rep.MacroPrecision, rep.MacroRecall, rep.MacroF1)
	return scores, nil
}
