// Command logistic_regression classifies iris flowers with logistic regression.
// Unlike most classifiers, it returns class probabilities, so the example looks at
// how confident each prediction is, and at how the regularization strength C trades
// a smoother, less confident model against a tighter fit to the training data.
//
//	go run ./examples/logistic_regression
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

var classNames = []string{"setosa", "versicolor", "virginica"}

func main() {
	if _, err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// newModel builds StandardScaler -> LogisticRegression with regularization strength c.
func newModel(c float64) (*pipeline.Pipeline, error) {
	lg := linear.NewLogisticRegression()
	lg.C = c
	return pipeline.MakePipeline(preprocessing.NewStandardScaler(), lg)
}

// run returns "test_accuracy" for C=1, "best_c" (the C with the best cross-validated
// accuracy), "confidence_low_c" and "confidence_high_c" (mean top-class probability
// on the test set at C=0.001 and C=100), and "new_flower_top_prob".
func run(w io.Writer) (map[string]float64, error) {
	X, y, err := datasets.LoadIris()
	if err != nil {
		return nil, err
	}
	Xtr, Xte, ytr, yte, err := utils.TrainTestSplit(X, y, 0.3, 42)
	if err != nil {
		return nil, err
	}
	scores := map[string]float64{}

	// --- 1. fit with the default C and look at the probabilities --------------
	model, err := newModel(1)
	if err != nil {
		return nil, err
	}
	if err := model.Fit(Xtr, ytr); err != nil {
		return nil, err
	}
	acc, err := model.Score(Xte, yte)
	if err != nil {
		return nil, err
	}
	scores["test_accuracy"] = acc
	fmt.Fprintf(w, "iris: %d training / %d test samples; multinomial logistic regression, C = 1\n", len(Xtr), len(Xte))
	fmt.Fprintf(w, "test accuracy: %.4f\n\n", acc)

	proba, err := model.PredictProba(Xte)
	if err != nil {
		return nil, err
	}
	pred, err := model.Predict(Xte)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(w, "probabilities for the five test flowers the model is least sure about:\n")
	fmt.Fprintf(w, "  %-12s %-12s %10s %11s %10s\n", "true", "predicted", classNames[0], classNames[1], classNames[2])
	for _, i := range leastConfident(proba, 5) {
		mark := ""
		if pred[i] != yte[i] {
			mark = "  <- wrong"
		}
		fmt.Fprintf(w, "  %-12s %-12s %10.3f %11.3f %10.3f%s\n", classNames[int(yte[i])], classNames[int(pred[i])],
			proba[i][0], proba[i][1], proba[i][2], mark)
	}

	// --- 2. the effect of C ----------------------------------------------------
	fmt.Fprintf(w, "\nregularization strength: smaller C = stronger penalty, smaller coefficients\n")
	fmt.Fprintf(w, "  %8s %10s %16s %12s\n", "C", "CV acc", "test confidence", "||coef||")
	cv := model_selection.NewStratifiedKFold(5)
	bestC, bestAcc := 0.0, -1.0
	for _, c := range []float64{0.001, 0.01, 0.1, 1, 10, 100} {
		c := c
		factory := func() (model_selection.Model, error) { return newModel(c) }
		foldScores, err := model_selection.CrossValScore(factory, Xtr, ytr, cv, nil)
		if err != nil {
			return nil, err
		}
		cvAcc := mean(foldScores)

		m, err := newModel(c)
		if err != nil {
			return nil, err
		}
		if err := m.Fit(Xtr, ytr); err != nil {
			return nil, err
		}
		p, err := m.PredictProba(Xte)
		if err != nil {
			return nil, err
		}
		conf := meanTopProbability(p)
		est, _ := m.Named("logisticregression")
		var sq float64
		for _, row := range est.(*linear.LogisticRegression).Coef() {
			for _, v := range row {
				sq += v * v
			}
		}
		fmt.Fprintf(w, "  %8g %10.4f %16.4f %12.3f\n", c, cvAcc, conf, math.Sqrt(sq))
		if cvAcc > bestAcc {
			bestC, bestAcc = c, cvAcc
		}
		switch c {
		case 0.001:
			scores["confidence_low_c"] = conf
		case 100:
			scores["confidence_high_c"] = conf
		}
	}
	scores["best_c"] = bestC
	fmt.Fprintf(w, "best cross-validated accuracy at C = %g\n", bestC)

	// --- 3. score a new measurement ---------------------------------------------
	flower := [][]float64{{6.1, 2.8, 4.6, 1.4}} // sepal length/width, petal length/width in cm
	p, err := model.PredictProba(flower)
	if err != nil {
		return nil, err
	}
	top := 0
	for k := range p[0] {
		if p[0][k] > p[0][top] {
			top = k
		}
	}
	scores["new_flower_top_prob"] = p[0][top]
	fmt.Fprintf(w, "\na new flower %v is %s (%.1f%%; versicolor %.1f%%, virginica %.1f%%)\n",
		flower[0], classNames[top], 100*p[0][top], 100*p[0][1], 100*p[0][2])
	return scores, nil
}

// leastConfident returns the indices of the k rows whose top probability is smallest.
func leastConfident(proba [][]float64, k int) []int {
	idx := make([]int, len(proba))
	top := make([]float64, len(proba))
	for i, row := range proba {
		idx[i] = i
		for _, p := range row {
			top[i] = math.Max(top[i], p)
		}
	}
	for i := 0; i < k; i++ { // partial selection sort: k is tiny
		for j := i + 1; j < len(idx); j++ {
			if top[idx[j]] < top[idx[i]] {
				idx[i], idx[j] = idx[j], idx[i]
			}
		}
	}
	return idx[:k]
}

func meanTopProbability(proba [][]float64) float64 {
	var s float64
	for _, row := range proba {
		best := 0.0
		for _, p := range row {
			best = math.Max(best, p)
		}
		s += best
	}
	return s / float64(len(proba))
}

func mean(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}
