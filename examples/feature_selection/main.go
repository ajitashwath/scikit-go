// Command feature_selection shows the three selectors on data where the answer
// is known. A classification set has 4 informative columns hidden among 16 noise
// columns and one constant column; a regression set depends on only 3 of its 8
// features. VarianceThreshold, SelectKBest and RFE each get a chance to find them.
//
//	go run ./examples/feature_selection
package main

import (
	"fmt"
	"io"
	"math/rand"
	"os"
	"slices"

	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/feature_selection"
	"github.com/ajitashwath/scikit-go/linear"
	"github.com/ajitashwath/scikit-go/metrics"
	"github.com/ajitashwath/scikit-go/neighbors"
	"github.com/ajitashwath/scikit-go/pipeline"
	"github.com/ajitashwath/scikit-go/preprocessing"
	"github.com/ajitashwath/scikit-go/svm"
	"github.com/ajitashwath/scikit-go/utils"
)

const (
	nInformative = 4
	nNoise       = 16
	noiseStd     = 6.0 // large next to the informative columns, so the noise hurts distance-based models
)

func main() {
	if _, err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// run returns "accuracy_all_features", "accuracy_selected", "selectkbest_hits"
// (how many of the informative columns SelectKBest kept), "pipeline_accuracy" and
// "rfe_hits" (how many of the 3 true regression features RFE kept).
func run(w io.Writer) (map[string]float64, error) {
	scores := map[string]float64{}

	// --- classification: informative + noise + constant columns --------------
	base, y, err := datasets.MakeClassification(600, nInformative, 3, 3)
	if err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(11))
	X := make([][]float64, len(base))
	for i, row := range base {
		x := append([]float64(nil), row...)
		for j := 0; j < nNoise; j++ {
			x = append(x, noiseStd*rng.NormFloat64())
		}
		x = append(x, 5) // a constant column
		X[i] = x
	}
	fmt.Fprintf(w, "classification data: %d samples, %d columns (%d informative, %d noise, 1 constant)\n\n",
		len(X), len(X[0]), nInformative, nNoise)

	// 1. VarianceThreshold drops columns that never change.
	vt := feature_selection.NewVarianceThreshold()
	X, err = vt.FitTransform(X)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(w, "VarianceThreshold: kept %d columns (dropped the constant one)\n", len(X[0]))

	Xtr, Xte, ytr, yte, err := utils.TrainTestSplit(X, y, 0.3, 42)
	if err != nil {
		return nil, err
	}

	// 2. SelectKBest ranks columns by ANOVA F-score against the label.
	sel := feature_selection.NewSelectKBest()
	sel.K = nInformative
	XtrSel, err := sel.FitTransform(Xtr, ytr)
	if err != nil {
		return nil, err
	}
	XteSel, err := sel.Transform(Xte)
	if err != nil {
		return nil, err
	}
	kept := sel.SupportIndices()
	hits := 0
	for _, j := range kept {
		if j < nInformative {
			hits++
		}
	}
	fmt.Fprintf(w, "SelectKBest(k=%d): kept columns %v (informative columns are 0-%d)\n\n", nInformative, kept, nInformative-1)
	scores["selectkbest_hits"] = float64(hits)

	// The payoff: the same classifier with and without the noise columns.
	accAll, err := knnAccuracy(Xtr, ytr, Xte, yte)
	if err != nil {
		return nil, err
	}
	accSel, err := knnAccuracy(XtrSel, ytr, XteSel, yte)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(w, "k-nearest-neighbors accuracy:\n  all %d columns       %.4f\n  selected %d columns  %.4f\n\n", len(Xtr[0]), accAll, len(kept), accSel)
	scores["accuracy_all_features"], scores["accuracy_selected"] = accAll, accSel

	// Selection can also live inside a Pipeline, so it is fit on training data only.
	sel2 := feature_selection.NewSelectKBest()
	sel2.K = nInformative
	p, err := pipeline.MakePipeline(preprocessing.NewStandardScaler(), sel2, svm.NewSVC())
	if err != nil {
		return nil, err
	}
	if err := p.Fit(Xtr, ytr); err != nil {
		return nil, err
	}
	pAcc, err := p.Score(Xte, yte)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(w, "Pipeline(StandardScaler, SelectKBest, SVC) accuracy: %.4f\n\n", pAcc)
	scores["pipeline_accuracy"] = pAcc

	// --- regression: RFE with a linear model --------------------------------
	const nFeatures = 8
	Xr := make([][]float64, 400)
	yr := make([]float64, len(Xr))
	for i := range Xr {
		row := make([]float64, nFeatures)
		for j := range row {
			row[j] = rng.NormFloat64()
		}
		Xr[i] = row
		yr[i] = 3*row[0] - 2*row[1] + 1.5*row[2] + 0.1*rng.NormFloat64()
	}
	rfe := feature_selection.NewRFE(linear.NewLinearRegression())
	rfe.NFeaturesToSelect = 3
	if err := rfe.Fit(Xr, yr); err != nil {
		return nil, err
	}
	rfeKept := rfe.SupportIndices()
	fmt.Fprintf(w, "regression data: y = 3*x0 - 2*x1 + 1.5*x2 + noise, with %d features\n", nFeatures)
	fmt.Fprintf(w, "RFE(LinearRegression, 3 features): kept columns %v\n", rfeKept)
	fmt.Fprintf(w, "  ranking per column (1 = kept): %v\n", rfe.Ranking())
	rfeHits := 0
	for _, j := range rfeKept {
		if slices.Contains([]int{0, 1, 2}, j) {
			rfeHits++
		}
	}
	scores["rfe_hits"] = float64(rfeHits)

	pred, err := rfe.Predict(Xr)
	if err != nil {
		return nil, err
	}
	r2, err := metrics.R2Score(yr, pred)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(w, "  R^2 of the model refit on those 3 columns: %.4f\n", r2)
	return scores, nil
}

func knnAccuracy(Xtr [][]float64, ytr []float64, Xte [][]float64, yte []float64) (float64, error) {
	knn := neighbors.NewKNeighborsClassifier()
	if err := knn.Fit(Xtr, ytr); err != nil {
		return 0, err
	}
	pred, err := knn.Predict(Xte)
	if err != nil {
		return 0, err
	}
	return metrics.AccuracyScore(yte, pred)
}
