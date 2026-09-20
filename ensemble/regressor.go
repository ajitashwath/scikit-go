package ensemble

import (
	"fmt"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/metrics"
	"github.com/ajitashwath/scikit-go/tree"
)

// Compile-time checks that RandomForestRegressor satisfies the core interfaces.
var (
	_ core.Estimator = (*RandomForestRegressor)(nil)
	_ core.Predictor = (*RandomForestRegressor)(nil)
	_ core.Saver     = (*RandomForestRegressor)(nil)
)

// RandomForestRegressor averages the predictions of NTrees regression trees,
// each grown on a bootstrap sample. It mirrors sklearn.ensemble.RandomForestRegressor.
//
// MaxFeatures is the number of features considered per split; 0 means all
// features (sklearn's default for regression). MaxDepth < 0 means unlimited.
// NJobs bounds the number of goroutines growing trees; 0 means GOMAXPROCS.
type RandomForestRegressor struct {
	NTrees          int
	Criterion       string
	MaxDepth        int
	MinSamplesSplit int
	MinSamplesLeaf  int
	MaxFeatures     int
	Bootstrap       bool
	Seed            int64
	NJobs           int

	trees     []*tree.DecisionTreeRegressor
	nFeatures int
	fitted    bool
}

// NewRandomForestRegressor returns an unfitted forest with sklearn's defaults.
func NewRandomForestRegressor() *RandomForestRegressor {
	return &RandomForestRegressor{
		NTrees:          100,
		Criterion:       "mse",
		MaxDepth:        -1,
		MinSamplesSplit: 2,
		MinSamplesLeaf:  1,
		MaxFeatures:     0,
		Bootstrap:       true,
	}
}

// Fit grows the forest. Each tree sees a bootstrap sample of the rows (or all
// rows when Bootstrap is false) and its own random-feature stream.
func (f *RandomForestRegressor) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("RandomForestRegressor.Fit: %w", err)
	}
	err := validateForest(forestParams{f.NTrees, f.MinSamplesSplit, f.MinSamplesLeaf, f.MaxFeatures})
	if err != nil {
		return fmt.Errorf("RandomForestRegressor.Fit: %w", err)
	}

	seeds := treeSeeds(f.Seed, f.NTrees)
	trees := make([]*tree.DecisionTreeRegressor, f.NTrees)
	err = runParallel(f.NTrees, f.NJobs, func(i int) error {
		t := tree.NewDecisionTreeRegressor()
		t.Criterion = f.Criterion
		t.MaxDepth = f.MaxDepth
		t.MinSamplesSplit = f.MinSamplesSplit
		t.MinSamplesLeaf = f.MinSamplesLeaf
		t.MaxFeatures = f.MaxFeatures
		t.Seed = seeds[i]
		Xt, yt := X, y
		if f.Bootstrap {
			Xt, yt = bootstrapSample(X, y, seeds[i])
		}
		if err := t.Fit(Xt, yt); err != nil {
			return err
		}
		trees[i] = t
		return nil
	})
	if err != nil {
		return fmt.Errorf("RandomForestRegressor.Fit: %w", err)
	}
	f.trees = trees
	f.nFeatures = len(X[0])
	f.fitted = true
	return nil
}

// Predict returns the mean prediction of the trees for each row of X.
func (f *RandomForestRegressor) Predict(X [][]float64) ([]float64, error) {
	if !f.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, f.nFeatures); err != nil {
		return nil, fmt.Errorf("RandomForestRegressor.Predict: %w", err)
	}
	sums := make([]float64, len(X))
	for _, t := range f.trees {
		preds, err := t.Predict(X)
		if err != nil {
			return nil, fmt.Errorf("RandomForestRegressor.Predict: %w", err)
		}
		for i, p := range preds {
			sums[i] += p
		}
	}
	n := float64(len(f.trees))
	for i := range sums {
		sums[i] /= n
	}
	return sums, nil
}

// Score returns the R^2 score of the predictions on X against y.
func (f *RandomForestRegressor) Score(X [][]float64, y []float64) (float64, error) {
	preds, err := f.Predict(X)
	if err != nil {
		return 0, err
	}
	score, err := metrics.R2Score(y, preds)
	if err != nil {
		return 0, fmt.Errorf("RandomForestRegressor.Score: %w", err)
	}
	return score, nil
}

// FeatureImportances returns the mean impurity-based importance across trees,
// normalized to sum to one. It returns nil before Fit.
func (f *RandomForestRegressor) FeatureImportances() []float64 {
	if !f.fitted {
		return nil
	}
	per := make([][]float64, len(f.trees))
	for i, t := range f.trees {
		per[i] = t.FeatureImportances()
	}
	return meanImportances(per, f.nFeatures)
}

// NEstimators returns the number of fitted trees (0 before Fit).
func (f *RandomForestRegressor) NEstimators() int { return len(f.trees) }
