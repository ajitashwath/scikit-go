package ensemble

import (
	"errors"
	"fmt"
	"math"

	"scikit-go/core"
	"scikit-go/internal/matutil"
	"scikit-go/metrics"
	"scikit-go/tree"
)

// Compile-time checks that RandomForestClassifier satisfies the core interfaces.
var (
	_ core.Estimator  = (*RandomForestClassifier)(nil)
	_ core.Predictor  = (*RandomForestClassifier)(nil)
	_ core.Classifier = (*RandomForestClassifier)(nil)
	_ core.Saver      = (*RandomForestClassifier)(nil)
)

// RandomForestClassifier averages the class probabilities of NTrees
// classification trees, each grown on a bootstrap sample. It mirrors
// sklearn.ensemble.RandomForestClassifier.
//
// MaxFeatures is the number of features considered per split; 0 means
// floor(sqrt(n_features)), sklearn's default for classification.
type RandomForestClassifier struct {
	NTrees          int
	Criterion       string
	MaxDepth        int
	MinSamplesSplit int
	MinSamplesLeaf  int
	MaxFeatures     int
	Bootstrap       bool
	Seed            int64
	NJobs           int

	trees     []*tree.DecisionTreeClassifier
	classes   []float64
	classIdx  [][]int // classIdx[t][k] is the forest class index of tree t's k-th class
	nFeatures int
	fitted    bool
}

// NewRandomForestClassifier returns an unfitted forest with sklearn's defaults.
func NewRandomForestClassifier() *RandomForestClassifier {
	return &RandomForestClassifier{
		NTrees:          100,
		Criterion:       "gini",
		MaxDepth:        -1,
		MinSamplesSplit: 2,
		MinSamplesLeaf:  1,
		MaxFeatures:     0,
		Bootstrap:       true,
	}
}

// defaultMaxFeatures resolves MaxFeatures == 0 to floor(sqrt(nFeatures)), at least 1.
func defaultMaxFeatures(nFeatures int) int {
	m := int(math.Sqrt(float64(nFeatures)))
	if m < 1 {
		m = 1
	}
	return m
}

// Fit grows the forest. Each tree sees a bootstrap sample of the rows (or all
// rows when Bootstrap is false) and its own random-feature stream.
func (f *RandomForestClassifier) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("RandomForestClassifier.Fit: %w", err)
	}
	err := validateForest(forestParams{f.NTrees, f.MinSamplesSplit, f.MinSamplesLeaf, f.MaxFeatures})
	if err != nil {
		return fmt.Errorf("RandomForestClassifier.Fit: %w", err)
	}
	maxFeatures := f.MaxFeatures
	if maxFeatures == 0 {
		maxFeatures = defaultMaxFeatures(len(X[0]))
	}

	seeds := treeSeeds(f.Seed, f.NTrees)
	trees := make([]*tree.DecisionTreeClassifier, f.NTrees)
	err = runParallel(f.NTrees, f.NJobs, func(i int) error {
		t := tree.NewDecisionTreeClassifier()
		t.Criterion = f.Criterion
		t.MaxDepth = f.MaxDepth
		t.MinSamplesSplit = f.MinSamplesSplit
		t.MinSamplesLeaf = f.MinSamplesLeaf
		t.MaxFeatures = maxFeatures
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
		return fmt.Errorf("RandomForestClassifier.Fit: %w", err)
	}

	f.trees = trees
	f.classes = uniqueSorted(y)
	f.classIdx = mapClasses(f.trees, f.classes)
	f.nFeatures = len(X[0])
	f.fitted = true
	return nil
}

// PredictProba returns, for each row, the mean of the trees' class probabilities.
// Columns follow Classes(); a bootstrap sample can miss a class entirely, in
// which case that tree contributes zero probability for it.
func (f *RandomForestClassifier) PredictProba(X [][]float64) ([][]float64, error) {
	if !f.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, f.nFeatures); err != nil {
		return nil, fmt.Errorf("RandomForestClassifier.PredictProba: %w", err)
	}
	out := make([][]float64, len(X))
	for i := range out {
		out[i] = make([]float64, len(f.classes))
	}
	for ti, t := range f.trees {
		proba, err := t.PredictProba(X)
		if err != nil {
			return nil, fmt.Errorf("RandomForestClassifier.PredictProba: %w", err)
		}
		idx := f.classIdx[ti]
		for i, row := range proba {
			for k, p := range row {
				out[i][idx[k]] += p
			}
		}
	}
	n := float64(len(f.trees))
	for i := range out {
		for k := range out[i] {
			out[i][k] /= n
		}
	}
	return out, nil
}

// Predict returns the class with the highest mean probability for each row,
// breaking ties toward the smallest label.
func (f *RandomForestClassifier) Predict(X [][]float64) ([]float64, error) {
	proba, err := f.PredictProba(X)
	if err != nil {
		if errors.Is(err, matutil.ErrNotFitted) {
			return nil, err
		}
		return nil, fmt.Errorf("RandomForestClassifier.Predict: %w", err)
	}
	preds := make([]float64, len(proba))
	for i, row := range proba {
		best := 0
		for k := 1; k < len(row); k++ {
			if row[k] > row[best] {
				best = k
			}
		}
		preds[i] = f.classes[best]
	}
	return preds, nil
}

// Score returns the accuracy of the predictions on X against y.
func (f *RandomForestClassifier) Score(X [][]float64, y []float64) (float64, error) {
	preds, err := f.Predict(X)
	if err != nil {
		return 0, err
	}
	score, err := metrics.AccuracyScore(y, preds)
	if err != nil {
		return 0, fmt.Errorf("RandomForestClassifier.Score: %w", err)
	}
	return score, nil
}

// FeatureImportances returns the mean impurity-based importance across trees,
// normalized to sum to one. It returns nil before Fit.
func (f *RandomForestClassifier) FeatureImportances() []float64 {
	if !f.fitted {
		return nil
	}
	per := make([][]float64, len(f.trees))
	for i, t := range f.trees {
		per[i] = t.FeatureImportances()
	}
	return meanImportances(per, f.nFeatures)
}

// Classes returns the sorted unique class labels seen at Fit time.
func (f *RandomForestClassifier) Classes() []float64 {
	if !f.fitted {
		return nil
	}
	return append([]float64(nil), f.classes...)
}

// NEstimators returns the number of fitted trees (0 before Fit).
func (f *RandomForestClassifier) NEstimators() int { return len(f.trees) }

// mapClasses maps each tree's class list onto indices of the forest's class list.
func mapClasses(trees []*tree.DecisionTreeClassifier, classes []float64) [][]int {
	index := make(map[float64]int, len(classes))
	for i, c := range classes {
		index[c] = i
	}
	out := make([][]int, len(trees))
	for ti, t := range trees {
		tc := t.Classes()
		out[ti] = make([]int, len(tc))
		for k, c := range tc {
			out[ti][k] = index[c]
		}
	}
	return out
}
