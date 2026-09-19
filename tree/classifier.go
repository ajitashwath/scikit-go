package tree

import (
	"fmt"
	"sort"

	"scikit-go/core"
	"scikit-go/internal/matutil"
	"scikit-go/metrics"
)

// Compile-time checks that DecisionTreeClassifier satisfies the core interfaces.
var (
	_ core.Estimator = (*DecisionTreeClassifier)(nil)
	_ core.Predictor = (*DecisionTreeClassifier)(nil)
	_ core.Classifier = (*DecisionTreeClassifier)(nil)
	_ core.Saver     = (*DecisionTreeClassifier)(nil)
)

// DecisionTreeClassifier fits a CART classification tree, mirroring
// sklearn.tree.DecisionTreeClassifier. Criterion is "gini" or "entropy";
// MaxDepth < 0 means unlimited (sklearn's None).
type DecisionTreeClassifier struct {
	Criterion       string
	MaxDepth        int
	MinSamplesSplit int
	MinSamplesLeaf  int
	MaxFeatures     int
	Seed            int64

	impl      *treeImpl
	classes   []float64
	nFeatures int
	fitted    bool
}

// NewDecisionTreeClassifier returns an unfitted DecisionTreeClassifier with
// sklearn's default hyperparameters.
func NewDecisionTreeClassifier() *DecisionTreeClassifier {
	return &DecisionTreeClassifier{
		Criterion:       "gini",
		MaxDepth:        -1,
		MinSamplesSplit: 2,
		MinSamplesLeaf:  1,
		MaxFeatures:     0,
		Seed:            0,
	}
}

func (c *DecisionTreeClassifier) criterion() (criterion, error) {
	switch c.Criterion {
	case "gini":
		return criterionGini, nil
	case "entropy":
		return criterionEntropy, nil
	default:
		return 0, fmt.Errorf("DecisionTreeClassifier.Fit: %w: %q", ErrInvalidCriterion, c.Criterion)
	}
}

// Fit grows the tree to minimize the chosen impurity criterion.
func (c *DecisionTreeClassifier) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("DecisionTreeClassifier.Fit: %w", err)
	}
	cr, err := c.criterion()
	if err != nil {
		return err
	}
	if err := validateParams(c.MinSamplesSplit, c.MinSamplesLeaf, c.MaxFeatures); err != nil {
		return fmt.Errorf("DecisionTreeClassifier.Fit: %w", err)
	}

	// Map labels to class indices over the sorted unique values (sklearn's classes_).
	classes := sortedUnique(y)
	yClass := make([]int, len(y))
	for i, v := range y {
		idx := sort.SearchFloat64s(classes, v)
		if idx >= len(classes) || classes[idx] != v {
			return fmt.Errorf("DecisionTreeClassifier.Fit: unexpected class label %v", v)
		}
		yClass[i] = idx
	}

	params := treeParams{
		criterion:       cr,
		maxDepth:        c.MaxDepth,
		minSamplesSplit: c.MinSamplesSplit,
		minSamplesLeaf:  c.MinSamplesLeaf,
		maxFeatures:     c.MaxFeatures,
		seed:            c.Seed,
	}
	c.impl = buildTree(X, y, classes, yClass, params)
	c.classes = classes
	c.nFeatures = len(X[0])
	c.fitted = true
	return nil
}

// Predict returns the majority class of each row's leaf.
func (c *DecisionTreeClassifier) Predict(X [][]float64) ([]float64, error) {
	if !c.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, c.nFeatures); err != nil {
		return nil, fmt.Errorf("DecisionTreeClassifier.Predict: %w", err)
	}
	preds := make([]float64, len(X))
	for i, row := range X {
		dist := c.impl.predictValue(row)
		preds[i] = c.classes[argmax(dist)]
	}
	return preds, nil
}

// PredictProba returns the per-class probability (leaf class distribution) of each row.
func (c *DecisionTreeClassifier) PredictProba(X [][]float64) ([][]float64, error) {
	if !c.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, c.nFeatures); err != nil {
		return nil, fmt.Errorf("DecisionTreeClassifier.PredictProba: %w", err)
	}
	out := make([][]float64, len(X))
	for i, row := range X {
		dist := c.impl.predictValue(row)
		out[i] = append([]float64(nil), dist...)
	}
	return out, nil
}

// Score returns the accuracy of the predictions on X against y.
func (c *DecisionTreeClassifier) Score(X [][]float64, y []float64) (float64, error) {
	preds, err := c.Predict(X)
	if err != nil {
		return 0, err
	}
	score, err := metrics.AccuracyScore(y, preds)
	if err != nil {
		return 0, fmt.Errorf("DecisionTreeClassifier.Score: %w", err)
	}
	return score, nil
}

// FeatureImportances returns the normalized impurity-based feature importances,
// matching sklearn's tree_.feature_importances_ (zeros for a single-node tree).
func (c *DecisionTreeClassifier) FeatureImportances() []float64 {
	if !c.fitted {
		return nil
	}
	return c.impl.featureImportances()
}

// Classes returns the sorted unique class labels learned at Fit time.
func (c *DecisionTreeClassifier) Classes() []float64 {
	if !c.fitted {
		return nil
	}
	return append([]float64(nil), c.classes...)
}

// Save writes the fitted tree to path in the versioned gob format.
func (c *DecisionTreeClassifier) Save(path string) error {
	return saveTree(path, "DecisionTreeClassifier", "classifier", c.Criterion,
		c.MaxDepth, c.MinSamplesSplit, c.MinSamplesLeaf, c.MaxFeatures, c.Seed, c.impl, c.classes)
}

// MarshalBinary encodes the fitted tree in the same versioned gob format as Save,
// so composite estimators can embed trees without touching the filesystem.
func (c *DecisionTreeClassifier) MarshalBinary() ([]byte, error) {
	return marshalTree("DecisionTreeClassifier", "classifier", c.Criterion,
		c.MaxDepth, c.MinSamplesSplit, c.MinSamplesLeaf, c.MaxFeatures, c.Seed, c.impl, c.classes)
}

// UnmarshalBinary restores a tree previously encoded by MarshalBinary or Save.
func (c *DecisionTreeClassifier) UnmarshalBinary(data []byte) error {
	payload, impl, err := unmarshalTree(data, "DecisionTreeClassifier", "classifier")
	if err != nil {
		return err
	}
	*c = DecisionTreeClassifier{
		Criterion:       payload.Criterion,
		MaxDepth:        payload.MaxDepth,
		MinSamplesSplit: payload.MinSamplesSplit,
		MinSamplesLeaf:  payload.MinSamplesLeaf,
		MaxFeatures:     payload.MaxFeatures,
		Seed:            payload.Seed,
		impl:            impl,
		classes:         payload.Classes,
		nFeatures:       payload.NFeatures,
		fitted:          true,
	}
	return nil
}

// LoadDecisionTreeClassifier reads a fitted tree previously written by Save.
func LoadDecisionTreeClassifier(path string) (*DecisionTreeClassifier, error) {
	payload, impl, err := loadTree(path, "DecisionTreeClassifier", "classifier")
	if err != nil {
		return nil, err
	}
	return &DecisionTreeClassifier{
		Criterion:       payload.Criterion,
		MaxDepth:        payload.MaxDepth,
		MinSamplesSplit: payload.MinSamplesSplit,
		MinSamplesLeaf:  payload.MinSamplesLeaf,
		MaxFeatures:     payload.MaxFeatures,
		Seed:            payload.Seed,
		impl:            impl,
		classes:         payload.Classes,
		nFeatures:       payload.NFeatures,
		fitted:          true,
	}, nil
}

func sortedUnique(y []float64) []float64 {
	classes := append([]float64(nil), y...)
	sort.Float64s(classes)
	out := classes[:0]
	var last float64
	for i, v := range classes {
		if i == 0 || v != last {
			out = append(out, v)
			last = v
		}
	}
	return out
}

func argmax(v []float64) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}