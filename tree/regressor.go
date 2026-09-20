package tree

import (
	"fmt"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/metrics"
)

// Compile-time checks that DecisionTreeRegressor satisfies the core interfaces.
var (
	_ core.Estimator = (*DecisionTreeRegressor)(nil)
	_ core.Predictor = (*DecisionTreeRegressor)(nil)
	_ core.Saver     = (*DecisionTreeRegressor)(nil)
)

// DecisionTreeRegressor fits a CART regression tree, mirroring
// sklearn.tree.DecisionTreeRegressor. Criterion is "mse" (squared error) or
// "mae" (mean absolute error); MaxDepth < 0 means unlimited (sklearn's None).
type DecisionTreeRegressor struct {
	Criterion       string
	MaxDepth        int
	MinSamplesSplit int
	MinSamplesLeaf  int
	MaxFeatures     int
	Seed            int64

	impl      *treeImpl
	nFeatures int
	fitted    bool
}

// NewDecisionTreeRegressor returns an unfitted DecisionTreeRegressor with
// sklearn's default hyperparameters.
func NewDecisionTreeRegressor() *DecisionTreeRegressor {
	return &DecisionTreeRegressor{
		Criterion:       "mse",
		MaxDepth:        -1,
		MinSamplesSplit: 2,
		MinSamplesLeaf:  1,
		MaxFeatures:     0,
		Seed:            0,
	}
}

func (r *DecisionTreeRegressor) criterion() (criterion, error) {
	switch r.Criterion {
	case "mse", "squared_error":
		return criterionMSE, nil
	case "mae":
		return criterionMAE, nil
	default:
		return 0, fmt.Errorf("DecisionTreeRegressor.Fit: %w: %q", ErrInvalidCriterion, r.Criterion)
	}
}

// Fit grows the tree to minimize the chosen impurity criterion.
func (r *DecisionTreeRegressor) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("DecisionTreeRegressor.Fit: %w", err)
	}
	cr, err := r.criterion()
	if err != nil {
		return err
	}
	if err := validateParams(r.MinSamplesSplit, r.MinSamplesLeaf, r.MaxFeatures); err != nil {
		return fmt.Errorf("DecisionTreeRegressor.Fit: %w", err)
	}
	params := treeParams{
		criterion:       cr,
		maxDepth:        r.MaxDepth,
		minSamplesSplit: r.MinSamplesSplit,
		minSamplesLeaf:  r.MinSamplesLeaf,
		maxFeatures:     r.MaxFeatures,
		seed:            r.Seed,
	}
	r.impl = buildTree(X, y, nil, nil, params)
	r.nFeatures = len(X[0])
	r.fitted = true
	return nil
}

// Predict returns the leaf mean (or median for mae) for each row of X.
func (r *DecisionTreeRegressor) Predict(X [][]float64) ([]float64, error) {
	if !r.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, r.nFeatures); err != nil {
		return nil, fmt.Errorf("DecisionTreeRegressor.Predict: %w", err)
	}
	preds := make([]float64, len(X))
	for i, row := range X {
		preds[i] = r.impl.predictValue(row)[0]
	}
	return preds, nil
}

// Score returns the R2 score of the predictions on X against y.
func (r *DecisionTreeRegressor) Score(X [][]float64, y []float64) (float64, error) {
	preds, err := r.Predict(X)
	if err != nil {
		return 0, err
	}
	score, err := metrics.R2Score(y, preds)
	if err != nil {
		return 0, fmt.Errorf("DecisionTreeRegressor.Score: %w", err)
	}
	return score, nil
}

// FeatureImportances returns the normalized impurity-based feature importances,
// matching sklearn's tree_.feature_importances_ (zeros for a single-node tree).
func (r *DecisionTreeRegressor) FeatureImportances() []float64 {
	if !r.fitted {
		return nil
	}
	return r.impl.featureImportances()
}

// Save writes the fitted tree to path in the versioned gob format.
func (r *DecisionTreeRegressor) Save(path string) error {
	return saveTree(path, "DecisionTreeRegressor", "regressor", r.Criterion,
		r.MaxDepth, r.MinSamplesSplit, r.MinSamplesLeaf, r.MaxFeatures, r.Seed, r.impl, nil)
}

// MarshalBinary encodes the fitted tree in the same versioned gob format as Save,
// so composite estimators can embed trees without touching the filesystem.
func (r *DecisionTreeRegressor) MarshalBinary() ([]byte, error) {
	return marshalTree("DecisionTreeRegressor", "regressor", r.Criterion,
		r.MaxDepth, r.MinSamplesSplit, r.MinSamplesLeaf, r.MaxFeatures, r.Seed, r.impl, nil)
}

// UnmarshalBinary restores a tree previously encoded by MarshalBinary or Save.
func (r *DecisionTreeRegressor) UnmarshalBinary(data []byte) error {
	payload, impl, err := unmarshalTree(data, "DecisionTreeRegressor", "regressor")
	if err != nil {
		return err
	}
	*r = DecisionTreeRegressor{
		Criterion:       payload.Criterion,
		MaxDepth:        payload.MaxDepth,
		MinSamplesSplit: payload.MinSamplesSplit,
		MinSamplesLeaf:  payload.MinSamplesLeaf,
		MaxFeatures:     payload.MaxFeatures,
		Seed:            payload.Seed,
		impl:            impl,
		nFeatures:       payload.NFeatures,
		fitted:          true,
	}
	return nil
}

// LoadDecisionTreeRegressor reads a fitted tree previously written by Save.
func LoadDecisionTreeRegressor(path string) (*DecisionTreeRegressor, error) {
	payload, impl, err := loadTree(path, "DecisionTreeRegressor", "regressor")
	if err != nil {
		return nil, err
	}
	return &DecisionTreeRegressor{
		Criterion:       payload.Criterion,
		MaxDepth:        payload.MaxDepth,
		MinSamplesSplit: payload.MinSamplesSplit,
		MinSamplesLeaf:  payload.MinSamplesLeaf,
		MaxFeatures:     payload.MaxFeatures,
		Seed:            payload.Seed,
		impl:            impl,
		nFeatures:       payload.NFeatures,
		fitted:          true,
	}, nil
}
