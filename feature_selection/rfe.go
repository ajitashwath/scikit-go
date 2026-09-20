package feature_selection

import (
	"errors"
	"fmt"
	"sort"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/linear"
)

// Compile-time checks that RFE satisfies the core interfaces.
var (
	_ core.Estimator = (*RFE)(nil)
	_ core.Predictor = (*RFE)(nil)
	_ core.Saver     = (*RFE)(nil)
)

// ErrNoImportances is returned when RFE's estimator exposes neither
// coefficients nor feature importances to rank features by.
var ErrNoImportances = errors.New("estimator exposes no feature importances")

// RFE (recursive feature elimination) repeatedly fits Estimator, ranks the
// remaining features by the squared magnitude of its coefficients (or by its
// feature importances) and drops the Step weakest until NFeaturesToSelect
// remain. It mirrors sklearn.feature_selection.RFE.
//
// Estimator must be a core.Estimator that is either a *linear.LinearRegression
// or provides FeatureImportances() []float64 (the trees and forests). It is
// refit in place; after Fit it holds the model trained on the selected
// features. NFeaturesToSelect == 0 keeps half of the features, sklearn's default.
type RFE struct {
	Estimator         core.Estimator
	NFeaturesToSelect int
	Step              int

	support
	ranking []int
}

// NewRFE returns an unfitted RFE wrapping estimator, with Step=1 and half of
// the features kept.
func NewRFE(estimator core.Estimator) *RFE {
	return &RFE{Estimator: estimator, Step: 1}
}

// featureImportances returns the estimator's importance of each of its input
// columns, squared so coefficient signs do not matter.
func featureImportances(est core.Estimator, nCols int) ([]float64, error) {
	var imp []float64
	switch e := est.(type) {
	case *linear.LinearRegression:
		imp = e.Coef
	case interface{ FeatureImportances() []float64 }:
		imp = e.FeatureImportances()
	default:
		return nil, fmt.Errorf("%w: %T", ErrNoImportances, est)
	}
	if len(imp) != nCols {
		return nil, fmt.Errorf("%w: got %d importances for %d features", ErrNoImportances, len(imp), nCols)
	}
	out := make([]float64, nCols)
	for j, v := range imp {
		out[j] = v * v
	}
	return out, nil
}

// selectColumns returns X restricted to the given columns.
func selectColumns(X [][]float64, cols []int) [][]float64 {
	out := make([][]float64, len(X))
	for i, row := range X {
		sel := make([]float64, len(cols))
		for k, j := range cols {
			sel[k] = row[j]
		}
		out[i] = sel
	}
	return out
}

// Fit runs the elimination loop and leaves Estimator fitted on the survivors.
func (r *RFE) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("RFE.Fit: %w", err)
	}
	if r.Estimator == nil {
		return fmt.Errorf("RFE.Fit: %w: Estimator is nil", ErrInvalidSelector)
	}
	p := len(X[0])
	if p < 2 {
		return fmt.Errorf("RFE.Fit: %w: need at least 2 features, got %d", ErrInvalidSelector, p)
	}
	if r.Step < 1 {
		return fmt.Errorf("RFE.Fit: %w: step must be >= 1, got %d", ErrInvalidSelector, r.Step)
	}
	if r.NFeaturesToSelect < 0 {
		return fmt.Errorf("RFE.Fit: %w: n_features_to_select must be >= 0, got %d", ErrInvalidSelector, r.NFeaturesToSelect)
	}
	target := r.NFeaturesToSelect
	if target == 0 {
		target = p / 2
	}

	mask := make([]bool, p)
	ranking := make([]int, p)
	for j := range mask {
		mask[j] = true
		ranking[j] = 1
	}
	remaining := p

	for remaining > target {
		cols := make([]int, 0, remaining)
		for j, keep := range mask {
			if keep {
				cols = append(cols, j)
			}
		}
		if err := r.Estimator.Fit(selectColumns(X, cols), y); err != nil {
			return fmt.Errorf("RFE.Fit: %w", err)
		}
		imp, err := featureImportances(r.Estimator, len(cols))
		if err != nil {
			return fmt.Errorf("RFE.Fit: %w", err)
		}
		order := make([]int, len(cols))
		for k := range order {
			order[k] = k
		}
		sort.SliceStable(order, func(a, b int) bool { return imp[order[a]] < imp[order[b]] })

		drop := r.Step
		if remaining-target < drop {
			drop = remaining - target
		}
		for _, k := range order[:drop] {
			mask[cols[k]] = false
		}
		remaining -= drop
		for j, keep := range mask {
			if !keep {
				ranking[j]++
			}
		}
	}

	cols := supportCols(mask)
	if err := r.Estimator.Fit(selectColumns(X, cols), y); err != nil {
		return fmt.Errorf("RFE.Fit: %w", err)
	}
	r.ranking = ranking
	r.set(mask)
	return nil
}

func supportCols(mask []bool) []int {
	cols := make([]int, 0, len(mask))
	for j, keep := range mask {
		if keep {
			cols = append(cols, j)
		}
	}
	return cols
}

// Transform keeps the selected features.
func (r *RFE) Transform(X [][]float64) ([][]float64, error) {
	return r.transform("RFE", X)
}

// FitTransform fits RFE to (X, y) and returns the reduced matrix. Unlike
// core.Transformer it needs the target, as sklearn's fit_transform does.
func (r *RFE) FitTransform(X [][]float64, y []float64) ([][]float64, error) {
	if err := r.Fit(X, y); err != nil {
		return nil, err
	}
	return r.Transform(X)
}

// Predict applies Estimator to the selected columns of X. It fails on an RFE
// restored by LoadRFE, which persists the selection but not the estimator.
func (r *RFE) Predict(X [][]float64) ([]float64, error) {
	if !r.fitted {
		return nil, matutil.ErrNotFitted
	}
	pred, ok := r.Estimator.(core.Predictor)
	if !ok {
		return nil, fmt.Errorf("RFE.Predict: %w: no fitted Estimator that can predict", ErrInvalidSelector)
	}
	reduced, err := r.transform("RFE", X)
	if err != nil {
		return nil, err
	}
	out, err := pred.Predict(reduced)
	if err != nil {
		return nil, fmt.Errorf("RFE.Predict: %w", err)
	}
	return out, nil
}

// Ranking returns each feature's rank: 1 for selected features, higher for
// features eliminated earlier. It returns nil before Fit.
func (r *RFE) Ranking() []int {
	if !r.fitted {
		return nil
	}
	return append([]int(nil), r.ranking...)
}

// Save writes the fitted selection to path in the versioned gob format. The
// wrapped Estimator is not persisted.
func (r *RFE) Save(path string) error {
	if !r.fitted {
		return matutil.ErrNotFitted
	}
	return saveSelector(path, "RFE", selectorGob{
		Kind:              "rfe",
		NFeatures:         r.nFeatures,
		Support:           r.mask,
		NFeaturesToSelect: r.NFeaturesToSelect,
		Step:              r.Step,
		Ranking:           r.ranking,
	})
}

// LoadRFE reads a fitted selection previously written by Save. The returned RFE
// can Transform but has no Estimator until one is assigned.
func LoadRFE(path string) (*RFE, error) {
	p, err := loadSelector(path, "RFE", "rfe")
	if err != nil {
		return nil, err
	}
	if len(p.Ranking) != p.NFeatures {
		return nil, fmt.Errorf("LoadRFE: ranking has %d entries for %d features", len(p.Ranking), p.NFeatures)
	}
	r := &RFE{NFeaturesToSelect: p.NFeaturesToSelect, Step: p.Step, ranking: p.Ranking}
	r.set(p.Support)
	return r, nil
}
