package feature_selection

import (
	"fmt"
	"math"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/internal/matutil"
)

// Compile-time checks that VarianceThreshold satisfies the core interfaces.
var (
	_ core.Estimator   = (*VarianceThreshold)(nil)
	_ core.Transformer = (*VarianceThreshold)(nil)
	_ core.Saver       = (*VarianceThreshold)(nil)
)

// VarianceThreshold removes features whose variance does not exceed Threshold,
// mirroring sklearn.feature_selection.VarianceThreshold. The default threshold
// of 0 drops constant features.
type VarianceThreshold struct {
	Threshold float64

	support
	variances []float64
}

// NewVarianceThreshold returns an unfitted selector with sklearn's default
// threshold of 0.
func NewVarianceThreshold() *VarianceThreshold {
	return &VarianceThreshold{}
}

// Fit learns each feature's variance from X. y is ignored and may be nil.
func (v *VarianceThreshold) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXMatrix(X); err != nil {
		return fmt.Errorf("VarianceThreshold.Fit: %w", err)
	}
	if v.Threshold < 0 {
		return fmt.Errorf("VarianceThreshold.Fit: %w: threshold must be >= 0, got %v", ErrInvalidSelector, v.Threshold)
	}
	n, p := len(X), len(X[0])
	variances := make([]float64, p)
	for j := 0; j < p; j++ {
		var mean float64
		lo, hi := X[0][j], X[0][j]
		for i := 0; i < n; i++ {
			mean += X[i][j]
			lo = math.Min(lo, X[i][j])
			hi = math.Max(hi, X[i][j])
		}
		mean /= float64(n)
		var ss float64
		for i := 0; i < n; i++ {
			d := X[i][j] - mean
			ss += d * d
		}
		variances[j] = ss / float64(n)
		if v.Threshold == 0 {
			// Use the peak-to-peak range so rounding error cannot make a
			// constant feature look like it has tiny positive variance.
			variances[j] = math.Min(variances[j], hi-lo)
		}
	}

	mask := make([]bool, p)
	kept := false
	for j, val := range variances {
		mask[j] = val > v.Threshold
		kept = kept || mask[j]
	}
	if !kept {
		return fmt.Errorf("VarianceThreshold.Fit: %w: no feature in X meets the variance threshold %.5f", ErrInvalidSelector, v.Threshold)
	}
	v.variances = variances
	v.set(mask)
	return nil
}

// Transform keeps the features whose variance exceeded the threshold at Fit time.
func (v *VarianceThreshold) Transform(X [][]float64) ([][]float64, error) {
	return v.transform("VarianceThreshold", X)
}

// FitTransform fits the selector to X and returns the reduced matrix.
func (v *VarianceThreshold) FitTransform(X [][]float64) ([][]float64, error) {
	if err := v.Fit(X, nil); err != nil {
		return nil, err
	}
	return v.Transform(X)
}

// Variances returns the per-feature variances learned at Fit time.
func (v *VarianceThreshold) Variances() []float64 {
	if !v.fitted {
		return nil
	}
	return append([]float64(nil), v.variances...)
}

// Save writes the fitted selector to path in the versioned gob format.
func (v *VarianceThreshold) Save(path string) error {
	if !v.fitted {
		return matutil.ErrNotFitted
	}
	return saveSelector(path, "VarianceThreshold", selectorGob{
		Kind:      "variance_threshold",
		NFeatures: v.nFeatures,
		Support:   v.mask,
		Threshold: v.Threshold,
		Variances: v.variances,
	})
}

// LoadVarianceThreshold reads a fitted selector previously written by Save.
func LoadVarianceThreshold(path string) (*VarianceThreshold, error) {
	p, err := loadSelector(path, "VarianceThreshold", "variance_threshold")
	if err != nil {
		return nil, err
	}
	v := &VarianceThreshold{Threshold: p.Threshold, variances: p.Variances}
	v.set(p.Support)
	return v, nil
}
