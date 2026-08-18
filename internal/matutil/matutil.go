package matutil

import (
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

var (
	// ErrEmptyInput is returned when X or y has zero samples.
	ErrEmptyInput = errors.New("matutil: input has zero samples")

	// ErrDimMismatch is returned when X and y have inconsistent sample counts.
	ErrDimMismatch = errors.New("matutil: X and y have mismatched sample counts")

	// ErrLengthMismatch is returned when two vectors that must have equal length differ.
	ErrLengthMismatch = errors.New("matutil: vectors have mismatched lengths")

	// ErrRaggedInput is returned when rows of X have inconsistent feature counts.
	ErrRaggedInput = errors.New("matutil: rows of X have inconsistent lengths")

	// ErrNotFitted is returned when Predict is called before Fit.
	ErrNotFitted = errors.New("matutil: estimator has not been fitted yet")

	// ErrContainsNaN is returned when input contains NaN values.
	ErrContainsNaN = errors.New("matutil: input contains NaN values")

	// ErrContainsInf is returned when input contains Inf values.
	ErrContainsInf = errors.New("matutil: input contains Inf values")
)

// ToDense converts a [][]float64 sample matrix into a gonum *mat.Dense
// Returns an error if X is empty or ragged
func ToDense(X [][]float64) (*mat.Dense, error) {
	n := len(X)
	if n == 0 {
		return nil, ErrEmptyInput
	}
	p := len(X[0])
	if p == 0 {
		return nil, ErrEmptyInput
	}

	flat := make([]float64, 0, n*p)
	for i, row := range X {
		if len(row) != p {
			return nil, fmt.Errorf("%w: Row 0 has %d features, Row %d has %d", ErrRaggedInput, p, i, len(row))
		}
		flat = append(flat, row...)
	}
	if err := checkFinite(flat); err != nil {
		return nil, err
	}
	return mat.NewDense(n, p, flat), nil
}

// ValidateXy checks that X and y are non-empty, non-ragged, dimensionally consistent and fall of NaN or inf
// This is standard validation every Fit method should run first
func ValidateXy(X [][]float64, y []float64) error {
	if len(X) == 0 || len(y) == 0 {
		return ErrEmptyInput
	}
	if len(X) != len(y) {
		return fmt.Errorf("%w: X has %d samples, y has %d", ErrDimMismatch, len(X), len(y))
	}
	p := len(X[0])
	if p == 0 {
		return ErrEmptyInput
	}
	for i, row := range X {
		if len(row) != p {
			return fmt.Errorf("%w: Row 0 has %d features, row %d has %d", ErrRaggedInput, p, i, len(row))
		}
	}
	if err := checkFinite(y); err != nil {
		return err
	}
	for _, row := range X {
		if err := checkFinite(row); err != nil {
			return err
		}
	}
	return nil
}

// ValidateX validates a prediction-time input matrix and checks that its feature count matches nFeatures
func ValidateX(X [][]float64, nFeatures int) error {
	if len(X) == 0 {
		return ErrEmptyInput
	}
	for i, row := range X {
		if len(row) != nFeatures {
			return fmt.Errorf("%w: Expected %d features (from training), Row %d has %d", ErrDimMismatch, nFeatures, i, len(row))
		}
		if err := checkFinite(row); err != nil {
			return err
		}
	}
	return nil
}

func checkFinite(vals []float64) error {
	for _, v := range vals {
		if math.IsNaN(v) {
			return ErrContainsNaN
		}
		if math.IsInf(v, 0) {
			return ErrContainsInf
		}
	}
	return nil
}

// DenseToSlice converts a gonum *mat.Dense back into [][]float64
func DenseToSlice(m *mat.Dense) [][]float64 {
	r, c := m.Dims()
	out := make([][]float64, r)
	for i := 0; i < r; i++ {
		row := make([]float64, c)
		for j := 0; j < c; j++ {
			row[j] = m.At(i, j)
		}
		out[i] = row
	}
	return out
}
