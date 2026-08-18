package preprocessing

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"

	"scikit-go/internal/matutil"

	"scikit-go/core"
)

// Compile-time check that StandardScaler satisfies core.Transformer
var _ core.Transformer = (*StandardScaler)(nil)

// StandardScaler standardizes features by removing mean and scaling to unit variance
// z = (x - mean) / scale
// where scale is population standard deviation (ddof = 0), matching scikit-learn's StandardScaler exactly
// Each feature (column) is scaled independently.

// Zero-variance columns are a documented sklearn edge case
// Rather than dividing by zero, sklearn leaves scale_ at 1.0 for those columns
// So a constant feature a column of zeros after scaling instead of NaN/Inf
type StandardScaler struct {
	Mean      []float64 // per-feature mean, learned at Fit time
	Scale     []float64 // per-feature standard deviation
	Var       []float64 // per-feature variance, exposed for parity with sklearn's var_
	nFeatures int
	fitted    bool
}

// NewStandardScaler returns an unfitted StandardScaler
func NewStandardScaler() *StandardScaler {
	return &StandardScaler{}
}

// Fit computes the per-feature mean and standard deviation from X
func (s *StandardScaler) Fit(X [][]float64) error {
	if len(X) == 0 || len(X[0]) == 0 {
		return fmt.Errorf("StandardScaler.Fit: %w", matutil.ErrEmptyInput)
	}
	n := len(X)
	p := len(X[0])
	for i, row := range X {
		if len(row) != p {
			return fmt.Errorf("StandardScaler.Fit: %w (Row 0 has %d features, Row %d has %d)", matutil.ErrRaggedInput, p, i, len(row))
		}
		for _, v := range row {
			if math.IsNaN(v) {
				return fmt.Errorf("StandardScaler.Fit: %w", matutil.ErrContainsNaN)
			}
			if math.IsInf(v, 0) {
				return fmt.Errorf("StandardScaler.Fit: %w", matutil.ErrContainsInf)
			}
		}
	}
	mean := make([]float64, p)
	for _, row := range X {
		for j, v := range row {
			mean[j] += v
		}
	}
	for j := range mean {
		mean[j] /= float64(n)
	}

	variance := make([]float64, p)
	for _, row := range X {
		for j, v := range row {
			d := v - mean[j]
			variance[j] += d * d
		}
	}

	scale := make([]float64, p)
	for j := range variance {
		variance[j] /= float64(n) // Population variance
		sd := math.Sqrt(variance[j])
		if sd == 0 {
			// Zero-variance columns keep scale = 1.0
			scale[j] = 1.0
		} else {
			scale[j] = sd
		}
	}
	s.Mean = mean
	s.Var = variance
	s.Scale = scale
	s.nFeatures = p
	s.fitted = true
	return nil
}

// Transform applies (x - mean) / scale to each feature of X using statistics learned at Fit time
// Returns matutil.ErrNotFitted if called before fit
func (s *StandardScaler) Transform(X [][]float64) ([][]float64, error) {
	if !s.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, s.nFeatures); err != nil {
		return nil, fmt.Errorf("StandardScaler.Transform: %w", err)
	}
	out := make([][]float64, len(X))
	for i, row := range X {
		scaledRow := make([]float64, len(row))
		for j, v := range row {
			scaledRow[j] = (v - s.Mean[j]) / s.Scale[j]
		}
		out[i] = scaledRow
	}
	return out, nil
}

// FitTransform fits the scaler to X and returns the transformed result in one call
func (s *StandardScaler) FitTransform(X [][]float64) ([][]float64, error) {
	if err := s.Fit(X); err != nil {
		return nil, err
	}
	return s.Transform(X)
}

// InverseTransform reverses the scaling
// x = z * scale + mean
// Useful for converting model outputs or coefficients back to original feature scale
func (s *StandardScaler) InverseTransform(X [][]float64) ([][]float64, error) {
	if !s.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, s.nFeatures); err != nil {
		return nil, fmt.Errorf("StandardScaler.InverseTransform: %w", err)
	}
	out := make([][]float64, len(X))
	for i, row := range X {
		origRow := make([]float64, len(row))
		for j, v := range row {
			origRow[j] = v*s.Scale[j] + s.Mean[j]
		}
		out[i] = origRow
	}
	return out, nil
}

type standardScalerGob struct {
	Version   int
	Mean      []float64
	Scale     []float64
	Var       []float64
	NFeatures int
}

const standardScalerFormatVersion = 1

func (s *StandardScaler) Save(path string) error {
	if !s.fitted {
		return matutil.ErrNotFitted
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("StandardScaler.Save: %w", err)
	}
	defer f.Close()

	payload := standardScalerGob{
		Version:   standardScalerFormatVersion,
		Mean:      s.Mean,
		Scale:     s.Scale,
		Var:       s.Var,
		NFeatures: s.nFeatures,
	}
	if err := gob.NewEncoder(f).Encode(payload); err != nil {
		return fmt.Errorf("StandardScaler.Save: Encode failed: %w", err)
	}
	return nil
}

// LoadStandardScaler reads a fitted scaler previously written by Save
func LoadStandardScaler(path string) (*StandardScaler, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("LoadStandardScaler: %w", err)
	}
	defer f.Close()

	var payload standardScalerGob
	if err := gob.NewDecoder(f).Decode(&payload); err != nil {
		return nil, fmt.Errorf("LoadStandardScaler: Decode failed: %w", err)
	}
	if payload.Version != standardScalerFormatVersion {
		return nil, fmt.Errorf("LoadStandardScaler: Unsupported format version %d (expected %d)", payload.Version, standardScalerFormatVersion)
	}

	return &StandardScaler{
		Mean:      payload.Mean,
		Scale:     payload.Scale,
		Var:       payload.Var,
		nFeatures: payload.NFeatures,
		fitted:    true,
	}, nil
}
