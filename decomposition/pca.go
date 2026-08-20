// Package decomposition implements dimensionality reduction estimators
// mirroring sklearn.decomposition: PCA via full SVD.
package decomposition

import (
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"

	"scikit-go/internal/matutil"
)

// ErrInvalidPCA is returned when PCA hyperparameters fail validation.
var ErrInvalidPCA = errors.New("invalid PCA parameters")

// PCA computes principal components via the full SVD of the column-centered
// data matrix, mirroring sklearn.decomposition.PCA with svd_solver="full".
// NComponents <= 0 keeps all components (min(n_samples, n_features)).
type PCA struct {
	NComponents int

	components [][]float64
	explainedVariance []float64
	explainedVarianceRatio []float64
	singularValues []float64
	mean []float64
	fitted bool
}

// NewPCA returns an unfitted PCA with sklearn's default hyperparameters.
func NewPCA() *PCA {
	return &PCA{NComponents: 0}
}

// Fit computes the top NComponents principal components of X.
func (p *PCA) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXMatrix(X); err != nil {
		return fmt.Errorf("PCA.Fit: %w", err)
	}
	nSamples := len(X)
	nFeatures := len(X[0])
	nComponents := p.NComponents
	if nComponents <= 0 {
		nComponents = min(nSamples, nFeatures)
	}
	if nComponents > min(nSamples, nFeatures) {
		return fmt.Errorf("PCA.Fit: %w: n_components=%d exceeds min(n_samples, n_features)=%d",
			ErrInvalidPCA, p.NComponents, min(nSamples, nFeatures))
	}

	// center data by column means
	mean := make([]float64, nFeatures)
	for i := range X {
		for j, v := range X[i] {
			mean[j] += v
		}
	}
	for j := range mean {
		mean[j] /= float64(nSamples)
	}

	dense, err := matutil.ToDense(X)
	if err != nil {
		return fmt.Errorf("PCA.Fit: %w", err)
	}
	centered := mat.DenseCopyOf(dense)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			centered.Set(i, j, centered.At(i, j)-mean[j])
		}
	}

	var svd mat.SVD
	if !svd.Factorize(centered, mat.SVDThin) {
		return fmt.Errorf("PCA.Fit: SVD failed to converge")
	}
	values := make([]float64, min(nSamples, nFeatures))
	svd.Values(values)
	var vt mat.Dense
	svd.VTo(&vt) // nFeatures x k, rows are right singular vectors

	// apply sklearn's svd_flip with u_based_decision=False: make the largest
	// absolute value in each component row positive
	components := make([][]float64, nComponents)
	for r := 0; r < nComponents; r++ {
		row := make([]float64, nFeatures)
		maxAbsIdx := 0
		for j := 0; j < nFeatures; j++ {
			row[j] = vt.At(j, r)
			if math.Abs(row[j]) > math.Abs(row[maxAbsIdx]) {
				maxAbsIdx = j
			}
		}
		sign := 1.0
		if row[maxAbsIdx] < 0 {
			sign = -1.0
		}
		for j := range row {
			row[j] *= sign
		}
		components[r] = row
	}

	explainedVariance := make([]float64, nComponents)
	var totalVar float64
	// sklearn normalizes by the total variance over ALL components, so the
	// ratios for the retained subset sum to less than 1.
	for r := 0; r < len(values); r++ {
		totalVar += values[r] * values[r] / float64(nSamples-1)
	}
	for r := 0; r < nComponents; r++ {
		explainedVariance[r] = values[r] * values[r] / float64(nSamples-1)
	}
	explainedRatio := make([]float64, nComponents)
	for r := range explainedVariance {
		if totalVar > 0 {
			explainedRatio[r] = explainedVariance[r] / totalVar
		}
	}

	p.components = components
	p.explainedVariance = explainedVariance
	p.explainedVarianceRatio = explainedRatio
	p.singularValues = append([]float64(nil), values[:nComponents]...)
	p.mean = mean
	p.fitted = true
	return nil
}

// Transform projects each row of X onto the principal components.
func (p *PCA) Transform(X [][]float64) ([][]float64, error) {
	if !p.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, len(p.mean)); err != nil {
		return nil, fmt.Errorf("PCA.Transform: %w", err)
	}
	out := make([][]float64, len(X))
	for i, row := range X {
		proj := make([]float64, len(p.components))
		for r, comp := range p.components {
			var sum float64
			for j, v := range row {
				sum += (v - p.mean[j]) * comp[j]
			}
			proj[r] = sum
		}
		out[i] = proj
	}
	return out, nil
}

// FitTransform fits the model and returns the transformed training data.
func (p *PCA) FitTransform(X [][]float64) ([][]float64, error) {
	if err := p.Fit(X, nil); err != nil {
		return nil, err
	}
	return p.Transform(X)
}

// InverseTransform reconstructs rows in the original feature space from their
// low-dimensional projections.
func (p *PCA) InverseTransform(Z [][]float64) ([][]float64, error) {
	if !p.fitted {
		return nil, matutil.ErrNotFitted
	}
	if len(Z) == 0 {
		return nil, matutil.ErrEmptyInput
	}
	for i, row := range Z {
		if len(row) != len(p.components) {
			return nil, fmt.Errorf("PCA.InverseTransform: %w: expected %d components, row %d has %d",
				matutil.ErrDimMismatch, len(p.components), i, len(row))
		}
	}
	out := make([][]float64, len(Z))
	for i, row := range Z {
		recon := make([]float64, len(p.mean))
		copy(recon, p.mean)
		for r, z := range row {
			for j, c := range p.components[r] {
				recon[j] += z * c
			}
		}
		out[i] = recon
	}
	return out, nil
}

// Components returns a copy of the principal axes (rows of components_).
func (p *PCA) Components() [][]float64 {
	if !p.fitted {
		return nil
	}
	out := make([][]float64, len(p.components))
	for i, c := range p.components {
		out[i] = append([]float64(nil), c...)
	}
	return out
}

// ExplainedVariance returns a copy of the explained variance of each component.
func (p *PCA) ExplainedVariance() []float64 {
	if !p.fitted {
		return nil
	}
	return append([]float64(nil), p.explainedVariance...)
}

// ExplainedVarianceRatio returns a copy of the fraction of total variance each
// component explains.
func (p *PCA) ExplainedVarianceRatio() []float64 {
	if !p.fitted {
		return nil
	}
	return append([]float64(nil), p.explainedVarianceRatio...)
}

// Mean returns a copy of the per-feature mean used to center the data.
func (p *PCA) Mean() []float64 {
	if !p.fitted {
		return nil
	}
	return append([]float64(nil), p.mean...)
}