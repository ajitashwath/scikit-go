// Linear Regression
package linear

import (
	"encoding/gob"
	"fmt"
	"os"

	"scikit-go/internal/matutil"

	"gonum.org/v1/gonum/mat"
)

// LinearRegression fits an ordinary least squares linear model
// y = X @ coef + intercept

// Implemented via QR decomposition of the design matrix
// X augmented with an intercept column, matching scikit-learn's default implementation
// scipy.linalg.lstsq
type LinearRegression struct {
	Coef      []float64
	Intercept float64

	nFeatures int
	fitted    bool
}

// NewLinearRegression returns an unfitted LinearRegression estimator
func NewLinearRegression() *LinearRegression {
	return &LinearRegression{}
}

// Fit computes the least-square solution for coef and intercept
func (lr *LinearRegression) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("LinearRegression.Fit: %w", err)
	}
	n := len(X)
	p := len(X[0])

	// Augmented design matrix [X | 1] so intercept is solved for directly as one more coefficient
	// Rather than centering data manually
	augmented := mat.NewDense(n, p+1, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			augmented.Set(i, j, X[i][j])
		}
		augmented.Set(i, p, 1.0)
	}

	yVec := mat.NewVecDense(n, y)
	var qr mat.QR
	qr.Factorize(augmented)

	var solution mat.Dense
	if err := qr.SolveTo(&solution, false, yVec); err != nil {
		return fmt.Errorf("LinearRegression.Fit : QR solve failed: %w", err)
	}

	coef := make([]float64, p)
	for j := 0; j < p; j++ {
		coef[j] = solution.At(j, 0)
	}
	lr.Coef = coef
	lr.Intercept = solution.At(p, 0)
	lr.nFeatures = p
	lr.fitted = true
	return nil
}

// Predict returns predictions for each row of X using the fitted coefficient and intercupt
// Returns matutil.ErrNotFitted if called
func (lr *LinearRegression) Predict(X [][]float64) ([]float64, error) {
	if !lr.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, lr.nFeatures); err != nil {
		return nil, fmt.Errorf("LinearRegression.Predict: %w", err)
	}

	preds := make([]float64, len(X))
	for i, row := range X {
		sum := lr.Intercept
		for j, v := range row {
			sum += v * lr.Coef[j]
		}
		preds[i] = sum
	}
	return preds, nil
}

// Score returns the R^2 (coefficient of determination) of the predictions
// on X against the true targets y, matching sklearn's .score() semantics.
func (lr *LinearRegression) Score(X [][]float64, y []float64) (float64, error) {
	preds, err := lr.Predict(X)
	if err != nil {
		return 0, err
	}
	if len(preds) != len(y) {
		return 0, fmt.Errorf("LinearRegression.Score: predictions and y length mismatch")
	}

	var mean float64
	for _, v := range y {
		mean += v
	}
	mean /= float64(len(y))

	var ssRes, ssTot float64
	for i := range y {
		diff := y[i] - preds[i]
		ssRes += diff * diff
		diffMean := y[i] - mean
		ssTot += diffMean * diffMean
	}

	if ssTot == 0 {
		if ssRes == 0 {
			return 1.0, nil
		}
		return 0.0, nil
	}

	return 1.0 - ssRes/ssTot, nil
}

// linearRegressionGob is the on-disk representation used by Save/Load.
// Kept as a separate exported-field struct (rather than gob-encoding
// LinearRegression directly) so the binary format is decoupled from the live struct's internal fields and can be versioned independently.
type linearRegressionGob struct {
	Version   int
	Coef      []float64
	Intercept float64
	NFeatures int
}

const linearRegressionFormatVersion = 1

// Save writes the fitted model to path in a versioned binary gob format.
func (lr *LinearRegression) Save(path string) error {
	if !lr.fitted {
		return matutil.ErrNotFitted
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("LinearRegression.Save: %w", err)
	}
	defer f.Close()

	payload := linearRegressionGob{
		Version:   linearRegressionFormatVersion,
		Coef:      lr.Coef,
		Intercept: lr.Intercept,
		NFeatures: lr.nFeatures,
	}

	if err := gob.NewEncoder(f).Encode(payload); err != nil {
		return fmt.Errorf("LinearRegression.Save: encode failed: %w", err)
	}
	return nil
}

// LoadLinearRegression reads a fitted model previously written by Save.
func LoadLinearRegression(path string) (*LinearRegression, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("LoadLinearRegression: %w", err)
	}
	defer f.Close()

	var payload linearRegressionGob
	if err := gob.NewDecoder(f).Decode(&payload); err != nil {
		return nil, fmt.Errorf("LoadLinearRegression: decode failed: %w", err)
	}
	if payload.Version != linearRegressionFormatVersion {
		return nil, fmt.Errorf("LoadLinearRegression: Unsupported format version %d (expected %d)", payload.Version, linearRegressionFormatVersion)
	}

	return &LinearRegression{
		Coef:      payload.Coef,
		Intercept: payload.Intercept,
		nFeatures: payload.NFeatures,
		fitted:    true,
	}, nil
}
