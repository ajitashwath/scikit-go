// Linear Regression
package linear

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/metrics"

	"gonum.org/v1/gonum/mat"
)

// LinearRegression fits an ordinary least squares linear model
// y = X @ coef + intercept
//
// It mirrors sklearn.linear_model.LinearRegression: the data are centered, and
// the coefficients are the minimum-norm least-squares solution computed from the
// SVD of the centered matrix, as scipy.linalg.lstsq does. That makes the fit
// well defined for constant or collinear columns and for more features than
// samples, where an ordinary QR solve fails.
//
// Tol is the cutoff for "small" singular values: those at most Tol times the
// largest are treated as zero when determining the effective rank. NewLinearRegression
// sets sklearn's default of 1e-6; Tol == 0 uses machine precision instead.
type LinearRegression struct {
	Coef      []float64
	Intercept float64
	Tol       float64

	nFeatures int
	rank      int
	fitted    bool
}

// NewLinearRegression returns an unfitted LinearRegression estimator
func NewLinearRegression() *LinearRegression {
	return &LinearRegression{Tol: 1e-6}
}

// Fit computes the minimum-norm least-squares solution for coef and intercept.
func (lr *LinearRegression) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("LinearRegression.Fit: %w", err)
	}
	if !(lr.Tol >= 0) { // also rejects NaN
		return fmt.Errorf("LinearRegression.Fit: tol must be >= 0, got %v", lr.Tol)
	}
	n := len(X)
	p := len(X[0])

	// Center X and y so the intercept drops out of the least-squares problem.
	xMean := make([]float64, p)
	var yMean float64
	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			xMean[j] += X[i][j]
		}
		yMean += y[i]
	}
	for j := range xMean {
		xMean[j] /= float64(n)
	}
	yMean /= float64(n)

	xc := mat.NewDense(n, p, nil)
	yc := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			xc.Set(i, j, X[i][j]-xMean[j])
		}
		yc.SetVec(i, y[i]-yMean)
	}

	var svd mat.SVD
	if ok := svd.Factorize(xc, mat.SVDThin); !ok {
		return fmt.Errorf("LinearRegression.Fit: SVD failed to converge")
	}
	sv := svd.Values(nil) // descending
	var u, v mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)

	cond := lr.Tol
	if cond == 0 {
		cond = 2.220446049250313e-16
	}
	cutoff := cond * sv[0]

	// coef = V * diag(1/s) * U' * yc, over singular values above the cutoff.
	coef := make([]float64, p)
	rank := 0
	for k, s := range sv {
		if s <= cutoff {
			continue
		}
		rank++
		var proj float64
		for i := 0; i < n; i++ {
			proj += u.At(i, k) * yc.AtVec(i)
		}
		proj /= s
		for j := 0; j < p; j++ {
			coef[j] += proj * v.At(j, k)
		}
	}

	intercept := yMean
	for j := 0; j < p; j++ {
		intercept -= xMean[j] * coef[j]
	}
	lr.Coef = coef
	lr.Intercept = intercept
	lr.nFeatures = p
	lr.rank = rank
	lr.fitted = true
	return nil
}

// Rank returns the effective rank of the centered design matrix found at Fit
// time (sklearn's rank_), or 0 before Fit.
func (lr *LinearRegression) Rank() int {
	if !lr.fitted {
		return 0
	}
	return lr.rank
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
	r2, err := metrics.R2Score(y, preds)
	if err != nil {
		return 0, fmt.Errorf("LinearRegression.Score: %w", err)
	}
	return r2, nil
}

// linearRegressionGob is the on-disk representation used by Save/Load.
// Kept as a separate exported-field struct (rather than gob-encoding
// LinearRegression directly) so the binary format is decoupled from the live struct's internal fields and can be versioned independently.
type linearRegressionGob struct {
	Version   int
	Coef      []float64
	Intercept float64
	NFeatures int
	Tol       float64 // added after version 1 shipped; gob leaves it zero in older files
	Rank      int
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
		Tol:       lr.Tol,
		Rank:      lr.rank,
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
	if payload.NFeatures < 1 || len(payload.Coef) != payload.NFeatures {
		return nil, fmt.Errorf("LoadLinearRegression: corrupt payload: %d coefficients for %d features",
			len(payload.Coef), payload.NFeatures)
	}

	return &LinearRegression{
		Coef:      payload.Coef,
		Intercept: payload.Intercept,
		Tol:       payload.Tol,
		nFeatures: payload.NFeatures,
		rank:      payload.Rank,
		fitted:    true,
	}, nil
}
