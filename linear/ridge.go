package linear

import (
	"fmt"
	"math"

	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/metrics"

	"gonum.org/v1/gonum/mat"
)

// Ridge is linear regression with an L2 penalty on the coefficients. It minimizes
//
//	||y - X @ coef - intercept||^2 + Alpha * ||coef||^2
//
// mirroring sklearn.linear_model.Ridge. The intercept is never penalized. Larger
// Alpha shrinks the coefficients harder; Alpha = 0 is ordinary least squares.
//
// The solution is computed from the SVD of the (centered) data rather than by
// solving the normal equations, so it is accurate for nearly collinear columns and
// works when there are more features than samples.
type Ridge struct {
	Alpha        float64
	FitIntercept bool // center the data and fit an intercept; false fits through the origin

	Coef      []float64
	Intercept float64

	nFeatures int
	fitted    bool
}

// NewRidge returns an unfitted Ridge with sklearn's defaults: Alpha 1 and an intercept.
func NewRidge() *Ridge {
	return &Ridge{Alpha: 1, FitIntercept: true}
}

// Fit computes Coef and Intercept.
func (r *Ridge) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("Ridge.Fit: %w", err)
	}
	if !(r.Alpha >= 0) || math.IsInf(r.Alpha, 0) {
		return fmt.Errorf("Ridge.Fit: alpha must be finite and >= 0, got %v", r.Alpha)
	}
	n, p := len(X), len(X[0])
	xc, yc, xMean, yMean := centered(X, y, r.FitIntercept)

	var svd mat.SVD
	if ok := svd.Factorize(xc, mat.SVDThin); !ok {
		return fmt.Errorf("Ridge.Fit: SVD failed to converge")
	}
	sv := svd.Values(nil) // descending
	var u, v mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)

	// coef = V diag(s / (s^2 + alpha)) U' yc. With alpha = 0 this is the least-squares
	// solution, where singular values that are numerically zero must be dropped.
	cutoff := 0.0
	if r.Alpha == 0 && len(sv) > 0 {
		cutoff = 2.220446049250313e-16 * float64(max(n, p)) * sv[0]
	}
	coef := make([]float64, p)
	for k, s := range sv {
		if s <= cutoff || s == 0 {
			continue
		}
		var proj float64
		for i := 0; i < n; i++ {
			proj += u.At(i, k) * yc[i]
		}
		proj *= s / (s*s + r.Alpha)
		for j := 0; j < p; j++ {
			coef[j] += proj * v.At(j, k)
		}
	}
	if !allFinite(coef) {
		return fmt.Errorf("Ridge.Fit: the solution is not finite; the data may contain extreme magnitudes")
	}

	r.Coef = coef
	r.Intercept = interceptFor(coef, xMean, yMean, r.FitIntercept)
	r.nFeatures = p
	r.fitted = true
	return nil
}

// Predict returns Coef . x + Intercept for each row of X.
func (r *Ridge) Predict(X [][]float64) ([]float64, error) {
	if !r.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, r.nFeatures); err != nil {
		return nil, fmt.Errorf("Ridge.Predict: %w", err)
	}
	return linearPredict(X, r.Coef, r.Intercept), nil
}

// Score returns the R^2 of the predictions on X against y.
func (r *Ridge) Score(X [][]float64, y []float64) (float64, error) {
	pred, err := r.Predict(X)
	if err != nil {
		return 0, err
	}
	r2, err := metrics.R2Score(y, pred)
	if err != nil {
		return 0, fmt.Errorf("Ridge.Score: %w", err)
	}
	return r2, nil
}

// centered returns X and y with their column means removed when fitIntercept is set
// (the means are returned too), or plain copies of the inputs otherwise. Removing the
// means is what takes the unpenalized intercept out of the optimization.
func centered(X [][]float64, y []float64, fitIntercept bool) (xc *mat.Dense, yc, xMean []float64, yMean float64) {
	n, p := len(X), len(X[0])
	xMean = make([]float64, p)
	if fitIntercept {
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
	}
	xc = mat.NewDense(n, p, nil)
	yc = make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			xc.Set(i, j, X[i][j]-xMean[j])
		}
		yc[i] = y[i] - yMean
	}
	return xc, yc, xMean, yMean
}

// interceptFor recovers the intercept from the coefficients fitted on centered data.
func interceptFor(coef, xMean []float64, yMean float64, fitIntercept bool) float64 {
	if !fitIntercept {
		return 0
	}
	b := yMean
	for j, c := range coef {
		b -= xMean[j] * c
	}
	return b
}

func linearPredict(X [][]float64, coef []float64, intercept float64) []float64 {
	out := make([]float64, len(X))
	for i, row := range X {
		sum := intercept
		for j, v := range row {
			sum += v * coef[j]
		}
		out[i] = sum
	}
	return out
}

// ridgeGob is the versioned on-disk payload.
type ridgeGob struct {
	Version      int
	Kind         string
	Coef         []float64
	Intercept    float64
	NFeatures    int
	Alpha        float64
	FitIntercept bool
}

const (
	ridgeFormatVersion = 1
	ridgeKind          = "ridge"
)

// Save writes the fitted model to path in the versioned gob format.
func (r *Ridge) Save(path string) error {
	if !r.fitted {
		return matutil.ErrNotFitted
	}
	return saveGob("Ridge.Save", path, ridgeGob{
		Version: ridgeFormatVersion, Kind: ridgeKind, Coef: r.Coef, Intercept: r.Intercept,
		NFeatures: r.nFeatures, Alpha: r.Alpha, FitIntercept: r.FitIntercept,
	})
}

// LoadRidge reads a fitted model previously written by Save.
func LoadRidge(path string) (*Ridge, error) {
	var g ridgeGob
	if err := loadGob("LoadRidge", path, &g); err != nil {
		return nil, err
	}
	if err := checkLoaded("LoadRidge", g.Version, ridgeFormatVersion, g.Kind, ridgeKind, g.NFeatures, g.Coef, g.Intercept); err != nil {
		return nil, err
	}
	if !(g.Alpha >= 0) || math.IsInf(g.Alpha, 0) {
		return nil, fmt.Errorf("LoadRidge: corrupt payload: alpha %v", g.Alpha)
	}
	return &Ridge{Alpha: g.Alpha, FitIntercept: g.FitIntercept, Coef: g.Coef, Intercept: g.Intercept,
		nFeatures: g.NFeatures, fitted: true}, nil
}
