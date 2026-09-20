package linear

import (
	"fmt"
	"math"

	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/metrics"
)

// ElasticNet is linear regression with a mix of L1 and L2 penalties. It minimizes
//
//	1/(2n) ||y - X @ coef - intercept||^2 + Alpha*L1Ratio*||coef||_1 + Alpha*(1-L1Ratio)/2 * ||coef||^2
//
// mirroring sklearn.linear_model.ElasticNet, including its cyclic coordinate descent
// and its duality-gap stopping rule. L1Ratio = 1 is the Lasso, which sets some
// coefficients exactly to zero; L1Ratio = 0 is a pure L2 (ridge-like) penalty. The
// intercept is never penalized.
//
// The solver stops when the duality gap is at most Tol*||y||^2 or after MaxIter
// passes over the features. Like sklearn it does not fail when MaxIter is hit; check
// Converged, and raise MaxIter or Tol if it is false.
type ElasticNet struct {
	Alpha        float64
	L1Ratio      float64 // in [0, 1]
	FitIntercept bool
	MaxIter      int
	Tol          float64

	Coef      []float64
	Intercept float64

	nFeatures int
	nIter     int
	converged bool
	fitted    bool
}

// NewElasticNet returns an unfitted ElasticNet with sklearn's defaults: Alpha 1,
// L1Ratio 0.5, an intercept, MaxIter 1000 and Tol 1e-4.
func NewElasticNet() *ElasticNet {
	return &ElasticNet{Alpha: 1, L1Ratio: 0.5, FitIntercept: true, MaxIter: 1000, Tol: 1e-4}
}

// Fit runs coordinate descent to compute Coef and Intercept.
func (e *ElasticNet) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("ElasticNet.Fit: %w", err)
	}
	if err := checkCDParams(e.Alpha, e.L1Ratio, e.MaxIter, e.Tol); err != nil {
		return fmt.Errorf("ElasticNet.Fit: %w", err)
	}
	res, err := fitCD(X, y, e.Alpha, e.L1Ratio, e.FitIntercept, e.MaxIter, e.Tol)
	if err != nil {
		return fmt.Errorf("ElasticNet.Fit: %w", err)
	}
	e.Coef, e.Intercept, e.nIter, e.converged = res.coef, res.intercept, res.nIter, res.converged
	e.nFeatures = len(X[0])
	e.fitted = true
	return nil
}

// Predict returns Coef . x + Intercept for each row of X.
func (e *ElasticNet) Predict(X [][]float64) ([]float64, error) {
	if !e.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, e.nFeatures); err != nil {
		return nil, fmt.Errorf("ElasticNet.Predict: %w", err)
	}
	return linearPredict(X, e.Coef, e.Intercept), nil
}

// Score returns the R^2 of the predictions on X against y.
func (e *ElasticNet) Score(X [][]float64, y []float64) (float64, error) {
	pred, err := e.Predict(X)
	if err != nil {
		return 0, err
	}
	r2, err := metrics.R2Score(y, pred)
	if err != nil {
		return 0, fmt.Errorf("ElasticNet.Score: %w", err)
	}
	return r2, nil
}

// NIter is the number of passes over the features that Fit used (0 before Fit).
func (e *ElasticNet) NIter() int { return e.nIter }

// Converged reports whether Fit reached Tol before running out of MaxIter passes.
func (e *ElasticNet) Converged() bool { return e.converged }

// Lasso is linear regression with an L1 penalty, which drives some coefficients
// exactly to zero. It minimizes
//
//	1/(2n) ||y - X @ coef - intercept||^2 + Alpha * ||coef||_1
//
// and mirrors sklearn.linear_model.Lasso: it is an ElasticNet with L1Ratio 1 and
// the same solver, stopping rule and Converged caveat. Alpha = 0 is ordinary least
// squares, but sklearn advises LinearRegression for that.
type Lasso struct {
	Alpha        float64
	FitIntercept bool
	MaxIter      int
	Tol          float64

	Coef      []float64
	Intercept float64

	nFeatures int
	nIter     int
	converged bool
	fitted    bool
}

// NewLasso returns an unfitted Lasso with sklearn's defaults: Alpha 1, an intercept,
// MaxIter 1000 and Tol 1e-4.
func NewLasso() *Lasso {
	return &Lasso{Alpha: 1, FitIntercept: true, MaxIter: 1000, Tol: 1e-4}
}

// Fit runs coordinate descent to compute Coef and Intercept.
func (l *Lasso) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("Lasso.Fit: %w", err)
	}
	if err := checkCDParams(l.Alpha, 1, l.MaxIter, l.Tol); err != nil {
		return fmt.Errorf("Lasso.Fit: %w", err)
	}
	res, err := fitCD(X, y, l.Alpha, 1, l.FitIntercept, l.MaxIter, l.Tol)
	if err != nil {
		return fmt.Errorf("Lasso.Fit: %w", err)
	}
	l.Coef, l.Intercept, l.nIter, l.converged = res.coef, res.intercept, res.nIter, res.converged
	l.nFeatures = len(X[0])
	l.fitted = true
	return nil
}

// Predict returns Coef . x + Intercept for each row of X.
func (l *Lasso) Predict(X [][]float64) ([]float64, error) {
	if !l.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, l.nFeatures); err != nil {
		return nil, fmt.Errorf("Lasso.Predict: %w", err)
	}
	return linearPredict(X, l.Coef, l.Intercept), nil
}

// Score returns the R^2 of the predictions on X against y.
func (l *Lasso) Score(X [][]float64, y []float64) (float64, error) {
	pred, err := l.Predict(X)
	if err != nil {
		return 0, err
	}
	r2, err := metrics.R2Score(y, pred)
	if err != nil {
		return 0, fmt.Errorf("Lasso.Score: %w", err)
	}
	return r2, nil
}

// NIter is the number of passes over the features that Fit used (0 before Fit).
func (l *Lasso) NIter() int { return l.nIter }

// Converged reports whether Fit reached Tol before running out of MaxIter passes.
func (l *Lasso) Converged() bool { return l.converged }

func checkCDParams(alpha, l1Ratio float64, maxIter int, tol float64) error {
	switch {
	case !(alpha >= 0) || math.IsInf(alpha, 0):
		return fmt.Errorf("alpha must be finite and >= 0, got %v", alpha)
	case !(l1Ratio >= 0 && l1Ratio <= 1):
		return fmt.Errorf("l1Ratio must be in [0, 1], got %v", l1Ratio)
	case maxIter < 1:
		return fmt.Errorf("maxIter must be >= 1, got %d", maxIter)
	case !(tol >= 0) || math.IsInf(tol, 0):
		return fmt.Errorf("tol must be finite and >= 0, got %v", tol)
	}
	return nil
}

type cdResult struct {
	coef      []float64
	intercept float64
	nIter     int
	converged bool
}

// fitCD centers the data (when fitIntercept) and runs coordinate descent on it.
func fitCD(X [][]float64, y []float64, alpha, l1Ratio float64, fitIntercept bool, maxIter int, tol float64) (cdResult, error) {
	n, p := len(X), len(X[0])
	xc, yc, xMean, yMean := centered(X, y, fitIntercept)
	cols := make([][]float64, p) // column-major copies: the solver sweeps columns
	for j := range cols {
		cols[j] = make([]float64, n)
		for i := 0; i < n; i++ {
			cols[j][i] = xc.At(i, j)
		}
	}
	w, nIter, converged := coordinateDescent(cols, yc, alpha, l1Ratio, maxIter, tol)
	if !allFinite(w) {
		return cdResult{}, fmt.Errorf("the solution is not finite; the data may contain extreme magnitudes")
	}
	return cdResult{coef: w, intercept: interceptFor(w, xMean, yMean, fitIntercept), nIter: nIter, converged: converged}, nil
}

// coordinateDescent minimizes the elastic-net objective over the columns of the
// centered design matrix, following sklearn 1.9's enet_coordinate_descent: the
// duality gap is checked once before the first pass (so an all-zero solution costs
// no iterations), then cyclic soft-thresholding updates run, and the gap is checked
// again whenever the largest coefficient change in a pass is small relative to the
// largest coefficient.
func coordinateDescent(cols [][]float64, y []float64, alpha, l1Ratio float64, maxIter int, tol float64) (w []float64, nIter int, converged bool) {
	n, p := len(y), len(cols)
	l1Reg := alpha * l1Ratio * float64(n)
	l2Reg := alpha * (1 - l1Ratio) * float64(n)

	w = make([]float64, p)
	resid := append([]float64(nil), y...) // y - X w
	norms := make([]float64, p)
	for j, c := range cols {
		norms[j] = dot(c, c)
	}
	gapTol := tol * dot(y, y)

	if dualityGap(cols, y, resid, w, l1Reg, l2Reg) <= gapTol {
		return w, 0, true
	}
	for iter := 0; iter < maxIter; iter++ {
		var wMax, dWMax float64
		for j, c := range cols {
			if norms[j] == 0 {
				continue
			}
			old := w[j]
			tmp := dot(c, resid) + old*norms[j]
			w[j] = softThreshold(tmp, l1Reg) / (norms[j] + l2Reg)
			if w[j] != old {
				axpy(old-w[j], c, resid)
			}
			dWMax = math.Max(dWMax, math.Abs(w[j]-old))
			wMax = math.Max(wMax, math.Abs(w[j]))
		}
		nIter = iter + 1
		if wMax == 0 || dWMax/wMax <= tol || iter == maxIter-1 {
			if dualityGap(cols, y, resid, w, l1Reg, l2Reg) <= gapTol {
				return w, nIter, true
			}
		}
	}
	return w, nIter, false
}

// dualityGap is sklearn 1.9's stopping quantity. With an L1 penalty it is the gap
// between the primal objective and the dual value at the residual rescaled to be dual
// feasible. With only an L2 penalty it uses the ridge dual instead, and with no
// penalty at all it falls back to the first-order condition ||X'R||^2.
func dualityGap(cols [][]float64, y, resid, w []float64, l1Reg, l2Reg float64) float64 {
	rNorm2 := dot(resid, resid)
	ry := dot(resid, y)

	if l1Reg == 0 {
		var xtr2 float64 // ||X'R||^2
		for _, c := range cols {
			g := dot(c, resid)
			xtr2 += g * g
		}
		if l2Reg == 0 {
			return xtr2
		}
		return rNorm2 + 0.5*l2Reg*dot(w, w) - ry + xtr2/(2*l2Reg)
	}

	var dualNorm float64 // max |X'R - l2Reg*w|
	for j, c := range cols {
		dualNorm = math.Max(dualNorm, math.Abs(dot(c, resid)-l2Reg*w[j]))
	}
	scale := 1.0
	if dualNorm > l1Reg {
		scale = l1Reg / dualNorm
	}
	var l1Norm float64
	for _, v := range w {
		l1Norm += math.Abs(v)
	}
	quad := rNorm2 + l2Reg*dot(w, w)
	primal := 0.5*quad + l1Reg*l1Norm
	dual := -0.5*scale*scale*quad + scale*ry
	return primal - dual
}

func softThreshold(x, t float64) float64 {
	switch {
	case x > t:
		return x - t
	case x < -t:
		return x + t
	}
	return 0
}

func dot(a, b []float64) float64 {
	var s float64
	for i, v := range a {
		s += v * b[i]
	}
	return s
}

// axpy sets y += a*x.
func axpy(a float64, x, y []float64) {
	for i, v := range x {
		y[i] += a * v
	}
}

// elasticNetGob is the versioned on-disk payload shared by ElasticNet and Lasso; Kind
// keeps one from being loaded as the other.
type elasticNetGob struct {
	Version      int
	Kind         string
	Coef         []float64
	Intercept    float64
	NFeatures    int
	Alpha        float64
	L1Ratio      float64
	FitIntercept bool
	MaxIter      int
	Tol          float64
	NIter        int
	Converged    bool
}

const (
	elasticNetFormatVersion = 1
	elasticNetKind          = "elastic_net"
	lassoKind               = "lasso"
)

func loadCD(op, path, kind string) (elasticNetGob, error) {
	var g elasticNetGob
	if err := loadGob(op, path, &g); err != nil {
		return g, err
	}
	if err := checkLoaded(op, g.Version, elasticNetFormatVersion, g.Kind, kind, g.NFeatures, g.Coef, g.Intercept); err != nil {
		return g, err
	}
	if err := checkCDParams(g.Alpha, g.L1Ratio, g.MaxIter, g.Tol); err != nil {
		return g, fmt.Errorf("%s: corrupt payload: %w", op, err)
	}
	return g, nil
}

// Save writes the fitted model to path in the versioned gob format.
func (e *ElasticNet) Save(path string) error {
	if !e.fitted {
		return matutil.ErrNotFitted
	}
	return saveGob("ElasticNet.Save", path, elasticNetGob{
		Version: elasticNetFormatVersion, Kind: elasticNetKind, Coef: e.Coef, Intercept: e.Intercept,
		NFeatures: e.nFeatures, Alpha: e.Alpha, L1Ratio: e.L1Ratio, FitIntercept: e.FitIntercept,
		MaxIter: e.MaxIter, Tol: e.Tol, NIter: e.nIter, Converged: e.converged,
	})
}

// LoadElasticNet reads a fitted model previously written by Save.
func LoadElasticNet(path string) (*ElasticNet, error) {
	g, err := loadCD("LoadElasticNet", path, elasticNetKind)
	if err != nil {
		return nil, err
	}
	return &ElasticNet{Alpha: g.Alpha, L1Ratio: g.L1Ratio, FitIntercept: g.FitIntercept, MaxIter: g.MaxIter,
		Tol: g.Tol, Coef: g.Coef, Intercept: g.Intercept, nFeatures: g.NFeatures, nIter: g.NIter,
		converged: g.Converged, fitted: true}, nil
}

// Save writes the fitted model to path in the versioned gob format.
func (l *Lasso) Save(path string) error {
	if !l.fitted {
		return matutil.ErrNotFitted
	}
	return saveGob("Lasso.Save", path, elasticNetGob{
		Version: elasticNetFormatVersion, Kind: lassoKind, Coef: l.Coef, Intercept: l.Intercept,
		NFeatures: l.nFeatures, Alpha: l.Alpha, L1Ratio: 1, FitIntercept: l.FitIntercept,
		MaxIter: l.MaxIter, Tol: l.Tol, NIter: l.nIter, Converged: l.converged,
	})
}

// LoadLasso reads a fitted model previously written by Save.
func LoadLasso(path string) (*Lasso, error) {
	g, err := loadCD("LoadLasso", path, lassoKind)
	if err != nil {
		return nil, err
	}
	return &Lasso{Alpha: g.Alpha, FitIntercept: g.FitIntercept, MaxIter: g.MaxIter, Tol: g.Tol, Coef: g.Coef,
		Intercept: g.Intercept, nFeatures: g.NFeatures, nIter: g.NIter, converged: g.Converged, fitted: true}, nil
}
