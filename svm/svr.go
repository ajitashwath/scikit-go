package svm

import (
	"fmt"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/metrics"
)

// Compile-time checks that SVR satisfies the core interfaces.
var (
	_ core.Estimator = (*SVR)(nil)
	_ core.Predictor = (*SVR)(nil)
	_ core.Saver     = (*SVR)(nil)
)

// SVR is an epsilon-support vector regressor, mirroring sklearn.svm.SVR. Errors
// smaller than Epsilon are not penalized. Kernel, Gamma, MaxIter and CacheSize
// behave as in SVC.
type SVR struct {
	C         float64
	Kernel    string
	Degree    int
	Gamma     float64
	Coef0     float64
	Epsilon   float64
	Tol       float64
	MaxIter   int
	CacheSize float64

	m      svrModel
	fitted bool
}

// svrModel is the fitted state: the support vectors with their dual
// coefficients alpha_i - alpha_i*, and the bias rho.
type svrModel struct {
	nFeatures int
	gamma     float64 // resolved
	sv        [][]float64
	svIdx     []int
	coef      []float64
	rho       float64
}

// NewSVR returns an unfitted regressor with sklearn's defaults
// (C=1, epsilon=0.1, rbf kernel, gamma="scale", tol=1e-3).
func NewSVR() *SVR {
	return &SVR{C: 1, Kernel: KernelRBF, Degree: 3, Epsilon: 0.1, Tol: 1e-3, MaxIter: -1, CacheSize: 200}
}

func (r *SVR) params() params {
	return params{c: r.C, kernel: r.Kernel, degree: r.Degree, gamma: r.Gamma, coef0: r.Coef0,
		tol: r.Tol, maxIter: r.MaxIter, cacheSize: r.CacheSize}
}

func (r *SVR) kernelFunc() kernelFunc {
	return kernelFunc{kind: r.Kernel, gamma: r.m.gamma, coef0: r.Coef0, degree: r.Degree}
}

// svrQ is the kernel matrix of the 2l-variable epsilon-SVR dual: variable i
// stands for sample i mod l with sign +1 (i < l) or -1.
type svrQ struct {
	X     [][]float64
	kf    kernelFunc
	cache *rowCache
	qd    []float64
	bufs  [2][]float64
	next  int
}

func newSVRQ(X [][]float64, kf kernelFunc, cacheMB float64) *svrQ {
	l := len(X)
	q := &svrQ{X: X, kf: kf, cache: newRowCache(l, cacheMB), qd: make([]float64, 2*l)}
	for i := range X {
		k := kf.eval(X[i], X[i])
		q.qd[i], q.qd[i+l] = k, k
	}
	q.bufs[0] = make([]float64, 2*l)
	q.bufs[1] = make([]float64, 2*l)
	return q
}

func (q *svrQ) diag() []float64 { return q.qd }

func (q *svrQ) row(i int) []float64 {
	l := len(q.X)
	real := i % l
	k := q.cache.get(real, func(dst []float64) {
		for j := range q.X {
			dst[j] = q.kf.eval(q.X[real], q.X[j])
		}
	})
	buf := q.bufs[q.next]
	q.next ^= 1
	for j := 0; j < l; j++ {
		buf[j] = k[j]
		buf[j+l] = -k[j]
	}
	if i >= l {
		for j := range buf {
			buf[j] = -buf[j]
		}
	}
	return buf
}

// Fit trains the regressor.
func (r *SVR) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("SVR.Fit: %w", err)
	}
	p := r.params()
	if err := p.validate(); err != nil {
		return fmt.Errorf("SVR.Fit: %w", err)
	}
	if !(r.Epsilon >= 0) {
		return fmt.Errorf("SVR.Fit: %w: epsilon must be >= 0, got %v", ErrInvalidSVM, r.Epsilon)
	}
	if _, err := newKernelFunc(r.Kernel, 1, r.Coef0, r.Degree); err != nil {
		return fmt.Errorf("SVR.Fit: %w", err)
	}

	l := len(X)
	gamma := resolveGamma(r.Gamma, X)
	kf := kernelFunc{kind: r.Kernel, gamma: gamma, coef0: r.Coef0, degree: r.Degree}

	linear := make([]float64, 2*l)
	sign := make([]float64, 2*l)
	for i := 0; i < l; i++ {
		linear[i] = r.Epsilon - y[i]
		sign[i] = 1
		linear[i+l] = r.Epsilon + y[i]
		sign[i+l] = -1
	}
	alpha, rho, _, err := solve(newSVRQ(X, kf, p.cacheSize), linear, sign, p.c, p.tol, p.maxIter)
	if err != nil {
		return fmt.Errorf("SVR.Fit: %w", err)
	}

	var sv [][]float64
	var svIdx []int
	var coef []float64
	for i := 0; i < l; i++ {
		if c := alpha[i] - alpha[i+l]; c != 0 {
			sv = append(sv, X[i])
			svIdx = append(svIdx, i)
			coef = append(coef, c)
		}
	}
	r.m = svrModel{nFeatures: len(X[0]), gamma: gamma, sv: sv, svIdx: svIdx, coef: coef, rho: rho}
	r.fitted = true
	return nil
}

// Predict returns the regression value of each row of X.
func (r *SVR) Predict(X [][]float64) ([]float64, error) {
	if !r.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, r.m.nFeatures); err != nil {
		return nil, fmt.Errorf("SVR.Predict: %w", err)
	}
	kf := r.kernelFunc()
	out := make([]float64, len(X))
	for i, x := range X {
		var s float64
		for k, v := range r.m.sv {
			s += r.m.coef[k] * kf.eval(v, x)
		}
		out[i] = s - r.m.rho
	}
	return out, nil
}

// Score returns the R^2 score of the predictions on X against y.
func (r *SVR) Score(X [][]float64, y []float64) (float64, error) {
	preds, err := r.Predict(X)
	if err != nil {
		return 0, err
	}
	score, err := metrics.R2Score(y, preds)
	if err != nil {
		return 0, fmt.Errorf("SVR.Score: %w", err)
	}
	return score, nil
}

// SupportVectors returns the support vectors.
func (r *SVR) SupportVectors() [][]float64 {
	if !r.fitted {
		return nil
	}
	return copyMatrix(r.m.sv)
}

// Support returns the training-set indices of the support vectors.
func (r *SVR) Support() []int {
	if !r.fitted {
		return nil
	}
	return append([]int(nil), r.m.svIdx...)
}

// DualCoef returns sklearn's dual_coef_: alpha_i - alpha_i* per support vector.
func (r *SVR) DualCoef() []float64 {
	if !r.fitted {
		return nil
	}
	return append([]float64(nil), r.m.coef...)
}

// Intercept returns sklearn's intercept_, the negated libsvm bias.
func (r *SVR) Intercept() float64 {
	if !r.fitted {
		return 0
	}
	return -r.m.rho
}

// GammaValue returns the gamma used by the fitted kernel, after resolving
// Gamma == 0 to sklearn's "scale" value.
func (r *SVR) GammaValue() float64 {
	if !r.fitted {
		return 0
	}
	return r.m.gamma
}
