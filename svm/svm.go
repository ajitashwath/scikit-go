// Package svm implements support vector machines mirroring sklearn.svm: SVC
// (C-support vector classification, one-vs-one for multiple classes, with
// optional Platt-scaled probabilities) and SVR (epsilon-support vector
// regression). Both are trained with a port of libsvm's SMO solver, so results
// agree with sklearn's up to the solver tolerance.
package svm

import (
	"errors"
	"fmt"
	"math"
)

// ErrInvalidSVM is returned when SVC or SVR hyperparameters fail validation.
var ErrInvalidSVM = errors.New("invalid SVM hyperparameters")

// params carries the hyperparameters shared by SVC and SVR.
type params struct {
	c         float64
	kernel    string
	degree    int
	gamma     float64
	coef0     float64
	tol       float64
	maxIter   int
	cacheSize float64
}

func (p params) validate() error {
	switch {
	case !(p.c > 0):
		return fmt.Errorf("%w: C must be > 0, got %v", ErrInvalidSVM, p.c)
	case !(p.tol > 0):
		return fmt.Errorf("%w: tol must be > 0, got %v", ErrInvalidSVM, p.tol)
	case p.gamma < 0 || math.IsNaN(p.gamma):
		return fmt.Errorf("%w: gamma must be >= 0 (0 means \"scale\"), got %v", ErrInvalidSVM, p.gamma)
	case p.kernel == KernelPoly && p.degree < 1:
		return fmt.Errorf("%w: degree must be >= 1, got %d", ErrInvalidSVM, p.degree)
	case p.maxIter == 0:
		return fmt.Errorf("%w: max_iter must be positive or negative for no limit, got 0", ErrInvalidSVM)
	case !(p.cacheSize > 0):
		return fmt.Errorf("%w: cache_size must be > 0, got %v", ErrInvalidSVM, p.cacheSize)
	}
	return nil
}

// resolveGamma returns the effective gamma: the configured value, or sklearn's
// "scale" heuristic 1 / (n_features * X.var()) when gamma is 0.
func resolveGamma(gamma float64, X [][]float64) float64 {
	if gamma > 0 {
		return gamma
	}
	n, p := len(X), len(X[0])
	var mean float64
	for _, row := range X {
		for _, v := range row {
			mean += v
		}
	}
	mean /= float64(n * p)
	var ss float64
	for _, row := range X {
		for _, v := range row {
			d := v - mean
			ss += d * d
		}
	}
	variance := ss / float64(n*p)
	if variance == 0 {
		return 1
	}
	return 1 / (float64(p) * variance)
}

// binaryModel is the solution of one two-class dual problem: coef[i] = alpha_i * y_i
// over the sub-problem's samples, and the bias rho, so that the decision value
// is sum_i coef[i] K(x_i, x) - rho, positive for the +1 class.
type binaryModel struct {
	coef []float64
	rho  float64
}

// svcQ is the signed kernel matrix Q_ij = y_i y_j K(x_i, x_j) of a C-SVC problem.
type svcQ struct {
	X     [][]float64
	y     []float64
	kf    kernelFunc
	cache *rowCache
	qd    []float64
	bufs  [2][]float64
	next  int
}

func newSVCQ(X [][]float64, y []float64, kf kernelFunc, cacheMB float64) *svcQ {
	n := len(X)
	q := &svcQ{X: X, y: y, kf: kf, cache: newRowCache(n, cacheMB), qd: make([]float64, n)}
	for i := range X {
		q.qd[i] = kf.eval(X[i], X[i])
	}
	q.bufs[0] = make([]float64, n)
	q.bufs[1] = make([]float64, n)
	return q
}

func (q *svcQ) diag() []float64 { return q.qd }

func (q *svcQ) row(i int) []float64 {
	k := q.cache.get(i, func(dst []float64) {
		for j := range q.X {
			dst[j] = q.kf.eval(q.X[i], q.X[j])
		}
	})
	buf := q.bufs[q.next]
	q.next ^= 1
	yi := q.y[i]
	for j := range buf {
		buf[j] = yi * q.y[j] * k[j]
	}
	return buf
}

// trainBinary solves the C-SVC dual for samples X with labels y in {-1, +1}.
func trainBinary(X [][]float64, y []float64, kf kernelFunc, p params) (binaryModel, error) {
	n := len(X)
	minusOnes := make([]float64, n)
	for i := range minusOnes {
		minusOnes[i] = -1
	}
	alpha, rho, _, err := solve(newSVCQ(X, y, kf, p.cacheSize), minusOnes, y, p.c, p.tol, p.maxIter)
	if err != nil {
		return binaryModel{}, err
	}
	coef := make([]float64, n)
	for i, a := range alpha {
		coef[i] = a * y[i]
	}
	return binaryModel{coef: coef, rho: rho}, nil
}

// decision evaluates the binary decision value at x.
func (m binaryModel) decision(X [][]float64, x []float64, kf kernelFunc) float64 {
	var s float64
	for i, c := range m.coef {
		if c != 0 {
			s += c * kf.eval(X[i], x)
		}
	}
	return s - m.rho
}
