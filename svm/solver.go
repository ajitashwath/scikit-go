package svm

import (
	"errors"
	"math"
)

// tau replaces non-positive curvature in the working-set and update formulas.
const tau = 1e-12

// ErrNumerical is returned when the kernel values or the solver's gradient stop
// being finite, or when the solver can no longer move (the curvature is so large
// that every step rounds to zero). Both happen when feature magnitudes are
// extreme; scaling the features (for example with StandardScaler) avoids it.
var ErrNumerical = errors.New("svm: numerical failure in the kernel matrix (overflow or badly scaled features); scale the features")

// ErrNotConverged is returned when the solver reaches its safety limit without
// meeting the tolerance. Only the automatic limit (MaxIter unset) raises it; an
// explicit positive MaxIter stops the solver and uses the model as it stands. The
// usual cause is badly scaled features, which make the dual problem so
// ill-conditioned that SMO crawls; StandardScaler normally fixes it.
var ErrNotConverged = errors.New("svm: solver did not converge; scale the features or set MaxIter")

// backstopIterations is the iteration limit applied when MaxIter is not set. Well
// scaled problems converge in a small multiple of n iterations, so this is far
// above what they need while still guaranteeing that no input spins for long.
func backstopIterations(n int) int {
	if limit := 1000 * n; limit > 1_000_000 {
		return limit
	}
	return 1_000_000
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// qMatrix supplies the rows of the dual problem's signed kernel matrix Q.
type qMatrix interface {
	// row returns Q[i][0..n). The slice may be an internal buffer that is
	// overwritten two calls later; the solver never holds more than two rows.
	row(i int) []float64
	// diag returns Q[i][i] for every i.
	diag() []float64
}

// solve minimizes 0.5 a'Qa + p'a subject to y'a = 0 and 0 <= a_i <= c, using
// libsvm's SMO with second-order working-set selection (WSS3). It stops when
// the maximal KKT violation drops below eps or after maxIter iterations
// (maxIter <= 0 applies a safety cap and reports ErrNotConverged if it is hit),
// and returns the multipliers and the bias rho. It returns ErrNumerical if the
// problem stops being finite or the solver can no longer move.
//
// This follows libsvm's Solver::Solve without shrinking; shrinking only skips
// work on variables already at their bounds and does not change the optimum.
func solve(q qMatrix, p, y []float64, c, eps float64, maxIter int) (alpha []float64, rho float64, iters int, err error) {
	n := len(p)
	alpha = make([]float64, n)
	g := append([]float64(nil), p...) // gradient; alpha starts at 0 so G = p
	qd := q.diag()
	for i := range qd {
		if !finite(qd[i]) || !finite(p[i]) {
			return nil, 0, 0, ErrNumerical
		}
	}
	automatic := maxIter <= 0
	if automatic {
		maxIter = backstopIterations(n)
	}

	isUpper := func(i int) bool { return alpha[i] >= c }
	isLower := func(i int) bool { return alpha[i] <= 0 }

	// selectWorkingSet picks the maximal violating pair (i, j); ok is false at optimality.
	selectWorkingSet := func() (out1, out2 int, ok, numeric bool) {
		gmax, gmax2 := math.Inf(-1), math.Inf(-1)
		gmaxIdx, gminIdx := -1, -1
		objDiffMin := math.Inf(1)

		for t := 0; t < n; t++ {
			if y[t] == 1 {
				if !isUpper(t) && -g[t] >= gmax {
					gmax = -g[t]
					gmaxIdx = t
				}
			} else {
				if !isLower(t) && g[t] >= gmax {
					gmax = g[t]
					gmaxIdx = t
				}
			}
		}
		i := gmaxIdx
		var qi []float64
		if i != -1 {
			qi = q.row(i)
		}
		for j := 0; j < n; j++ {
			if y[j] == 1 {
				if isLower(j) {
					continue
				}
				gradDiff := gmax + g[j]
				if g[j] >= gmax2 {
					gmax2 = g[j]
				}
				if gradDiff > 0 {
					quad := qd[i] + qd[j] - 2.0*y[i]*qi[j]
					var objDiff float64
					if quad > 0 {
						objDiff = -(gradDiff * gradDiff) / quad
					} else {
						objDiff = -(gradDiff * gradDiff) / tau
					}
					if objDiff <= objDiffMin {
						gminIdx = j
						objDiffMin = objDiff
					}
				}
			} else {
				if isUpper(j) {
					continue
				}
				gradDiff := gmax - g[j]
				if -g[j] >= gmax2 {
					gmax2 = -g[j]
				}
				if gradDiff > 0 {
					quad := qd[i] + qd[j] + 2.0*y[i]*qi[j]
					var objDiff float64
					if quad > 0 {
						objDiff = -(gradDiff * gradDiff) / quad
					} else {
						objDiff = -(gradDiff * gradDiff) / tau
					}
					if objDiff <= objDiffMin {
						gminIdx = j
						objDiffMin = objDiff
					}
				}
			}
		}
		if gminIdx != -1 && !finite(gmax+gmax2) {
			return 0, 0, false, true
		}
		if gmax+gmax2 < eps || gminIdx == -1 {
			return 0, 0, false, false
		}
		return gmaxIdx, gminIdx, true, false
	}

	for iters < maxIter {
		i, j, ok, numeric := selectWorkingSet()
		if numeric {
			return nil, 0, iters, ErrNumerical
		}
		if !ok {
			break
		}
		iters++

		qi := q.row(i)
		qj := q.row(j)
		oldAi, oldAj := alpha[i], alpha[j]

		if y[i] != y[j] {
			quad := qd[i] + qd[j] + 2*qi[j]
			if quad <= 0 {
				quad = tau
			}
			delta := (-g[i] - g[j]) / quad
			diff := alpha[i] - alpha[j]
			alpha[i] += delta
			alpha[j] += delta
			if diff > 0 {
				if alpha[j] < 0 {
					alpha[j] = 0
					alpha[i] = diff
				}
			} else if alpha[i] < 0 {
				alpha[i] = 0
				alpha[j] = -diff
			}
			if diff > 0 { // C_i - C_j == 0 with a single C
				if alpha[i] > c {
					alpha[i] = c
					alpha[j] = c - diff
				}
			} else if alpha[j] > c {
				alpha[j] = c
				alpha[i] = c + diff
			}
		} else {
			quad := qd[i] + qd[j] - 2*qi[j]
			if quad <= 0 {
				quad = tau
			}
			delta := (g[i] - g[j]) / quad
			sum := alpha[i] + alpha[j]
			alpha[i] -= delta
			alpha[j] += delta
			if sum > c {
				if alpha[i] > c {
					alpha[i] = c
					alpha[j] = sum - c
				}
			} else if alpha[j] < 0 {
				alpha[j] = 0
				alpha[i] = sum
			}
			if sum > c {
				if alpha[j] > c {
					alpha[j] = c
					alpha[i] = sum - c
				}
			} else if alpha[i] < 0 {
				alpha[i] = 0
				alpha[j] = sum
			}
		}

		if !finite(alpha[i]) || !finite(alpha[j]) {
			return nil, 0, iters, ErrNumerical
		}
		dAi, dAj := alpha[i]-oldAi, alpha[j]-oldAj
		if dAi == 0 && dAj == 0 {
			// The chosen pair violates the KKT conditions by more than eps yet
			// cannot move. Nothing changes, so the same pair would be chosen
			// again forever; this is a precision failure, not convergence.
			return nil, 0, iters, ErrNumerical
		}
		for k := 0; k < n; k++ {
			g[k] += qi[k]*dAi + qj[k]*dAj
		}
	}

	if automatic && iters >= maxIter {
		if _, _, more, _ := selectWorkingSet(); more {
			return nil, 0, iters, ErrNotConverged
		}
	}
	return alpha, calculateRho(alpha, g, y, c), iters, nil
}

// calculateRho mirrors libsvm's Solver::calculate_rho: the mean y*G over free
// variables, or the midpoint of the bound-derived interval when none is free.
func calculateRho(alpha, g, y []float64, c float64) float64 {
	nrFree := 0
	ub, lb := math.Inf(1), math.Inf(-1)
	var sumFree float64
	for i := range alpha {
		yG := y[i] * g[i]
		switch {
		case alpha[i] >= c:
			if y[i] == -1 {
				ub = math.Min(ub, yG)
			} else {
				lb = math.Max(lb, yG)
			}
		case alpha[i] <= 0:
			if y[i] == 1 {
				ub = math.Min(ub, yG)
			} else {
				lb = math.Max(lb, yG)
			}
		default:
			nrFree++
			sumFree += yG
		}
	}
	if nrFree > 0 {
		return sumFree / float64(nrFree)
	}
	return (ub + lb) / 2
}
