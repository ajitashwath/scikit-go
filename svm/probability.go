package svm

import (
	"math"
	"math/rand"
)

// Platt scaling for SVC.PredictProba: a sigmoid P(y=1|f) = 1/(1+exp(A f + B)) is
// fitted to cross-validated decision values of every one-vs-one pair, and the
// pairwise probabilities are coupled into class probabilities. The routines
// follow libsvm (sigmoid_train, sigmoid_predict, multiclass_probability); the
// cross-validation folds use a seeded shuffle rather than libsvm's C rand(),
// so probabilities agree with sklearn's only approximately.

const probFolds = 5

// sigmoidTrain fits A and B by regularized Newton iterations with backtracking
// line search, as libsvm's sigmoid_train does.
func sigmoidTrain(dec, labels []float64) (a, b float64) {
	var prior1, prior0 float64
	for _, l := range labels {
		if l > 0 {
			prior1++
		} else {
			prior0++
		}
	}
	const (
		maxIter = 100
		minStep = 1e-10
		sigma   = 1e-12
		eps     = 1e-5
	)
	hiTarget := (prior1 + 1) / (prior1 + 2)
	loTarget := 1 / (prior0 + 2)
	t := make([]float64, len(dec))
	for i, l := range labels {
		if l > 0 {
			t[i] = hiTarget
		} else {
			t[i] = loTarget
		}
	}

	a, b = 0, math.Log((prior0+1)/(prior1+1))
	objective := func(a, b float64) float64 {
		var f float64
		for i := range dec {
			fApB := dec[i]*a + b
			if fApB >= 0 {
				f += t[i]*fApB + math.Log1p(math.Exp(-fApB))
			} else {
				f += (t[i]-1)*fApB + math.Log1p(math.Exp(fApB))
			}
		}
		return f
	}
	fval := objective(a, b)

	for iter := 0; iter < maxIter; iter++ {
		h11, h22, h21, g1, g2 := sigma, sigma, 0.0, 0.0, 0.0
		for i := range dec {
			fApB := dec[i]*a + b
			var p, q float64
			if fApB >= 0 {
				p = math.Exp(-fApB) / (1 + math.Exp(-fApB))
				q = 1 / (1 + math.Exp(-fApB))
			} else {
				p = 1 / (1 + math.Exp(fApB))
				q = math.Exp(fApB) / (1 + math.Exp(fApB))
			}
			d2 := p * q
			h11 += dec[i] * dec[i] * d2
			h22 += d2
			h21 += dec[i] * d2
			d1 := t[i] - p
			g1 += dec[i] * d1
			g2 += d1
		}
		if math.Abs(g1) < eps && math.Abs(g2) < eps {
			break
		}
		det := h11*h22 - h21*h21
		dA := -(h22*g1 - h21*g2) / det
		dB := -(-h21*g1 + h11*g2) / det
		gd := g1*dA + g2*dB

		step := 1.0
		for step >= minStep {
			newA, newB := a+step*dA, b+step*dB
			newf := objective(newA, newB)
			if newf < fval+0.0001*step*gd {
				a, b, fval = newA, newB, newf
				break
			}
			step /= 2
		}
		if step < minStep {
			break
		}
	}
	return a, b
}

func sigmoidPredict(dec, a, b float64) float64 {
	fApB := dec*a + b
	if fApB >= 0 {
		return math.Exp(-fApB) / (1 + math.Exp(-fApB))
	}
	return 1 / (1 + math.Exp(fApB))
}

// multiclassProbability couples pairwise probabilities r[i][j] = P(class i | i or j)
// into class probabilities using libsvm's second method (Wu, Lin & Weng).
func multiclassProbability(k int, r [][]float64) []float64 {
	maxIter := 100
	if k > maxIter {
		maxIter = k
	}
	eps := 0.005 / float64(k)

	q := make([][]float64, k)
	for t := range q {
		q[t] = make([]float64, k)
	}
	p := make([]float64, k)
	qp := make([]float64, k)
	for t := 0; t < k; t++ {
		p[t] = 1 / float64(k)
		for j := 0; j < t; j++ {
			q[t][t] += r[j][t] * r[j][t]
			q[t][j] = q[j][t]
		}
		for j := t + 1; j < k; j++ {
			q[t][t] += r[j][t] * r[j][t]
			q[t][j] = -r[j][t] * r[t][j]
		}
	}
	for iter := 0; iter < maxIter; iter++ {
		var pQp float64
		for t := 0; t < k; t++ {
			qp[t] = 0
			for j := 0; j < k; j++ {
				qp[t] += q[t][j] * p[j]
			}
			pQp += p[t] * qp[t]
		}
		var maxError float64
		for t := 0; t < k; t++ {
			maxError = math.Max(maxError, math.Abs(qp[t]-pQp))
		}
		if maxError < eps {
			break
		}
		for t := 0; t < k; t++ {
			diff := (-qp[t] + pQp) / q[t][t]
			p[t] += diff
			pQp = (pQp + diff*(diff*q[t][t]+2*qp[t])) / (1 + diff) / (1 + diff)
			for j := 0; j < k; j++ {
				qp[j] = (qp[j] + diff*q[t][j]) / (1 + diff)
				p[j] /= 1 + diff
			}
		}
	}
	return p
}

// fitSigmoid runs libsvm's svm_binary_svc_probability for one pair: it obtains
// held-out decision values by 5-fold cross-validation of the pair's samples and
// fits the sigmoid to them.
func fitSigmoid(X [][]float64, y []float64, kf kernelFunc, p params, rng *rand.Rand) (a, b float64, err error) {
	n := len(X)
	perm := rng.Perm(n)
	dec := make([]float64, n)

	for f := 0; f < probFolds; f++ {
		begin, end := f*n/probFolds, (f+1)*n/probFolds
		var subX [][]float64
		var subY []float64
		for j := 0; j < n; j++ {
			if j < begin || j >= end {
				subX = append(subX, X[perm[j]])
				subY = append(subY, y[perm[j]])
			}
		}
		var pos, neg int
		for _, l := range subY {
			if l > 0 {
				pos++
			} else {
				neg++
			}
		}
		switch {
		case pos == 0 && neg == 0:
			for j := begin; j < end; j++ {
				dec[perm[j]] = 0
			}
		case pos > 0 && neg == 0:
			for j := begin; j < end; j++ {
				dec[perm[j]] = 1
			}
		case pos == 0 && neg > 0:
			for j := begin; j < end; j++ {
				dec[perm[j]] = -1
			}
		default:
			m, err := trainBinary(subX, subY, kf, p)
			if err != nil {
				return 0, 0, err
			}
			for j := begin; j < end; j++ {
				dec[perm[j]] = m.decision(subX, X[perm[j]], kf)
			}
		}
	}
	a, b = sigmoidTrain(dec, y)
	return a, b, nil
}
