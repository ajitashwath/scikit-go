package linear

import "math"

// lbfgs minimizes a smooth function with limited-memory BFGS and a strong-Wolfe
// line search (Nocedal & Wright, algorithms 7.4, 3.5 and 3.6). It is deliberately
// small: it exists so LogisticRegression does not need an optimization dependency.
//
// fg must return the value at x and write the gradient there into g. The search stops
// when the largest gradient component is below gtol (converged), after maxIter
// iterations, or when no further progress can be made; in every case x holds the best
// point found.
func lbfgs(fg func(x, g []float64) float64, x0 []float64, gtol float64, maxIter int) (x []float64, iters int, converged bool) {
	const memory = 10
	n := len(x0)
	x = append([]float64(nil), x0...)
	g := make([]float64, n)
	f := fg(x, g)
	if math.IsNaN(f) || math.IsInf(f, 0) || !allFinite(g) {
		return x, 0, false
	}
	if maxAbs(g) < gtol {
		return x, 0, true
	}

	var sHist, yHist [][]float64
	var rho []float64
	d := make([]float64, n)
	alpha := make([]float64, memory)

	for iters = 0; iters < maxIter; {
		// Search direction d = -H g from the two-loop recursion.
		for i := range d {
			d[i] = -g[i]
		}
		k := len(sHist)
		for i := k - 1; i >= 0; i-- {
			alpha[i] = rho[i] * dot(sHist[i], d)
			axpy(-alpha[i], yHist[i], d)
		}
		if k > 0 {
			scale := dot(sHist[k-1], yHist[k-1]) / dot(yHist[k-1], yHist[k-1])
			for i := range d {
				d[i] *= scale
			}
		}
		for i := 0; i < k; i++ {
			beta := rho[i] * dot(yHist[i], d)
			axpy(alpha[i]-beta, sHist[i], d)
		}

		dphi0 := dot(g, d)
		if !(dphi0 < 0) { // not a descent direction (round-off): restart from steepest descent
			for i := range d {
				d[i] = -g[i]
			}
			dphi0 = -dot(g, g)
			sHist, yHist, rho = nil, nil, nil
		}
		step := 1.0
		if len(sHist) == 0 {
			step = math.Min(1, 1/math.Sqrt(dot(g, g)))
		}

		pt, ok := wolfeSearch(fg, x, d, f, dphi0, step)
		if !ok {
			if len(sHist) > 0 { // forget the curvature model and try steepest descent once more
				sHist, yHist, rho = nil, nil, nil
				continue
			}
			return x, iters, false
		}
		iters++

		s := make([]float64, n)
		yv := make([]float64, n)
		for i := range s {
			s[i] = pt.a * d[i]
			yv[i] = pt.g[i] - g[i]
		}
		if sy := dot(s, yv); sy > 1e-12*math.Sqrt(dot(s, s)*dot(yv, yv)) {
			if len(sHist) == memory {
				sHist, yHist, rho = sHist[1:], yHist[1:], rho[1:]
			}
			sHist, yHist, rho = append(sHist, s), append(yHist, yv), append(rho, 1/sy)
		}
		for i := range x {
			x[i] += s[i]
		}
		f = pt.f
		copy(g, pt.g)
		if maxAbs(g) < gtol {
			return x, iters, true
		}
	}
	return x, iters, false
}

// lsPoint is a point on the line x + a*d.
type lsPoint struct {
	a, f, dphi float64
	g          []float64
}

// wolfeSearch finds a step a along d satisfying the strong Wolfe conditions.
// f0 and dphi0 are the value and slope at a = 0. The sufficient-decrease test has a
// tiny slack so that round-off in f near the optimum cannot make it reject steps that
// are in fact fine; the curvature test (which uses gradients) does the real work there.
func wolfeSearch(fg func(x, g []float64) float64, x, d []float64, f0, dphi0, a1 float64) (lsPoint, bool) {
	const (
		c1, c2   = 1e-4, 0.9
		maxEvals = 30
		maxStep  = 1e10
	)
	slack := 1e-14 * math.Max(1, math.Abs(f0))
	evals := 0
	xt := make([]float64, len(x))
	eval := func(a float64) lsPoint {
		evals++
		for i := range xt {
			xt[i] = x[i] + a*d[i]
		}
		g := make([]float64, len(x))
		f := fg(xt, g)
		if math.IsNaN(f) || !allFinite(g) {
			f = math.Inf(1)
		}
		return lsPoint{a: a, f: f, dphi: dot(g, d), g: g}
	}
	tooHigh := func(p lsPoint) bool { return p.f > f0+c1*p.a*dphi0+slack }

	zoom := func(lo, hi lsPoint) (lsPoint, bool) {
		for evals < maxEvals {
			p := eval((lo.a + hi.a) / 2)
			switch {
			case tooHigh(p) || p.f > lo.f+slack:
				hi = p
			default:
				if math.Abs(p.dphi) <= -c2*dphi0 {
					return p, true
				}
				if p.dphi*(hi.a-lo.a) >= 0 {
					hi = lo
				}
				lo = p
			}
			if math.Abs(hi.a-lo.a) < 1e-16*math.Max(1, lo.a) {
				break
			}
		}
		if lo.a > 0 && lo.f <= f0+slack { // the best sufficient-decrease point seen
			return lo, true
		}
		return lsPoint{}, false
	}

	prev := lsPoint{a: 0, f: f0, dphi: dphi0}
	a := a1
	for i := 0; evals < maxEvals; i++ {
		p := eval(a)
		if tooHigh(p) || (i > 0 && p.f >= prev.f+slack) {
			return zoom(prev, p)
		}
		if math.Abs(p.dphi) <= -c2*dphi0 {
			return p, true
		}
		if p.dphi >= 0 {
			return zoom(p, prev)
		}
		prev = p
		a = math.Min(2*a, maxStep)
	}
	return lsPoint{}, false
}

func maxAbs(v []float64) float64 {
	var m float64
	for _, x := range v {
		m = math.Max(m, math.Abs(x))
	}
	return m
}
