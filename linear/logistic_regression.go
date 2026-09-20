package linear

import (
	"fmt"
	"math"
	"sort"

	"github.com/ajitashwath/scikit-go/internal/matutil"
)

// LogisticRegression is L2-regularized logistic regression for two or more
// classes, mirroring sklearn.linear_model.LogisticRegression with its default
// lbfgs solver. It minimizes
//
//	C * sum_i logloss_i + 1/2 * ||coef||^2
//
// The intercepts are not penalized. With two classes there is one coefficient row and
// the model is the usual sigmoid; with more it is multinomial (softmax) with one row per
// class. C is the inverse regularization strength, so smaller C means stronger
// regularization; C = +Inf removes the penalty (which does not converge if the classes
// are linearly separable, and then stops after MaxIter).
//
// The objective is strictly convex, so it has a single optimum that any solver reaches.
// This port uses its own small L-BFGS (see lbfgs.go) rather than scipy's, so it agrees with sklearn to the
// solver tolerance and not bit for bit, and its defaults are tighter than sklearn's
// (Tol 1e-6 rather than 1e-4, MaxIter 1000 rather than 100). Like sklearn, Fit does not
// fail when MaxIter is reached; check Converged.
//
// Tol is a bound on the largest component of the gradient of the mean objective
// (1/n)*sum(logloss_i) + 1/(2*C*n)*||coef||^2, as in sklearn.
type LogisticRegression struct {
	C            float64
	FitIntercept bool
	MaxIter      int
	Tol          float64

	classes   []float64
	coef      [][]float64 // 1 row for two classes, else one per class
	intercept []float64
	nFeatures int
	nIter     int
	converged bool
	fitted    bool
}

// NewLogisticRegression returns an unfitted LogisticRegression with C 1, an
// intercept, MaxIter 1000 and Tol 1e-6.
func NewLogisticRegression() *LogisticRegression {
	return &LogisticRegression{C: 1, FitIntercept: true, MaxIter: 1000, Tol: 1e-6}
}

func (m *LogisticRegression) checkParams() error {
	switch {
	case !(m.C > 0):
		return fmt.Errorf("C must be > 0, got %v", m.C)
	case m.MaxIter < 1:
		return fmt.Errorf("maxIter must be >= 1, got %d", m.MaxIter)
	case !(m.Tol > 0) || math.IsInf(m.Tol, 0):
		return fmt.Errorf("tol must be finite and > 0, got %v", m.Tol)
	}
	return nil
}

// Fit finds the coefficients and intercepts. y holds the class labels, which can be
// any distinct float64 values; they are reported back sorted by Classes.
func (m *LogisticRegression) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("LogisticRegression.Fit: %w", err)
	}
	if err := m.checkParams(); err != nil {
		return fmt.Errorf("LogisticRegression.Fit: %w", err)
	}
	classes, yIdx := encodeLabels(y)
	if len(classes) < 2 {
		return fmt.Errorf("LogisticRegression.Fit: need at least 2 classes, y has %d", len(classes))
	}

	prob := newLogisticProblem(X, yIdx, len(classes), m.C, m.FitIntercept)
	theta, iters, converged := lbfgs(prob.eval, make([]float64, prob.dim()), m.Tol, m.MaxIter)
	// Hitting MaxIter (converged == false) still leaves the best point found, like sklearn.
	if !allFinite(theta) {
		return fmt.Errorf("LogisticRegression.Fit: the solution is not finite; the data may contain extreme magnitudes")
	}

	m.classes = classes
	m.coef, m.intercept = prob.unpack(theta)
	m.nFeatures = len(X[0])
	m.nIter, m.converged = iters, converged
	m.fitted = true
	return nil
}

// DecisionFunction returns the confidence scores w.x + b: one column for two classes
// (positive means the larger class), otherwise one column per class.
func (m *LogisticRegression) DecisionFunction(X [][]float64) ([][]float64, error) {
	if !m.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, m.nFeatures); err != nil {
		return nil, fmt.Errorf("LogisticRegression.DecisionFunction: %w", err)
	}
	out := make([][]float64, len(X))
	for i, row := range X {
		out[i] = m.logits(row)
	}
	return out, nil
}

// PredictProba returns the probability of every class, in Classes order.
func (m *LogisticRegression) PredictProba(X [][]float64) ([][]float64, error) {
	z, err := m.DecisionFunction(X)
	if err != nil {
		return nil, err
	}
	out := make([][]float64, len(z))
	for i, zi := range z {
		if len(zi) == 1 {
			p := sigmoid(zi[0])
			out[i] = []float64{1 - p, p}
		} else {
			out[i] = softmax(zi)
		}
	}
	return out, nil
}

// Predict returns the most probable class label for each row of X.
func (m *LogisticRegression) Predict(X [][]float64) ([]float64, error) {
	z, err := m.DecisionFunction(X)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(z))
	for i, zi := range z {
		if len(zi) == 1 {
			out[i] = m.classes[boolToInt(zi[0] > 0)]
			continue
		}
		best := 0
		for k, v := range zi {
			if v > zi[best] {
				best = k
			}
		}
		out[i] = m.classes[best]
	}
	return out, nil
}

// Score returns the mean accuracy on X, y.
func (m *LogisticRegression) Score(X [][]float64, y []float64) (float64, error) {
	pred, err := m.Predict(X)
	if err != nil {
		return 0, err
	}
	if len(pred) != len(y) {
		return 0, fmt.Errorf("LogisticRegression.Score: %w: X has %d samples, y has %d", matutil.ErrDimMismatch, len(pred), len(y))
	}
	correct := 0
	for i := range y {
		if pred[i] == y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(y)), nil
}

// Classes returns the sorted class labels seen by Fit (nil before Fit).
func (m *LogisticRegression) Classes() []float64 { return append([]float64(nil), m.classes...) }

// Coef returns the coefficients as a matrix with one row of NFeatures values for two
// classes and one row per class otherwise, as sklearn's coef_ (nil before Fit).
func (m *LogisticRegression) Coef() [][]float64 {
	if !m.fitted {
		return nil
	}
	out := make([][]float64, len(m.coef))
	for i, row := range m.coef {
		out[i] = append([]float64(nil), row...)
	}
	return out
}

// Intercept returns one intercept per coefficient row (zeros if FitIntercept is false).
func (m *LogisticRegression) Intercept() []float64 {
	if !m.fitted {
		return nil
	}
	return append([]float64(nil), m.intercept...)
}

// NIter is the number of L-BFGS iterations Fit used (0 before Fit).
func (m *LogisticRegression) NIter() int { return m.nIter }

// Converged reports whether Fit reached Tol before running out of iterations.
func (m *LogisticRegression) Converged() bool { return m.converged }

func (m *LogisticRegression) logits(x []float64) []float64 {
	z := make([]float64, len(m.coef))
	for k, row := range m.coef {
		s := m.intercept[k]
		for j, v := range x {
			s += v * row[j]
		}
		z[k] = s
	}
	return z
}

// encodeLabels returns the sorted distinct labels and each sample's index into them.
func encodeLabels(y []float64) (classes []float64, idx []int) {
	classes = append([]float64(nil), y...)
	sort.Float64s(classes)
	uniq := classes[:0]
	for i, c := range classes {
		if i == 0 || c != classes[i-1] {
			uniq = append(uniq, c)
		}
	}
	classes = uniq
	idx = make([]int, len(y))
	for i, v := range y {
		idx[i] = sort.SearchFloat64s(classes, v)
	}
	return classes, idx
}

// logisticProblem is the regularized negative log-likelihood and its gradient. The
// parameters are laid out per output row as [weights (p), intercept (1 if fitted)].
type logisticProblem struct {
	X      [][]float64
	yIdx   []int
	n, p   int
	nOut   int // 1 for two classes, else the number of classes
	stride int // parameters per output row
	lambda float64
	fitInt bool
}

func newLogisticProblem(X [][]float64, yIdx []int, nClasses int, c float64, fitIntercept bool) *logisticProblem {
	pr := &logisticProblem{X: X, yIdx: yIdx, n: len(X), p: len(X[0]), nOut: nClasses, fitInt: fitIntercept}
	if nClasses == 2 {
		pr.nOut = 1
	}
	pr.stride = pr.p
	if fitIntercept {
		pr.stride++
	}
	if !math.IsInf(c, 1) {
		pr.lambda = 1 / (c * float64(pr.n))
	}
	return pr
}

func (pr *logisticProblem) dim() int { return pr.nOut * pr.stride }

// eval returns the objective at x and writes its gradient into g.
func (pr *logisticProblem) eval(x, g []float64) float64 {
	for i := range g {
		g[i] = 0
	}
	invN := 1 / float64(pr.n)
	var loss float64
	z := make([]float64, pr.nOut)
	for i, row := range pr.X {
		for o := 0; o < pr.nOut; o++ {
			w := x[o*pr.stride : o*pr.stride+pr.p]
			s := dot(w, row)
			if pr.fitInt {
				s += x[o*pr.stride+pr.p]
			}
			z[o] = s
		}
		if pr.nOut == 1 { // binary: sigmoid
			y := float64(pr.yIdx[i])
			loss += softplus(z[0]) - y*z[0]
			r := (sigmoid(z[0]) - y) * invN
			axpy(r, row, g[:pr.p])
			if pr.fitInt {
				g[pr.p] += r
			}
			continue
		}
		lse := logSumExp(z) // multinomial: softmax
		loss += lse - z[pr.yIdx[i]]
		for o := 0; o < pr.nOut; o++ {
			r := math.Exp(z[o] - lse)
			if o == pr.yIdx[i] {
				r--
			}
			r *= invN
			axpy(r, row, g[o*pr.stride:o*pr.stride+pr.p])
			if pr.fitInt {
				g[o*pr.stride+pr.p] += r
			}
		}
	}
	f := loss * invN
	for o := 0; o < pr.nOut; o++ {
		w := x[o*pr.stride : o*pr.stride+pr.p]
		f += 0.5 * pr.lambda * dot(w, w)
		axpy(pr.lambda, w, g[o*pr.stride:o*pr.stride+pr.p])
	}
	return f
}

// unpack splits a parameter vector into coefficient rows and intercepts.
func (pr *logisticProblem) unpack(x []float64) (coef [][]float64, intercept []float64) {
	coef = make([][]float64, pr.nOut)
	intercept = make([]float64, pr.nOut)
	for o := range coef {
		coef[o] = append([]float64(nil), x[o*pr.stride:o*pr.stride+pr.p]...)
		if pr.fitInt {
			intercept[o] = x[o*pr.stride+pr.p]
		}
	}
	return coef, intercept
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// softplus is log(1 + e^z), computed without overflow.
func softplus(z float64) float64 {
	return math.Max(z, 0) + math.Log1p(math.Exp(-math.Abs(z)))
}

func sigmoid(z float64) float64 {
	if z >= 0 {
		return 1 / (1 + math.Exp(-z))
	}
	e := math.Exp(z)
	return e / (1 + e)
}

func logSumExp(z []float64) float64 {
	m := z[0]
	for _, v := range z[1:] {
		m = math.Max(m, v)
	}
	var s float64
	for _, v := range z {
		s += math.Exp(v - m)
	}
	return m + math.Log(s)
}

func softmax(z []float64) []float64 {
	lse := logSumExp(z)
	out := make([]float64, len(z))
	for i, v := range z {
		out[i] = math.Exp(v - lse)
	}
	return out
}

// logisticGob is the versioned on-disk payload.
type logisticGob struct {
	Version      int
	Classes      []float64
	Coef         [][]float64
	Intercept    []float64
	NFeatures    int
	C            float64
	FitIntercept bool
	MaxIter      int
	Tol          float64
	NIter        int
	Converged    bool
}

const logisticFormatVersion = 1

// Save writes the fitted model to path in the versioned gob format.
func (m *LogisticRegression) Save(path string) error {
	if !m.fitted {
		return matutil.ErrNotFitted
	}
	return saveGob("LogisticRegression.Save", path, logisticGob{
		Version: logisticFormatVersion, Classes: m.classes, Coef: m.coef, Intercept: m.intercept,
		NFeatures: m.nFeatures, C: m.C, FitIntercept: m.FitIntercept, MaxIter: m.MaxIter, Tol: m.Tol,
		NIter: m.nIter, Converged: m.converged,
	})
}

// LoadLogisticRegression reads a fitted model previously written by Save.
func LoadLogisticRegression(path string) (*LogisticRegression, error) {
	const op = "LoadLogisticRegression"
	var g logisticGob
	if err := loadGob(op, path, &g); err != nil {
		return nil, err
	}
	if g.Version != logisticFormatVersion {
		return nil, fmt.Errorf("%s: unsupported format version %d (expected %d)", op, g.Version, logisticFormatVersion)
	}
	k := len(g.Classes)
	rows := k
	if k == 2 {
		rows = 1
	}
	switch {
	case k < 2 || !sort.Float64sAreSorted(g.Classes) || !allFinite(g.Classes):
		return nil, fmt.Errorf("%s: corrupt payload: invalid class list", op)
	case g.NFeatures < 1 || len(g.Coef) != rows || len(g.Intercept) != rows:
		return nil, fmt.Errorf("%s: corrupt payload: %d coefficient rows and %d intercepts for %d classes", op, len(g.Coef), len(g.Intercept), k)
	}
	for i := 1; i < k; i++ {
		if g.Classes[i] == g.Classes[i-1] {
			return nil, fmt.Errorf("%s: corrupt payload: duplicate class label", op)
		}
	}
	for _, row := range g.Coef {
		if len(row) != g.NFeatures || !allFinite(row) {
			return nil, fmt.Errorf("%s: corrupt payload: bad coefficient row", op)
		}
	}
	if !allFinite(g.Intercept) {
		return nil, fmt.Errorf("%s: corrupt payload: non-finite intercepts", op)
	}
	m := &LogisticRegression{C: g.C, FitIntercept: g.FitIntercept, MaxIter: g.MaxIter, Tol: g.Tol}
	if err := m.checkParams(); err != nil {
		return nil, fmt.Errorf("%s: corrupt payload: %w", op, err)
	}
	m.classes, m.coef, m.intercept = g.Classes, g.Coef, g.Intercept
	m.nFeatures, m.nIter, m.converged, m.fitted = g.NFeatures, g.NIter, g.Converged, true
	return m, nil
}
