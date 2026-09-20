package svm

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/metrics"
)

// Compile-time checks that SVC satisfies the core interfaces.
var (
	_ core.Estimator  = (*SVC)(nil)
	_ core.Predictor  = (*SVC)(nil)
	_ core.Classifier = (*SVC)(nil)
	_ core.Saver      = (*SVC)(nil)
)

// SVC is a C-support vector classifier, mirroring sklearn.svm.SVC.
//
// Multiclass problems are solved one-vs-one; Predict takes the majority vote
// of the pairwise classifiers, breaking ties toward the smaller label.
//
// Kernel is "linear", "poly", "rbf" or "sigmoid". Gamma == 0 means sklearn's
// "scale" (1 / (n_features * X.var())); pass 1/n_features for "auto". MaxIter < 0
// applies a generous safety limit and Fit returns ErrNotConverged if it is hit
// (badly scaled features are the usual cause); an explicit positive MaxIter
// instead stops the solver there and uses the model as it stands. CacheSize is the kernel cache in megabytes. Setting
// Probability trains Platt-scaling sigmoids (five times slower) so that
// PredictProba works; Seed drives its cross-validation shuffles.
type SVC struct {
	C           float64
	Kernel      string
	Degree      int
	Gamma       float64
	Coef0       float64
	Tol         float64
	MaxIter     int
	CacheSize   float64
	Probability bool
	Seed        int64

	m      svcModel
	fitted bool
}

// svcModel is the fitted state, in libsvm's layout: support vectors grouped by
// class, and coef[r] holding the dual coefficients of pairs (i, j) with j-1 == r
// for class i's vectors and with i == r for class j's vectors.
type svcModel struct {
	classes   []float64
	nFeatures int
	gamma     float64 // resolved
	sv        [][]float64
	svIdx     []int // training-set index of each support vector
	nSV       []int // support vectors per class
	coef      [][]float64
	rho       []float64 // one per class pair, in (0,1),(0,2),...,(1,2),... order
	probA     []float64
	probB     []float64
}

// NewSVC returns an unfitted classifier with sklearn's defaults
// (C=1, rbf kernel, gamma="scale", tol=1e-3).
func NewSVC() *SVC {
	return &SVC{C: 1, Kernel: KernelRBF, Degree: 3, Tol: 1e-3, MaxIter: -1, CacheSize: 200}
}

func (s *SVC) params() params {
	return params{c: s.C, kernel: s.Kernel, degree: s.Degree, gamma: s.Gamma, coef0: s.Coef0,
		tol: s.Tol, maxIter: s.MaxIter, cacheSize: s.CacheSize}
}

func (s *SVC) kernelFunc() kernelFunc {
	return kernelFunc{kind: s.Kernel, gamma: s.m.gamma, coef0: s.Coef0, degree: s.Degree}
}

// Fit trains the classifier.
func (s *SVC) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("SVC.Fit: %w", err)
	}
	p := s.params()
	if err := p.validate(); err != nil {
		return fmt.Errorf("SVC.Fit: %w", err)
	}
	if _, err := newKernelFunc(s.Kernel, 1, s.Coef0, s.Degree); err != nil {
		return fmt.Errorf("SVC.Fit: %w", err)
	}

	classes := sortedUnique(y)
	k := len(classes)
	if k < 2 {
		return fmt.Errorf("SVC.Fit: %w: need samples of at least 2 classes, got %d", ErrInvalidSVM, k)
	}
	byClass := make([][]int, k)
	for i, label := range y {
		c := sort.SearchFloat64s(classes, label)
		byClass[c] = append(byClass[c], i)
	}

	gamma := resolveGamma(s.Gamma, X)
	kf := kernelFunc{kind: s.Kernel, gamma: gamma, coef0: s.Coef0, degree: s.Degree}
	rng := rand.New(rand.NewSource(s.Seed))

	// Train every one-vs-one pair.
	type pair struct {
		a, b  int
		model binaryModel
	}
	var pairs []pair
	var probA, probB []float64
	for a := 0; a < k; a++ {
		for b := a + 1; b < k; b++ {
			idx := append(append([]int(nil), byClass[a]...), byClass[b]...)
			subX := make([][]float64, len(idx))
			subY := make([]float64, len(idx))
			for r, i := range idx {
				subX[r] = X[i]
				if r < len(byClass[a]) {
					subY[r] = 1
				} else {
					subY[r] = -1
				}
			}
			if s.Probability {
				pa, pb, err := fitSigmoid(subX, subY, kf, p, rng)
				if err != nil {
					return fmt.Errorf("SVC.Fit: %w", err)
				}
				probA = append(probA, pa)
				probB = append(probB, pb)
			}
			model, err := trainBinary(subX, subY, kf, p)
			if err != nil {
				return fmt.Errorf("SVC.Fit: %w", err)
			}
			pairs = append(pairs, pair{a, b, model})
		}
	}

	// A sample is a support vector if it has a nonzero coefficient in any pair.
	nonzero := make([]bool, len(y))
	for _, pr := range pairs {
		idx := append(append([]int(nil), byClass[pr.a]...), byClass[pr.b]...)
		for r, i := range idx {
			if pr.model.coef[r] != 0 {
				nonzero[i] = true
			}
		}
	}
	var sv [][]float64
	var svIdx []int
	nSV := make([]int, k)
	svPos := make([]int, len(y)) // position of each support vector in sv
	nzStart := make([]int, k)
	for c := 0; c < k; c++ {
		nzStart[c] = len(sv)
		for _, i := range byClass[c] {
			if nonzero[i] {
				svPos[i] = len(sv)
				sv = append(sv, X[i])
				svIdx = append(svIdx, i)
				nSV[c]++
			}
		}
	}
	coef := make([][]float64, k-1)
	for r := range coef {
		coef[r] = make([]float64, len(sv))
	}
	rho := make([]float64, len(pairs))
	for pi, pr := range pairs {
		rho[pi] = pr.model.rho
		na := len(byClass[pr.a])
		for r, i := range byClass[pr.a] {
			if nonzero[i] {
				coef[pr.b-1][svPos[i]] = pr.model.coef[r]
			}
		}
		for r, i := range byClass[pr.b] {
			if nonzero[i] {
				coef[pr.a][svPos[i]] = pr.model.coef[na+r]
			}
		}
	}

	s.m = svcModel{classes: classes, nFeatures: len(X[0]), gamma: gamma, sv: sv, svIdx: svIdx,
		nSV: nSV, coef: coef, rho: rho, probA: probA, probB: probB}
	s.fitted = true
	return nil
}

// pairDecisions returns the libsvm decision value of every class pair at x
// (positive votes for the pair's first class).
func (s *SVC) pairDecisions(x []float64) []float64 {
	m := &s.m
	kf := s.kernelFunc()
	kv := make([]float64, len(m.sv))
	for i, v := range m.sv {
		kv[i] = kf.eval(v, x)
	}
	start := make([]int, len(m.classes))
	for c := 1; c < len(start); c++ {
		start[c] = start[c-1] + m.nSV[c-1]
	}
	dec := make([]float64, 0, len(m.rho))
	pi := 0
	for i := 0; i < len(m.classes); i++ {
		for j := i + 1; j < len(m.classes); j++ {
			var sum float64
			si, sj := start[i], start[j]
			for q := 0; q < m.nSV[i]; q++ {
				sum += m.coef[j-1][si+q] * kv[si+q]
			}
			for q := 0; q < m.nSV[j]; q++ {
				sum += m.coef[i][sj+q] * kv[sj+q]
			}
			dec = append(dec, sum-m.rho[pi])
			pi++
		}
	}
	return dec
}

func (s *SVC) checkPredict(name string, X [][]float64) error {
	if !s.fitted {
		return matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, s.m.nFeatures); err != nil {
		return fmt.Errorf("SVC.%s: %w", name, err)
	}
	return nil
}

// Predict returns the one-vs-one majority-vote class of each row of X.
func (s *SVC) Predict(X [][]float64) ([]float64, error) {
	if err := s.checkPredict("Predict", X); err != nil {
		return nil, err
	}
	k := len(s.m.classes)
	preds := make([]float64, len(X))
	for r, x := range X {
		votes := make([]int, k)
		pi := 0
		dec := s.pairDecisions(x)
		for i := 0; i < k; i++ {
			for j := i + 1; j < k; j++ {
				if dec[pi] > 0 {
					votes[i]++
				} else {
					votes[j]++
				}
				pi++
			}
		}
		best := 0
		for c := 1; c < k; c++ {
			if votes[c] > votes[best] {
				best = c
			}
		}
		preds[r] = s.m.classes[best]
	}
	return preds, nil
}

// DecisionFunction returns sklearn's decision_function. For two classes it is
// a single column, positive for Classes()[1]. For more classes it has one
// column per class (the one-vs-rest aggregation of the pairwise votes and
// confidences, sklearn's default decision_function_shape="ovr").
func (s *SVC) DecisionFunction(X [][]float64) ([][]float64, error) {
	if err := s.checkPredict("DecisionFunction", X); err != nil {
		return nil, err
	}
	k := len(s.m.classes)
	out := make([][]float64, len(X))
	for r, x := range X {
		dec := s.pairDecisions(x)
		if k == 2 {
			out[r] = []float64{-dec[0]}
			continue
		}
		votes := make([]float64, k)
		conf := make([]float64, k)
		pi := 0
		for i := 0; i < k; i++ {
			for j := i + 1; j < k; j++ {
				conf[i] += dec[pi]
				conf[j] -= dec[pi]
				if dec[pi] < 0 {
					votes[j]++
				} else {
					votes[i]++
				}
				pi++
			}
		}
		row := make([]float64, k)
		for c := range row {
			row[c] = votes[c] + conf[c]/(3*(math.Abs(conf[c])+1))
		}
		out[r] = row
	}
	return out, nil
}

// PredictProba returns class probabilities from the Platt-scaled pairwise
// classifiers. It requires the classifier to have been fitted with Probability
// set. Columns follow Classes().
func (s *SVC) PredictProba(X [][]float64) ([][]float64, error) {
	if err := s.checkPredict("PredictProba", X); err != nil {
		return nil, err
	}
	if len(s.m.probA) == 0 {
		return nil, fmt.Errorf("SVC.PredictProba: %w: fit with Probability=true to enable probability estimates", ErrInvalidSVM)
	}
	k := len(s.m.classes)
	const minProb = 1e-7
	out := make([][]float64, len(X))
	for r, x := range X {
		dec := s.pairDecisions(x)
		pw := make([][]float64, k)
		for i := range pw {
			pw[i] = make([]float64, k)
		}
		pi := 0
		for i := 0; i < k; i++ {
			for j := i + 1; j < k; j++ {
				v := math.Min(math.Max(sigmoidPredict(dec[pi], s.m.probA[pi], s.m.probB[pi]), minProb), 1-minProb)
				pw[i][j] = v
				pw[j][i] = 1 - v
				pi++
			}
		}
		out[r] = multiclassProbability(k, pw)
	}
	return out, nil
}

// Score returns the accuracy of the predictions on X against y.
func (s *SVC) Score(X [][]float64, y []float64) (float64, error) {
	preds, err := s.Predict(X)
	if err != nil {
		return 0, err
	}
	score, err := metrics.AccuracyScore(y, preds)
	if err != nil {
		return 0, fmt.Errorf("SVC.Score: %w", err)
	}
	return score, nil
}

// Classes returns the sorted unique class labels learned at Fit time.
func (s *SVC) Classes() []float64 {
	if !s.fitted {
		return nil
	}
	return append([]float64(nil), s.m.classes...)
}

// SupportVectors returns the support vectors, grouped by class.
func (s *SVC) SupportVectors() [][]float64 {
	if !s.fitted {
		return nil
	}
	return copyMatrix(s.m.sv)
}

// Support returns the training-set indices of the support vectors.
func (s *SVC) Support() []int {
	if !s.fitted {
		return nil
	}
	return append([]int(nil), s.m.svIdx...)
}

// NSupport returns the number of support vectors of each class.
func (s *SVC) NSupport() []int {
	if !s.fitted {
		return nil
	}
	return append([]int(nil), s.m.nSV...)
}

// DualCoef returns sklearn's dual_coef_: (n_classes-1) x n_support_vectors
// products alpha_i*y_i. For two classes the sign is flipped, as sklearn does,
// so that the decision function is positive for Classes()[1].
func (s *SVC) DualCoef() [][]float64 {
	if !s.fitted {
		return nil
	}
	out := copyMatrix(s.m.coef)
	if len(s.m.classes) == 2 {
		for i := range out {
			for j := range out[i] {
				out[i][j] = -out[i][j]
			}
		}
	}
	return out
}

// Intercept returns sklearn's intercept_ (one value per class pair). For two
// classes it is the negated libsvm bias, matching DualCoef's flipped sign.
func (s *SVC) Intercept() []float64 {
	if !s.fitted {
		return nil
	}
	out := make([]float64, len(s.m.rho))
	flip := len(s.m.classes) == 2
	for i, r := range s.m.rho {
		if flip {
			out[i] = r
		} else {
			out[i] = -r
		}
	}
	return out
}

// GammaValue returns the gamma used by the fitted kernel, after resolving
// Gamma == 0 to sklearn's "scale" value.
func (s *SVC) GammaValue() float64 {
	if !s.fitted {
		return 0
	}
	return s.m.gamma
}

func sortedUnique(y []float64) []float64 {
	c := append([]float64(nil), y...)
	sort.Float64s(c)
	out := c[:0]
	for i, v := range c {
		if i == 0 || v != out[len(out)-1] {
			out = append(out, v)
		}
	}
	return out
}

func copyMatrix(m [][]float64) [][]float64 {
	out := make([][]float64, len(m))
	for i, row := range m {
		out[i] = append([]float64(nil), row...)
	}
	return out
}
