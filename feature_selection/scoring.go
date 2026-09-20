package feature_selection

import (
	"fmt"
	"math"
	"sort"

	"github.com/ajitashwath/scikit-go/internal/matutil"

	"gonum.org/v1/gonum/mathext"
)

// ScoreFunc scores every feature of X against the target y, returning one
// score and one p-value per column. NaN scores are allowed (constant features);
// selectors rank them last.
type ScoreFunc func(X [][]float64, y []float64) (scores, pvalues []float64, err error)

// fSurvival is the survival function of the F distribution with (d1, d2)
// degrees of freedom, matching scipy.stats.f.sf.
func fSurvival(f, d1, d2 float64) float64 {
	switch {
	case math.IsNaN(f):
		return math.NaN()
	case f <= 0:
		return 1
	case math.IsInf(f, 1):
		return 0
	}
	return mathext.RegIncBeta(d2/2, d1/2, d2/(d2+d1*f))
}

// FClassif computes the one-way ANOVA F-statistic of each feature across the
// classes of y, like sklearn.feature_selection.f_classif. A feature that is
// constant within every class and differs between them gets an infinite score
// and a zero p-value; a feature constant overall gets a NaN score.
func FClassif(X [][]float64, y []float64) ([]float64, []float64, error) {
	if err := matutil.ValidateXy(X, y); err != nil {
		return nil, nil, fmt.Errorf("FClassif: %w", err)
	}
	n, p := len(X), len(X[0])
	classes := uniqueSorted(y)
	k := len(classes)
	if k < 2 {
		return nil, nil, fmt.Errorf("FClassif: %w: need at least 2 classes, got %d", ErrInvalidSelector, k)
	}
	if n <= k {
		return nil, nil, fmt.Errorf("FClassif: %w: need more samples (%d) than classes (%d)", ErrInvalidSelector, n, k)
	}
	group := make(map[float64]int, k)
	for i, c := range classes {
		group[c] = i
	}
	counts := make([]float64, k)
	for _, label := range y {
		counts[group[label]]++
	}

	dfBetween, dfWithin := float64(k-1), float64(n-k)
	scores := make([]float64, p)
	pvalues := make([]float64, p)
	sums := make([]float64, k)
	for j := 0; j < p; j++ {
		for g := range sums {
			sums[g] = 0
		}
		var total float64
		for i := 0; i < n; i++ {
			v := X[i][j]
			sums[group[y[i]]] += v
			total += v
		}
		grand := total / float64(n)
		means := make([]float64, k)
		var ssBetween float64
		for g := 0; g < k; g++ {
			means[g] = sums[g] / counts[g]
			d := means[g] - grand
			ssBetween += counts[g] * d * d
		}
		var ssWithin float64
		for i := 0; i < n; i++ {
			d := X[i][j] - means[group[y[i]]]
			ssWithin += d * d
		}
		f := (ssBetween / dfBetween) / (ssWithin / dfWithin)
		scores[j] = f
		pvalues[j] = fSurvival(f, dfBetween, dfWithin)
	}
	return scores, pvalues, nil
}

// FRegression computes the univariate linear-regression F-statistic of each
// feature against y (with centering), like sklearn.feature_selection.f_regression
// with force_finite=True: perfect correlation yields the largest float64 and a
// zero p-value, and a constant feature or target yields a score of 0 and a
// p-value of 1.
func FRegression(X [][]float64, y []float64) ([]float64, []float64, error) {
	if err := matutil.ValidateXy(X, y); err != nil {
		return nil, nil, fmt.Errorf("FRegression: %w", err)
	}
	n, p := len(X), len(X[0])
	if n < 3 {
		return nil, nil, fmt.Errorf("FRegression: %w: need at least 3 samples, got %d", ErrInvalidSelector, n)
	}

	var yMean float64
	for _, v := range y {
		yMean += v
	}
	yMean /= float64(n)
	yc := make([]float64, n)
	var yNorm float64
	for i, v := range y {
		yc[i] = v - yMean
		yNorm += yc[i] * yc[i]
	}
	yNorm = math.Sqrt(yNorm)

	dof := float64(n - 2)
	scores := make([]float64, p)
	pvalues := make([]float64, p)
	for j := 0; j < p; j++ {
		var mean float64
		for i := 0; i < n; i++ {
			mean += X[i][j]
		}
		mean /= float64(n)
		// Centered cross-product and scaled standard deviation via moments,
		// as r_regression does.
		var dot, sq float64
		for i := 0; i < n; i++ {
			dot += yc[i] * X[i][j]
			sq += X[i][j] * X[i][j]
		}
		xNorm := math.Sqrt(sq - float64(n)*mean*mean)
		corr := dot / xNorm / yNorm
		if math.IsNaN(corr) || math.IsInf(corr, 0) {
			scores[j], pvalues[j] = 0, 1
			continue
		}
		c2 := corr * corr
		f := c2 / (1 - c2) * dof
		switch {
		case math.IsInf(f, 1):
			scores[j], pvalues[j] = math.MaxFloat64, 0
		case math.IsNaN(f):
			scores[j], pvalues[j] = 0, 1
		default:
			scores[j], pvalues[j] = f, fSurvival(f, 1, dof)
		}
	}
	return scores, pvalues, nil
}

func uniqueSorted(y []float64) []float64 {
	seen := make(map[float64]struct{}, 8)
	for _, v := range y {
		seen[v] = struct{}{}
	}
	out := make([]float64, 0, len(seen))
	for v := range seen {
		out = append(out, v)
	}
	sort.Float64s(out)
	return out
}
