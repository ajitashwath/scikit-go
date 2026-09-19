package feature_selection

import (
	"fmt"
	"math"
	"sort"

	"scikit-go/core"
	"scikit-go/internal/matutil"
)

// Compile-time checks that SelectKBest satisfies the core interfaces.
var (
	_ core.Estimator = (*SelectKBest)(nil)
	_ core.Saver     = (*SelectKBest)(nil)
)

// Names of the built-in scoring functions accepted by SelectKBest.Score.
const (
	ScoreFClassif    = "f_classif"
	ScoreFRegression = "f_regression"
)

// SelectKBest keeps the K features with the highest univariate score against
// the target, mirroring sklearn.feature_selection.SelectKBest.
//
// Score names a built-in scorer ("f_classif" for classification targets,
// "f_regression" for continuous ones). A non-nil ScoreFunc overrides it; it is
// not persisted by Save, so a loaded selector keeps its mask but must be given
// the function again before refitting.
type SelectKBest struct {
	K         int
	Score     string
	ScoreFunc ScoreFunc

	support
	scores  []float64
	pvalues []float64
}

// NewSelectKBest returns an unfitted selector with sklearn's defaults
// (K=10, f_classif).
func NewSelectKBest() *SelectKBest {
	return &SelectKBest{K: 10, Score: ScoreFClassif}
}

func (s *SelectKBest) scorer() (ScoreFunc, error) {
	if s.ScoreFunc != nil {
		return s.ScoreFunc, nil
	}
	switch s.Score {
	case ScoreFClassif, "":
		return FClassif, nil
	case ScoreFRegression:
		return FRegression, nil
	default:
		return nil, fmt.Errorf("SelectKBest.Fit: %w: unsupported score %q", ErrInvalidSelector, s.Score)
	}
}

// Fit scores every feature against y and selects the K best. Ties are resolved
// in favor of the later column, as sklearn's stable argsort does.
func (s *SelectKBest) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("SelectKBest.Fit: %w", err)
	}
	score, err := s.scorer()
	if err != nil {
		return err
	}
	p := len(X[0])
	if s.K < 0 || s.K > p {
		return fmt.Errorf("SelectKBest.Fit: %w: k=%d must be between 0 and n_features=%d", ErrInvalidSelector, s.K, p)
	}
	scores, pvalues, err := score(X, y)
	if err != nil {
		return fmt.Errorf("SelectKBest.Fit: %w", err)
	}
	if len(scores) != p || len(pvalues) != p {
		return fmt.Errorf("SelectKBest.Fit: %w: score function returned %d scores for %d features", ErrInvalidSelector, len(scores), p)
	}

	// NaN scores rank below every real score.
	ranked := make([]float64, p)
	for j, v := range scores {
		if math.IsNaN(v) {
			v = -math.MaxFloat64
		}
		ranked[j] = v
	}
	order := make([]int, p)
	for j := range order {
		order[j] = j
	}
	sort.SliceStable(order, func(a, b int) bool { return ranked[order[a]] < ranked[order[b]] })
	mask := make([]bool, p)
	for _, j := range order[p-s.K:] {
		mask[j] = true
	}

	s.scores = scores
	s.pvalues = pvalues
	s.set(mask)
	return nil
}

// Transform keeps the K selected features.
func (s *SelectKBest) Transform(X [][]float64) ([][]float64, error) {
	return s.transform("SelectKBest", X)
}

// FitTransform fits the selector to (X, y) and returns the reduced matrix.
// Unlike core.Transformer it needs the target, as sklearn's fit_transform does.
func (s *SelectKBest) FitTransform(X [][]float64, y []float64) ([][]float64, error) {
	if err := s.Fit(X, y); err != nil {
		return nil, err
	}
	return s.Transform(X)
}

// Scores returns the per-feature scores computed at Fit time (NaN where the
// scorer produced none), or nil before Fit.
func (s *SelectKBest) Scores() []float64 {
	if !s.fitted {
		return nil
	}
	return append([]float64(nil), s.scores...)
}

// PValues returns the per-feature p-values computed at Fit time, or nil before Fit.
func (s *SelectKBest) PValues() []float64 {
	if !s.fitted {
		return nil
	}
	return append([]float64(nil), s.pvalues...)
}

// Save writes the fitted selector to path in the versioned gob format.
func (s *SelectKBest) Save(path string) error {
	if !s.fitted {
		return matutil.ErrNotFitted
	}
	return saveSelector(path, "SelectKBest", selectorGob{
		Kind:      "select_k_best",
		NFeatures: s.nFeatures,
		Support:   s.mask,
		K:         s.K,
		Score:     s.Score,
		Scores:    s.scores,
		PValues:   s.pvalues,
	})
}

// LoadSelectKBest reads a fitted selector previously written by Save.
func LoadSelectKBest(path string) (*SelectKBest, error) {
	p, err := loadSelector(path, "SelectKBest", "select_k_best")
	if err != nil {
		return nil, err
	}
	s := &SelectKBest{K: p.K, Score: p.Score, scores: p.Scores, pvalues: p.PValues}
	s.set(p.Support)
	return s, nil
}
