package model_selection

import (
	"fmt"
	"math"
	"sort"

	"github.com/ajitashwath/scikit-go/internal/matutil"
)

// ParamGrid maps each parameter name to the values to try. Every combination is
// evaluated. Values can be anything the ModelBuilder understands.
type ParamGrid map[string][]any

// ModelBuilder builds a new, unfitted model for one combination of parameters. It is
// called once per combination and fold, always with the full set of parameter names.
//
// For a bare estimator, set its fields from params. For a pipeline, use the
// "<step>__<Field>" names that Pipeline.SetParams understands:
//
//	func(params map[string]any) (model_selection.Model, error) {
//		p, err := pipeline.MakePipeline(preprocessing.NewStandardScaler(), svm.NewSVC())
//		if err != nil {
//			return nil, err
//		}
//		return p, p.SetParams(params) // e.g. {"svc__C": 10.0, "svc__Kernel": "rbf"}
//	}
type ModelBuilder func(params map[string]any) (Model, error)

// CVResult is the cross-validated outcome for one parameter combination.
type CVResult struct {
	Params     map[string]any
	FoldScores []float64
	MeanScore  float64
	StdScore   float64 // population standard deviation of FoldScores, like sklearn's std_test_score
	Rank       int     // 1 is best; combinations with equal MeanScore share a rank
}

// GridSearchCV exhaustively cross-validates every combination in Grid and, when
// Refit is set, fits the best one on the whole training set. It mirrors
// sklearn.model_selection.GridSearchCV.
//
// It is not itself serializable, since it holds functions; save BestModel() instead.
// Combinations are evaluated one after another on one goroutine.
type GridSearchCV struct {
	NewModel ModelBuilder
	Grid     ParamGrid
	CV       Splitter // nil means a 5-fold KFold
	Scorer   Scorer   // nil means each model's own Score method
	Refit    bool     // fit the best combination on all of X, y so Predict and Score work

	results   []CVResult
	bestIndex int
	best      Model
	fitted    bool
}

// NewGridSearchCV returns a search with sklearn's defaults: a 5-fold KFold and Refit on.
func NewGridSearchCV(newModel ModelBuilder, grid ParamGrid) *GridSearchCV {
	return &GridSearchCV{NewModel: newModel, Grid: grid, CV: NewKFold(5), Refit: true}
}

// Fit cross-validates every parameter combination, in sklearn's order (parameter
// names sorted, the last name varying fastest), then refits the best if Refit is set.
// The best combination is the one with the highest MeanScore; ties (see scoreTolerance)
// go to the earliest.
// An error from any fit or score aborts the search.
func (g *GridSearchCV) Fit(X [][]float64, y []float64) error {
	g.fitted, g.best, g.results = false, nil, nil
	if g.NewModel == nil {
		return fmt.Errorf("GridSearchCV.Fit: %w: NewModel is nil", ErrInvalidParams)
	}
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("GridSearchCV.Fit: %w", err)
	}
	combos, err := paramCombinations(g.Grid)
	if err != nil {
		return fmt.Errorf("GridSearchCV.Fit: %w", err)
	}

	results := make([]CVResult, len(combos))
	for i, params := range combos {
		params := params
		factory := func() (Model, error) { return g.NewModel(copyParams(params)) }
		scores, err := CrossValScore(factory, X, y, g.CV, g.Scorer)
		if err != nil {
			return fmt.Errorf("GridSearchCV.Fit: parameters %v: %w", params, err)
		}
		mean, std := meanStd(scores)
		results[i] = CVResult{Params: params, FoldScores: scores, MeanScore: mean, StdScore: std}
	}
	rankResults(results)

	best := 0
	for i, r := range results {
		if r.Rank < results[best].Rank {
			best = i
		}
	}

	var refit Model
	if g.Refit {
		if refit, err = g.NewModel(copyParams(results[best].Params)); err != nil {
			return fmt.Errorf("GridSearchCV.Fit: building the best model: %w", err)
		}
		if refit == nil {
			return fmt.Errorf("GridSearchCV.Fit: %w: NewModel returned a nil model", ErrInvalidParams)
		}
		if err := refit.Fit(X, y); err != nil {
			return fmt.Errorf("GridSearchCV.Fit: refitting the best model: %w", err)
		}
	}
	g.results, g.bestIndex, g.best, g.fitted = results, best, refit, true
	return nil
}

// Predict predicts with the refit best model. It needs Refit to have been set.
func (g *GridSearchCV) Predict(X [][]float64) ([]float64, error) {
	if err := g.refitted("Predict"); err != nil {
		return nil, err
	}
	return g.best.Predict(X)
}

// Score scores the refit best model on X, y with Scorer, or with the model's own
// Score method when Scorer is nil.
func (g *GridSearchCV) Score(X [][]float64, y []float64) (float64, error) {
	if err := g.refitted("Score"); err != nil {
		return 0, err
	}
	return scoreModel(g.best, X, y, g.Scorer)
}

func (g *GridSearchCV) refitted(op string) error {
	if !g.fitted {
		return fmt.Errorf("GridSearchCV.%s: %w", op, matutil.ErrNotFitted)
	}
	if g.best == nil {
		return fmt.Errorf("GridSearchCV.%s: %w: Refit was not set, so there is no best model", op, ErrInvalidParams)
	}
	return nil
}

// Results returns one CVResult per parameter combination, in evaluation order.
func (g *GridSearchCV) Results() []CVResult { return append([]CVResult(nil), g.results...) }

// BestIndex is the index in Results of the best combination.
func (g *GridSearchCV) BestIndex() int { return g.bestIndex }

// BestParams returns a copy of the best combination's parameters (nil before Fit).
func (g *GridSearchCV) BestParams() map[string]any {
	if !g.fitted {
		return nil
	}
	return copyParams(g.results[g.bestIndex].Params)
}

// BestScore is the best combination's mean cross-validated score (NaN before Fit).
func (g *GridSearchCV) BestScore() float64 {
	if !g.fitted {
		return math.NaN()
	}
	return g.results[g.bestIndex].MeanScore
}

// BestModel returns the model refit on the whole training set, or nil if Refit was
// not set or Fit has not run. It can be saved if its type supports Save.
func (g *GridSearchCV) BestModel() Model { return g.best }

// paramCombinations lists every combination like sklearn's ParameterGrid: names
// sorted, and the last name changes fastest.
func paramCombinations(grid ParamGrid) ([]map[string]any, error) {
	names := make([]string, 0, len(grid))
	for name, values := range grid {
		if len(values) == 0 {
			return nil, fmt.Errorf("%w: parameter %q has no values", ErrInvalidParams, name)
		}
		names = append(names, name)
	}
	sort.Strings(names)

	combos := []map[string]any{{}}
	for _, name := range names {
		next := make([]map[string]any, 0, len(combos)*len(grid[name]))
		for _, base := range combos {
			for _, v := range grid[name] {
				c := copyParams(base)
				c[name] = v
				next = append(next, c)
			}
		}
		combos = next
	}
	return combos, nil
}

func copyParams(p map[string]any) map[string]any {
	c := make(map[string]any, len(p))
	for k, v := range p {
		c[k] = v
	}
	return c
}

func meanStd(v []float64) (mean, std float64) {
	for _, x := range v {
		mean += x
	}
	mean /= float64(len(v))
	for _, x := range v {
		std += (x - mean) * (x - mean)
	}
	return mean, math.Sqrt(std / float64(len(v)))
}

// scoreTolerance is the relative difference below which two mean scores count as tied.
// Fold scores such as {1, 0.95, 0.90} averaged in a different order differ in the last
// bit; treating that as a real difference would make the winner depend on summation
// order. (sklearn ranks on exact floats and so can be swayed by that noise.)
const scoreTolerance = 1e-12

// rankResults sets Rank like scipy's rankdata(-mean, method="min"): tied means share
// the lowest rank of their group. Means within scoreTolerance of each other are tied,
// and a NaN mean ranks below every real score.
func rankResults(results []CVResult) {
	key := func(m float64) float64 {
		if math.IsNaN(m) {
			return math.Inf(-1)
		}
		return m
	}
	better := func(a, b float64) bool { // is a strictly better than b?
		a, b = key(a), key(b)
		if math.IsInf(a, 0) || math.IsInf(b, 0) {
			return a > b
		}
		return a-b > scoreTolerance*math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
	}
	for i := range results {
		rank := 1
		for j := range results {
			if better(results[j].MeanScore, results[i].MeanScore) {
				rank++
			}
		}
		results[i].Rank = rank
	}
}
