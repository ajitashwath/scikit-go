package model_selection

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/metrics"
	"github.com/ajitashwath/scikit-go/neighbors"
	"github.com/ajitashwath/scikit-go/pipeline"
	"github.com/ajitashwath/scikit-go/preprocessing"
	"github.com/ajitashwath/scikit-go/svm"
)

// GridSearchCV is itself a Model, so searches can be nested or cross-validated.
var _ Model = (*GridSearchCV)(nil)

// paramInt converts a value that came out of JSON (float64) or Go (int) to an int.
func paramInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case float64:
		return int(n)
	}
	panic("not a number")
}

func knnClassifierBuilder(params map[string]any) (Model, error) {
	m := neighbors.NewKNeighborsClassifier()
	m.NNeighbors = paramInt(params["n_neighbors"])
	if w, ok := params["weights"]; ok {
		m.Weights = w.(string)
	}
	return m, nil
}

func knnRegressorBuilder(params map[string]any) (Model, error) {
	m := neighbors.NewKNeighborsRegressor()
	m.NNeighbors = paramInt(params["n_neighbors"])
	return m, nil
}

func TestGridSearchCV_AgainstSklearn(t *testing.T) {
	builders := map[string]ModelBuilder{"knn_classifier": knnClassifierBuilder, "knn_regressor": knnRegressorBuilder}
	for name, tc := range loadFixtures(t).Grid {
		grid := ParamGrid{}
		for k, vs := range tc.Grid {
			grid[k] = vs
		}
		g := NewGridSearchCV(builders[name], grid)
		g.CV = tc.CV.splitter()
		if err := g.Fit(tc.X, tc.Y); err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		res := g.Results()
		if len(res) != len(tc.Params) {
			t.Fatalf("%s: %d combinations, want %d", name, len(res), len(tc.Params))
		}
		for i, r := range res {
			// Same combination in the same position: the enumeration order matches sklearn's.
			for k, want := range tc.Params[i] {
				if paramKey(r.Params[k]) != paramKey(want) {
					t.Errorf("%s: combination %d param %s = %v, want %v", name, i, k, r.Params[k], want)
				}
			}
			assertClose(t, name+" fold scores", r.FoldScores, tc.FoldScores[i], 1e-9)
			if math.Abs(r.MeanScore-tc.MeanTestScore[i]) > 1e-9 {
				t.Errorf("%s: mean[%d] = %.12f, want %.12f", name, i, r.MeanScore, tc.MeanTestScore[i])
			}
			if math.Abs(r.StdScore-tc.StdTestScore[i]) > 1e-9 {
				t.Errorf("%s: std[%d] = %.12f, want %.12f", name, i, r.StdScore, tc.StdTestScore[i])
			}
			if r.Rank != tc.RankTestScore[i] {
				t.Errorf("%s: rank[%d] = %d, want %d", name, i, r.Rank, tc.RankTestScore[i])
			}
		}
		if g.BestIndex() != tc.BestIndex {
			t.Errorf("%s: best index %d, want %d", name, g.BestIndex(), tc.BestIndex)
		}
		if math.Abs(g.BestScore()-tc.BestScore) > 1e-9 {
			t.Errorf("%s: best score %.12f, want %.12f", name, g.BestScore(), tc.BestScore)
		}

		pred, err := g.Predict(tc.PredictX)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		assertClose(t, name+" refit predictions", pred, tc.Predict, 1e-9)
	}
}

// paramKey makes an int from Go and the float64 from JSON comparable.
func paramKey(v any) any {
	if f, ok := v.(float64); ok {
		return int(f)
	}
	return v
}

func TestParamCombinations_OrderMatchesSklearn(t *testing.T) {
	got, err := paramCombinations(ParamGrid{"b": {1, 2}, "a": {"x", "y", "z"}})
	if err != nil {
		t.Fatal(err)
	}
	// Names sorted (a, b); the last name (b) varies fastest.
	want := []map[string]any{
		{"a": "x", "b": 1}, {"a": "x", "b": 2},
		{"a": "y", "b": 1}, {"a": "y", "b": 2},
		{"a": "z", "b": 1}, {"a": "z", "b": 2},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v\nwant %v", got, want)
	}

	// An empty grid is one combination with no parameters, as in sklearn.
	got, err = paramCombinations(ParamGrid{})
	if err != nil || len(got) != 1 || len(got[0]) != 0 {
		t.Errorf("empty grid: got %v, %v; want one empty combination", got, err)
	}
	if _, err := paramCombinations(ParamGrid{"a": {}}); !errors.Is(err, ErrInvalidParams) {
		t.Errorf("empty value list: got %v, want ErrInvalidParams", err)
	}
}

func TestRankResults_TiesShareTheLowestRank(t *testing.T) {
	res := []CVResult{{MeanScore: 0.5}, {MeanScore: 0.9}, {MeanScore: 0.9}, {MeanScore: 0.1}, {MeanScore: math.NaN()}}
	rankResults(res)
	want := []int{3, 1, 1, 4, 5}
	for i, r := range res {
		if r.Rank != want[i] {
			t.Errorf("rank[%d] = %d, want %d", i, r.Rank, want[i])
		}
	}
}

// The same fold scores summed in a different order differ in the last bit; that is a tie.
func TestRankResults_LastBitDifferencesAreTies(t *testing.T) {
	a := 0.97142857142857131
	b := 0.97142857142857153 // 2e-16 above a: the mean of the same folds in another order
	if a == b {
		t.Fatal("test values must differ")
	}
	res := []CVResult{{MeanScore: a}, {MeanScore: b}, {MeanScore: 0.9714}}
	rankResults(res)
	if res[0].Rank != 1 || res[1].Rank != 1 {
		t.Errorf("ranks %d, %d for means that differ by one ulp; want a shared rank 1", res[0].Rank, res[1].Rank)
	}
	if res[2].Rank != 3 {
		t.Errorf("a mean 3e-6 lower got rank %d, want 3 (a real difference must still separate)", res[2].Rank)
	}
}

func TestGridSearchCV_TiesPickTheEarliestCombination(t *testing.T) {
	// Every combination scores the same, so the first one must win.
	X, y := dummyX(20), make([]float64, 20)
	g := NewGridSearchCV(func(p map[string]any) (Model, error) { return &noScoreModel{}, nil }, ParamGrid{"a": {1, 2, 3}})
	g.Scorer = func(a, b []float64) (float64, error) { return 1, nil }
	if err := g.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	if g.BestIndex() != 0 {
		t.Errorf("best index = %d, want 0 (earliest of the ties)", g.BestIndex())
	}
	if got := g.BestParams()["a"]; got != 1 {
		t.Errorf("best params a = %v, want 1", got)
	}
}

func TestGridSearchCV_BestParamsIsACopy(t *testing.T) {
	tc := loadFixtures(t).Grid["knn_regressor"]
	g := NewGridSearchCV(knnRegressorBuilder, ParamGrid{"n_neighbors": {2, 4}})
	if err := g.Fit(tc.X, tc.Y); err != nil {
		t.Fatal(err)
	}
	p := g.BestParams()
	p["n_neighbors"] = 999
	if g.BestParams()["n_neighbors"] == 999 {
		t.Error("modifying the returned BestParams changed the search's own results")
	}
}

func TestGridSearchCV_BuilderCannotCorruptResults(t *testing.T) {
	tc := loadFixtures(t).Grid["knn_regressor"]
	builder := func(params map[string]any) (Model, error) {
		m, err := knnRegressorBuilder(params)
		params["n_neighbors"] = -1 // a careless builder
		return m, err
	}
	g := NewGridSearchCV(builder, ParamGrid{"n_neighbors": {2, 4}})
	if err := g.Fit(tc.X, tc.Y); err != nil {
		t.Fatal(err)
	}
	for i, r := range g.Results() {
		if r.Params["n_neighbors"] == -1 {
			t.Errorf("result %d had its params overwritten by the builder", i)
		}
	}
}

func TestGridSearchCV_RefitFalse(t *testing.T) {
	tc := loadFixtures(t).Grid["knn_regressor"]
	g := NewGridSearchCV(knnRegressorBuilder, ParamGrid{"n_neighbors": {2, 4}})
	g.Refit = false
	if err := g.Fit(tc.X, tc.Y); err != nil {
		t.Fatal(err)
	}
	if g.BestModel() != nil {
		t.Error("BestModel should be nil without Refit")
	}
	if g.BestParams() == nil || len(g.Results()) != 2 {
		t.Error("the search results should still be available without Refit")
	}
	if _, err := g.Predict(tc.X); !errors.Is(err, ErrInvalidParams) {
		t.Errorf("Predict without Refit: got %v, want ErrInvalidParams", err)
	}
	if _, err := g.Score(tc.X, tc.Y); !errors.Is(err, ErrInvalidParams) {
		t.Errorf("Score without Refit: got %v, want ErrInvalidParams", err)
	}
}

func TestGridSearchCV_NotFitted(t *testing.T) {
	g := NewGridSearchCV(knnRegressorBuilder, ParamGrid{"n_neighbors": {2}})
	if _, err := g.Predict([][]float64{{1}}); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Predict: got %v, want ErrNotFitted", err)
	}
	if _, err := g.Score([][]float64{{1}}, []float64{1}); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Score: got %v, want ErrNotFitted", err)
	}
	if g.BestParams() != nil || !math.IsNaN(g.BestScore()) || g.BestModel() != nil {
		t.Error("best-result accessors should be empty before Fit")
	}
}

func TestGridSearchCV_FitErrors(t *testing.T) {
	X, y := dummyX(20), make([]float64, 20)
	boom := errors.New("boom")
	failingModel := func(p map[string]any) (Model, error) { return nil, boom }
	grid := ParamGrid{"a": {1}}

	cases := []struct {
		name string
		g    *GridSearchCV
		X    [][]float64
		y    []float64
		want error
	}{
		{"nil builder", NewGridSearchCV(nil, grid), X, y, ErrInvalidParams},
		{"builder error", NewGridSearchCV(failingModel, grid), X, y, boom},
		{"empty parameter values", NewGridSearchCV(knnRegressorBuilder, ParamGrid{"n_neighbors": {}}), X, y, ErrInvalidParams},
		{"bad data", NewGridSearchCV(knnRegressorBuilder, ParamGrid{"n_neighbors": {2}}), nil, nil, matutil.ErrEmptyInput},
		{"more folds than samples", func() *GridSearchCV {
			g := NewGridSearchCV(knnRegressorBuilder, ParamGrid{"n_neighbors": {2}})
			g.CV = NewKFold(50)
			return g
		}(), X, y, ErrInvalidParams},
	}
	for _, tc := range cases {
		if err := tc.g.Fit(tc.X, tc.y); !errors.Is(err, tc.want) {
			t.Errorf("%s: got %v, want %v", tc.name, err, tc.want)
		}
	}
}

// A failed refit must leave no half-fitted state behind.
func TestGridSearchCV_FailedFitResetsState(t *testing.T) {
	tc := loadFixtures(t).Grid["knn_regressor"]
	g := NewGridSearchCV(knnRegressorBuilder, ParamGrid{"n_neighbors": {2, 4}})
	if err := g.Fit(tc.X, tc.Y); err != nil {
		t.Fatal(err)
	}
	if err := g.Fit(nil, nil); err == nil {
		t.Fatal("expected an error fitting empty data")
	}
	if _, err := g.Predict(tc.X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("after a failed Fit, Predict returned %v, want ErrNotFitted", err)
	}
}

func TestGridSearchCV_ScoreUsesScorerWhenSet(t *testing.T) {
	tc := loadFixtures(t).Grid["knn_regressor"]
	g := NewGridSearchCV(knnRegressorBuilder, ParamGrid{"n_neighbors": {2, 4, 8}})
	if err := g.Fit(tc.X, tc.Y); err != nil {
		t.Fatal(err)
	}
	r2, err := g.Score(tc.X, tc.Y)
	if err != nil {
		t.Fatal(err)
	}
	direct, _ := g.BestModel().(interface {
		Score([][]float64, []float64) (float64, error)
	}).Score(tc.X, tc.Y)
	if r2 != direct {
		t.Errorf("Score = %v, want the best model's own score %v", r2, direct)
	}

	g.Scorer = metrics.MeanAbsoluteError
	mae, err := g.Score(tc.X, tc.Y)
	if err != nil {
		t.Fatal(err)
	}
	if mae == r2 || mae <= 0 {
		t.Errorf("with a MAE scorer got %v (R^2 was %v); the scorer was not used", mae, r2)
	}
}

// Grid search over a Pipeline, using its "<step>__<Field>" parameter names.
func TestGridSearchCV_WithPipeline(t *testing.T) {
	tc := loadFixtures(t).CrossVal["knn_stratified"]
	build := func(params map[string]any) (Model, error) {
		p, err := pipeline.MakePipeline(preprocessing.NewStandardScaler(), svm.NewSVC())
		if err != nil {
			return nil, err
		}
		return p, p.SetParams(params)
	}
	g := NewGridSearchCV(build, ParamGrid{
		"svc__C":      {0.1, 1.0, 10.0},
		"svc__Kernel": {"linear", "rbf"},
	})
	g.CV = NewStratifiedKFold(3)
	if err := g.Fit(tc.X, tc.Y); err != nil {
		t.Fatal(err)
	}
	if n := len(g.Results()); n != 6 {
		t.Fatalf("got %d combinations, want 6", n)
	}
	best := g.BestParams()
	if _, ok := best["svc__C"]; !ok {
		t.Errorf("best params %v are missing svc__C", best)
	}
	for _, r := range g.Results() {
		if r.MeanScore > g.BestScore() {
			t.Errorf("combination %v scored %.4f, above the reported best %.4f", r.Params, r.MeanScore, g.BestScore())
		}
	}
	acc, err := g.Score(tc.X, tc.Y)
	if err != nil {
		t.Fatal(err)
	}
	if acc < 0.7 {
		t.Errorf("refit pipeline accuracy on its training data = %.3f, want >= 0.7", acc)
	}
}

// A search can be cross-validated like any other model (nested cross-validation).
func TestGridSearchCV_NestedCrossValidation(t *testing.T) {
	tc := loadFixtures(t).CrossVal["knn_kfold"]
	outer := func() (Model, error) {
		g := NewGridSearchCV(knnClassifierBuilder, ParamGrid{"n_neighbors": {1, 5, 9}})
		g.CV = NewKFold(3)
		return g, nil
	}
	scores, err := CrossValScore(outer, tc.X, tc.Y, NewKFold(3), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != 3 {
		t.Fatalf("got %d outer scores, want 3", len(scores))
	}
	for i, s := range scores {
		if s < 0.4 || s > 1 {
			t.Errorf("outer fold %d accuracy = %.3f, want within [0.4, 1]", i, s)
		}
	}
}
