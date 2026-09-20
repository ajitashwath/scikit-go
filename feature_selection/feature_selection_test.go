package feature_selection

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/linear"
	"github.com/ajitashwath/scikit-go/neighbors"
	"github.com/ajitashwath/scikit-go/tree"
)

const tol = 1e-8

type fixtureFile struct {
	VarianceThreshold struct {
		X     [][]float64 `json:"X"`
		Cases map[string]struct {
			Threshold   float64     `json:"threshold"`
			Variances   []float64   `json:"variances"`
			Support     []bool      `json:"support"`
			Transformed [][]float64 `json:"transformed"`
		} `json:"cases"`
	} `json:"variance_threshold"`
	FClassif struct {
		X           [][]float64 `json:"X"`
		Y           []float64   `json:"y"`
		Scores      []float64   `json:"scores"`
		PValues     []float64   `json:"pvalues"`
		Support     []bool      `json:"k3_support"`
		Transformed [][]float64 `json:"k3_transformed"`
	} `json:"f_classif"`
	FRegression struct {
		X           [][]float64 `json:"X"`
		Y           []float64   `json:"y"`
		Scores      []float64   `json:"scores"`
		PValues     []float64   `json:"pvalues"`
		Support     []bool      `json:"k2_support"`
		Transformed [][]float64 `json:"k2_transformed"`
	} `json:"f_regression"`
	RFE struct {
		X     [][]float64 `json:"X"`
		Y     []float64   `json:"y"`
		Cases map[string]struct {
			NFeaturesToSelect int         `json:"n_features_to_select"`
			Step              int         `json:"step"`
			Support           []bool      `json:"support"`
			Ranking           []int       `json:"ranking"`
			Transformed       [][]float64 `json:"transformed"`
			Predictions       []float64   `json:"predictions"`
		} `json:"cases"`
	} `json:"rfe"`
}

func loadFixtures(t *testing.T) *fixtureFile {
	t.Helper()
	path := filepath.Join("testdata", "feature_selection_fixtures.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read fixtures at %s: %v", path, err)
	}
	var fx fixtureFile
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("failed to parse fixtures: %v", err)
	}
	return &fx
}

func assertClose(t *testing.T, name string, got, want, tolerance float64) {
	t.Helper()
	// Relative tolerance for large values such as F statistics.
	if math.Abs(got-want) > tolerance*(1+math.Abs(want)) {
		t.Errorf("%s: got %.12g, want %.12g", name, got, want)
	}
}

func assertMask(t *testing.T, name string, got, want []bool) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("%s[%d]: got %v, want %v (full got %v, want %v)", name, i, got[i], want[i], got, want)
			return
		}
	}
}

func assertMatrix(t *testing.T, name string, got, want [][]float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d rows, want %d", name, len(got), len(want))
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("%s row %d: %d cols, want %d", name, i, len(got[i]), len(want[i]))
		}
		for j := range want[i] {
			assertClose(t, name, got[i][j], want[i][j], tol)
		}
	}
}

func TestVarianceThreshold_AgainstSklearn(t *testing.T) {
	fx := loadFixtures(t).VarianceThreshold
	for name, c := range fx.Cases {
		c := c
		t.Run(name, func(t *testing.T) {
			vt := NewVarianceThreshold()
			vt.Threshold = c.Threshold
			out, err := vt.FitTransform(fx.X)
			if err != nil {
				t.Fatalf("FitTransform: %v", err)
			}
			for j, want := range c.Variances {
				assertClose(t, "variances", vt.Variances()[j], want, tol)
			}
			assertMask(t, "support", vt.Support(), c.Support)
			assertMatrix(t, "transformed", out, c.Transformed)
		})
	}
}

func TestVarianceThreshold_NoFeatureMeetsThreshold(t *testing.T) {
	X := [][]float64{{1, 2}, {1, 2}, {1, 2}}
	if err := NewVarianceThreshold().Fit(X, nil); !errors.Is(err, ErrInvalidSelector) {
		t.Errorf("constant data: got %v, want ErrInvalidSelector", err)
	}
	vt := NewVarianceThreshold()
	vt.Threshold = -1
	if err := vt.Fit([][]float64{{1}, {2}}, nil); !errors.Is(err, ErrInvalidSelector) {
		t.Errorf("negative threshold: got %v, want ErrInvalidSelector", err)
	}
}

// Constant columns whose mean is not exactly representable accumulate rounding
// error; the peak-to-peak guard must still treat them as constant.
func TestVarianceThreshold_ConstantColumnWithRoundingError(t *testing.T) {
	X := make([][]float64, 50)
	for i := range X {
		X[i] = []float64{0.1, float64(i)}
	}
	vt := NewVarianceThreshold()
	if err := vt.Fit(X, nil); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	assertMask(t, "support", vt.Support(), []bool{false, true})
}

func TestSelectKBest_FClassif_AgainstSklearn(t *testing.T) {
	fx := loadFixtures(t).FClassif
	scores, pvalues, err := FClassif(fx.X, fx.Y)
	if err != nil {
		t.Fatalf("FClassif: %v", err)
	}
	for j := range fx.Scores {
		assertClose(t, "score", scores[j], fx.Scores[j], 1e-9)
		assertClose(t, "pvalue", pvalues[j], fx.PValues[j], 1e-9)
	}

	sel := NewSelectKBest()
	sel.K = 3
	out, err := sel.FitTransform(fx.X, fx.Y)
	if err != nil {
		t.Fatalf("FitTransform: %v", err)
	}
	assertMask(t, "support", sel.Support(), fx.Support)
	assertMatrix(t, "transformed", out, fx.Transformed)
	for j := range fx.Scores {
		assertClose(t, "Scores()", sel.Scores()[j], fx.Scores[j], 1e-9)
		assertClose(t, "PValues()", sel.PValues()[j], fx.PValues[j], 1e-9)
	}
}

func TestSelectKBest_FRegression_AgainstSklearn(t *testing.T) {
	fx := loadFixtures(t).FRegression
	scores, pvalues, err := FRegression(fx.X, fx.Y)
	if err != nil {
		t.Fatalf("FRegression: %v", err)
	}
	for j := range fx.Scores {
		assertClose(t, "score", scores[j], fx.Scores[j], 1e-9)
		assertClose(t, "pvalue", pvalues[j], fx.PValues[j], 1e-9)
	}

	sel := NewSelectKBest()
	sel.K = 2
	sel.Score = ScoreFRegression
	out, err := sel.FitTransform(fx.X, fx.Y)
	if err != nil {
		t.Fatalf("FitTransform: %v", err)
	}
	assertMask(t, "support", sel.Support(), fx.Support)
	assertMatrix(t, "transformed", out, fx.Transformed)
}

func TestFClassif_ConstantAndSeparableFeatures(t *testing.T) {
	// Column 0 is constant; column 1 separates the classes perfectly and is
	// constant inside each class; column 2 is noise.
	X := [][]float64{
		{5, 0, 1}, {5, 0, 3}, {5, 0, 2},
		{5, 10, 2}, {5, 10, 1}, {5, 10, 3},
	}
	y := []float64{0, 0, 0, 1, 1, 1}
	scores, pvalues, err := FClassif(X, y)
	if err != nil {
		t.Fatalf("FClassif: %v", err)
	}
	if !math.IsNaN(scores[0]) {
		t.Errorf("constant feature score: got %v, want NaN", scores[0])
	}
	if !math.IsInf(scores[1], 1) || pvalues[1] != 0 {
		t.Errorf("perfectly separating feature: got score %v p %v, want +Inf and 0", scores[1], pvalues[1])
	}
	if scores[2] != 0 || pvalues[2] != 1 {
		t.Errorf("noise feature with equal class means: got score %v p %v, want 0 and 1", scores[2], pvalues[2])
	}

	sel := NewSelectKBest()
	sel.K = 1
	if err := sel.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	assertMask(t, "support", sel.Support(), []bool{false, true, false})
	sel.K = 2
	if err := sel.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	// NaN ranks below real scores, so the constant column is dropped first.
	assertMask(t, "support k=2", sel.Support(), []bool{false, true, true})
}

func TestFRegression_ConstantAndPerfectFeatures(t *testing.T) {
	X := [][]float64{{7, 1}, {7, 2}, {7, 3}, {7, 4}, {7, 5}}
	y := []float64{2, 4, 6, 8, 10}
	scores, pvalues, err := FRegression(X, y)
	if err != nil {
		t.Fatalf("FRegression: %v", err)
	}
	if scores[0] != 0 || pvalues[0] != 1 {
		t.Errorf("constant feature: got %v, %v want 0, 1", scores[0], pvalues[0])
	}
	if scores[1] < 1e15 || pvalues[1] > 1e-12 {
		t.Errorf("perfect correlation: got %v, %v want a huge score and ~0 p-value", scores[1], pvalues[1])
	}
	if _, _, err := FRegression(X[:2], y[:2]); !errors.Is(err, ErrInvalidSelector) {
		t.Errorf("2 samples: got %v, want ErrInvalidSelector", err)
	}
}

func TestFClassif_InvalidTargets(t *testing.T) {
	X := [][]float64{{1}, {2}, {3}}
	if _, _, err := FClassif(X, []float64{1, 1, 1}); !errors.Is(err, ErrInvalidSelector) {
		t.Errorf("single class: got %v, want ErrInvalidSelector", err)
	}
	if _, _, err := FClassif(X, []float64{0, 1, 2}); !errors.Is(err, ErrInvalidSelector) {
		t.Errorf("as many classes as samples: got %v, want ErrInvalidSelector", err)
	}
}

func TestSelectKBest_TiesPreferLaterColumn(t *testing.T) {
	// Identical columns get identical scores; sklearn's stable argsort keeps
	// the later ones, and so must we.
	X := [][]float64{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}
	y := []float64{1, 2, 3, 4}
	sel := NewSelectKBest()
	sel.K = 2
	sel.Score = ScoreFRegression
	if err := sel.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	assertMask(t, "support", sel.Support(), []bool{false, true, true})
}

func TestSelectKBest_CustomScoreFuncAndKZero(t *testing.T) {
	X := [][]float64{{1, 9}, {2, 8}, {3, 7}}
	y := []float64{0, 1, 0}
	sel := NewSelectKBest()
	sel.K = 1
	sel.ScoreFunc = func(X [][]float64, y []float64) ([]float64, []float64, error) {
		return []float64{1, 2}, []float64{0.5, 0.5}, nil
	}
	if err := sel.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	assertMask(t, "support", sel.Support(), []bool{false, true})

	sel.K = 0
	if err := sel.Fit(X, y); err != nil {
		t.Fatalf("Fit k=0: %v", err)
	}
	out, err := sel.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}
	if len(out) != 3 || len(out[0]) != 0 {
		t.Errorf("k=0 should keep no columns, got shape %dx%d", len(out), len(out[0]))
	}

	sel.ScoreFunc = func(X [][]float64, y []float64) ([]float64, []float64, error) {
		return []float64{1}, []float64{1}, nil
	}
	sel.K = 1
	if err := sel.Fit(X, y); !errors.Is(err, ErrInvalidSelector) {
		t.Errorf("wrong score length: got %v, want ErrInvalidSelector", err)
	}
}

func TestSelectKBest_InvalidConfiguration(t *testing.T) {
	X := [][]float64{{1, 2}, {2, 1}, {3, 5}, {4, 3}}
	y := []float64{0, 1, 0, 1}
	for _, k := range []int{-1, 3} {
		sel := NewSelectKBest()
		sel.K = k
		if err := sel.Fit(X, y); !errors.Is(err, ErrInvalidSelector) {
			t.Errorf("k=%d: got %v, want ErrInvalidSelector", k, err)
		}
	}
	sel := NewSelectKBest()
	sel.K = 1
	sel.Score = "chi2"
	if err := sel.Fit(X, y); !errors.Is(err, ErrInvalidSelector) {
		t.Errorf("unknown score: got %v, want ErrInvalidSelector", err)
	}
}

func TestRFE_AgainstSklearn(t *testing.T) {
	fx := loadFixtures(t).RFE
	for name, c := range fx.Cases {
		c := c
		t.Run(name, func(t *testing.T) {
			var est interface {
				Fit([][]float64, []float64) error
				Predict([][]float64) ([]float64, error)
			}
			if name == "tree_step1" {
				tr := tree.NewDecisionTreeRegressor()
				tr.MaxDepth = 3
				est = tr
			} else {
				est = linear.NewLinearRegression()
			}
			rfe := NewRFE(est)
			rfe.NFeaturesToSelect = c.NFeaturesToSelect
			rfe.Step = c.Step
			out, err := rfe.FitTransform(fx.X, fx.Y)
			if err != nil {
				t.Fatalf("FitTransform: %v", err)
			}
			assertMask(t, "support", rfe.Support(), c.Support)
			got := rfe.Ranking()
			for j, want := range c.Ranking {
				if got[j] != want {
					t.Errorf("ranking: got %v, want %v", got, c.Ranking)
					break
				}
			}
			assertMatrix(t, "transformed", out, c.Transformed)
			pred, err := rfe.Predict(fx.X)
			if err != nil {
				t.Fatalf("Predict: %v", err)
			}
			for i, want := range c.Predictions {
				assertClose(t, "predictions", pred[i], want, 1e-6)
			}
		})
	}
}

func TestRFE_InvalidConfiguration(t *testing.T) {
	X := [][]float64{{1, 2, 3}, {2, 1, 5}, {3, 5, 1}, {4, 3, 2}, {5, 9, 4}}
	y := []float64{1, 2, 3, 4, 5}

	if err := (&RFE{Step: 1}).Fit(X, y); !errors.Is(err, ErrInvalidSelector) {
		t.Errorf("nil estimator: got %v, want ErrInvalidSelector", err)
	}
	r := NewRFE(linear.NewLinearRegression())
	r.Step = 0
	if err := r.Fit(X, y); !errors.Is(err, ErrInvalidSelector) {
		t.Errorf("step=0: got %v, want ErrInvalidSelector", err)
	}
	r = NewRFE(linear.NewLinearRegression())
	r.NFeaturesToSelect = -1
	if err := r.Fit(X, y); !errors.Is(err, ErrInvalidSelector) {
		t.Errorf("n_features_to_select<0: got %v, want ErrInvalidSelector", err)
	}
	r = NewRFE(linear.NewLinearRegression())
	if err := r.Fit([][]float64{{1}, {2}, {3}}, []float64{1, 2, 3}); !errors.Is(err, ErrInvalidSelector) {
		t.Errorf("single feature: got %v, want ErrInvalidSelector", err)
	}

	knn := neighbors.NewKNeighborsRegressor()
	knn.NNeighbors = 2
	r = NewRFE(knn)
	r.NFeaturesToSelect = 1
	if err := r.Fit(X, y); !errors.Is(err, ErrNoImportances) {
		t.Errorf("estimator without importances: got %v, want ErrNoImportances", err)
	}
}

func TestRFE_KeepingAllFeaturesStillFitsEstimator(t *testing.T) {
	X := [][]float64{{1, 2}, {2, 1}, {3, 5}, {4, 3}, {5, 9}}
	y := []float64{1, 2, 3, 4, 5}
	r := NewRFE(linear.NewLinearRegression())
	r.NFeaturesToSelect = 5 // more than n_features: nothing is eliminated
	if err := r.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	assertMask(t, "support", r.Support(), []bool{true, true})
	if _, err := r.Predict(X); err != nil {
		t.Errorf("Predict: %v", err)
	}
}

func TestSelectors_BeforeFit(t *testing.T) {
	X := [][]float64{{1, 2}}
	if _, err := NewVarianceThreshold().Transform(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("VarianceThreshold.Transform: got %v", err)
	}
	if _, err := NewSelectKBest().Transform(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("SelectKBest.Transform: got %v", err)
	}
	r := NewRFE(linear.NewLinearRegression())
	if _, err := r.Transform(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("RFE.Transform: got %v", err)
	}
	if _, err := r.Predict(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("RFE.Predict: got %v", err)
	}
	if NewVarianceThreshold().Support() != nil || NewSelectKBest().Scores() != nil || r.Ranking() != nil {
		t.Error("accessors should return nil before Fit")
	}
}

func TestSelectors_InvalidInput(t *testing.T) {
	good := [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
	y := []float64{0, 1, 0, 1}
	cases := []struct {
		name string
		X    [][]float64
		y    []float64
		want error
	}{
		{"empty", nil, nil, matutil.ErrEmptyInput},
		{"ragged", [][]float64{{1, 2}, {3}, {5, 6}, {7, 8}}, y, matutil.ErrRaggedInput},
		{"nan", [][]float64{{1, math.NaN()}, {3, 4}, {5, 6}, {7, 8}}, y, matutil.ErrContainsNaN},
		{"inf", [][]float64{{1, math.Inf(1)}, {3, 4}, {5, 6}, {7, 8}}, y, matutil.ErrContainsInf},
		{"mismatched", good, y[:3], matutil.ErrDimMismatch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sel := NewSelectKBest()
			sel.K = 1
			if err := sel.Fit(tc.X, tc.y); !errors.Is(err, tc.want) {
				t.Errorf("SelectKBest: got %v, want %v", err, tc.want)
			}
			rfe := NewRFE(linear.NewLinearRegression())
			if err := rfe.Fit(tc.X, tc.y); !errors.Is(err, tc.want) {
				t.Errorf("RFE: got %v, want %v", err, tc.want)
			}
			if tc.name != "mismatched" { // VarianceThreshold ignores y
				if err := NewVarianceThreshold().Fit(tc.X, tc.y); !errors.Is(err, tc.want) {
					t.Errorf("VarianceThreshold: got %v, want %v", err, tc.want)
				}
			}
		})
	}
}

func TestSelectors_TransformWrongFeatureCount(t *testing.T) {
	X := [][]float64{{1, 2, 3}, {2, 5, 1}, {3, 1, 4}, {4, 8, 1}}
	y := []float64{0, 1, 0, 1}
	sel := NewSelectKBest()
	sel.K = 2
	if err := sel.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if _, err := sel.Transform([][]float64{{1, 2}}); !errors.Is(err, matutil.ErrDimMismatch) {
		t.Errorf("got %v, want ErrDimMismatch", err)
	}
	if _, err := sel.Transform(nil); !errors.Is(err, matutil.ErrEmptyInput) {
		t.Errorf("got %v, want ErrEmptyInput", err)
	}
}

func TestSelectors_SaveLoadRoundTrip(t *testing.T) {
	fx := loadFixtures(t)
	dir := t.TempDir()

	vt := NewVarianceThreshold()
	vt.Threshold = 0.5
	if err := vt.Fit(fx.VarianceThreshold.X, nil); err != nil {
		t.Fatalf("VarianceThreshold.Fit: %v", err)
	}
	if err := vt.Save(filepath.Join(dir, "vt.gob")); err != nil {
		t.Fatalf("VarianceThreshold.Save: %v", err)
	}
	vt2, err := LoadVarianceThreshold(filepath.Join(dir, "vt.gob"))
	if err != nil {
		t.Fatalf("LoadVarianceThreshold: %v", err)
	}
	assertMask(t, "vt support", vt2.Support(), vt.Support())
	if vt2.Threshold != 0.5 || len(vt2.Variances()) != len(vt.Variances()) {
		t.Errorf("VarianceThreshold state not restored: %+v", vt2)
	}

	sel := NewSelectKBest()
	sel.K = 3
	if err := sel.Fit(fx.FClassif.X, fx.FClassif.Y); err != nil {
		t.Fatalf("SelectKBest.Fit: %v", err)
	}
	if err := sel.Save(filepath.Join(dir, "skb.gob")); err != nil {
		t.Fatalf("SelectKBest.Save: %v", err)
	}
	sel2, err := LoadSelectKBest(filepath.Join(dir, "skb.gob"))
	if err != nil {
		t.Fatalf("LoadSelectKBest: %v", err)
	}
	want, _ := sel.Transform(fx.FClassif.X)
	got, err := sel2.Transform(fx.FClassif.X)
	if err != nil {
		t.Fatalf("Transform after load: %v", err)
	}
	assertMatrix(t, "skb transform", got, want)
	if sel2.K != 3 || sel2.Score != ScoreFClassif || len(sel2.Scores()) != 8 || len(sel2.PValues()) != 8 {
		t.Errorf("SelectKBest state not restored: %+v", sel2)
	}

	rfe := NewRFE(linear.NewLinearRegression())
	rfe.NFeaturesToSelect = 3
	if err := rfe.Fit(fx.RFE.X, fx.RFE.Y); err != nil {
		t.Fatalf("RFE.Fit: %v", err)
	}
	if err := rfe.Save(filepath.Join(dir, "rfe.gob")); err != nil {
		t.Fatalf("RFE.Save: %v", err)
	}
	rfe2, err := LoadRFE(filepath.Join(dir, "rfe.gob"))
	if err != nil {
		t.Fatalf("LoadRFE: %v", err)
	}
	wantX, _ := rfe.Transform(fx.RFE.X)
	gotX, err := rfe2.Transform(fx.RFE.X)
	if err != nil {
		t.Fatalf("Transform after load: %v", err)
	}
	assertMatrix(t, "rfe transform", gotX, wantX)
	for j, r := range rfe.Ranking() {
		if rfe2.Ranking()[j] != r {
			t.Errorf("ranking not restored: %v vs %v", rfe2.Ranking(), rfe.Ranking())
			break
		}
	}
	if _, err := rfe2.Predict(fx.RFE.X); err == nil {
		t.Error("Predict on a loaded RFE without an estimator should fail")
	}

	// A file of one kind must not load as another.
	if _, err := LoadSelectKBest(filepath.Join(dir, "vt.gob")); err == nil {
		t.Error("loading a VarianceThreshold file as SelectKBest should fail")
	}
}

func TestSelectors_SaveBeforeFitAndBadFiles(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.gob")
	if err := NewVarianceThreshold().Save(path); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("VarianceThreshold.Save: got %v", err)
	}
	if err := NewSelectKBest().Save(path); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("SelectKBest.Save: got %v", err)
	}
	if err := NewRFE(linear.NewLinearRegression()).Save(path); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("RFE.Save: got %v", err)
	}
	if _, err := LoadVarianceThreshold(filepath.Join(dir, "missing.gob")); err == nil {
		t.Error("loading a nonexistent file should fail")
	}
	junk := filepath.Join(dir, "junk.gob")
	if err := os.WriteFile(junk, []byte("not a gob"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadRFE(junk); err == nil {
		t.Error("loading a corrupt file should fail")
	}
}

// RFE can rank features with any of the linear models, not just LinearRegression.
func TestRFE_WithRegularizedLinearModels(t *testing.T) {
	// Regression: only columns 0-2 drive y among 8 columns.
	X := make([][]float64, 150)
	y := make([]float64, len(X))
	yc := make([]float64, len(X))
	for i := range X {
		row := make([]float64, 8)
		for j := range row {
			row[j] = math.Sin(float64(i*(j+2))*0.9) + 0.4*math.Cos(float64(i*(j+5))*1.7)
		}
		X[i] = row
		y[i] = 3*row[0] - 2*row[1] + 1.5*row[2]
		if row[0]-row[1] > 0 { // classification labels depend on columns 0 and 1 only
			yc[i] = 1
		}
	}
	ridge, lasso, enet := linear.NewRidge(), linear.NewLasso(), linear.NewElasticNet()
	lasso.Alpha, enet.Alpha = 0.01, 0.01
	for name, est := range map[string]core.Estimator{"ridge": ridge, "lasso": lasso, "elastic_net": enet} {
		r := NewRFE(est)
		r.NFeaturesToSelect = 3
		if err := r.Fit(X, y); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got := r.SupportIndices(); !reflect.DeepEqual(got, []int{0, 1, 2}) {
			t.Errorf("%s: RFE kept %v, want [0 1 2]", name, got)
		}
	}

	r := NewRFE(linear.NewLogisticRegression())
	r.NFeaturesToSelect = 2
	if err := r.Fit(X, yc); err != nil {
		t.Fatal(err)
	}
	if got := r.SupportIndices(); !reflect.DeepEqual(got, []int{0, 1}) {
		t.Errorf("logistic RFE kept %v, want [0 1]", got)
	}
}
