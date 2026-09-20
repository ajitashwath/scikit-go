package ensemble

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/internal/matutil"
)

type predFixture struct {
	Classes     []float64   `json:"classes"`
	PredTest    []float64   `json:"pred_test"`
	PredTrain   []float64   `json:"pred_train"`
	ProbaTest   [][]float64 `json:"proba_test"`
	Importances []float64   `json:"importances"`
	R2Test      float64     `json:"r2_test"`
	AccTest     float64     `json:"accuracy_test"`
}

type dataFixture struct {
	X            [][]float64 `json:"X"`
	Y            []float64   `json:"y"`
	XTest        [][]float64 `json:"X_test"`
	YTest        []float64   `json:"y_test"`
	NoBootstrap  predFixture `json:"nobootstrap"`
	WithBootstrp predFixture `json:"bootstrap"`
}

func loadEnsembleFixtures(t *testing.T) map[string]dataFixture {
	t.Helper()
	path := filepath.Join("testdata", "ensemble_fixtures.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read fixtures at %s: %v", path, err)
	}
	var fx map[string]dataFixture
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("failed to parse fixtures: %v", err)
	}
	return fx
}

func assertClose(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %.10f, want %.10f (diff %.2e)", name, got, want, math.Abs(got-want))
	}
}

// With bootstrap off and every feature considered, each tree is the same CART
// tree, so the forest is deterministic and must match sklearn exactly.
func TestRandomForestRegressor_AgainstSklearn(t *testing.T) {
	fx := loadEnsembleFixtures(t)["regression"]

	t.Run("nobootstrap", func(t *testing.T) {
		rf := NewRandomForestRegressor()
		rf.NTrees = 5
		rf.Bootstrap = false
		if err := rf.Fit(fx.X, fx.Y); err != nil {
			t.Fatalf("Fit: %v", err)
		}
		pred, err := rf.Predict(fx.XTest)
		if err != nil {
			t.Fatalf("Predict: %v", err)
		}
		for i, want := range fx.NoBootstrap.PredTest {
			assertClose(t, "pred_test", pred[i], want, 1e-6)
		}
		for i, want := range fx.NoBootstrap.Importances {
			assertClose(t, "importances", rf.FeatureImportances()[i], want, 1e-6)
		}
	})

	// Bootstrap forests differ in RNG stream from sklearn; the held-out R^2 of
	// both must land in the same neighbourhood.
	t.Run("bootstrap", func(t *testing.T) {
		rf := NewRandomForestRegressor()
		rf.NTrees = 100
		rf.MaxDepth = 8
		rf.Seed = 0
		if err := rf.Fit(fx.X, fx.Y); err != nil {
			t.Fatalf("Fit: %v", err)
		}
		r2, err := rf.Score(fx.XTest, fx.YTest)
		if err != nil {
			t.Fatalf("Score: %v", err)
		}
		assertClose(t, "r2_test", r2, fx.WithBootstrp.R2Test, 0.05)
		if argmax(rf.FeatureImportances()) != argmax(fx.WithBootstrp.Importances) {
			t.Errorf("most important feature differs: got %v, sklearn %v",
				rf.FeatureImportances(), fx.WithBootstrp.Importances)
		}
	})
}

func TestRandomForestClassifier_AgainstSklearn(t *testing.T) {
	fx := loadEnsembleFixtures(t)["classification"]

	t.Run("nobootstrap", func(t *testing.T) {
		rf := NewRandomForestClassifier()
		rf.NTrees = 5
		rf.Bootstrap = false
		rf.MaxFeatures = len(fx.X[0])
		if err := rf.Fit(fx.X, fx.Y); err != nil {
			t.Fatalf("Fit: %v", err)
		}
		classes := rf.Classes()
		for i, want := range fx.NoBootstrap.Classes {
			assertClose(t, "classes", classes[i], want, 0)
		}
		pred, err := rf.Predict(fx.XTest)
		if err != nil {
			t.Fatalf("Predict: %v", err)
		}
		for i, want := range fx.NoBootstrap.PredTest {
			assertClose(t, "pred_test", pred[i], want, 0)
		}
		proba, err := rf.PredictProba(fx.XTest)
		if err != nil {
			t.Fatalf("PredictProba: %v", err)
		}
		for i, row := range fx.NoBootstrap.ProbaTest {
			for k, want := range row {
				assertClose(t, "proba_test", proba[i][k], want, 1e-6)
			}
		}
		for i, want := range fx.NoBootstrap.Importances {
			assertClose(t, "importances", rf.FeatureImportances()[i], want, 1e-6)
		}
	})

	t.Run("bootstrap", func(t *testing.T) {
		rf := NewRandomForestClassifier()
		rf.NTrees = 100
		rf.MaxDepth = 8
		rf.Seed = 0
		if err := rf.Fit(fx.X, fx.Y); err != nil {
			t.Fatalf("Fit: %v", err)
		}
		acc, err := rf.Score(fx.XTest, fx.YTest)
		if err != nil {
			t.Fatalf("Score: %v", err)
		}
		assertClose(t, "accuracy_test", acc, fx.WithBootstrp.AccTest, 0.1)
	})
}

func argmax(v []float64) int {
	best := 0
	for i := range v {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func TestRandomForest_DeterministicAcrossWorkerCounts(t *testing.T) {
	X, y, err := datasets.MakeClassification(120, 6, 3, 7)
	if err != nil {
		t.Fatalf("MakeClassification: %v", err)
	}
	fit := func(jobs int, seed int64) []float64 {
		rf := NewRandomForestClassifier()
		rf.NTrees = 20
		rf.Seed = seed
		rf.NJobs = jobs
		if err := rf.Fit(X, y); err != nil {
			t.Fatalf("Fit: %v", err)
		}
		p, err := rf.PredictProba(X)
		if err != nil {
			t.Fatalf("PredictProba: %v", err)
		}
		var flat []float64
		for _, row := range p {
			flat = append(flat, row...)
		}
		return flat
	}
	serial, parallel, again := fit(1, 3), fit(8, 3), fit(1, 3)
	other := fit(1, 4)
	same := func(a, b []float64) bool {
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
		return true
	}
	if !same(serial, parallel) {
		t.Error("forest depends on the number of workers")
	}
	if !same(serial, again) {
		t.Error("forest is not reproducible for a fixed seed")
	}
	if same(serial, other) {
		t.Error("different seeds produced identical forests")
	}
}

// A bootstrap sample can miss a rare class; probabilities must still line up
// with the forest-wide class list and sum to one.
func TestRandomForestClassifier_MissingClassInBootstrap(t *testing.T) {
	var X [][]float64
	var y []float64
	for i := 0; i < 30; i++ {
		X = append(X, []float64{float64(i), float64(i % 3)})
		y = append(y, float64(i%2))
	}
	X = append(X, []float64{100, 1})
	y = append(y, 2) // single sample of class 2

	rf := NewRandomForestClassifier()
	rf.NTrees = 40
	if err := rf.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if got := len(rf.Classes()); got != 3 {
		t.Fatalf("Classes: got %d classes, want 3", got)
	}
	missing := 0
	for _, idx := range rf.classIdx {
		if len(idx) < 3 {
			missing++
		}
	}
	if missing == 0 {
		t.Fatal("test data never produced a bootstrap sample missing class 2; pick a rarer class")
	}
	proba, err := rf.PredictProba(X)
	if err != nil {
		t.Fatalf("PredictProba: %v", err)
	}
	for i, row := range proba {
		if len(row) != 3 {
			t.Fatalf("row %d has %d columns, want 3", i, len(row))
		}
		var sum float64
		for _, p := range row {
			sum += p
		}
		assertClose(t, "row sum", sum, 1, 1e-9)
	}
}

func TestRandomForest_PredictBeforeFit(t *testing.T) {
	X := [][]float64{{1, 2}}
	if _, err := NewRandomForestRegressor().Predict(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("regressor Predict: got %v, want ErrNotFitted", err)
	}
	if _, err := NewRandomForestClassifier().Predict(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("classifier Predict: got %v, want ErrNotFitted", err)
	}
	if _, err := NewRandomForestClassifier().PredictProba(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("classifier PredictProba: got %v, want ErrNotFitted", err)
	}
	if NewRandomForestRegressor().FeatureImportances() != nil {
		t.Error("FeatureImportances before Fit should be nil")
	}
}

func TestRandomForest_InvalidInput(t *testing.T) {
	good := [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
	y := []float64{0, 1, 0, 1}
	cases := []struct {
		name string
		X    [][]float64
		y    []float64
		want error
	}{
		{"empty", nil, nil, matutil.ErrEmptyInput},
		{"mismatched", good, y[:3], matutil.ErrDimMismatch},
		{"ragged", [][]float64{{1, 2}, {3}, {5, 6}, {7, 8}}, y, matutil.ErrRaggedInput},
		{"nan", [][]float64{{1, math.NaN()}, {3, 4}, {5, 6}, {7, 8}}, y, matutil.ErrContainsNaN},
		{"inf", [][]float64{{1, math.Inf(1)}, {3, 4}, {5, 6}, {7, 8}}, y, matutil.ErrContainsInf},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := NewRandomForestRegressor().Fit(tc.X, tc.y); !errors.Is(err, tc.want) {
				t.Errorf("regressor: got %v, want %v", err, tc.want)
			}
			if err := NewRandomForestClassifier().Fit(tc.X, tc.y); !errors.Is(err, tc.want) {
				t.Errorf("classifier: got %v, want %v", err, tc.want)
			}
		})
	}
}

func TestRandomForest_InvalidHyperparameters(t *testing.T) {
	X := [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
	y := []float64{0, 1, 0, 1}
	mutations := map[string]func(n, split, leaf, feats *int){
		"n_trees=0":           func(n, _, _, _ *int) { *n = 0 },
		"min_samples_split=1": func(_, s, _, _ *int) { *s = 1 },
		"min_samples_leaf=0":  func(_, _, l, _ *int) { *l = 0 },
		"max_features=-1":     func(_, _, _, f *int) { *f = -1 },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			r := NewRandomForestRegressor()
			mutate(&r.NTrees, &r.MinSamplesSplit, &r.MinSamplesLeaf, &r.MaxFeatures)
			if err := r.Fit(X, y); !errors.Is(err, ErrInvalidForest) {
				t.Errorf("regressor: got %v, want ErrInvalidForest", err)
			}
			c := NewRandomForestClassifier()
			mutate(&c.NTrees, &c.MinSamplesSplit, &c.MinSamplesLeaf, &c.MaxFeatures)
			if err := c.Fit(X, y); !errors.Is(err, ErrInvalidForest) {
				t.Errorf("classifier: got %v, want ErrInvalidForest", err)
			}
		})
	}
	bad := NewRandomForestRegressor()
	bad.Criterion = "nonsense"
	if err := bad.Fit(X, y); err == nil {
		t.Error("an unknown criterion should fail Fit")
	}
}

func TestRandomForest_PredictWrongFeatureCount(t *testing.T) {
	X, y, _ := datasets.MakeRegression(40, 3, 0.1, 1)
	rf := NewRandomForestRegressor()
	rf.NTrees = 5
	if err := rf.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if _, err := rf.Predict([][]float64{{1, 2}}); !errors.Is(err, matutil.ErrDimMismatch) {
		t.Errorf("got %v, want ErrDimMismatch", err)
	}
	if _, err := rf.Predict(nil); !errors.Is(err, matutil.ErrEmptyInput) {
		t.Errorf("got %v, want ErrEmptyInput", err)
	}
}

func TestRandomForest_FeatureImportancesSumToOne(t *testing.T) {
	X, y, _ := datasets.MakeRegression(80, 5, 0.1, 2)
	rf := NewRandomForestRegressor()
	rf.NTrees = 10
	if err := rf.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	var sum float64
	for _, v := range rf.FeatureImportances() {
		if v < 0 {
			t.Errorf("negative importance %v", v)
		}
		sum += v
	}
	assertClose(t, "importance sum", sum, 1, 1e-9)
}

func TestRandomForestRegressor_SaveLoadRoundTrip(t *testing.T) {
	X, y, _ := datasets.MakeRegression(60, 4, 0.5, 3)
	rf := NewRandomForestRegressor()
	rf.NTrees = 8
	rf.MaxDepth = 5
	rf.Seed = 11
	if err := rf.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	path := filepath.Join(t.TempDir(), "rfr.gob")
	if err := rf.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadRandomForestRegressor(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	want, _ := rf.Predict(X)
	got, err := loaded.Predict(X)
	if err != nil {
		t.Fatalf("Predict after load: %v", err)
	}
	for i := range want {
		assertClose(t, "pred", got[i], want[i], 0)
	}
	if loaded.NTrees != 8 || loaded.MaxDepth != 5 || loaded.Seed != 11 || loaded.NEstimators() != 8 {
		t.Errorf("hyperparameters not restored: %+v", loaded)
	}
}

func TestRandomForestClassifier_SaveLoadRoundTrip(t *testing.T) {
	X, y, _ := datasets.MakeClassification(90, 5, 3, 4)
	rf := NewRandomForestClassifier()
	rf.NTrees = 8
	rf.Seed = 5
	if err := rf.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	path := filepath.Join(t.TempDir(), "rfc.gob")
	if err := rf.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadRandomForestClassifier(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	want, _ := rf.PredictProba(X)
	got, err := loaded.PredictProba(X)
	if err != nil {
		t.Fatalf("PredictProba after load: %v", err)
	}
	for i := range want {
		for k := range want[i] {
			assertClose(t, "proba", got[i][k], want[i][k], 0)
		}
	}
	// A regressor file must not load as a classifier and vice versa.
	if _, err := LoadRandomForestRegressor(path); err == nil {
		t.Error("loading a classifier file as a regressor should fail")
	}
}

func TestRandomForest_SaveBeforeFit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.gob")
	if err := NewRandomForestRegressor().Save(path); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("regressor: got %v, want ErrNotFitted", err)
	}
	if err := NewRandomForestClassifier().Save(path); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("classifier: got %v, want ErrNotFitted", err)
	}
}

func TestLoadRandomForest_BadFiles(t *testing.T) {
	dir := t.TempDir()
	if _, err := LoadRandomForestRegressor(filepath.Join(dir, "missing.gob")); err == nil {
		t.Error("loading a nonexistent file should fail")
	}
	junk := filepath.Join(dir, "junk.gob")
	if err := os.WriteFile(junk, []byte("not a gob"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadRandomForestClassifier(junk); err == nil {
		t.Error("loading a corrupt file should fail")
	}
}
