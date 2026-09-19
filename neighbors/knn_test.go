package neighbors

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type knnFixture struct {
	X         [][]float64 `json:"X"`
	Y         []float64   `json:"y"`
	XTest     [][]float64 `json:"X_test"`
	PredTrain []float64   `json:"pred_train"`
	PredTest  []float64   `json:"pred_test"`
	ProbaTest [][]float64 `json:"proba_test"`
	Classes   []float64   `json:"classes"`
}

func loadKNNFixtures(t *testing.T) map[string]knnFixture {
	t.Helper()
	path := filepath.Join("testdata", "neighbors_fixtures.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read fixtures: %v", err)
	}
	var fixtures map[string]knnFixture
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatalf("failed to parse fixtures: %v", err)
	}
	return fixtures
}

func assertClose(t *testing.T, name string, got, want float64, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %.12f, want %.12f (diff %.2e)", name, got, want, math.Abs(got-want))
	}
}

func assertSliceClose(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch, got %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		assertClose(t, fmt.Sprintf("%s[%d]", name, i), got[i], want[i], tol)
	}
}

func assertMatrixClose(t *testing.T, name string, got, want [][]float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: row count mismatch, got %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("%s[%d]: col count mismatch, got %d, want %d", name, i, len(got[i]), len(want[i]))
		}
		for j := range want[i] {
			assertClose(t, fmt.Sprintf("%s[%d][%d]", name, i, j), got[i][j], want[i][j], tol)
		}
	}
}

func TestKNeighborsRegressor_AgainstSklearn(t *testing.T) {
	fixtures := loadKNNFixtures(t)
	const tol = 1e-9

	specs := []struct {
		key string
		k   int
		w   string
		p   float64
	}{
		{"knn_reg_k3_uniform", 3, "uniform", 2},
		{"knn_reg_k5_uniform", 5, "uniform", 2},
		{"knn_reg_k5_distance", 5, "distance", 2},
		{"knn_reg_k7_manhattan", 7, "uniform", 1},
		{"knn_reg_tie", 3, "uniform", 2},
	}
	for _, spec := range specs {
		t.Run(spec.key, func(t *testing.T) {
			fx := fixtures[spec.key]
			model := NewKNeighborsRegressor()
			model.NNeighbors = spec.k
			model.Weights = spec.w
			model.P = spec.p
			if err := model.Fit(fx.X, fx.Y); err != nil {
				t.Fatalf("Fit: %v", err)
			}
			predTrain, err := model.Predict(fx.X)
			if err != nil {
				t.Fatalf("Predict(train): %v", err)
			}
			assertSliceClose(t, "pred_train", predTrain, fx.PredTrain, tol)
			predTest, err := model.Predict(fx.XTest)
			if err != nil {
				t.Fatalf("Predict(test): %v", err)
			}
			assertSliceClose(t, "pred_test", predTest, fx.PredTest, tol)
		})
	}
}

func TestKNeighborsClassifier_AgainstSklearn(t *testing.T) {
	fixtures := loadKNNFixtures(t)
	const tol = 1e-9

	specs := []struct {
		key string
		k   int
		w   string
		p   float64
	}{
		{"knn_cls_k3_uniform", 3, "uniform", 2},
		{"knn_cls_k5_uniform", 5, "uniform", 2},
		{"knn_cls_k5_distance", 5, "distance", 2},
		{"knn_cls_k7_manhattan", 7, "uniform", 1},
		{"knn_cls_tie", 3, "uniform", 2},
	}
	for _, spec := range specs {
		t.Run(spec.key, func(t *testing.T) {
			fx := fixtures[spec.key]
			model := NewKNeighborsClassifier()
			model.NNeighbors = spec.k
			model.Weights = spec.w
			model.P = spec.p
			if err := model.Fit(fx.X, fx.Y); err != nil {
				t.Fatalf("Fit: %v", err)
			}
			if len(model.Classes()) != len(fx.Classes) {
				t.Fatalf("classes: got %d, want %d", len(model.Classes()), len(fx.Classes))
			}
			assertSliceClose(t, "classes", model.Classes(), fx.Classes, tol)
			predTrain, err := model.Predict(fx.X)
			if err != nil {
				t.Fatalf("Predict(train): %v", err)
			}
			assertSliceClose(t, "pred_train", predTrain, fx.PredTrain, tol)
			predTest, err := model.Predict(fx.XTest)
			if err != nil {
				t.Fatalf("Predict(test): %v", err)
			}
			assertSliceClose(t, "pred_test", predTest, fx.PredTest, tol)
			proba, err := model.PredictProba(fx.XTest)
			if err != nil {
				t.Fatalf("PredictProba(test): %v", err)
			}
			assertMatrixClose(t, "proba_test", proba, fx.ProbaTest, tol)
		})
	}
}

func TestKNeighbors_Validation(t *testing.T) {
	X := [][]float64{{1, 2}, {3, 4}}
	y := []float64{1.0, 2.0}

	model := NewKNeighborsRegressor()
	if _, err := model.Predict(X); err == nil {
		t.Error("expected error predicting before fit")
	}

	model.NNeighbors = 0
	if err := model.Fit(X, y); err == nil {
		t.Error("expected error for n_neighbors = 0")
	}

	model = NewKNeighborsRegressor()
	model.Weights = "bogus"
	if err := model.Fit(X, y); err == nil {
		t.Error("expected error for invalid weights")
	}

	model = NewKNeighborsRegressor()
	model.P = 0.5
	if err := model.Fit(X, y); err == nil {
		t.Error("expected error for p < 1")
	}

	classifier := NewKNeighborsClassifier()
	if _, err := classifier.Predict(X); err == nil {
		t.Error("expected error predicting before fit")
	}
}

func TestKNeighbors_DistanceWeightsZeroDist(t *testing.T) {
	X := [][]float64{{1, 1}, {2, 2}, {10, 10}}
	y := []float64{1.0, 2.0, 20.0}
	model := NewKNeighborsRegressor()
	model.NNeighbors = 3
	model.Weights = "distance"
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	preds, err := model.Predict([][]float64{{2, 2}})
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	assertClose(t, "zero-dist neighbor wins", preds[0], 2.0, 1e-12)
}

func TestKNeighbors_ClassificationTieBreaksLowestClass(t *testing.T) {
	X := [][]float64{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
		{5, 5}, {5, 6}, {6, 5}, {6, 6},
	}
	y := []float64{0, 0, 1, 1, 0, 1, 1, 1}
	model := NewKNeighborsClassifier()
	model.NNeighbors = 2
	model.Weights = "uniform"
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	// Query equidistant from (0,0)[class 0] and (1,1)[class 1]: tie between
	// classes, lowest class label (0) must win.
	preds, err := model.Predict([][]float64{{0.5, 0.5}})
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	assertClose(t, "tie broken to lowest class", preds[0], 0.0, 0)
}

func TestKNeighbors_SaveLoad(t *testing.T) {
	X := [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
	y := []float64{1.0, 2.0, 3.0, 4.0}
	reg := NewKNeighborsRegressor()
	reg.NNeighbors = 2
	reg.Weights = "distance"
	if err := reg.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	path := filepath.Join(t.TempDir(), "reg.gob")
	if err := reg.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadKNeighborsRegressor(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	predsWant, _ := reg.Predict(X)
	predsGot, err := loaded.Predict(X)
	if err != nil {
		t.Fatalf("Predict after load: %v", err)
	}
	assertSliceClose(t, "loaded preds", predsGot, predsWant, 1e-12)

	cls := NewKNeighborsClassifier()
	cls.NNeighbors = 2
	cls.Weights = "distance"
	yc := []float64{0.0, 1.0, 0.0, 1.0}
	if err := cls.Fit(X, yc); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	cpath := filepath.Join(t.TempDir(), "cls.gob")
	if err := cls.Save(cpath); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loadedCls, err := LoadKNeighborsClassifier(cpath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	predsWant, _ = cls.Predict(X)
	predsGot, err = loadedCls.Predict(X)
	if err != nil {
		t.Fatalf("Predict after load: %v", err)
	}
	assertSliceClose(t, "loaded class preds", predsGot, predsWant, 1e-12)

	// Cross-kind load must fail.
	if _, err := LoadKNeighborsRegressor(cpath); err == nil {
		t.Error("expected error loading classifier file as regressor")
	}
}
func TestKNN_NNeighborsExceedsSamples(t *testing.T) {
	X := [][]float64{{0}, {1}, {2}}
	y := []float64{0, 1, 0}

	c := NewKNeighborsClassifier()
	c.NNeighbors = 10
	if err := c.Fit(X, y); !errors.Is(err, ErrInvalidKNN) {
		t.Fatalf("classifier Fit: got %v, want ErrInvalidKNN", err)
	}

	r := NewKNeighborsRegressor()
	r.NNeighbors = 4
	if err := r.Fit(X, y); !errors.Is(err, ErrInvalidKNN) {
		t.Fatalf("regressor Fit: got %v, want ErrInvalidKNN", err)
	}

	// n_neighbors == n_samples is the largest valid value and must not panic.
	c.NNeighbors = 3
	if err := c.Fit(X, y); err != nil {
		t.Fatalf("Fit with n_neighbors == n_samples: %v", err)
	}
	if _, err := c.Predict(X); err != nil {
		t.Fatalf("Predict: %v", err)
	}
}
