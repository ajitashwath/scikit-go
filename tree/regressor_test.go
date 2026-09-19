package tree

import (
	"math"
	"path/filepath"
	"testing"
)

// TestDecisionTreeRegressor_AgainstSklearn checks prediction and feature-importance
// parity with sklearn.tree.DecisionTreeRegressor on the committed fixtures.
func TestDecisionTreeRegressor_AgainstSklearn(t *testing.T) {
	fixtures := loadTreeFixtures(t)
	const tol = 1e-9
	for _, name := range []string{"reg_mse", "reg_mae", "reg_mse_depth3"} {
		fx := fixtures[name]
		model := NewDecisionTreeRegressor()
		switch name {
		case "reg_mse":
			model.Criterion = "mse"
		case "reg_mse_depth3":
			model.Criterion = "mse"
			model.MaxDepth = 3
		case "reg_mae":
			model.Criterion = "mae"
		}
		if err := model.Fit(fx.X, fx.Y); err != nil {
			t.Fatalf("%s Fit: %v", name, err)
		}
		preds, err := model.Predict(fx.X)
		if err != nil {
			t.Fatalf("%s Predict(train): %v", name, err)
		}
		assertSliceClose(t, name+" pred_train", preds, fx.PredTrain, tol)

		testPreds, err := model.Predict(fx.XTest)
		if err != nil {
			t.Fatalf("%s Predict(test): %v", name, err)
		}
		assertSliceClose(t, name+" pred_test", testPreds, fx.PredTest, tol)

		importances := model.FeatureImportances()
		assertSliceClose(t, name+" importances", importances, fx.FeatureImportances, 1e-6)
	}
}

func TestDecisionTreeRegressor_ConstantTarget(t *testing.T) {
	// A constant target must yield a single leaf predicting the constant.
	model := NewDecisionTreeRegressor()
	X := [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
	y := []float64{9, 9, 9, 9}
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	preds, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	for i, p := range preds {
		if p != 9 {
			t.Errorf("pred[%d] = %v, want 9", i, p)
		}
	}
	imp := model.FeatureImportances()
	for i, v := range imp {
		if v != 0 {
			t.Errorf("importances[%d] = %v, want 0 for single-node tree", i, v)
		}
	}
}

func TestDecisionTreeRegressor_StepFunction(t *testing.T) {
	// y = sign(x - 0.5): a single threshold split on one feature must recover it
	// perfectly with enough depth.
	model := NewDecisionTreeRegressor()
	X := make([][]float64, 100)
	y := make([]float64, 100)
	for i := 0; i < 100; i++ {
		x := float64(i-50) / 100.0
		X[i] = []float64{x}
		if x > 0.5 {
			y[i] = 1.0
		} else {
			y[i] = -1.0
		}
	}
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	preds, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	for i, p := range preds {
		if math.Abs(p-y[i]) > 1e-12 {
			t.Errorf("pred[%d] = %v, want %v", i, p, y[i])
		}
	}
}

func TestDecisionTreeRegressor_MAEPredictsMedian(t *testing.T) {
	// With max_depth=1 the mae criterion must predict the leaf statistical median.
	model := NewDecisionTreeRegressor()
	model.Criterion = "mae"
	model.MaxDepth = 1
	X := [][]float64{{0}, {1}, {2}, {3}, {100}}
	y := []float64{1, 3, 5, 7, 9}
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	// Best split is between x=1 and x=2 (threshold 1.5): left {1,3}, right {5,7,9}.
	preds, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if preds[0] != 2 || preds[1] != 2 {
		t.Errorf("left leaf preds = %v, want [2 2] (median of [1 3])", preds[:2])
	}
	if preds[4] != 7 {
		t.Errorf("right leaf pred = %v, want 7 (median of [5 7 9])", preds[4])
	}
}

func TestDecisionTreeRegressor_MinSamplesLeaf(t *testing.T) {
	model := NewDecisionTreeRegressor()
	model.MinSamplesLeaf = 10
	X := make([][]float64, 100)
	y := make([]float64, 100)
	for i := 0; i < 100; i++ {
		X[i] = []float64{float64(i)}
		y[i] = float64(i)
	}
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	// No node may have fewer than min_samples_leaf samples.
	for i, n := range model.impl.nodes {
		if n.NSamples < 10 {
			t.Errorf("node %d has %d samples < min_samples_leaf=10", i, n.NSamples)
		}
	}
}

func TestDecisionTreeRegressor_SaveLoadRoundTrip(t *testing.T) {
	model := NewDecisionTreeRegressor()
	model.Criterion = "mae"
	model.MaxDepth = 4
	X := make([][]float64, 50)
	y := make([]float64, 50)
	for i := 0; i < 50; i++ {
		X[i] = []float64{float64(i % 7), float64(i % 3)}
		y[i] = float64(i % 5)
	}
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	tmp := filepath.Join(t.TempDir(), "tree.bin")
	if err := model.Save(tmp); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadDecisionTreeRegressor(tmp)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	origPreds, _ := model.Predict(X)
	loadedPreds, err := loaded.Predict(X)
	if err != nil {
		t.Fatalf("loaded Predict: %v", err)
	}
	assertSliceClose(t, "loaded preds", loadedPreds, origPreds, 1e-12)
	if loaded.MaxDepth != 4 || loaded.Criterion != "mae" {
		t.Errorf("params not preserved: %+v", loaded)
	}
}

func TestDecisionTreeRegressor_SaveBeforeFit(t *testing.T) {
	model := NewDecisionTreeRegressor()
	if err := model.Save(filepath.Join(t.TempDir(), "x.bin")); err == nil {
		t.Fatal("expected error saving unfitted model")
	}
}

func TestDecisionTreeRegressor_PredictBeforeFit(t *testing.T) {
	model := NewDecisionTreeRegressor()
	if _, err := model.Predict([][]float64{{1, 2}}); err == nil {
		t.Fatal("expected error predicting before fit")
	}
}

func TestDecisionTreeRegressor_EdgeCases(t *testing.T) {
	model := NewDecisionTreeRegressor()
	if err := model.Fit([][]float64{}, []float64{}); err == nil {
		t.Error("expected error on empty input")
	}
	if err := model.Fit([][]float64{{1}, {2}}, []float64{1, 2, 3}); err == nil {
		t.Error("expected error on mismatched dims")
	}
	if err := model.Fit([][]float64{{1, 2}, {3}}, []float64{1, 2}); err == nil {
		t.Error("expected error on ragged input")
	}
	if err := model.Fit([][]float64{{math.NaN()}, {1}}, []float64{1, 2}); err == nil {
		t.Error("expected error on NaN input")
	}
}

func TestDecisionTreeRegressor_InvalidCriterion(t *testing.T) {
	model := NewDecisionTreeRegressor()
	model.Criterion = "bogus"
	if err := model.Fit([][]float64{{1}, {2}}, []float64{1, 2}); err == nil {
		t.Error("expected error on invalid criterion")
	}
}

func TestDecisionTreeRegressor_PredictWrongFeatureCount(t *testing.T) {
	model := NewDecisionTreeRegressor()
	if err := model.Fit([][]float64{{1, 2}, {3, 4}}, []float64{1, 2}); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if _, err := model.Predict([][]float64{{1, 2, 3}}); err == nil {
		t.Error("expected error on wrong feature count")
	}
}
func TestTree_MarshalBinaryRoundTrip(t *testing.T) {
	X := [][]float64{{0, 1}, {1, 0}, {2, 3}, {3, 2}, {4, 5}, {5, 4}}
	y := []float64{0, 0, 1, 1, 2, 2}

	reg := NewDecisionTreeRegressor()
	if err := reg.Fit(X, y); err != nil {
		t.Fatalf("regressor Fit: %v", err)
	}
	data, err := reg.MarshalBinary()
	if err != nil {
		t.Fatalf("regressor MarshalBinary: %v", err)
	}
	var reg2 DecisionTreeRegressor
	if err := reg2.UnmarshalBinary(data); err != nil {
		t.Fatalf("regressor UnmarshalBinary: %v", err)
	}
	want, _ := reg.Predict(X)
	got, err := reg2.Predict(X)
	if err != nil {
		t.Fatalf("regressor Predict after unmarshal: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("regressor pred[%d]: got %v, want %v", i, got[i], want[i])
		}
	}

	clf := NewDecisionTreeClassifier()
	if err := clf.Fit(X, y); err != nil {
		t.Fatalf("classifier Fit: %v", err)
	}
	data, err = clf.MarshalBinary()
	if err != nil {
		t.Fatalf("classifier MarshalBinary: %v", err)
	}
	var clf2 DecisionTreeClassifier
	if err := clf2.UnmarshalBinary(data); err != nil {
		t.Fatalf("classifier UnmarshalBinary: %v", err)
	}
	wantP, _ := clf.PredictProba(X)
	gotP, err := clf2.PredictProba(X)
	if err != nil {
		t.Fatalf("classifier PredictProba after unmarshal: %v", err)
	}
	for i := range wantP {
		for j := range wantP[i] {
			if gotP[i][j] != wantP[i][j] {
				t.Errorf("classifier proba[%d][%d]: got %v, want %v", i, j, gotP[i][j], wantP[i][j])
			}
		}
	}

	// A classifier payload must not decode into a regressor.
	if err := reg2.UnmarshalBinary(mustMarshal(t, clf)); err == nil {
		t.Error("decoding a classifier payload into a regressor should fail")
	}
	// An unfitted tree cannot be marshaled.
	if _, err := NewDecisionTreeRegressor().MarshalBinary(); err == nil {
		t.Error("marshaling an unfitted tree should fail")
	}
}

func mustMarshal(t *testing.T, c *DecisionTreeClassifier) []byte {
	t.Helper()
	data, err := c.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	return data
}
