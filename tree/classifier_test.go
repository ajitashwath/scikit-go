package tree

import (
	"errors"
	"math"
	"path/filepath"
	"testing"
)

// TestDecisionTreeClassifier_AgainstSklearn checks prediction, proba and
// importance parity with sklearn.tree.DecisionTreeClassifier.
func TestDecisionTreeClassifier_AgainstSklearn(t *testing.T) {
	fixtures := loadTreeFixtures(t)
	const tol = 1e-9
	for _, name := range []string{"cls_gini", "cls_entropy", "cls_xor_gini", "cls_xor_entropy", "cls_gini_depth2"} {
		fx := fixtures[name]
		model := NewDecisionTreeClassifier()
		switch name {
		case "cls_gini", "cls_xor_gini":
			model.Criterion = "gini"
		case "cls_entropy", "cls_xor_entropy":
			model.Criterion = "entropy"
		case "cls_gini_depth2":
			model.Criterion = "gini"
			model.MaxDepth = 2
		}
		if err := model.Fit(fx.X, fx.Y); err != nil {
			t.Fatalf("%s Fit: %v", name, err)
		}
		preds, err := model.Predict(fx.X)
		if err != nil {
			t.Fatalf("%s Predict(train): %v", name, err)
		}
		assertSliceClose(t, name+" pred_train", preds, fx.PredTrain, tol)

		if len(fx.XTest) > 0 {
			testPreds, err := model.Predict(fx.XTest)
			if err != nil {
				t.Fatalf("%s Predict(test): %v", name, err)
			}
			assertSliceClose(t, name+" pred_test", testPreds, fx.PredTest, tol)

			proba, err := model.PredictProba(fx.XTest)
			if err != nil {
				t.Fatalf("%s PredictProba: %v", name, err)
			}
			assertMatrixClose(t, name+" proba_test", proba, fx.ProbaTest, tol)
		}

		assertSliceClose(t, name+" classes", model.Classes(), fx.Classes, 1e-12)
		importances := model.FeatureImportances()
		assertSliceClose(t, name+" importances", importances, fx.FeatureImportances, 1e-6)
	}
}

func TestDecisionTreeClassifier_XORPerfect(t *testing.T) {
	// XOR is exactly representable by a depth-3 tree: predictions must be exact.
	fx := loadTreeFixtures(t)["cls_xor_gini"]
	model := NewDecisionTreeClassifier()
	model.Criterion = "gini"
	if err := model.Fit(fx.X, fx.Y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	preds, err := model.Predict(fx.X)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	for i, p := range preds {
		if p != fx.Y[i] {
			t.Errorf("pred[%d] = %v, want %v", i, p, fx.Y[i])
		}
	}
	// proba at a leaf must be 1.0 for the winning class.
	proba, err := model.PredictProba(fx.X)
	if err != nil {
		t.Fatalf("PredictProba: %v", err)
	}
	for i, row := range proba {
		best := argmax(row)
		if row[best] < 1-1e-12 {
			t.Errorf("proba[%d] = %v, expected a pure leaf", i, row)
		}
	}
}

func TestDecisionTreeClassifier_ConstantTarget(t *testing.T) {
	model := NewDecisionTreeClassifier()
	X := [][]float64{{1}, {2}, {3}}
	y := []float64{4, 4, 4}
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	preds, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	for _, p := range preds {
		if p != 4 {
			t.Errorf("pred = %v, want 4", p)
		}
	}
}

func TestDecisionTreeClassifier_MinSamplesLeaf(t *testing.T) {
	model := NewDecisionTreeClassifier()
	model.MinSamplesLeaf = 5
	X := make([][]float64, 40)
	y := make([]float64, 40)
	for i := 0; i < 40; i++ {
		X[i] = []float64{float64(i)}
		y[i] = float64(i % 2)
	}
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	for i, n := range model.impl.nodes {
		if n.NSamples < 5 {
			t.Errorf("node %d has %d samples < min_samples_leaf=5", i, n.NSamples)
		}
	}
}

func TestDecisionTreeClassifier_SaveLoadRoundTrip(t *testing.T) {
	model := NewDecisionTreeClassifier()
	model.Criterion = "entropy"
	X := make([][]float64, 40)
	y := make([]float64, 40)
	for i := 0; i < 40; i++ {
		X[i] = []float64{float64(i % 4), float64(i % 3)}
		y[i] = float64(i % 3)
	}
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	tmp := filepath.Join(t.TempDir(), "clf.bin")
	if err := model.Save(tmp); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadDecisionTreeClassifier(tmp)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	origPreds, _ := model.Predict(X)
	loadedPreds, err := loaded.Predict(X)
	if err != nil {
		t.Fatalf("loaded Predict: %v", err)
	}
	assertSliceClose(t, "loaded preds", loadedPreds, origPreds, 1e-12)
	loadedProbas, err := loaded.PredictProba(X)
	if err != nil {
		t.Fatalf("loaded PredictProba: %v", err)
	}
	origProbas, _ := model.PredictProba(X)
	assertMatrixClose(t, "loaded probas", loadedProbas, origProbas, 1e-12)
	if loaded.Criterion != "entropy" {
		t.Errorf("criterion not preserved: %q", loaded.Criterion)
	}
}

func TestDecisionTreeClassifier_LoadWrongKind(t *testing.T) {
	model := NewDecisionTreeRegressor()
	X := [][]float64{{1}, {2}, {3}}
	y := []float64{1, 2, 3}
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	tmp := filepath.Join(t.TempDir(), "reg.bin")
	if err := model.Save(tmp); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if _, err := LoadDecisionTreeClassifier(tmp); err == nil {
		t.Fatal("expected error loading a regressor payload as a classifier")
	}
}

func TestDecisionTreeClassifier_SaveBeforeFit(t *testing.T) {
	model := NewDecisionTreeClassifier()
	if err := model.Save(filepath.Join(t.TempDir(), "x.bin")); err == nil {
		t.Fatal("expected error saving unfitted model")
	}
}

func TestDecisionTreeClassifier_EdgeCases(t *testing.T) {
	model := NewDecisionTreeClassifier()
	if err := model.Fit([][]float64{}, []float64{}); err == nil {
		t.Error("expected error on empty input")
	}
	if err := model.Fit([][]float64{{1}, {2}}, []float64{1, 2, 3}); err == nil {
		t.Error("expected error on mismatched dims")
	}
	if err := model.Fit([][]float64{{1, 2}, {3}}, []float64{1, 2}); err == nil {
		t.Error("expected error on ragged input")
	}
	if err := model.Fit([][]float64{{math.Inf(1)}, {1}}, []float64{1, 2}); err == nil {
		t.Error("expected error on Inf input")
	}
	if err := model.Fit([][]float64{{1}, {2}}, []float64{1, 2}); err != nil {
		t.Errorf("unexpected error on valid input: %v", err)
	}
}

func TestDecisionTreeClassifier_InvalidCriterion(t *testing.T) {
	model := NewDecisionTreeClassifier()
	model.Criterion = "bogus"
	if err := model.Fit([][]float64{{1}, {2}}, []float64{1, 2}); err == nil {
		t.Error("expected error on invalid criterion")
	}
}
func TestTree_InvalidHyperparameters(t *testing.T) {
	X := [][]float64{{0}, {1}, {2}, {3}}
	y := []float64{0, 0, 1, 1}
	cases := []struct {
		name   string
		mutate func(split, leaf, features *int)
	}{
		{"min_samples_split=1", func(s, _, _ *int) { *s = 1 }},
		{"min_samples_split=0", func(s, _, _ *int) { *s = 0 }},
		{"min_samples_leaf=0", func(_, l, _ *int) { *l = 0 }},
		{"min_samples_leaf<0", func(_, l, _ *int) { *l = -3 }},
		{"max_features<0", func(_, _, f *int) { *f = -1 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := NewDecisionTreeClassifier()
			tc.mutate(&c.MinSamplesSplit, &c.MinSamplesLeaf, &c.MaxFeatures)
			if err := c.Fit(X, y); !errors.Is(err, ErrInvalidParams) {
				t.Errorf("classifier: got %v, want ErrInvalidParams", err)
			}
			r := NewDecisionTreeRegressor()
			tc.mutate(&r.MinSamplesSplit, &r.MinSamplesLeaf, &r.MaxFeatures)
			if err := r.Fit(X, y); !errors.Is(err, ErrInvalidParams) {
				t.Errorf("regressor: got %v, want ErrInvalidParams", err)
			}
		})
	}
}

func TestTree_MaxFeaturesAboveNFeatures(t *testing.T) {
	X := [][]float64{{0, 5}, {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0}}
	y := []float64{0, 0, 0, 1, 1, 1}
	c := NewDecisionTreeClassifier()
	c.MaxFeatures = 50
	if err := c.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if _, err := c.Predict(X); err != nil {
		t.Fatalf("Predict: %v", err)
	}
}
