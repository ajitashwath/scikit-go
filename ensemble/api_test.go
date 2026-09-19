package ensemble

import (
	"testing"

	"scikit-go/datasets"
)

func TestForest_AccessorsBeforeAndAfterFit(t *testing.T) {
	c, r := NewRandomForestClassifier(), NewRandomForestRegressor()
	if c.Classes() != nil || c.FeatureImportances() != nil || c.NEstimators() != 0 || r.NEstimators() != 0 {
		t.Error("accessors should return zero values before Fit")
	}

	X, y, err := datasets.MakeClassification(60, 4, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	c.NTrees = 7
	if err := c.Fit(X, y); err != nil {
		t.Fatalf("classifier Fit: %v", err)
	}
	if c.NEstimators() != 7 || len(c.Classes()) != 3 {
		t.Errorf("classifier: %d trees, %d classes; want 7 and 3", c.NEstimators(), len(c.Classes()))
	}
	// Classes returns a copy.
	c.Classes()[0] = 1e9
	if c.Classes()[0] == 1e9 {
		t.Error("Classes returned the internal slice")
	}

	Xr, yr, err := datasets.MakeRegression(60, 4, 0.1, 2)
	if err != nil {
		t.Fatal(err)
	}
	r.NTrees = 4
	if err := r.Fit(Xr, yr); err != nil {
		t.Fatalf("regressor Fit: %v", err)
	}
	if r.NEstimators() != 4 {
		t.Errorf("regressor NEstimators = %d, want 4", r.NEstimators())
	}
}

func TestForest_ScoreAgainstTrainingData(t *testing.T) {
	X, y, _ := datasets.MakeClassification(80, 4, 2, 3)
	c := NewRandomForestClassifier()
	c.NTrees = 30
	if err := c.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	acc, err := c.Score(X, y)
	if err != nil {
		t.Fatalf("classifier Score: %v", err)
	}
	if acc < 0.9 {
		t.Errorf("training accuracy %v is implausibly low for a forest", acc)
	}
	if _, err := c.Score(X, y[:5]); err == nil {
		t.Error("mismatched lengths should fail")
	}

	Xr, yr, _ := datasets.MakeRegression(80, 4, 0.5, 4)
	r := NewRandomForestRegressor()
	r.NTrees = 30
	if err := r.Fit(Xr, yr); err != nil {
		t.Fatal(err)
	}
	r2, err := r.Score(Xr, yr)
	if err != nil {
		t.Fatalf("regressor Score: %v", err)
	}
	if r2 < 0.8 {
		t.Errorf("training R2 %v is implausibly low for a forest", r2)
	}
	if _, err := r.Score(Xr, yr[:5]); err == nil {
		t.Error("mismatched lengths should fail")
	}
}

func TestDefaultMaxFeatures(t *testing.T) {
	for n, want := range map[int]int{1: 1, 2: 1, 3: 1, 4: 2, 9: 3, 10: 3, 100: 10} {
		if got := defaultMaxFeatures(n); got != want {
			t.Errorf("defaultMaxFeatures(%d) = %d, want %d", n, got, want)
		}
	}
}
