package tree

import (
	"errors"
	"math"
	"testing"

	"github.com/ajitashwath/scikit-go/internal/matutil"
)

func TestDecisionTreeClassifier_ScoreAndAccessors(t *testing.T) {
	X := [][]float64{{0, 9}, {1, 8}, {2, 7}, {3, 6}, {4, 5}, {5, 4}}
	y := []float64{0, 0, 0, 1, 1, 1}
	c := NewDecisionTreeClassifier()
	if _, err := c.Score(X, y); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Score before Fit: got %v, want ErrNotFitted", err)
	}
	if c.FeatureImportances() != nil || c.Classes() != nil {
		t.Error("accessors should return nil before Fit")
	}
	if err := c.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	acc, err := c.Score(X, y)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if acc != 1 {
		t.Errorf("accuracy on separable training data: got %v, want 1", acc)
	}
	acc, err = c.Score([][]float64{{0, 9}, {5, 4}}, []float64{1, 1})
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if acc != 0.5 {
		t.Errorf("accuracy: got %v, want 0.5", acc)
	}
	if _, err := c.Score(X, y[:2]); err == nil {
		t.Error("mismatched lengths should fail")
	}

	var sum float64
	for _, v := range c.FeatureImportances() {
		sum += v
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Errorf("feature importances sum to %v, want 1", sum)
	}
	classes := c.Classes()
	classes[0] = 1e9
	if c.Classes()[0] == 1e9 {
		t.Error("Classes returned the internal slice")
	}
}

func TestDecisionTreeRegressor_ScoreAndAccessors(t *testing.T) {
	X := [][]float64{{0}, {1}, {2}, {3}, {4}, {5}}
	y := []float64{0, 1, 4, 9, 16, 25}
	r := NewDecisionTreeRegressor()
	if _, err := r.Score(X, y); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Score before Fit: got %v, want ErrNotFitted", err)
	}
	if r.FeatureImportances() != nil {
		t.Error("FeatureImportances should be nil before Fit")
	}
	if err := r.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	r2, err := r.Score(X, y)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if math.Abs(r2-1) > 1e-12 {
		t.Errorf("R2 of a fully grown tree on its training data: got %v, want 1", r2)
	}
	if _, err := r.Score(X, y[:2]); err == nil {
		t.Error("mismatched lengths should fail")
	}
	if imp := r.FeatureImportances(); len(imp) != 1 || math.Abs(imp[0]-1) > 1e-12 {
		t.Errorf("a single feature carries all the importance, got %v", imp)
	}
}
