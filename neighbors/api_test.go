package neighbors

import (
	"errors"
	"math"
	"testing"

	"scikit-go/internal/matutil"
)

func TestMinkowskiDistance(t *testing.T) {
	a, b := []float64{0, 0}, []float64{3, 4}
	cases := []struct {
		p    float64
		want float64
	}{
		{1, 7},
		{2, 5},
		{3, math.Cbrt(91)}, // (3^3 + 4^3)^(1/3)
		{1.5, math.Pow(math.Pow(3, 1.5)+math.Pow(4, 1.5), 1/1.5)},
	}
	for _, tc := range cases {
		if got := MinkowskiDistance(a, b, tc.p); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("p=%v: got %v, want %v", tc.p, got, tc.want)
		}
	}
	if got := MinkowskiDistance(a, a, 3); got != 0 {
		t.Errorf("distance to itself: got %v, want 0", got)
	}
	if MinkowskiDistance(a, b, 2) != MinkowskiDistance(b, a, 2) {
		t.Error("distance is not symmetric")
	}
}

var knnX = [][]float64{{0, 0}, {0, 1}, {1, 0}, {5, 5}, {5, 6}, {6, 5}}
var knnY = []float64{0, 0, 0, 1, 1, 1}

func TestKNeighborsClassifier_Score(t *testing.T) {
	c := NewKNeighborsClassifier()
	c.NNeighbors = 3
	if _, err := c.Score(knnX, knnY); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Score before Fit: got %v, want ErrNotFitted", err)
	}
	if err := c.Fit(knnX, knnY); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	got, err := c.Score(knnX, knnY)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got != 1.0 {
		t.Errorf("accuracy on separable training data: got %v, want 1", got)
	}
	// Half wrong labels -> accuracy 0.5.
	got, err = c.Score([][]float64{{0, 0}, {5, 5}}, []float64{0, 0})
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got != 0.5 {
		t.Errorf("accuracy: got %v, want 0.5", got)
	}
	if _, err := c.Score(knnX, knnY[:2]); err == nil {
		t.Error("mismatched lengths should fail")
	}
}

func TestKNeighborsRegressor_Score(t *testing.T) {
	r := NewKNeighborsRegressor()
	r.NNeighbors = 1
	if _, err := r.Score(knnX, knnY); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Score before Fit: got %v, want ErrNotFitted", err)
	}
	if err := r.Fit(knnX, knnY); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	got, err := r.Score(knnX, knnY)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if math.Abs(got-1) > 1e-12 {
		t.Errorf("R2 with k=1 on the training data: got %v, want 1", got)
	}
	if _, err := r.Score(knnX, knnY[:2]); err == nil {
		t.Error("mismatched lengths should fail")
	}
}

func TestKNN_DistanceWeightingExactMatch(t *testing.T) {
	// A query equal to a training point has distance 0; with distance weights the
	// coincident neighbor must take all the weight instead of dividing by zero.
	r := NewKNeighborsRegressor()
	r.NNeighbors, r.Weights = 3, "distance"
	if err := r.Fit(knnX, []float64{10, 20, 30, 40, 50, 60}); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	pred, err := r.Predict([][]float64{{5, 5}})
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if pred[0] != 40 || math.IsNaN(pred[0]) {
		t.Errorf("got %v, want exactly 40", pred[0])
	}
}
