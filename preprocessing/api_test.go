package preprocessing

import (
	"math"
	"testing"
)

func TestStandardScaler_FitTransformMatchesFitThenTransform(t *testing.T) {
	X := [][]float64{{1, 10, 5}, {2, 20, 5}, {3, 30, 5}, {4, 44, 5}}
	a := NewStandardScaler()
	got, err := a.FitTransform(X)
	if err != nil {
		t.Fatalf("FitTransform: %v", err)
	}
	b := NewStandardScaler()
	if err := b.Fit(X); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	want, err := b.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}
	for i := range got {
		for j := range got[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("[%d][%d]: %v vs %v", i, j, got[i][j], want[i][j])
			}
		}
	}
	// A constant column keeps scale 1 and standardizes to zeros instead of NaN.
	for i := range got {
		if got[i][2] != 0 || math.IsNaN(got[i][2]) {
			t.Errorf("constant column row %d: got %v, want 0", i, got[i][2])
		}
	}
	// Non-constant columns end up with zero mean and unit variance.
	for j := 0; j < 2; j++ {
		var mean, sq float64
		for i := range got {
			mean += got[i][j]
		}
		mean /= float64(len(got))
		for i := range got {
			sq += (got[i][j] - mean) * (got[i][j] - mean)
		}
		if math.Abs(mean) > 1e-12 || math.Abs(sq/float64(len(got))-1) > 1e-12 {
			t.Errorf("column %d: mean %v, variance %v", j, mean, sq/float64(len(got)))
		}
	}
	back, err := a.InverseTransform(got)
	if err != nil {
		t.Fatalf("InverseTransform: %v", err)
	}
	for i := range X {
		for j := range X[i] {
			if math.Abs(back[i][j]-X[i][j]) > 1e-12 {
				t.Fatalf("round trip [%d][%d]: got %v, want %v", i, j, back[i][j], X[i][j])
			}
		}
	}
}
