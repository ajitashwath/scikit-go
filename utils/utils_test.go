package utils

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"scikit-go/internal/matutil"
)

func makeData(n, p int) ([][]float64, []float64) {
	r := rand.New(rand.NewSource(1))
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		X[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			X[i][j] = r.Float64()
		}
		y[i] = r.Float64()
	}
	return X, y
}

func TestCheckConsistentLength_Valid(t *testing.T) {
	X, y := makeData(10, 3)
	if err := CheckConsistentLength(X, y); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}

func TestCheckConsistentLength_WrapsMatutilErrors(t *testing.T) {
	X, y := makeData(10, 3)

	cases := []struct {
		name string
		X    [][]float64
		y    []float64
		want error
	}{
		{"empty X", nil, y, matutil.ErrEmptyInput},
		{"empty y", X, nil, matutil.ErrEmptyInput},
		{"length mismatch", X, y[:5], matutil.ErrDimMismatch},
		{"ragged", [][]float64{{1, 2}, {1}}, []float64{1, 2}, matutil.ErrRaggedInput},
		{"nan y", X, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, math.NaN()}, matutil.ErrContainsNaN},
		{"inf X", [][]float64{{1, math.Inf(1)}, {1, 2}}, []float64{1, 2}, matutil.ErrContainsInf},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := CheckConsistentLength(c.X, c.y)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !errors.Is(err, c.want) {
				t.Errorf("expected wrapped %v, got %v", c.want, err)
			}
		})
	}
}

func TestShuffle_Deterministic(t *testing.T) {
	X, y := makeData(20, 3)
	s1x, s1y, err := Shuffle(X, y, 42)
	if err != nil {
		t.Fatalf("Shuffle: %v", err)
	}
	s2x, s2y, err := Shuffle(X, y, 42)
	if err != nil {
		t.Fatalf("Shuffle: %v", err)
	}
	assertSlicesEqual(t, "X", s1x, s2x)
	assertFloatSlicesEqual(t, "y", s1y, s2y)
}

func TestShuffle_PreservesRows(t *testing.T) {
	X, y := makeData(20, 3)
	sx, sy, err := Shuffle(X, y, 7)
	if err != nil {
		t.Fatalf("Shuffle: %v", err)
	}
	if len(sx) != len(X) || len(sy) != len(y) {
		t.Fatalf("length changed: got %d/%d, want %d/%d", len(sx), len(sy), len(X), len(y))
	}
	// Every original (row, label) pair must still exist.
	remaining := map[string]bool{}
	for i := range X {
		remaining[fmt.Sprintf("%v|%v", X[i], y[i])] = false
	}
	for i := range sx {
		key := fmt.Sprintf("%v|%v", sx[i], sy[i])
		if _, ok := remaining[key]; !ok {
			t.Fatalf("row %d (%v, %v) not present in original data", i, sx[i], sy[i])
		}
		remaining[key] = true
	}
	for k, seen := range remaining {
		if !seen {
			t.Fatalf("original row %v was lost", k)
		}
	}
	// Order must actually change (probability of identity permutation is ~0).
	same := true
	for i := range X {
		if fmt.Sprintf("%v", X[i]) != fmt.Sprintf("%v", sx[i]) {
			same = false
			break
		}
	}
	if same {
		t.Error("expected shuffle to reorder rows")
	}
}

func TestShuffle_LeavesInputsUntouched(t *testing.T) {
	X, y := makeData(10, 2)
	origX := deepCopy(X)
	origY := append([]float64(nil), y...)
	if _, _, err := Shuffle(X, y, 5); err != nil {
		t.Fatalf("Shuffle: %v", err)
	}
	assertSlicesEqual(t, "X input", X, origX)
	assertFloatSlicesEqual(t, "y input", y, origY)
}

func TestShuffle_InvalidInputs(t *testing.T) {
	X, y := makeData(5, 2)
	if _, _, err := Shuffle(X, y[:3], 1); err == nil {
		t.Error("expected error for mismatched lengths, got nil")
	}
}

func TestTrainTestSplit_Sizes(t *testing.T) {
	X, y := makeData(100, 4)
	testSize := 0.25
	XTr, XTe, yTr, yTe, err := TrainTestSplit(X, y, testSize, 1)
	if err != nil {
		t.Fatalf("TrainTestSplit: %v", err)
	}
	if len(XTe) != 25 {
		t.Errorf("test size: got %d, want %d (ceil(0.25*100))", len(XTe), 25)
	}
	if len(XTr) != 75 {
		t.Errorf("train size: got %d, want %d", len(XTr), 75)
	}
	if len(yTe) != 25 || len(yTr) != 75 {
		t.Errorf("y sizes: got %d/%d, want 25/75", len(yTe), len(yTr))
	}
	if len(XTr)+len(XTe) != len(X) {
		t.Errorf("split lost samples: %d + %d != %d", len(XTr), len(XTe), len(X))
	}
}

func TestTrainTestSplit_CeilSemantics(t *testing.T) {
	// sklearn: n_test = int(ceil(test_size * n_samples)); 0.2 * 11 = 2.2 -> 3.
	X, y := makeData(11, 2)
	_, XTe, _, _, err := TrainTestSplit(X, y, 0.2, 1)
	if err != nil {
		t.Fatalf("TrainTestSplit: %v", err)
	}
	if len(XTe) != 3 {
		t.Errorf("test size: got %d, want %d", len(XTe), 3)
	}
}

func TestTrainTestSplit_DeterministicAndExhaustive(t *testing.T) {
	X, y := makeData(50, 3)
	XTr1, XTe1, yTr1, yTe1, err := TrainTestSplit(X, y, 0.2, 99)
	if err != nil {
		t.Fatalf("TrainTestSplit: %v", err)
	}
	XTr2, XTe2, yTr2, yTe2, err := TrainTestSplit(X, y, 0.2, 99)
	if err != nil {
		t.Fatalf("TrainTestSplit: %v", err)
	}
	assertSlicesEqual(t, "XTr", XTr1, XTr2)
	assertSlicesEqual(t, "XTe", XTe1, XTe2)
	assertFloatSlicesEqual(t, "yTr", yTr1, yTr2)
	assertFloatSlicesEqual(t, "yTe", yTe1, yTe2)

	// Train + test must be a partition of the original rows.
	seen := map[string]bool{}
	for i := range XTr1 {
		seen[fmt.Sprintf("%v|%v", XTr1[i], yTr1[i])] = true
	}
	for i := range XTe1 {
		seen[fmt.Sprintf("%v|%v", XTe1[i], yTe1[i])] = true
	}
	if len(seen) != len(X) {
		t.Errorf("partition covers %d distinct rows, want %d", len(seen), len(X))
	}
	for i := range X {
		key := fmt.Sprintf("%v|%v", X[i], y[i])
		if !seen[key] {
			t.Fatalf("row %d missing from split", i)
		}
	}
}

func TestTrainTestSplit_InvalidParams(t *testing.T) {
	X, y := makeData(10, 2)
	for _, ts := range []float64{0.0, 1.0, -0.5, 1.5} {
		if _, _, _, _, err := TrainTestSplit(X, y, ts, 1); err == nil {
			t.Errorf("expected error for testSize=%v, got nil", ts)
		}
	}
	if _, _, _, _, err := TrainTestSplit(X, y[:3], 0.2, 1); err == nil {
		t.Error("expected error for mismatched lengths, got nil")
	}
}

func TestTrainTestSplit_TooSmallForSplit(t *testing.T) {
	// n=1, testSize=0.5 -> ceil(0.5)=1 == n -> cannot split.
	X, y := makeData(1, 2)
	if _, _, _, _, err := TrainTestSplit(X, y, 0.5, 1); err == nil {
		t.Error("expected error for unsplittable data, got nil")
	}
}

func deepCopy(X [][]float64) [][]float64 {
	out := make([][]float64, len(X))
	for i, row := range X {
		out[i] = append([]float64(nil), row...)
	}
	return out
}

func assertSlicesEqual(t *testing.T, name string, a, b [][]float64) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s: length mismatch, got %d, want %d", name, len(a), len(b))
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			t.Fatalf("%s[%d]: inner length mismatch %d vs %d", name, i, len(a[i]), len(b[i]))
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				t.Errorf("%s[%d][%d]: got %v, want %v", name, i, j, a[i][j], b[i][j])
			}
		}
	}
}

func assertFloatSlicesEqual(t *testing.T, name string, a, b []float64) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s: length mismatch, got %d, want %d", name, len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("%s[%d]: got %v, want %v", name, i, a[i], b[i])
		}
	}
}
