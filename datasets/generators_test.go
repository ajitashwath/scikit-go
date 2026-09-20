package datasets

import (
	"math"
	"testing"

	"github.com/ajitashwath/scikit-go/linear"
	"github.com/ajitashwath/scikit-go/metrics"
)

// TestMakeRegression_Shapes checks the deterministic shapes of MakeRegression.
func TestMakeRegression_Shapes(t *testing.T) {
	X, y, err := MakeRegression(100, 5, 0.1, 42)
	if err != nil {
		t.Fatalf("MakeRegression: %v", err)
	}
	if len(X) != 100 || len(y) != 100 {
		t.Fatalf("expected 100 samples, got X=%d y=%d", len(X), len(y))
	}
	for i, row := range X {
		if len(row) != 5 {
			t.Fatalf("row %d has %d features, want 5", i, len(row))
		}
	}
}

// TestMakeRegression_Deterministic checks that the same seed reproduces the same data.
func TestMakeRegression_Deterministic(t *testing.T) {
	X1, y1, err := MakeRegression(50, 4, 0.1, 7)
	if err != nil {
		t.Fatalf("MakeRegression: %v", err)
	}
	X2, y2, err := MakeRegression(50, 4, 0.1, 7)
	if err != nil {
		t.Fatalf("MakeRegression: %v", err)
	}
	assertMatrixEqual(t, X1, X2)
	assertSliceClose(t, "y", y1, y2)
}

// TestMakeRegression_NoNoiseFitsExactly checks that a noiseless dataset is perfectly
// explainable by a linear model: y must lie exactly on some hyperplane, so the
// residual R2 vs a LinearRegression fit is ~1.0.
func TestMakeRegression_NoNoiseFitsExactly(t *testing.T) {
	X, y, err := MakeRegression(200, 6, 0.0, 3)
	if err != nil {
		t.Fatalf("MakeRegression: %v", err)
	}
	r2 := fitR2(t, X, y)
	if math.Abs(r2-1.0) > 1e-9 {
		t.Fatalf("expected R2 ~ 1.0 for noiseless data, got %v", r2)
	}
}

// TestMakeRegression_NoiseDegradesFit checks that adding noise lowers the R2.
func TestMakeRegression_NoiseDegradesFit(t *testing.T) {
	X, y, err := MakeRegression(500, 4, 1.0, 9)
	if err != nil {
		t.Fatalf("MakeRegression: %v", err)
	}
	r2 := fitR2(t, X, y)
	if r2 > 0.99 {
		t.Fatalf("expected noisy data to have R2 < 0.99, got %v", r2)
	}
	if r2 < 0.3 {
		t.Fatalf("expected noisy data to retain signal (R2 > 0.3), got %v", r2)
	}
}

func TestMakeRegression_InvalidParams(t *testing.T) {
	for _, args := range []struct {
		n, p  int
		noise float64
	}{
		{0, 5, 0.1},
		{5, 0, 0.1},
		{5, 5, -1.0},
	} {
		if _, _, err := MakeRegression(args.n, args.p, args.noise, 1); err == nil {
			t.Errorf("MakeRegression(%d, %d, %v) expected error, got nil", args.n, args.p, args.noise)
		}
	}
}

func TestMakeClassification_Balanced(t *testing.T) {
	nSamples, nClasses := 60, 3
	X, y, err := MakeClassification(nSamples, 4, nClasses, 5)
	if err != nil {
		t.Fatalf("MakeClassification: %v", err)
	}
	if len(X) != nSamples || len(y) != nSamples {
		t.Fatalf("expected %d samples, got X=%d y=%d", nSamples, len(X), len(y))
	}
	counts := make([]int, nClasses)
	for _, l := range y {
		if int(l) < 0 || int(l) >= nClasses {
			t.Fatalf("unexpected class label %v", l)
		}
		counts[int(l)]++
	}
	for c := 0; c < nClasses; c++ {
		want := nSamples / nClasses
		if counts[c] != want {
			t.Errorf("class %d has %d samples, want %d", c, counts[c], want)
		}
	}
}

func TestMakeClassification_Separable(t *testing.T) {
	// Two classes with well-separated centers must be linearly separable.
	X, y, err := MakeClassification(200, 2, 2, 11)
	if err != nil {
		t.Fatalf("MakeClassification: %v", err)
	}
	if accuracyOfLinearSeparation(t, X, y) != 1.0 {
		t.Fatal("expected the generated classes to be linearly separable")
	}
}

func TestMakeClassification_Deterministic(t *testing.T) {
	X1, y1, err := MakeClassification(30, 3, 2, 2)
	if err != nil {
		t.Fatalf("MakeClassification: %v", err)
	}
	X2, y2, err := MakeClassification(30, 3, 2, 2)
	if err != nil {
		t.Fatalf("MakeClassification: %v", err)
	}
	assertMatrixEqual(t, X1, X2)
	assertSliceClose(t, "y", y1, y2)
}

func TestMakeClassification_InvalidParams(t *testing.T) {
	if _, _, err := MakeClassification(10, 2, 20, 1); err == nil {
		t.Error("expected error when nClasses > nSamples, got nil")
	}
	if _, _, err := MakeClassification(0, 2, 2, 1); err == nil {
		t.Error("expected error for zero samples, got nil")
	}
}

func TestMakeBlobs_BalancedClusters(t *testing.T) {
	nSamples, nCenters := 90, 3
	X, y, err := MakeBlobs(nSamples, 2, nCenters, 4)
	if err != nil {
		t.Fatalf("MakeBlobs: %v", err)
	}
	if len(X) != nSamples || len(y) != nSamples {
		t.Fatalf("expected %d samples, got X=%d y=%d", nSamples, len(X), len(y))
	}
	counts := make([]int, nCenters)
	for _, l := range y {
		if int(l) < 0 || int(l) >= nCenters {
			t.Fatalf("unexpected cluster label %v", l)
		}
		counts[int(l)]++
	}
	for c := 0; c < nCenters; c++ {
		if counts[c] != nSamples/nCenters {
			t.Errorf("cluster %d has %d samples, want %d", c, counts[c], nSamples/nCenters)
		}
	}
}

func TestMakeBlobs_UnitVarianceClusters(t *testing.T) {
	// With clusterStd=1 (default), each cluster's per-feature variance is ~1.
	nPerCluster := 2000
	X, y, err := MakeBlobs(nPerCluster*3, 2, 3, 8)
	if err != nil {
		t.Fatalf("MakeBlobs: %v", err)
	}
	for c := 0; c < 3; c++ {
		sums := make([]float64, 2)
		count := 0
		for i, l := range y {
			if int(l) == c {
				for j := range sums {
					sums[j] += X[i][j]
				}
				count++
			}
		}
		means := make([]float64, 2)
		for j := range means {
			means[j] = sums[j] / float64(count)
		}
		var varSum float64
		for i, l := range y {
			if int(l) == c {
				for j := range means {
					d := X[i][j] - means[j]
					varSum += d * d
				}
			}
		}
		avgVar := varSum / float64(count*2)
		if math.Abs(avgVar-1.0) > 0.1 {
			t.Errorf("cluster %d per-feature variance %v is not ~1.0", c, avgVar)
		}
	}
}

func TestMakeBlobs_Deterministic(t *testing.T) {
	X1, y1, err := MakeBlobs(40, 2, 2, 6)
	if err != nil {
		t.Fatalf("MakeBlobs: %v", err)
	}
	X2, y2, err := MakeBlobs(40, 2, 2, 6)
	if err != nil {
		t.Fatalf("MakeBlobs: %v", err)
	}
	assertMatrixEqual(t, X1, X2)
	assertSliceClose(t, "y", y1, y2)
}

func TestMakeBlobs_InvalidParams(t *testing.T) {
	if _, _, err := MakeBlobs(10, 2, 0, 1); err == nil {
		t.Error("expected error for zero centers, got nil")
	}
	if _, _, err := MakeBlobs(0, 2, 2, 1); err == nil {
		t.Error("expected error for zero samples, got nil")
	}
}

func TestMakeMoons_Shapes(t *testing.T) {
	n := 100
	X, y, err := MakeMoons(n, 0.1, 1)
	if err != nil {
		t.Fatalf("MakeMoons: %v", err)
	}
	if len(X) != n || len(y) != n {
		t.Fatalf("expected %d samples, got X=%d y=%d", n, len(X), len(y))
	}
	for _, l := range y {
		if l != 0 && l != 1 {
			t.Fatalf("unexpected label %v", l)
		}
	}
	if countLabel(y, 0) != n/2 {
		t.Errorf("expected %d outer samples, got %d", n/2, countLabel(y, 0))
	}
}

func TestMakeMoons_Deterministic(t *testing.T) {
	X1, y1, err := MakeMoons(60, 0.05, 3)
	if err != nil {
		t.Fatalf("MakeMoons: %v", err)
	}
	X2, y2, err := MakeMoons(60, 0.05, 3)
	if err != nil {
		t.Fatalf("MakeMoons: %v", err)
	}
	assertMatrixEqual(t, X1, X2)
	assertSliceClose(t, "y", y1, y2)
}

func TestMakeMoons_InvalidParams(t *testing.T) {
	if _, _, err := MakeMoons(0, 0.1, 1); err == nil {
		t.Error("expected error for zero samples, got nil")
	}
	if _, _, err := MakeMoons(10, -0.1, 1); err == nil {
		t.Error("expected error for negative noise, got nil")
	}
}

func countLabel(y []float64, label float64) int {
	n := 0
	for _, l := range y {
		if l == label {
			n++
		}
	}
	return n
}

func assertMatrixEqual(t *testing.T, a, b [][]float64) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("row count mismatch: %d vs %d", len(a), len(b))
	}
	for i := range a {
		assertSliceClose(t, "row", a[i], b[i])
	}
}

// fitR2 fits a LinearRegression to X/y and returns its R2 score.
func fitR2(t *testing.T, X [][]float64, y []float64) float64 {
	t.Helper()
	model := linear.NewLinearRegression()
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	preds, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	r2, err := metrics.R2Score(y, preds)
	if err != nil {
		t.Fatalf("R2Score failed: %v", err)
	}
	return r2
}

// accuracyOfLinearSeparation reports the best training accuracy of a linear
// classifier fit to a two-class dataset (via LinearRegression thresholding).
func accuracyOfLinearSeparation(t *testing.T, X [][]float64, y []float64) float64 {
	t.Helper()
	model := linear.NewLinearRegression()
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	preds, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	correct := 0
	for i, p := range preds {
		predClass := 0.0
		if p >= 0.5 {
			predClass = 1.0
		}
		if predClass == y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(y))
}
