package datasets

import (
	"sort"
	"testing"
)

func TestLoadDiabetes_AgainstSklearn(t *testing.T) {
	X, y, err := LoadDiabetes()
	if err != nil {
		t.Fatalf("LoadDiabetes: %v", err)
	}
	fx := loadJSON(t, "datasets_fixtures.json")["diabetes"].(map[string]interface{})

	if len(X) != fixtureInt(t, fx, "n_samples") {
		t.Fatalf("sample count: got %d, want %d", len(X), fixtureInt(t, fx, "n_samples"))
	}
	if len(X[0]) != fixtureInt(t, fx, "n_features") {
		t.Fatalf("feature count: got %d, want %d", len(X[0]), fixtureInt(t, fx, "n_features"))
	}
	if len(y) != len(X) {
		t.Fatalf("target length %d does not match sample count %d", len(y), len(X))
	}

	means := featureMeans(X)
	assertSliceClose(t, "feature_means", means, fixtureFloatSlice(t, fx, "feature_means"))
	assertClose(t, "y_mean", vectorMean(y), fixtureFloat(t, fx, "y_mean"))
	assertClose(t, "y_min", vectorMin(y), fixtureFloat(t, fx, "y_min"))
	assertClose(t, "y_max", vectorMax(y), fixtureFloat(t, fx, "y_max"))
}

func TestLoadIris_AgainstSklearn(t *testing.T) {
	X, y, err := LoadIris()
	if err != nil {
		t.Fatalf("LoadIris: %v", err)
	}
	fx := loadJSON(t, "datasets_fixtures.json")["iris"].(map[string]interface{})

	if len(X) != fixtureInt(t, fx, "n_samples") {
		t.Fatalf("sample count: got %d, want %d", len(X), fixtureInt(t, fx, "n_samples"))
	}
	if len(X[0]) != fixtureInt(t, fx, "n_features") {
		t.Fatalf("feature count: got %d, want %d", len(X[0]), fixtureInt(t, fx, "n_features"))
	}
	if len(y) != len(X) {
		t.Fatalf("target length %d does not match sample count %d", len(y), len(X))
	}

	means := featureMeans(X)
	assertSliceClose(t, "feature_means", means, fixtureFloatSlice(t, fx, "feature_means"))
	assertClose(t, "y_mean", vectorMean(y), fixtureFloat(t, fx, "y_mean"))
	assertClose(t, "y_min", vectorMin(y), fixtureFloat(t, fx, "y_min"))
	assertClose(t, "y_max", vectorMax(y), fixtureFloat(t, fx, "y_max"))

	// Class indices must be 0, 1, 2 with sklearn's class counts (50 each).
	classCounts := make([]int, 3)
	for _, l := range y {
		if int(l) < 0 || int(l) > 2 {
			t.Fatalf("unexpected class label %v", l)
		}
		classCounts[int(l)]++
	}
	wantCounts := fixtureIntSlice(t, fx, "class_counts")
	for c := range wantCounts {
		if classCounts[c] != wantCounts[c] {
			t.Errorf("class %d count: got %d, want %d", c, classCounts[c], wantCounts[c])
		}
	}
}

// TestMakeMoons_NoNoise_AgainstSklearn checks bit-compatibility of the deterministic
// no-noise moon geometry with sklearn's make_moons(noise=0). sklearn shuffles the
// rows with its own RNG, so the comparison is order-insensitive: we sort the
// (point, label) rows of both outputs and compare element-wise.
func TestMakeMoons_NoNoise_AgainstSklearn(t *testing.T) {
	fx := loadJSON(t, "moons_fixtures.json")
	wantX := fixtureFloatMatrix(t, fx, "X")
	wantY := fixtureFloatSlice(t, fx, "y")

	gotX, gotY, err := MakeMoons(100, 0.0, 0)
	if err != nil {
		t.Fatalf("MakeMoons: %v", err)
	}

	sortByPoint(gotX, gotY)
	sortByPoint(wantX, wantY)

	assertMatrixClose(t, "X", gotX, wantX, 1e-12)
	assertSliceClose(t, "y", gotY, wantY)
}

func lessPoint(a, b []float64) bool {
	if a[0] != b[0] {
		return a[0] < b[0]
	}
	return a[1] < b[1]
}

// sortByPoint sorts X and y together by the point coordinates of X.
func sortByPoint(X [][]float64, y []float64) {
	order := make([]int, len(X))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return lessPoint(X[order[a]], X[order[b]])
	})
	sortedX := make([][]float64, len(X))
	sortedY := make([]float64, len(y))
	for i, o := range order {
		sortedX[i] = X[o]
		sortedY[i] = y[o]
	}
	copy(X, sortedX)
	copy(y, sortedY)
}

func fixtureIntSlice(t *testing.T, fx map[string]interface{}, key string) []int {
	t.Helper()
	raw, ok := fx[key].([]interface{})
	if !ok {
		t.Fatalf("fixture key %q is not an array: %v", key, fx[key])
	}
	out := make([]int, len(raw))
	for i, v := range raw {
		f, ok := v.(float64)
		if !ok {
			t.Fatalf("fixture key %q[%d] is not numeric: %v", key, i, v)
		}
		out[i] = int(f)
	}
	return out
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
			diff := got[i][j] - want[i][j]
			if diff < 0 {
				diff = -diff
			}
			if diff > tol {
				t.Errorf("%s[%d][%d]: got %.15f, want %.15f (diff %.2e)", name, i, j, got[i][j], want[i][j], diff)
			}
		}
	}
}

func featureMeans(X [][]float64) []float64 {
	p := len(X[0])
	means := make([]float64, p)
	for _, row := range X {
		for j, v := range row {
			means[j] += v
		}
	}
	for j := range means {
		means[j] /= float64(len(X))
	}
	return means
}

func vectorMean(v []float64) float64 {
	var sum float64
	for _, x := range v {
		sum += x
	}
	return sum / float64(len(v))
}

func vectorMin(v []float64) float64 {
	m := v[0]
	for _, x := range v[1:] {
		if x < m {
			m = x
		}
	}
	return m
}

func vectorMax(v []float64) float64 {
	m := v[0]
	for _, x := range v[1:] {
		if x > m {
			m = x
		}
	}
	return m
}
