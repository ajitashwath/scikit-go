package metrics

import (
	"errors"
	"math"
	"testing"
)

func TestClustering_AgainstSklearn(t *testing.T) {
	for _, name := range []string{"clustering", "clustering_perfect"} {
		fx := exampleFixture(t, name)
		labelsTrue := fixtureSlice(t, fx, "labels_true")
		labelsPred := fixtureSlice(t, fx, "labels_pred")

		t.Run(name, func(t *testing.T) {
			got, err := AdjustedRandIndex(labelsTrue, labelsPred)
			if err != nil {
				t.Fatalf("AdjustedRandIndex: %v", err)
			}
			assertClose(t, "adjusted_rand", got, fixtureFloat(t, fx, "adjusted_rand"))

			got, err = HomogeneityScore(labelsTrue, labelsPred)
			if err != nil {
				t.Fatalf("HomogeneityScore: %v", err)
			}
			assertClose(t, "homogeneity", got, fixtureFloat(t, fx, "homogeneity"))

			got, err = CompletenessScore(labelsTrue, labelsPred)
			if err != nil {
				t.Fatalf("CompletenessScore: %v", err)
			}
			assertClose(t, "completeness", got, fixtureFloat(t, fx, "completeness"))

			got, err = VMeasure(labelsTrue, labelsPred)
			if err != nil {
				t.Fatalf("VMeasure: %v", err)
			}
			assertClose(t, "v_measure", got, fixtureFloat(t, fx, "v_measure"))

			got, err = AdjustedMutualInfo(labelsTrue, labelsPred)
			if err != nil {
				t.Fatalf("AdjustedMutualInfo: %v", err)
			}
			assertClose(t, "adjusted_mutual_info", got, fixtureFloat(t, fx, "adjusted_mutual_info"))
		})
	}
}

func TestSilhouette_AgainstSklearn(t *testing.T) {
	fx := exampleFixture(t, "clustering")
	X := fixtureMatrix(t, fx, "X_silhouette")
	labels := fixtureSlice(t, fx, "labels_silhouette")

	got, err := SilhouetteScore(X, labels)
	if err != nil {
		t.Fatalf("SilhouetteScore: %v", err)
	}
	assertClose(t, "silhouette", got, fixtureFloat(t, fx, "silhouette"))
}

func fixtureMatrix(t *testing.T, fx map[string]interface{}, key string) [][]float64 {
	t.Helper()
	raw, ok := fx[key].([]interface{})
	if !ok {
		t.Fatalf("fixture key %q is not an array", key)
	}
	out := make([][]float64, len(raw))
	for i, row := range raw {
		rowRaw, ok := row.([]interface{})
		if !ok {
			t.Fatalf("fixture key %q[%d] is not an array", key, i)
		}
		out[i] = make([]float64, len(rowRaw))
		for j, v := range rowRaw {
			f, ok := v.(float64)
			if !ok {
				t.Fatalf("fixture key %q[%d][%d] is not a float: %v", key, i, j, v)
			}
			out[i][j] = f
		}
	}
	return out
}

// sklearn requires 2 <= n_labels <= n_samples-1 and raises otherwise.
func TestSilhouette_InvalidClusterCount(t *testing.T) {
	X := [][]float64{{1, 2}, {3, 4}}
	cases := map[string]struct {
		X      [][]float64
		labels []float64
	}{
		"single cluster":        {X, []float64{0, 0}},
		"as many as samples":    {X, []float64{0, 1}},
		"single sample":         {[][]float64{{1, 2}}, []float64{0}},
		"singleton per cluster": {[][]float64{{0}, {1}, {2}}, []float64{5, 6, 7}},
	}
	for name, tc := range cases {
		if _, err := SilhouetteScore(tc.X, tc.labels); !errors.Is(err, ErrInvalidClusterCount) {
			t.Errorf("%s: got %v, want ErrInvalidClusterCount", name, err)
		}
	}
}

// Singleton clusters are valid as long as another cluster has several members;
// they contribute a silhouette of 0.
func TestSilhouette_SingletonClusterScoresZero(t *testing.T) {
	X := [][]float64{{0}, {1}, {10}}
	got, err := SilhouetteScore(X, []float64{0, 0, 1})
	if err != nil {
		t.Fatalf("SilhouetteScore: %v", err)
	}
	// Samples 0 and 1: a = 1, b = (10-0 or 10-1); the singleton {10} scores 0.
	want := ((10.0-1)/10.0 + (9.0-1)/9.0) / 3
	assertClose(t, "mean silhouette", got, want)
}

func TestSilhouette_Validation(t *testing.T) {
	// Ragged X.
	if _, err := SilhouetteScore([][]float64{{1, 2}, {3}}, []float64{0, 0}); err == nil {
		t.Error("expected error for ragged X, got nil")
	}
	// Length mismatch.
	if _, err := SilhouetteScore([][]float64{{1, 2}, {3, 4}}, []float64{0}); err == nil {
		t.Error("expected error for label length mismatch, got nil")
	}
	// NaN in X.
	if _, err := SilhouetteScore([][]float64{{1, math.NaN()}, {3, 4}}, []float64{0, 0}); err == nil {
		t.Error("expected error for NaN in X, got nil")
	}
}

func TestClustering_Validation(t *testing.T) {
	if _, err := AdjustedRandIndex([]float64{0, 1}, []float64{0}); err == nil {
		t.Error("expected error for length mismatch, got nil")
	}
	if _, err := AdjustedRandIndex(nil, nil); err == nil {
		t.Error("expected error for empty input, got nil")
	}
	if _, err := HomogeneityScore([]float64{math.NaN()}, []float64{0}); err == nil {
		t.Error("expected error for NaN label, got nil")
	}
}

func TestClustering_SingleClusterEach(t *testing.T) {
	// Both labelings are a single cluster: perfect match for ARI and AMI.
	labelsTrue := []float64{1, 1, 1, 1}
	labelsPred := []float64{0, 0, 0, 0}
	ari, err := AdjustedRandIndex(labelsTrue, labelsPred)
	if err != nil {
		t.Fatalf("AdjustedRandIndex: %v", err)
	}
	assertClose(t, "ari", ari, 1.0)
	ami, err := AdjustedMutualInfo(labelsTrue, labelsPred)
	if err != nil {
		t.Fatalf("AdjustedMutualInfo: %v", err)
	}
	assertClose(t, "ami", ami, 1.0)
	// Homogeneity/completeness with zero-entropy true labels are perfect.
	h, err := HomogeneityScore(labelsTrue, labelsPred)
	if err != nil {
		t.Fatalf("HomogeneityScore: %v", err)
	}
	assertClose(t, "homogeneity", h, 1.0)
	c, err := CompletenessScore(labelsTrue, labelsPred)
	if err != nil {
		t.Fatalf("CompletenessScore: %v", err)
	}
	assertClose(t, "completeness", c, 1.0)
}

func TestClustering_OneClusterEach(t *testing.T) {
	// One labeling has a single cluster, the other does not: AMI must be 0.0.
	labelsTrue := []float64{0, 0, 0, 0}
	labelsPred := []float64{0, 1, 2, 3}
	ami, err := AdjustedMutualInfo(labelsTrue, labelsPred)
	if err != nil {
		t.Fatalf("AdjustedMutualInfo: %v", err)
	}
	assertClose(t, "ami", ami, 0.0)
	ari, err := AdjustedRandIndex(labelsTrue, labelsPred)
	if err != nil {
		t.Fatalf("AdjustedRandIndex: %v", err)
	}
	assertClose(t, "ari", ari, 0.0)
}
