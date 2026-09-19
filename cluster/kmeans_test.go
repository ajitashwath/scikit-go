package cluster

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type kmeansFixture struct {
	X       [][]float64 `json:"X"`
	XTest   [][]float64 `json:"X_test"`
	Labels  []float64   `json:"labels"`
	Centers [][]float64 `json:"centers"`
	Inertia float64     `json:"inertia"`
	NIter   int         `json:"n_iter"`
	Pred    []float64   `json:"pred"`
}

func loadKMeansFixtures(t *testing.T) map[string]kmeansFixture {
	t.Helper()
	path := filepath.Join("testdata", "cluster_fixtures.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read fixtures: %v", err)
	}
	var fixtures map[string]kmeansFixture
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatalf("failed to parse fixtures: %v", err)
	}
	return fixtures
}

func assertClose(t *testing.T, name string, got, want float64, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %.12f, want %.12f (diff %.2e)", name, got, want, math.Abs(got-want))
	}
}

func TestKMeans_AgainstSklearn(t *testing.T) {
	fixtures := loadKMeansFixtures(t)
	const centerTol = 1e-6
	const inertiaTol = 1e-6

	specs := []struct {
		key      string
		nClusters int
		hasPred  bool
	}{
		{"kmeans_blobs3", 3, true},
		{"kmeans_blobs5", 5, false},
		{"kmeans_blobs5_k2", 2, false},
	}
	for _, spec := range specs {
		t.Run(spec.key, func(t *testing.T) {
			fx := fixtures[spec.key]
			model := NewKMeans()
			model.NClusters = spec.nClusters
			model.Seed = 0
			if err := model.Fit(fx.X, nil); err != nil {
				t.Fatalf("Fit: %v", err)
			}

			// Align our cluster ordering to sklearn's before comparing labels.
			// Recover the permutation matching each Go center to the sklearn
			// center minimizing distance.
			centers := model.ClusterCenters()
			used := make([]bool, len(fx.Centers))
			permOrder := make([]int, len(fx.Centers))
			for g := 0; g < len(fx.Centers); g++ {
				bestS, bestD := -1, math.Inf(1)
				for s := 0; s < len(fx.Centers); s++ {
					if used[s] {
						continue
					}
					d := sqDist(centers[g], fx.Centers[s])
					if d < bestD {
						bestD = d
						bestS = s
					}
				}
				used[bestS] = true
				permOrder[g] = bestS
			}

			gotLabels := model.Labels()
			for i := range fx.Labels {
				mapped := float64(permOrder[int(gotLabels[i])])
				if mapped != fx.Labels[i] {
					t.Errorf("labels[%d]: got %d, want %d", i, int(mapped), int(fx.Labels[i]))
					break
				}
			}

			// Reorder centers to the aligned permutation and compare.
			for g := range centers {
				for j := range centers[g] {
					assertClose(t, fmt.Sprintf("center[%d][%d]", g, j),
						centers[g][j], fx.Centers[permOrder[g]][j], centerTol)
				}
			}

			assertClose(t, "inertia", model.Inertia(), fx.Inertia, inertiaTol)
			if spec.hasPred {
				pred, err := model.Predict(fx.XTest)
				if err != nil {
					t.Fatalf("Predict: %v", err)
				}
				if len(pred) != len(fx.Pred) {
					t.Fatalf("pred length mismatch: got %d, want %d", len(pred), len(fx.Pred))
				}
				// Relabel preds through the center-derived permutation.
				for i := range fx.Pred {
					mapped := float64(permOrder[int(pred[i])])
					if mapped != fx.Pred[i] {
						t.Errorf("pred[%d]: got %v, want %v", i, mapped, fx.Pred[i])
						break
					}
				}
			}
		})
	}
}

func TestKMeans_Validation(t *testing.T) {
	X := [][]float64{{1, 2}, {3, 4}, {5, 6}}
	model := NewKMeans()
	if _, err := model.Predict(X); err == nil {
		t.Error("expected error predicting before fit")
	}
	model.NClusters = 0
	if err := model.Fit(X, nil); err == nil {
		t.Error("expected error for n_clusters = 0")
	}
	model = NewKMeans()
	model.NClusters = 10
	if err := model.Fit(X, nil); err == nil {
		t.Error("expected error for n_clusters > n_samples")
	}
	model = NewKMeans()
	model.MaxIter = 0
	if err := model.Fit(X, nil); err == nil {
		t.Error("expected error for max_iter = 0")
	}
}

func TestKMeans_EmptyClusterReseed(t *testing.T) {
	// Data that collapses to fewer points than clusters forces empty-cluster
	// re-seeding during Lloyd iterations.
	X := [][]float64{
		{0, 0}, {0, 0}, {0, 0}, {0, 0},
		{10, 10}, {10, 10}, {10, 10}, {10, 10},
	}
	model := NewKMeans()
	model.NClusters = 5
	model.NInit = 5
	model.MaxIter = 50
	model.Seed = 0
	if err := model.Fit(X, nil); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if math.IsInf(model.Inertia(), 0) {
		t.Fatal("inertia must be finite after empty-cluster reseeding")
	}
	for _, c := range model.ClusterCenters() {
		for _, v := range c {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("center contains non-finite value %v", v)
			}
		}
	}
}

func TestKMeans_Deterministic(t *testing.T) {
	X := [][]float64{
		{1, 1}, {1, 2}, {2, 1}, {2, 2},
		{9, 9}, {9, 10}, {10, 9}, {10, 10},
	}
	a := NewKMeans()
	a.NClusters = 2
	a.Seed = 7
	if err := a.Fit(X, nil); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	b := NewKMeans()
	b.NClusters = 2
	b.Seed = 7
	if err := b.Fit(X, nil); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if a.Inertia() != b.Inertia() {
		t.Errorf("inertia not deterministic: %v vs %v", a.Inertia(), b.Inertia())
	}
}

func TestKMeans_SaveLoad(t *testing.T) {
	X := [][]float64{
		{1, 1}, {1, 2}, {2, 1}, {2, 2},
		{9, 9}, {9, 10}, {10, 9}, {10, 10},
	}
	model := NewKMeans()
	model.NClusters = 2
	model.Seed = 3
	if err := model.Fit(X, nil); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	path := filepath.Join(t.TempDir(), "kmeans.gob")
	if err := model.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadKMeans(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Inertia() != model.Inertia() {
		t.Errorf("inertia mismatch after load: %v vs %v", loaded.Inertia(), model.Inertia())
	}
	predWant, _ := model.Predict(X)
	predGot, err := loaded.Predict(X)
	if err != nil {
		t.Fatalf("Predict after load: %v", err)
	}
	for i := range predWant {
		if predGot[i] != predWant[i] {
			t.Errorf("pred[%d] mismatch after load: got %v, want %v", i, predGot[i], predWant[i])
		}
	}
}
func TestReseedEmptyClusters_MultipleEmpty(t *testing.T) {
	X := [][]float64{{0, 0}, {1, 0}, {10, 0}, {20, 0}, {50, 0}}
	// Clusters 1 and 3 are empty; recomputeCenters leaves zero placeholders there.
	centers := [][]float64{{0.5, 0}, {0, 0}, {10, 0}, {0, 0}}
	counts := []int{2, 0, 1, 0}

	if !reseedEmptyClusters(X, centers, counts) {
		t.Fatal("expected reseeding to be reported")
	}
	// The farthest point from the live centers {0.5,0} and {10,0} is {50,0};
	// the next, now also measuring against {50,0}, is {20,0}. Neither empty
	// cluster may keep its zero placeholder, and they must not collide.
	if centers[1][0] != 50 || centers[3][0] != 20 {
		t.Fatalf("empty clusters reseeded to %v and %v, want [50 0] and [20 0]", centers[1], centers[3])
	}
	if centers[0][0] != 0.5 || centers[2][0] != 10 {
		t.Fatalf("non-empty centers were modified: %v", centers)
	}
}

func TestReseedEmptyClusters_NoneEmpty(t *testing.T) {
	X := [][]float64{{0, 0}, {10, 10}}
	centers := [][]float64{{0, 0}, {10, 10}}
	if reseedEmptyClusters(X, centers, []int{1, 1}) {
		t.Fatal("no cluster was empty, reseed must report false")
	}
}
