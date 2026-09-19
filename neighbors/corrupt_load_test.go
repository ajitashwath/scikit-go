package neighbors

import (
	"encoding/gob"
	"os"
	"path/filepath"
	"testing"
)

// mutatePayload decodes the gob file at path into a T, lets f break it, and writes
// the result to a new file, so each test starts from a valid saved model.
func mutatePayload[T any](t *testing.T, path string, f func(*T)) string {
	t.Helper()
	in, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	var payload T
	if err := gob.NewDecoder(in).Decode(&payload); err != nil {
		t.Fatal(err)
	}
	f(&payload)
	out := filepath.Join(t.TempDir(), "mutated.gob")
	w, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if err := gob.NewEncoder(w).Encode(payload); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestLoadKNN_CorruptPayload(t *testing.T) {
	X := [][]float64{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}}
	y := []float64{0, 0, 0, 1, 1, 1}
	c := NewKNeighborsClassifier()
	if err := c.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	r := NewKNeighborsRegressor()
	if err := r.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	cPath, rPath := filepath.Join(dir, "c.gob"), filepath.Join(dir, "r.gob")
	if err := c.Save(cPath); err != nil {
		t.Fatal(err)
	}
	if err := r.Save(rPath); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadKNeighborsClassifier(cPath); err != nil {
		t.Fatalf("the unmodified classifier file must load: %v", err)
	}
	if _, err := LoadKNeighborsRegressor(rPath); err != nil {
		t.Fatalf("the unmodified regressor file must load: %v", err)
	}
	cases := map[string]func(*knnGob){
		"no samples":          func(g *knnGob) { g.X = nil; g.Y = nil },
		"targets mismatch":    func(g *knnGob) { g.Y = g.Y[:3] },
		"no features":         func(g *knnGob) { g.NFeatures = 0 },
		"ragged row":          func(g *knnGob) { g.X[2] = g.X[2][:1] },
		"wrong feature count": func(g *knnGob) { g.NFeatures = 3 },
		"n_neighbors > n":     func(g *knnGob) { g.NNeighbors = 99 },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := LoadKNeighborsClassifier(mutatePayload(t, cPath, mutate)); err == nil {
				t.Error("classifier: expected an error for a corrupt payload")
			}
			if _, err := LoadKNeighborsRegressor(mutatePayload(t, rPath, mutate)); err == nil {
				t.Error("regressor: expected an error for a corrupt payload")
			}
		})
	}
}
