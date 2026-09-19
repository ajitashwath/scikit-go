package cluster

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

func TestLoadKMeans_CorruptPayload(t *testing.T) {
	k := NewKMeans()
	k.NClusters = 2
	if err := k.Fit([][]float64{{0, 0}, {0, 1}, {9, 9}, {9, 8}}, nil); err != nil {
		t.Fatal(err)
	}
	valid := filepath.Join(t.TempDir(), "km.gob")
	if err := k.Save(valid); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadKMeans(valid); err != nil {
		t.Fatalf("the unmodified file must load: %v", err)
	}
	cases := map[string]func(*kmeansGob){
		"no centers":        func(g *kmeansGob) { g.Centers = nil },
		"count mismatch":    func(g *kmeansGob) { g.NClusters = 5 },
		"ragged center":     func(g *kmeansGob) { g.Centers[1] = g.Centers[1][:1] },
		"empty center rows": func(g *kmeansGob) { g.Centers = [][]float64{{}, {}}; g.NClusters = 2 },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := LoadKMeans(mutatePayload(t, valid, mutate)); err == nil {
				t.Error("expected an error for a corrupt payload")
			}
		})
	}
}
