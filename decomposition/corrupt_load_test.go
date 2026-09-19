package decomposition

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

func TestLoadPCA_CorruptPayload(t *testing.T) {
	p := NewPCA()
	p.NComponents = 2
	if err := p.Fit([][]float64{{1, 2, 3}, {2, 1, 0}, {3, 5, 1}, {4, 3, 9}, {0, 1, 2}}, nil); err != nil {
		t.Fatal(err)
	}
	valid := filepath.Join(t.TempDir(), "pca.gob")
	if err := p.Save(valid); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPCA(valid); err != nil {
		t.Fatalf("the unmodified file must load: %v", err)
	}
	cases := map[string]func(*pcaGob){
		"no components":       func(g *pcaGob) { g.Components = nil },
		"no mean":             func(g *pcaGob) { g.Mean = nil },
		"short component row": func(g *pcaGob) { g.Components[0] = g.Components[0][:1] },
		"short mean":          func(g *pcaGob) { g.Mean = g.Mean[:2] },
		"short variance":      func(g *pcaGob) { g.ExplainedVariance = g.ExplainedVariance[:1] },
		"short ratio":         func(g *pcaGob) { g.ExplainedVarianceRatio = nil },
		"short singular":      func(g *pcaGob) { g.SingularValues = g.SingularValues[:1] },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := LoadPCA(mutatePayload(t, valid, mutate)); err == nil {
				t.Error("expected an error for a corrupt payload")
			}
		})
	}
}
