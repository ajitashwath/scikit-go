package preprocessing

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

func TestLoadStandardScaler_CorruptPayload(t *testing.T) {
	s := NewStandardScaler()
	if _, err := s.FitTransform([][]float64{{1, 2}, {3, 5}, {4, 9}}); err != nil {
		t.Fatal(err)
	}
	valid := filepath.Join(t.TempDir(), "scaler.gob")
	if err := s.Save(valid); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadStandardScaler(valid); err != nil {
		t.Fatalf("the unmodified file must load: %v", err)
	}
	cases := map[string]func(*standardScalerGob){
		"no features":   func(g *standardScalerGob) { g.NFeatures = 0 },
		"short mean":    func(g *standardScalerGob) { g.Mean = g.Mean[:1] },
		"short scale":   func(g *standardScalerGob) { g.Scale = nil },
		"short var":     func(g *standardScalerGob) { g.Var = g.Var[:1] },
		"zero scale":    func(g *standardScalerGob) { g.Scale[0] = 0 },
		"more features": func(g *standardScalerGob) { g.NFeatures = 5 },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := LoadStandardScaler(mutatePayload(t, valid, mutate)); err == nil {
				t.Error("expected an error for a corrupt payload")
			}
		})
	}
}
