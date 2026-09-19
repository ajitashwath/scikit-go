package linear

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

func TestLoadLinearRegression_CorruptPayload(t *testing.T) {
	m := NewLinearRegression()
	if err := m.Fit([][]float64{{1, 2}, {2, 1}, {3, 5}, {4, 3}}, []float64{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	valid := filepath.Join(t.TempDir(), "lr.gob")
	if err := m.Save(valid); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadLinearRegression(valid)
	if err != nil {
		t.Fatalf("the unmodified file must load: %v", err)
	}
	if loaded.Tol != m.Tol || loaded.Rank() != m.Rank() {
		t.Errorf("tol/rank not restored: tol %v rank %d, want %v and %d", loaded.Tol, loaded.Rank(), m.Tol, m.Rank())
	}
	cases := map[string]func(*linearRegressionGob){
		"no features":        func(g *linearRegressionGob) { g.NFeatures = 0 },
		"short coefficients": func(g *linearRegressionGob) { g.Coef = g.Coef[:1] },
		"no coefficients":    func(g *linearRegressionGob) { g.Coef = nil },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := LoadLinearRegression(mutatePayload(t, valid, mutate)); err == nil {
				t.Error("expected an error for a corrupt payload")
			}
		})
	}
}
