package tree

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

// A tree with an out-of-range or backward child index would panic or loop forever
// during prediction, so Load must reject it up front.
func TestLoadTree_CorruptPayload(t *testing.T) {
	X := [][]float64{{0, 5}, {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0}}
	y := []float64{0, 0, 0, 1, 1, 1}
	c := NewDecisionTreeClassifier()
	if err := c.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	r := NewDecisionTreeRegressor()
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
	if _, err := LoadDecisionTreeClassifier(cPath); err != nil {
		t.Fatalf("the unmodified classifier file must load: %v", err)
	}
	if _, err := LoadDecisionTreeRegressor(rPath); err != nil {
		t.Fatalf("the unmodified regressor file must load: %v", err)
	}
	if len(c.impl.nodes) < 3 {
		t.Fatalf("test tree is too small to corrupt: %d nodes", len(c.impl.nodes))
	}

	cases := map[string]func(*treeGob){
		"no nodes":             func(g *treeGob) { g.Nodes = nil },
		"no features":          func(g *treeGob) { g.NFeatures = 0 },
		"child points back":    func(g *treeGob) { g.Nodes[0].Left = 0 },
		"self-referencing":     func(g *treeGob) { g.Nodes[0].Right = 0 },
		"child out of range":   func(g *treeGob) { g.Nodes[0].Left = len(g.Nodes) + 3 },
		"negative child":       func(g *treeGob) { g.Nodes[0].Right = -4 },
		"split feature range":  func(g *treeGob) { g.Nodes[0].Feature = g.NFeatures },
		"empty node value":     func(g *treeGob) { g.Nodes[1].Value = nil },
		"oversized node value": func(g *treeGob) { g.Nodes[1].Value = append(g.Nodes[1].Value, 0, 0, 0) },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := LoadDecisionTreeClassifier(mutatePayload(t, cPath, mutate)); err == nil {
				t.Error("classifier: expected an error for a corrupt payload")
			}
			if _, err := LoadDecisionTreeRegressor(mutatePayload(t, rPath, mutate)); err == nil {
				t.Error("regressor: expected an error for a corrupt payload")
			}
		})
	}
	t.Run("classifier without classes", func(t *testing.T) {
		if _, err := LoadDecisionTreeClassifier(mutatePayload(t, cPath, func(g *treeGob) { g.Classes = nil })); err == nil {
			t.Error("expected an error")
		}
	})
}
