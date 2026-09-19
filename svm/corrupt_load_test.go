package svm

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

func TestLoadSVM_CorruptPayload(t *testing.T) {
	X := [][]float64{{0, 0}, {0, 1}, {1, 0}, {5, 5}, {5, 6}, {6, 5}, {10, 0}, {10, 1}, {11, 0}}
	y := []float64{0, 0, 0, 1, 1, 1, 2, 2, 2}
	s := NewSVC()
	s.Probability = true
	if err := s.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	r := NewSVR()
	if err := r.Fit(X, []float64{1, 2, 1, 3, 4, 3, 5, 6, 5}); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	sPath, rPath := filepath.Join(dir, "svc.gob"), filepath.Join(dir, "svr.gob")
	if err := s.Save(sPath); err != nil {
		t.Fatal(err)
	}
	if err := r.Save(rPath); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadSVC(sPath); err != nil {
		t.Fatalf("the unmodified SVC file must load: %v", err)
	}
	if _, err := LoadSVR(rPath); err != nil {
		t.Fatalf("the unmodified SVR file must load: %v", err)
	}

	svcCases := map[string]func(*svmGob){
		"one class":              func(g *svmGob) { g.Classes = g.Classes[:1] },
		"nsv length":             func(g *svmGob) { g.NSV = g.NSV[:2] },
		"nsv sum":                func(g *svmGob) { g.NSV[0]++ },
		"negative nsv":           func(g *svmGob) { g.NSV[0] = -1 },
		"short coef row":         func(g *svmGob) { g.Coef[0] = g.Coef[0][:1] },
		"missing coef row":       func(g *svmGob) { g.Coef = g.Coef[:1] },
		"support vector width":   func(g *svmGob) { g.SV[0] = g.SV[0][:1] },
		"no features":            func(g *svmGob) { g.NFeatures = 0 },
		"rho count":              func(g *svmGob) { g.Rho = g.Rho[:1] },
		"probA/probB mismatch":   func(g *svmGob) { g.ProbB = g.ProbB[:1] },
		"probA wrong pair count": func(g *svmGob) { g.ProbA, g.ProbB = g.ProbA[:1], g.ProbB[:1] },
	}
	for name, mutate := range svcCases {
		t.Run("svc "+name, func(t *testing.T) {
			if _, err := LoadSVC(mutatePayload(t, sPath, mutate)); err == nil {
				t.Error("expected an error for a corrupt payload")
			}
		})
	}
	svrCases := map[string]func(*svmGob){
		"no coef":              func(g *svmGob) { g.Coef = nil },
		"short coef":           func(g *svmGob) { g.Coef[0] = g.Coef[0][:1] },
		"support vector width": func(g *svmGob) { g.SV[0] = g.SV[0][:1] },
		"no features":          func(g *svmGob) { g.NFeatures = 0 },
		"no rho":               func(g *svmGob) { g.Rho = nil },
	}
	for name, mutate := range svrCases {
		t.Run("svr "+name, func(t *testing.T) {
			if _, err := LoadSVR(mutatePayload(t, rPath, mutate)); err == nil {
				t.Error("expected an error for a corrupt payload")
			}
		})
	}
}
