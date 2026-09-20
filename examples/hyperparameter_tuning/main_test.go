package main

import (
	"strings"
	"testing"
)

func TestRun(t *testing.T) {
	var out strings.Builder
	s, err := run(&out)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"cv_knn", "cv_tree", "cv_forest", "cv_svc"} {
		if got := s[key]; got < 0.8 || got > 1 {
			t.Errorf("%s = %.4f, want within [0.8, 1]", key, got)
		}
	}
	if s["grid_combinations"] != 8 {
		t.Errorf("grid evaluated %v combinations, want 8 (4 C values x 2 kernels)", s["grid_combinations"])
	}
	// The search picks the best cross-validated accuracy, which cannot be below the
	// untuned default SVC's (the grid contains C=1, the default, with the rbf kernel).
	if s["grid_best_cv"] < s["cv_svc"]-1e-9 {
		t.Errorf("tuned CV accuracy %.4f is below the untuned SVC's %.4f", s["grid_best_cv"], s["cv_svc"])
	}
	if s["test_accuracy"] < 0.85 {
		t.Errorf("held-out test accuracy = %.4f, want >= 0.85", s["test_accuracy"])
	}
	for _, want := range []string{"grid search over 8 combinations", "best: C=", "held-out test accuracy"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output is missing %q", want)
		}
	}
}
