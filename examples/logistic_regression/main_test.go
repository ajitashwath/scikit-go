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
	if s["test_accuracy"] < 0.9 {
		t.Errorf("test accuracy = %.4f, want >= 0.9", s["test_accuracy"])
	}
	// Weaker regularization (bigger C) makes the model more confident.
	if s["confidence_high_c"] <= s["confidence_low_c"] {
		t.Errorf("confidence at C=100 (%.4f) is not above confidence at C=0.001 (%.4f)",
			s["confidence_high_c"], s["confidence_low_c"])
	}
	// Near-uniform predictions at C=0.001 (3 classes, so 1/3 is the floor).
	if s["confidence_low_c"] > 0.6 {
		t.Errorf("confidence at C=0.001 = %.4f, want a heavily shrunk model (< 0.6)", s["confidence_low_c"])
	}
	if s["best_c"] < 0.1 {
		t.Errorf("best C = %v, want a moderate value (>= 0.1)", s["best_c"])
	}
	if p := s["new_flower_top_prob"]; p < 1.0/3 || p > 1 {
		t.Errorf("top probability %v is not a probability over 3 classes", p)
	}
	for _, want := range []string{"probabilities", "regularization strength", "a new flower", "versicolor"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output is missing %q", want)
		}
	}
}

func TestLeastConfident(t *testing.T) {
	proba := [][]float64{{0.9, 0.05, 0.05}, {0.4, 0.35, 0.25}, {0.6, 0.3, 0.1}, {0.34, 0.33, 0.33}}
	got := leastConfident(proba, 2)
	if got[0] != 3 || got[1] != 1 {
		t.Errorf("leastConfident = %v, want [3 1]", got)
	}
}
