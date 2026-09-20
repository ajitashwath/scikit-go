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
	lin, rbf := s["SVC (linear kernel)"], s["SVC (rbf kernel)"]
	if rbf <= lin {
		t.Errorf("rbf accuracy %.4f is not above linear accuracy %.4f; the moons should not be linearly separable", rbf, lin)
	}
	if rbf < 0.88 || s["KNeighborsClassifier"] < 0.88 || s["RandomForestClassifier"] < 0.88 {
		t.Errorf("nonlinear models scored %v, want >= 0.88 each", s)
	}
	// The text plot must contain both decision regions and both kinds of test point.
	text := out.String()
	for _, r := range []string{".", "+", "o", "x"} {
		if !strings.Contains(text[strings.Index(text, "decision regions"):], r) {
			t.Errorf("plot is missing %q", r)
		}
	}
}
