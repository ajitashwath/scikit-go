package main

import (
	"strings"
	"testing"
)

func TestRun(t *testing.T) {
	var out strings.Builder
	scores, err := run(&out)
	if err != nil {
		t.Fatal(err)
	}
	// Diabetes is a hard, noisy target: R^2 around 0.45-0.50 is what sklearn gets too.
	for _, name := range []string{"LinearRegression", "StandardScaler + KNN", "RandomForestRegressor", "StandardScaler + SVR"} {
		if got, ok := scores[name]; !ok || got < 0.35 || got > 0.75 {
			t.Errorf("%s R^2 = %.4f (present=%v), want within [0.35, 0.75]", name, got, ok)
		}
	}
	// bmi and s5 are the well-known strongest predictors.
	text := out.String()
	for _, want := range []string{"bmi", "s5", "feature importances"} {
		if !strings.Contains(text, want) {
			t.Errorf("output is missing %q", want)
		}
	}
}

func TestTopK(t *testing.T) {
	got := topK([]float64{1, -5, 3, 0}, 2, nil)
	if got[0] != 2 || got[1] != 0 {
		t.Errorf("topK by value = %v, want [2 0]", got)
	}
	got = topK([]float64{1, -5, 3, 0}, 2, func(x float64) float64 {
		if x < 0 {
			return -x
		}
		return x
	})
	if got[0] != 1 || got[1] != 2 {
		t.Errorf("topK by magnitude = %v, want [1 2]", got)
	}
	if n := len(topK([]float64{1, 2}, 5, nil)); n != 2 {
		t.Errorf("topK with k > len returned %d indices, want 2", n)
	}
}
