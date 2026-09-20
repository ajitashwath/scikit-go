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
	if scores["best_k"] != trueSegments {
		t.Errorf("best k = %v, want %d (the number of generated segments)", scores["best_k"], trueSegments)
	}
	if scores["silhouette"] < 0.6 {
		t.Errorf("silhouette = %.4f, want >= 0.6", scores["silhouette"])
	}
	if scores["adjusted_rand_index"] < 0.95 {
		t.Errorf("adjusted Rand index = %.4f, want >= 0.95", scores["adjusted_rand_index"])
	}
	for _, name := range featureNames {
		if !strings.Contains(out.String(), name) {
			t.Errorf("output is missing feature %q", name)
		}
	}
}
