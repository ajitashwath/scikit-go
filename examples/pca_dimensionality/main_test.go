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
	// Known values for standardized iris: PC1+PC2 explain 95.8% of the variance (sklearn agrees).
	if got := s["variance_kept_2"]; got < 0.957 || got > 0.959 {
		t.Errorf("variance kept by 2 components = %.4f, want about 0.9581", got)
	}
	if s["reconstruction_rmse_2"] <= 0 || s["reconstruction_rmse_2"] > 0.3 {
		t.Errorf("reconstruction RMSE = %.4f, want in (0, 0.3]", s["reconstruction_rmse_2"])
	}
	if s["accuracy_4d"] < 0.85 || s["accuracy_2d"] < 0.75 {
		t.Errorf("accuracies = %.4f (4d), %.4f (2d), want >= 0.85 and >= 0.75", s["accuracy_4d"], s["accuracy_2d"])
	}
	if !strings.Contains(out.String(), "cumulative") {
		t.Error("output is missing the cumulative variance table")
	}
}
