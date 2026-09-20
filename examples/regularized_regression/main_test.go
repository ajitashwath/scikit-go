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
	for _, key := range []string{"ols", "ridge", "lasso", "elastic_net"} {
		if got := s[key+"_r2"]; got < 0.4 || got > 0.65 {
			t.Errorf("%s test R^2 = %.4f, want within [0.4, 0.65] (diabetes is a hard target)", key, got)
		}
	}
	if s["lasso_zeros"] < 2 {
		t.Errorf("Lasso zeroed %v coefficients, want at least 2", s["lasso_zeros"])
	}
	// Sparsity should cost almost nothing: within 0.03 R^2 of OLS.
	if s["lasso_r2"] < s["ols_r2"]-0.03 {
		t.Errorf("Lasso R^2 %.4f fell well below OLS %.4f", s["lasso_r2"], s["ols_r2"])
	}
	if s["ridge_shrink"] >= 1 || s["ridge_shrink"] < 0.5 {
		t.Errorf("ridge coefficient length is %.3f of OLS's, want in [0.5, 1)", s["ridge_shrink"])
	}
	for _, key := range []string{"ridge", "lasso", "elastic_net"} {
		if a := s["tuned_"+key+"_alpha"]; a < 0.01 || a > 64 {
			t.Errorf("tuned %s alpha = %v is outside the searched grid", key, a)
		}
		if got := s["tuned_"+key+"_r2"]; got < 0.4 {
			t.Errorf("tuned %s test R^2 = %.4f, want >= 0.4", key, got)
		}
	}
	for _, want := range []string{"coefficients", "zero coefs", "choosing alpha", "bmi"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output is missing %q", want)
		}
	}
}
