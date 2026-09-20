package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRun(t *testing.T) {
	dir := t.TempDir()
	var out strings.Builder
	s, err := run(&out, dir)
	if err != nil {
		t.Fatal(err)
	}
	if s["agreement"] != 1 {
		t.Errorf("loaded model agreed on %.4f of predictions, want 1", s["agreement"])
	}
	if s["corrupt_rejected"] != 1 {
		t.Error("a truncated model file was not rejected")
	}
	if _, err := os.Stat(filepath.Join(dir, "iris_model.gob")); err != nil {
		t.Errorf("model file was not written: %v", err)
	}
	if strings.Contains(out.String(), "BUG") {
		t.Errorf("output reports a bug:\n%s", out.String())
	}
}
