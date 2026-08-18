package tree

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type treeFixture struct {
	X                 [][]float64 `json:"X"`
	Y                 []float64   `json:"y"`
	XTest             [][]float64 `json:"X_test"`
	PredTrain         []float64   `json:"pred_train"`
	PredTest          []float64   `json:"pred_test"`
	ProbaTest         [][]float64 `json:"proba_test"`
	Classes           []float64   `json:"classes"`
	FeatureImportances []float64  `json:"feature_importances"`
}

func loadTreeFixtures(t *testing.T) map[string]treeFixture {
	t.Helper()
	path := filepath.Join("testdata", "tree_fixtures.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read fixtures: %v", err)
	}
	var fixtures map[string]treeFixture
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatalf("failed to parse fixtures: %v", err)
	}
	return fixtures
}

func assertClose(t *testing.T, name string, got, want float64, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %.12f, want %.12f (diff %.2e)", name, got, want, math.Abs(got-want))
	}
}

func assertSliceClose(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch, got %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		assertClose(t, fmt.Sprintf("%s[%d]", name, i), got[i], want[i], tol)
	}
}

func assertMatrixClose(t *testing.T, name string, got, want [][]float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: row count mismatch, got %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("%s[%d]: col count mismatch, got %d, want %d", name, i, len(got[i]), len(want[i]))
		}
		for j := range want[i] {
			assertClose(t, fmt.Sprintf("%s[%d][%d]", name, i, j), got[i][j], want[i][j], tol)
		}
	}
}