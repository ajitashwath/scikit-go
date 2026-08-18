package metrics

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const floatTolerance = 1e-10

func assertClose(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > floatTolerance {
		t.Errorf("%s: got %.12f, want %.12f (diff %.2e)", name, got, want, math.Abs(got-want))
	}
}

func assertMatrixClose(t *testing.T, name string, got, want [][]int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: row count mismatch, got %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("%s[%d]: col count mismatch, got %d, want %d", name, i, len(got[i]), len(want[i]))
		}
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Errorf("%s[%d][%d]: got %d, want %d", name, i, j, got[i][j], want[i][j])
			}
		}
	}
}

func loadMetricsFixtures(t *testing.T) map[string]map[string]interface{} {
	t.Helper()
	path := filepath.Join("testdata", "metrics_fixtures.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read fixtures at %s: %v", path, err)
	}
	var fixtures map[string]map[string]interface{}
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatalf("failed to parse fixtures: %v", err)
	}
	return fixtures
}

func fixtureFloat(t *testing.T, fx map[string]interface{}, key string) float64 {
	t.Helper()
	v, ok := fx[key].(float64)
	if !ok {
		t.Fatalf("fixture key %q is not a float: %v", key, fx[key])
	}
	return v
}

func fixtureSlice(t *testing.T, fx map[string]interface{}, key string) []float64 {
	t.Helper()
	raw, ok := fx[key].([]interface{})
	if !ok {
		t.Fatalf("fixture key %q is not an array: %v", key, fx[key])
	}
	out := make([]float64, len(raw))
	for i, v := range raw {
		f, ok := v.(float64)
		if !ok {
			t.Fatalf("fixture key %q[%d] is not a float: %v", key, i, v)
		}
		out[i] = f
	}
	return out
}

func fixtureIntMatrix(t *testing.T, fx map[string]interface{}, key string) [][]int {
	t.Helper()
	raw, ok := fx[key].([]interface{})
	if !ok {
		t.Fatalf("fixture key %q is not an array: %v", key, fx[key])
	}
	out := make([][]int, len(raw))
	for i, row := range raw {
		rowRaw, ok := row.([]interface{})
		if !ok {
			t.Fatalf("fixture key %q[%d] is not an array", key, i)
		}
		out[i] = make([]int, len(rowRaw))
		for j, v := range rowRaw {
			f, ok := v.(float64)
			if !ok {
				t.Fatalf("fixture key %q[%d][%d] is not numeric: %v", key, i, j, v)
			}
			out[i][j] = int(f)
		}
	}
	return out
}

func exampleFixture(t *testing.T, name string) map[string]interface{} {
	t.Helper()
	fx, ok := loadMetricsFixtures(t)[name]
	if !ok {
		t.Fatalf("fixture %q not found", name)
	}
	return fx
}

func TestValidateVectors(t *testing.T) {
	if _, err := R2Score(nil, []float64{1}); err == nil {
		t.Error("expected error for nil y_true, got nil")
	}
	if _, err := R2Score([]float64{1}, nil); err == nil {
		t.Error("expected error for nil y_pred, got nil")
	}
	if _, err := R2Score([]float64{1}, []float64{1, 2}); err == nil {
		t.Error("expected error for length mismatch, got nil")
	}
}

func TestValidateLabels(t *testing.T) {
	if _, err := AccuracyScore([]float64{math.NaN()}, []float64{0}); err == nil {
		t.Error("expected error for NaN label, got nil")
	}
	if _, err := AccuracyScore([]float64{0}, []float64{math.Inf(1)}); err == nil {
		t.Error("expected error for Inf label, got nil")
	}
	if _, err := AccuracyScore([]float64{}, []float64{}); err == nil {
		t.Error("expected error for empty input, got nil")
	}
}

func fmtName(prefix string, i int) string {
	return fmt.Sprintf("%s[%d]", prefix, i)
}
