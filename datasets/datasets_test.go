package datasets

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

func assertSliceClose(t *testing.T, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch, got %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		assertClose(t, fmt.Sprintf("%s[%d]", name, i), got[i], want[i])
	}
}

// loadJSON reads a fixture file from testdata into a generic map.
func loadJSON(t *testing.T, name string) map[string]interface{} {
	t.Helper()
	path := filepath.Join("testdata", name)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read fixtures at %s: %v", path, err)
	}
	var out map[string]interface{}
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("failed to parse %s: %v", path, err)
	}
	return out
}

func fixtureInt(t *testing.T, fx map[string]interface{}, key string) int {
	t.Helper()
	v, ok := fx[key].(float64)
	if !ok {
		t.Fatalf("fixture key %q is not numeric: %v", key, fx[key])
	}
	return int(v)
}

func fixtureFloat(t *testing.T, fx map[string]interface{}, key string) float64 {
	t.Helper()
	v, ok := fx[key].(float64)
	if !ok {
		t.Fatalf("fixture key %q is not numeric: %v", key, fx[key])
	}
	return v
}

func fixtureFloatSlice(t *testing.T, fx map[string]interface{}, key string) []float64 {
	t.Helper()
	raw, ok := fx[key].([]interface{})
	if !ok {
		t.Fatalf("fixture key %q is not an array: %v", key, fx[key])
	}
	out := make([]float64, len(raw))
	for i, v := range raw {
		f, ok := v.(float64)
		if !ok {
			t.Fatalf("fixture key %q[%d] is not numeric: %v", key, i, v)
		}
		out[i] = f
	}
	return out
}

func fixtureFloatMatrix(t *testing.T, fx map[string]interface{}, key string) [][]float64 {
	t.Helper()
	raw, ok := fx[key].([]interface{})
	if !ok {
		t.Fatalf("fixture key %q is not an array: %v", key, fx[key])
	}
	out := make([][]float64, len(raw))
	for i, row := range raw {
		rowRaw, ok := row.([]interface{})
		if !ok {
			t.Fatalf("fixture key %q[%d] is not an array", key, i)
		}
		out[i] = make([]float64, len(rowRaw))
		for j, v := range rowRaw {
			f, ok := v.(float64)
			if !ok {
				t.Fatalf("fixture key %q[%d][%d] is not numeric: %v", key, i, j, v)
			}
			out[i][j] = f
		}
	}
	return out
}
