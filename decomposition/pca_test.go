package decomposition

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type pcaFixture struct {
	X                      [][]float64 `json:"X"`
	XTest                  [][]float64 `json:"X_test"`
	Components             [][]float64 `json:"components"`
	ExplainedVariance      []float64   `json:"explained_variance"`
	ExplainedVarianceRatio []float64   `json:"explained_variance_ratio"`
	SingularValues         []float64   `json:"singular_values"`
	Mean                   []float64   `json:"mean"`
	Transformed            [][]float64 `json:"transformed"`
	TransformedTest        [][]float64 `json:"transformed_test"`
	Inverse                [][]float64 `json:"inverse"`
}

func loadPCAFixtures(t *testing.T) map[string]pcaFixture {
	t.Helper()
	path := filepath.Join("testdata", "decomposition_fixtures.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read fixtures: %v", err)
	}
	var fixtures map[string]pcaFixture
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

func TestPCA_AgainstSklearn(t *testing.T) {
	fixtures := loadPCAFixtures(t)
	const tol = 1e-6

	specs := []struct {
		key         string
		nComponents int
	}{
		{"pca_k2", 2},
		{"pca_k4", 4},
		{"pca_k3_n6", 3},
	}
	for _, spec := range specs {
		t.Run(spec.key, func(t *testing.T) {
			fx := fixtures[spec.key]
			model := NewPCA()
			model.NComponents = spec.nComponents
			if err := model.Fit(fx.X, nil); err != nil {
				t.Fatalf("Fit: %v", err)
			}

			assertSliceClose(t, "mean", model.Mean(), fx.Mean, tol)
			assertSliceClose(t, "explained_variance", model.ExplainedVariance(), fx.ExplainedVariance, tol)
			assertSliceClose(t, "explained_variance_ratio", model.ExplainedVarianceRatio(), fx.ExplainedVarianceRatio, tol)
			assertMatrixClose(t, "components", model.Components(), fx.Components, tol)

			transformed, err := model.Transform(fx.X)
			if err != nil {
				t.Fatalf("Transform: %v", err)
			}
			assertMatrixClose(t, "transformed", transformed, fx.Transformed, tol)

			if fx.TransformedTest != nil {
				transformedTest, err := model.Transform(fx.XTest)
				if err != nil {
					t.Fatalf("Transform(test): %v", err)
				}
				assertMatrixClose(t, "transformed_test", transformedTest, fx.TransformedTest, tol)

				inverse, err := model.InverseTransform(transformedTest)
				if err != nil {
					t.Fatalf("InverseTransform: %v", err)
				}
				assertMatrixClose(t, "inverse", inverse, fx.Inverse, tol)
			}
		})
	}
}

func TestPCA_DefaultComponents(t *testing.T) {
	X := [][]float64{
		{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}, {5, 6, 7, 8},
	}
	model := NewPCA()
	if err := model.Fit(X, nil); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if len(model.Components()) != 4 {
		t.Fatalf("expected min(n,p)=4 components, got %d", len(model.Components()))
	}
}

func TestPCA_Validation(t *testing.T) {
	X := [][]float64{{1, 2}, {3, 4}}
	model := NewPCA()
	if _, err := model.Transform(X); err == nil {
		t.Error("expected error transforming before fit")
	}
	model.NComponents = 10
	if err := model.Fit(X, nil); err == nil {
		t.Error("expected error for n_components > min(n,p)")
	}
}

func TestPCA_RoundTrip(t *testing.T) {
	X := [][]float64{
		{1, 2, 3}, {2, 4, 6}, {3, 6, 9}, {4, 8, 12},
	}
	model := NewPCA()
	model.NComponents = 2
	if err := model.Fit(X, nil); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	transformed, err := model.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}
	recon, err := model.InverseTransform(transformed)
	if err != nil {
		t.Fatalf("InverseTransform: %v", err)
	}
	// Rank-deficient data: full reconstruction is exact.
	for i := range recon {
		for j := range recon[i] {
			if math.Abs(recon[i][j]-X[i][j]) > 1e-8 {
				t.Errorf("round trip[%d][%d]: got %v, want %v", i, j, recon[i][j], X[i][j])
			}
		}
	}
}

func TestPCA_SaveLoad(t *testing.T) {
	X := [][]float64{
		{1, 2, 3}, {2, 4, 6}, {3, 6, 9}, {4, 8, 12}, {5, 10, 15},
	}
	model := NewPCA()
	model.NComponents = 2
	if err := model.Fit(X, nil); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	path := filepath.Join(t.TempDir(), "pca.gob")
	if err := model.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadPCA(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	assertMatrixClose(t, "components", loaded.Components(), model.Components(), 1e-12)
	transformedWant, _ := model.Transform(X)
	transformedGot, err := loaded.Transform(X)
	if err != nil {
		t.Fatalf("Transform after load: %v", err)
	}
	assertMatrixClose(t, "transformed", transformedGot, transformedWant, 1e-12)
}
