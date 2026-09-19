package decomposition

import (
	"errors"
	"math"
	"testing"

	"scikit-go/internal/matutil"
)

var pcaData = [][]float64{
	{2.5, 2.4, 1}, {0.5, 0.7, 2}, {2.2, 2.9, 0}, {1.9, 2.2, 3},
	{3.1, 3.0, 1}, {2.3, 2.7, 2}, {2, 1.6, 0}, {1, 1.1, 2},
}

func TestPCA_FitTransformMatchesFitThenTransform(t *testing.T) {
	a := NewPCA()
	a.NComponents = 2
	viaFitTransform, err := a.FitTransform(pcaData)
	if err != nil {
		t.Fatalf("FitTransform: %v", err)
	}
	b := NewPCA()
	b.NComponents = 2
	if err := b.Fit(pcaData, nil); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	viaTransform, err := b.Transform(pcaData)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}
	for i := range viaFitTransform {
		for j := range viaFitTransform[i] {
			if viaFitTransform[i][j] != viaTransform[i][j] {
				t.Fatalf("[%d][%d]: %v vs %v", i, j, viaFitTransform[i][j], viaTransform[i][j])
			}
		}
	}
	if len(viaFitTransform[0]) != 2 {
		t.Errorf("got %d output components, want 2", len(viaFitTransform[0]))
	}
}

func TestPCA_AccessorsAndReconstruction(t *testing.T) {
	p := NewPCA() // all components
	Z, err := p.FitTransform(pcaData)
	if err != nil {
		t.Fatalf("FitTransform: %v", err)
	}
	ratio := p.ExplainedVarianceRatio()
	var sum float64
	for i, r := range ratio {
		sum += r
		if i > 0 && r > ratio[i-1]+1e-12 {
			t.Errorf("explained variance ratio is not descending: %v", ratio)
		}
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("ratios over all components sum to %v, want 1", sum)
	}
	if len(p.ExplainedVariance()) != len(ratio) || len(p.Mean()) != 3 || len(p.Components()) != len(ratio) {
		t.Errorf("accessor shapes: variance %d, mean %d, components %d, ratio %d",
			len(p.ExplainedVariance()), len(p.Mean()), len(p.Components()), len(ratio))
	}

	// Keeping every component, InverseTransform must reconstruct the data.
	back, err := p.InverseTransform(Z)
	if err != nil {
		t.Fatalf("InverseTransform: %v", err)
	}
	for i := range pcaData {
		for j := range pcaData[i] {
			if math.Abs(back[i][j]-pcaData[i][j]) > 1e-9 {
				t.Fatalf("reconstruction [%d][%d]: got %v, want %v", i, j, back[i][j], pcaData[i][j])
			}
		}
	}

	// Accessors return copies.
	p.Mean()[0] = 1e9
	p.Components()[0][0] = 1e9
	p.ExplainedVariance()[0] = 1e9
	p.ExplainedVarianceRatio()[0] = 1e9
	if p.Mean()[0] == 1e9 || p.Components()[0][0] == 1e9 || p.ExplainedVariance()[0] == 1e9 || p.ExplainedVarianceRatio()[0] == 1e9 {
		t.Error("an accessor returned the internal slice")
	}
}

func TestPCA_BeforeFitAndBadInput(t *testing.T) {
	p := NewPCA()
	if p.Components() != nil || p.ExplainedVariance() != nil || p.ExplainedVarianceRatio() != nil || p.Mean() != nil {
		t.Error("accessors should return nil before Fit")
	}
	if _, err := p.Transform(pcaData); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Transform: got %v", err)
	}
	if _, err := p.InverseTransform([][]float64{{1}}); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("InverseTransform: got %v", err)
	}
	if _, err := p.FitTransform(nil); !errors.Is(err, matutil.ErrEmptyInput) {
		t.Errorf("FitTransform(nil): got %v", err)
	}

	p.NComponents = 2
	if err := p.Fit(pcaData, nil); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if _, err := p.InverseTransform(nil); !errors.Is(err, matutil.ErrEmptyInput) {
		t.Errorf("InverseTransform(nil): got %v", err)
	}
	if _, err := p.InverseTransform([][]float64{{1, 2, 3}}); !errors.Is(err, matutil.ErrDimMismatch) {
		t.Errorf("InverseTransform with the wrong width: got %v", err)
	}
	if _, err := p.Transform([][]float64{{1, 2}}); !errors.Is(err, matutil.ErrDimMismatch) {
		t.Errorf("Transform with the wrong width: got %v", err)
	}
}
