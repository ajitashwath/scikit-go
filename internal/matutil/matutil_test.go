package matutil

import (
	"errors"
	"math"
	"testing"
)

func TestToDense(t *testing.T) {
	m, err := ToDense([][]float64{{1, 2, 3}, {4, 5, 6}})
	if err != nil {
		t.Fatalf("ToDense: %v", err)
	}
	if r, c := m.Dims(); r != 2 || c != 3 || m.At(1, 2) != 6 {
		t.Errorf("wrong matrix: %dx%d, m[1][2]=%v", r, c, m.At(1, 2))
	}

	cases := map[string]struct {
		X    [][]float64
		want error
	}{
		"no rows":    {nil, ErrEmptyInput},
		"no columns": {[][]float64{{}}, ErrEmptyInput},
		"ragged":     {[][]float64{{1, 2}, {3}}, ErrRaggedInput},
		"nan":        {[][]float64{{1, math.NaN()}}, ErrContainsNaN},
		"inf":        {[][]float64{{math.Inf(-1), 1}}, ErrContainsInf},
	}
	for name, tc := range cases {
		if _, err := ToDense(tc.X); !errors.Is(err, tc.want) {
			t.Errorf("%s: got %v, want %v", name, err, tc.want)
		}
	}
}

func TestDenseToSliceRoundTrip(t *testing.T) {
	in := [][]float64{{1.5, -2}, {3, 4.25}, {0, 9}}
	m, err := ToDense(in)
	if err != nil {
		t.Fatal(err)
	}
	out := DenseToSlice(m)
	for i := range in {
		for j := range in[i] {
			if out[i][j] != in[i][j] {
				t.Errorf("[%d][%d]: got %v, want %v", i, j, out[i][j], in[i][j])
			}
		}
	}
	out[0][0] = 99 // the slice must be a copy, not a view of the matrix
	if m.At(0, 0) != 1.5 {
		t.Error("DenseToSlice returned a view of the matrix rather than a copy")
	}
}

func TestValidateXy(t *testing.T) {
	good := [][]float64{{1, 2}, {3, 4}}
	cases := map[string]struct {
		X    [][]float64
		y    []float64
		want error
	}{
		"ok":            {good, []float64{0, 1}, nil},
		"empty X":       {nil, []float64{1}, ErrEmptyInput},
		"empty y":       {good, nil, ErrEmptyInput},
		"length":        {good, []float64{1}, ErrDimMismatch},
		"ragged":        {[][]float64{{1, 2}, {3}}, []float64{0, 1}, ErrRaggedInput},
		"zero features": {[][]float64{{}, {}}, []float64{0, 1}, ErrEmptyInput},
		"nan in X":      {[][]float64{{1, math.NaN()}, {3, 4}}, []float64{0, 1}, ErrContainsNaN},
		"inf in X":      {[][]float64{{1, 2}, {3, math.Inf(1)}}, []float64{0, 1}, ErrContainsInf},
		"nan in y":      {good, []float64{0, math.NaN()}, ErrContainsNaN},
		"inf in y":      {good, []float64{math.Inf(1), 0}, ErrContainsInf},
	}
	for name, tc := range cases {
		err := ValidateXy(tc.X, tc.y)
		if tc.want == nil && err != nil || tc.want != nil && !errors.Is(err, tc.want) {
			t.Errorf("%s: got %v, want %v", name, err, tc.want)
		}
	}
}

func TestValidateXMatrix(t *testing.T) {
	if err := ValidateXMatrix([][]float64{{1, 2}, {3, 4}}); err != nil {
		t.Errorf("valid matrix: %v", err)
	}
	if err := ValidateXMatrix(nil); !errors.Is(err, ErrEmptyInput) {
		t.Errorf("empty: got %v", err)
	}
	if err := ValidateXMatrix([][]float64{{1, 2}, {3}}); !errors.Is(err, ErrRaggedInput) {
		t.Errorf("ragged: got %v", err)
	}
	if err := ValidateXMatrix([][]float64{{1, math.NaN()}}); !errors.Is(err, ErrContainsNaN) {
		t.Errorf("nan: got %v", err)
	}
}

func TestValidateX(t *testing.T) {
	if err := ValidateX([][]float64{{1, 2}, {3, 4}}, 2); err != nil {
		t.Errorf("valid: %v", err)
	}
	if err := ValidateX(nil, 2); !errors.Is(err, ErrEmptyInput) {
		t.Errorf("empty: got %v", err)
	}
	if err := ValidateX([][]float64{{1, 2}, {3}}, 2); !errors.Is(err, ErrDimMismatch) {
		t.Errorf("wrong width: got %v", err)
	}
	if err := ValidateX([][]float64{{1, 2, 3}}, 2); !errors.Is(err, ErrDimMismatch) {
		t.Errorf("too wide: got %v", err)
	}
	if err := ValidateX([][]float64{{1, math.Inf(1)}}, 2); !errors.Is(err, ErrContainsInf) {
		t.Errorf("inf: got %v", err)
	}
}
