package metrics

import (
	"math"
	"testing"
)

// With a single label present there are no positives to find, so precision,
// recall and F1 are 0 (or 1 when that one label is the positive class); an error
// is only right when two or more labels exist and the positive label is not one
// of them. These are the cases the sklearn differential test uncovered.
func TestPrecisionRecallF1_SingleLabel(t *testing.T) {
	cases := []struct {
		name         string
		yTrue, yPred []float64
		pos          []float64
		want         float64
	}{
		{"all negative, default pos_label", []float64{0, 0, 0}, []float64{0, 0, 0}, nil, 0},
		{"single label that is not 1", []float64{5, 5}, []float64{5, 5}, nil, 0},
		{"single label that is the positive class", []float64{1, 1}, []float64{1, 1}, nil, 1},
		{"single label, explicit absent pos_label", []float64{0, 0}, []float64{0, 0}, []float64{7}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for name, fn := range map[string]func(a, b []float64, pos ...float64) (float64, error){
				"precision": PrecisionScore, "recall": RecallScore, "f1": F1Score,
			} {
				got, err := fn(tc.yTrue, tc.yPred, tc.pos...)
				if err != nil {
					t.Fatalf("%s: unexpected error %v", name, err)
				}
				if got != tc.want {
					t.Errorf("%s: got %v, want %v", name, got, tc.want)
				}
			}
		})
	}
}

func TestPrecisionRecallF1_MissingPositiveLabelWithTwoLabelsIsAnError(t *testing.T) {
	yTrue, yPred := []float64{2, 3, 2}, []float64{2, 3, 3}
	if _, err := PrecisionScore(yTrue, yPred); err == nil {
		t.Error("default pos_label=1 with labels {2,3} should fail")
	}
	if _, err := RecallScore(yTrue, yPred, 9); err == nil {
		t.Error("pos_label=9 with labels {2,3} should fail")
	}
	if _, err := F1Score(yTrue, yPred, 3); err != nil {
		t.Errorf("pos_label=3 is valid: %v", err)
	}
}

// R^2 is undefined for fewer than two samples; like sklearn we return NaN.
func TestR2Score_FewerThanTwoSamplesIsNaN(t *testing.T) {
	for _, pair := range [][2][]float64{
		{{3}, {3}},
		{{3}, {-4.4}},
	} {
		got, err := R2Score(pair[0], pair[1])
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !math.IsNaN(got) {
			t.Errorf("R2Score(%v, %v) = %v, want NaN", pair[0], pair[1], got)
		}
	}
	if _, err := R2Score(nil, nil); err == nil {
		t.Error("empty input should still be an error, not NaN")
	}
}

func TestVMeasure_ZeroWhenHomogeneityAndCompletenessAreZero(t *testing.T) {
	// Independent labelings share no information: both components are 0.
	got, err := VMeasure([]float64{0, 0, 1, 1}, []float64{0, 1, 0, 1})
	if err != nil {
		t.Fatalf("VMeasure: %v", err)
	}
	if got != 0 {
		t.Errorf("got %v, want 0", got)
	}
}

func TestRootMeanSquaredError_Validation(t *testing.T) {
	if got, err := RootMeanSquaredError([]float64{1, 2}, []float64{1, 4}); err != nil || math.Abs(got-math.Sqrt(2)) > 1e-12 {
		t.Errorf("got %v, %v; want sqrt(2)", got, err)
	}
	if _, err := RootMeanSquaredError([]float64{1}, []float64{1, 2}); err == nil {
		t.Error("length mismatch should fail")
	}
	if _, err := RootMeanSquaredError(nil, nil); err == nil {
		t.Error("empty input should fail")
	}
}
