package metrics

import "testing"

func TestRegression_AgainstSklearn(t *testing.T) {
	for _, name := range []string{"regression", "constant_target"} {
		fx := exampleFixture(t, name)
		yTrue := fixtureSlice(t, fx, "y_true")
		yPred := fixtureSlice(t, fx, "y_pred")

		t.Run(name, func(t *testing.T) {
			got, err := MeanSquaredError(yTrue, yPred)
			if err != nil {
				t.Fatalf("MeanSquaredError: %v", err)
			}
			assertClose(t, "mse", got, fixtureFloat(t, fx, "mean_squared_error"))

			got, err = RootMeanSquaredError(yTrue, yPred)
			if err != nil {
				t.Fatalf("RootMeanSquaredError: %v", err)
			}
			assertClose(t, "rmse", got, fixtureFloat(t, fx, "root_mean_squared_error"))

			got, err = MeanAbsoluteError(yTrue, yPred)
			if err != nil {
				t.Fatalf("MeanAbsoluteError: %v", err)
			}
			assertClose(t, "mae", got, fixtureFloat(t, fx, "mean_absolute_error"))

			got, err = R2Score(yTrue, yPred)
			if err != nil {
				t.Fatalf("R2Score: %v", err)
			}
			assertClose(t, "r2", got, fixtureFloat(t, fx, "r2_score"))

			got, err = ExplainedVarianceScore(yTrue, yPred)
			if err != nil {
				t.Fatalf("ExplainedVarianceScore: %v", err)
			}
			assertClose(t, "evs", got, fixtureFloat(t, fx, "explained_variance_score"))

			got, err = MaxError(yTrue, yPred)
			if err != nil {
				t.Fatalf("MaxError: %v", err)
			}
			assertClose(t, "max_error", got, fixtureFloat(t, fx, "max_error"))
		})
	}
}

func TestRegression_PerfectFit(t *testing.T) {
	y := []float64{1, 2, 3, 4, 5}
	r2, err := R2Score(y, y)
	if err != nil {
		t.Fatalf("R2Score: %v", err)
	}
	assertClose(t, "r2 perfect", r2, 1.0)
	mse, err := MeanSquaredError(y, y)
	if err != nil {
		t.Fatalf("MeanSquaredError: %v", err)
	}
	assertClose(t, "mse perfect", mse, 0.0)
}

func TestRegression_ConstantTargetNonPerfect(t *testing.T) {
	// Constant y_true with non-perfect predictions must score 0.0 for R2 and EVS,
	// matching sklearn's denominator guard.
	yTrue := []float64{3, 3, 3, 3}
	yPred := []float64{1, 2, 3, 4}
	r2, err := R2Score(yTrue, yPred)
	if err != nil {
		t.Fatalf("R2Score: %v", err)
	}
	assertClose(t, "r2 constant", r2, 0.0)
	evs, err := ExplainedVarianceScore(yTrue, yPred)
	if err != nil {
		t.Fatalf("ExplainedVarianceScore: %v", err)
	}
	assertClose(t, "evs constant", evs, 0.0)
}
