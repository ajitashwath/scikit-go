package metrics

import "math"

// MeanSquaredError returns the mean of the squared differences between predictions
// and true targets, matching sklearn.metrics.mean_squared_error.
func MeanSquaredError(yTrue, yPred []float64) (float64, error) {
	if err := validateVectors(yTrue, yPred); err != nil {
		return 0, err
	}
	var sum float64
	for i := range yTrue {
		d := yTrue[i] - yPred[i]
		sum += d * d
	}
	return sum / float64(len(yTrue)), nil
}

// RootMeanSquaredError returns the square root of MeanSquaredError, matching
// sklearn.metrics.root_mean_squared_error.
func RootMeanSquaredError(yTrue, yPred []float64) (float64, error) {
	mse, err := MeanSquaredError(yTrue, yPred)
	if err != nil {
		return 0, err
	}
	return math.Sqrt(mse), nil
}

// MeanAbsoluteError returns the mean of the absolute differences between predictions
// and true targets, matching sklearn.metrics.mean_absolute_error.
func MeanAbsoluteError(yTrue, yPred []float64) (float64, error) {
	if err := validateVectors(yTrue, yPred); err != nil {
		return 0, err
	}
	var sum float64
	for i := range yTrue {
		sum += math.Abs(yTrue[i] - yPred[i])
	}
	return sum / float64(len(yTrue)), nil
}

// MaxError returns the worst-case (largest absolute) error between predictions and
// true targets, matching sklearn.metrics.max_error.
func MaxError(yTrue, yPred []float64) (float64, error) {
	if err := validateVectors(yTrue, yPred); err != nil {
		return 0, err
	}
	var m float64
	for i := range yTrue {
		if d := math.Abs(yTrue[i] - yPred[i]); d > m {
			m = d
		}
	}
	return m, nil
}

// R2Score returns the coefficient of determination, matching
// sklearn.metrics.r2_score (including the ssTot == 0 edge cases: a perfect fit
// on constant targets scores 1.0, a non-perfect fit on constant targets scores 0.0).
func R2Score(yTrue, yPred []float64) (float64, error) {
	if err := validateVectors(yTrue, yPred); err != nil {
		return 0, err
	}
	var mean float64
	for _, v := range yTrue {
		mean += v
	}
	mean /= float64(len(yTrue))

	var ssRes, ssTot float64
	for i := range yTrue {
		diff := yTrue[i] - yPred[i]
		ssRes += diff * diff
		diffMean := yTrue[i] - mean
		ssTot += diffMean * diffMean
	}
	if ssTot == 0 {
		if ssRes == 0 {
			return 1.0, nil
		}
		return 0.0, nil
	}
	return 1.0 - ssRes/ssTot, nil
}

// ExplainedVarianceScore returns the explained variance regression score, matching
// sklearn.metrics.explained_variance_score: 1 - Var(y_true - y_pred) / Var(y_true).
func ExplainedVarianceScore(yTrue, yPred []float64) (float64, error) {
	if err := validateVectors(yTrue, yPred); err != nil {
		return 0, err
	}
	n := len(yTrue)
	var meanTrue, meanDiff float64
	for i := range yTrue {
		meanTrue += yTrue[i]
		meanDiff += yTrue[i] - yPred[i]
	}
	meanTrue /= float64(n)
	meanDiff /= float64(n)

	var numerator, denominator float64
	for i := range yTrue {
		d := (yTrue[i] - yPred[i]) - meanDiff
		numerator += d * d
		t := yTrue[i] - meanTrue
		denominator += t * t
	}
	if denominator == 0 {
		if numerator == 0 {
			return 1.0, nil
		}
		return 0.0, nil
	}
	return 1.0 - numerator/denominator, nil
}
