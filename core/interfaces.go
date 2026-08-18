package core

import "context"

// Estimator is satisfied by anything that can be fit to data
// X is a slice of samples, each a slice of feature values ([n_samples][n_features])
// y is the target vector, length n_samples

type Estimator interface {
	Fit(X [][]float64, y []float64) error
}

// Predictor is satisfied by anything that produces predictions for new data
type Predictor interface {
	Predict(X [][]float64) ([]float64, error)
}

// Transformer is satisfied by preprocessing steps
type Transformer interface {
	FitTransform(X [][]float64) ([][]float64, error)
	Transform(X [][]float64) ([][]float64, error)
}

// ContextEstimator is an optional extension for long-running training that supports cancellation and deadlines
// Estimators implement this in addition to Estimator when their Fit is expensive enough to warrant it
type ContextEstimator interface {
	FitContext(ctx context.Context, X [][]float64, y []float64) error
}

// Saver is satisfied by estimators that support binary serialization
type Saver interface {
	Save(path string) error
}
