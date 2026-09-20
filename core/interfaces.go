package core

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

// Classifier is satisfied by estimators that produce per-class probability estimates.
type Classifier interface {
	PredictProba(X [][]float64) ([][]float64, error)
}

// Clusterer is satisfied by clustering estimators that can label data and expose
// the labels assigned during Fit.
type Clusterer interface {
	FitPredict(X [][]float64) ([]float64, error)
	Labels() []float64
}

// Saver is satisfied by estimators that support binary serialization
type Saver interface {
	Save(path string) error
}
