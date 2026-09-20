package neighbors

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/metrics"
)

// Compile-time checks that KNeighborsRegressor satisfies the core interfaces.
var (
	_ core.Estimator = (*KNeighborsRegressor)(nil)
	_ core.Predictor = (*KNeighborsRegressor)(nil)
	_ core.Saver     = (*KNeighborsRegressor)(nil)
)

// KNeighborsRegressor fits a k-nearest-neighbors regression model, mirroring
// sklearn.neighbors.KNeighborsRegressor with brute-force search and Minkowski
// distances. Weights is "uniform" or "distance"; P is the Minkowski exponent
// (2 = Euclidean, 1 = Manhattan).
type KNeighborsRegressor struct {
	NNeighbors int
	Weights    string
	P          float64

	model *knnModel
}

// NewKNeighborsRegressor returns an unfitted KNeighborsRegressor with sklearn's
// default hyperparameters.
func NewKNeighborsRegressor() *KNeighborsRegressor {
	return &KNeighborsRegressor{
		NNeighbors: 5,
		Weights:    "uniform",
		P:          2,
	}
}

// Fit stores the training data and validates hyperparameters.
func (r *KNeighborsRegressor) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("KNeighborsRegressor.Fit: %w", err)
	}
	params, err := knnParamsFrom(r.NNeighbors, r.Weights, r.P, len(X))
	if err != nil {
		return err
	}
	r.model = &knnModel{
		X:          X,
		y:          y,
		nFeatures:  len(X[0]),
		nNeighbors: params.nNeighbors,
		weights:    params.weights,
		p:          params.p,
		fitted:     true,
	}
	return nil
}

// Predict returns the (possibly distance-weighted) mean of the target values of
// the k nearest training rows for each row of X.
func (r *KNeighborsRegressor) Predict(X [][]float64) ([]float64, error) {
	if !r.fitted() {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, r.model.nFeatures); err != nil {
		return nil, fmt.Errorf("KNeighborsRegressor.Predict: %w", err)
	}
	distances, indices := r.model.kneighborsDistances(X)
	weights := r.model.neighborWeights(distances)
	preds := make([]float64, len(X))
	if weights == nil {
		for i, idx := range indices {
			var sum float64
			for _, j := range idx {
				sum += r.model.y[j]
			}
			preds[i] = sum / float64(len(idx))
		}
		return preds, nil
	}
	for i, idx := range indices {
		var num, denom float64
		for k, j := range idx {
			num += r.model.y[j] * weights[i][k]
			denom += weights[i][k]
		}
		preds[i] = num / denom
	}
	return preds, nil
}

// Score returns the R2 score of the predictions on X against y.
func (r *KNeighborsRegressor) Score(X [][]float64, y []float64) (float64, error) {
	preds, err := r.Predict(X)
	if err != nil {
		return 0, err
	}
	score, err := metrics.R2Score(y, preds)
	if err != nil {
		return 0, fmt.Errorf("KNeighborsRegressor.Score: %w", err)
	}
	return score, nil
}

func (r *KNeighborsRegressor) fitted() bool {
	return r.model != nil && r.model.fitted
}

// Save writes the fitted model to path in the versioned gob format.
func (r *KNeighborsRegressor) Save(path string) error {
	if !r.fitted() {
		return matutil.ErrNotFitted
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("KNeighborsRegressor.Save: %w", err)
	}
	defer f.Close()
	payload := knnGob{
		Version:    knnFormatVersion,
		Kind:       "regressor",
		NNeighbors: r.model.nNeighbors,
		Weights:    r.model.weights,
		P:          r.model.p,
		NFeatures:  r.model.nFeatures,
		X:          r.model.X,
		Y:          r.model.y,
	}
	if err := gob.NewEncoder(f).Encode(payload); err != nil {
		return fmt.Errorf("KNeighborsRegressor.Save: encode failed: %w", err)
	}
	return nil
}

// LoadKNeighborsRegressor reads a fitted model previously written by Save.
func LoadKNeighborsRegressor(path string) (*KNeighborsRegressor, error) {
	model, err := loadKNN(path, "KNeighborsRegressor", "regressor")
	if err != nil {
		return nil, err
	}
	return &KNeighborsRegressor{
		NNeighbors: model.nNeighbors,
		Weights:    model.weights,
		P:          model.p,
		model:      model,
	}, nil
}
