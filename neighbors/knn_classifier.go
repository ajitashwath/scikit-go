package neighbors

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/metrics"
)

// Compile-time checks that KNeighborsClassifier satisfies the core interfaces.
var (
	_ core.Estimator  = (*KNeighborsClassifier)(nil)
	_ core.Predictor  = (*KNeighborsClassifier)(nil)
	_ core.Classifier = (*KNeighborsClassifier)(nil)
	_ core.Saver      = (*KNeighborsClassifier)(nil)
)

// KNeighborsClassifier fits a k-nearest-neighbors classifier, mirroring
// sklearn.neighbors.KNeighborsClassifier with brute-force search and Minkowski
// distances. Weights is "uniform" or "distance"; P is the Minkowski exponent.
type KNeighborsClassifier struct {
	NNeighbors int
	Weights    string
	P          float64

	model *knnModel
}

// NewKNeighborsClassifier returns an unfitted KNeighborsClassifier with
// sklearn's default hyperparameters.
func NewKNeighborsClassifier() *KNeighborsClassifier {
	return &KNeighborsClassifier{
		NNeighbors: 5,
		Weights:    "uniform",
		P:          2,
	}
}

// Fit stores the training data and validates hyperparameters.
func (c *KNeighborsClassifier) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("KNeighborsClassifier.Fit: %w", err)
	}
	params, err := knnParamsFrom(c.NNeighbors, c.Weights, c.P, len(X))
	if err != nil {
		return err
	}
	classes, yClass := classEncoding(y)
	c.model = &knnModel{
		X:          X,
		y:          y,
		yClass:     yClass,
		classes:    classes,
		nFeatures:  len(X[0]),
		nNeighbors: params.nNeighbors,
		weights:    params.weights,
		p:          params.p,
		fitted:     true,
	}
	return nil
}

// Predict returns the majority class (weighted if Weights is "distance") among
// the k nearest training rows for each row of X. Ties are broken by the class
// with the smallest label value, matching sklearn.
func (c *KNeighborsClassifier) Predict(X [][]float64) ([]float64, error) {
	if !c.fitted() {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, c.model.nFeatures); err != nil {
		return nil, fmt.Errorf("KNeighborsClassifier.Predict: %w", err)
	}
	distances, indices := c.model.kneighborsDistances(X)
	weights := c.model.neighborWeights(distances)
	preds := make([]float64, len(X))
	for i, idx := range indices {
		counts := make([]float64, len(c.model.classes))
		if weights == nil {
			for _, j := range idx {
				counts[c.model.yClass[j]]++
			}
		} else {
			for k, j := range idx {
				counts[c.model.yClass[j]] += weights[i][k]
			}
		}
		preds[i] = c.model.classes[argmaxFirst(counts)]
	}
	return preds, nil
}

// PredictProba returns the fraction of votes (weighted if Weights is "distance")
// each class receives among the k nearest training rows.
func (c *KNeighborsClassifier) PredictProba(X [][]float64) ([][]float64, error) {
	if !c.fitted() {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, c.model.nFeatures); err != nil {
		return nil, fmt.Errorf("KNeighborsClassifier.PredictProba: %w", err)
	}
	distances, indices := c.model.kneighborsDistances(X)
	weights := c.model.neighborWeights(distances)
	proba := make([][]float64, len(X))
	for i, idx := range indices {
		counts := make([]float64, len(c.model.classes))
		if weights == nil {
			for _, j := range idx {
				counts[c.model.yClass[j]]++
			}
		} else {
			for k, j := range idx {
				counts[c.model.yClass[j]] += weights[i][k]
			}
		}
		var total float64
		for _, v := range counts {
			total += v
		}
		row := make([]float64, len(counts))
		for j, v := range counts {
			row[j] = v / total
		}
		proba[i] = row
	}
	return proba, nil
}

// Score returns the mean accuracy of the predictions on X against y.
func (c *KNeighborsClassifier) Score(X [][]float64, y []float64) (float64, error) {
	preds, err := c.Predict(X)
	if err != nil {
		return 0, err
	}
	score, err := metrics.AccuracyScore(y, preds)
	if err != nil {
		return 0, fmt.Errorf("KNeighborsClassifier.Score: %w", err)
	}
	return score, nil
}

func (c *KNeighborsClassifier) fitted() bool {
	return c.model != nil && c.model.fitted
}

// Classes returns the unique class labels in ascending order.
func (c *KNeighborsClassifier) Classes() []float64 {
	if !c.fitted() {
		return nil
	}
	out := make([]float64, len(c.model.classes))
	copy(out, c.model.classes)
	return out
}

// Save writes the fitted model to path in the versioned gob format.
func (c *KNeighborsClassifier) Save(path string) error {
	if !c.fitted() {
		return matutil.ErrNotFitted
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("KNeighborsClassifier.Save: %w", err)
	}
	defer f.Close()
	payload := knnGob{
		Version:    knnFormatVersion,
		Kind:       "classifier",
		NNeighbors: c.model.nNeighbors,
		Weights:    c.model.weights,
		P:          c.model.p,
		NFeatures:  c.model.nFeatures,
		X:          c.model.X,
		Y:          c.model.y,
		Classes:    c.model.classes,
	}
	if err := gob.NewEncoder(f).Encode(payload); err != nil {
		return fmt.Errorf("KNeighborsClassifier.Save: encode failed: %w", err)
	}
	return nil
}

// LoadKNeighborsClassifier reads a fitted model previously written by Save.
func LoadKNeighborsClassifier(path string) (*KNeighborsClassifier, error) {
	model, err := loadKNN(path, "KNeighborsClassifier", "classifier")
	if err != nil {
		return nil, err
	}
	classes, yClass := classEncoding(model.y)
	model.classes = classes
	model.yClass = yClass
	return &KNeighborsClassifier{
		NNeighbors: model.nNeighbors,
		Weights:    model.weights,
		P:          model.p,
		model:      model,
	}, nil
}
