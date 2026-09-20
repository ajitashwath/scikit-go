package model_selection

import (
	"fmt"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/internal/matutil"
)

// Model is anything that can be fit and then asked for predictions. Every
// supervised estimator in scikit-go, and every pipeline.Pipeline, satisfies it.
type Model interface {
	core.Estimator
	core.Predictor
}

// ModelFactory returns a new, unfitted Model. Estimators are mutable and have no
// clone method, so cross-validation asks for a fresh one for every fold.
//
//	func() (model_selection.Model, error) { return neighbors.NewKNeighborsClassifier(), nil }
type ModelFactory func() (Model, error)

// Scorer turns true and predicted targets into a score where larger is better,
// for example metrics.AccuracyScore or metrics.R2Score.
type Scorer func(yTrue, yPred []float64) (float64, error)

// scoringModel is implemented by every estimator that has a default score
// (accuracy for classifiers, R^2 for regressors).
type scoringModel interface {
	Score(X [][]float64, y []float64) (float64, error)
}

// CrossValScore fits a fresh model on each training fold and returns its score on
// the matching test fold, one score per split, mirroring sklearn's cross_val_score.
//
// cv nil means a 5-fold KFold. Note that sklearn's default for classifiers is
// stratified; here pass a StratifiedKFold explicitly when you want that.
// scorer nil means each model's own Score method (accuracy or R^2); it is an error
// for the model to have none.
func CrossValScore(newModel ModelFactory, X [][]float64, y []float64, cv Splitter, scorer Scorer) ([]float64, error) {
	if newModel == nil {
		return nil, fmt.Errorf("CrossValScore: %w: newModel is nil", ErrInvalidParams)
	}
	if err := matutil.ValidateXy(X, y); err != nil {
		return nil, fmt.Errorf("CrossValScore: %w", err)
	}
	if cv == nil {
		cv = NewKFold(5)
	}
	splits, err := cv.Split(X, y)
	if err != nil {
		return nil, fmt.Errorf("CrossValScore: %w", err)
	}

	scores := make([]float64, len(splits))
	for i, sp := range splits {
		if len(sp.Train) == 0 || len(sp.Test) == 0 {
			return nil, fmt.Errorf("CrossValScore: %w: split %d has an empty train or test set", ErrInvalidParams, i)
		}
		model, err := newModel()
		if err != nil {
			return nil, fmt.Errorf("CrossValScore: creating model for split %d: %w", i, err)
		}
		if model == nil {
			return nil, fmt.Errorf("CrossValScore: %w: newModel returned a nil model", ErrInvalidParams)
		}
		Xtr, ytr := takeRows(X, y, sp.Train)
		Xte, yte := takeRows(X, y, sp.Test)
		if err := model.Fit(Xtr, ytr); err != nil {
			return nil, fmt.Errorf("CrossValScore: fitting split %d: %w", i, err)
		}
		scores[i], err = scoreModel(model, Xte, yte, scorer)
		if err != nil {
			return nil, fmt.Errorf("CrossValScore: scoring split %d: %w", i, err)
		}
	}
	return scores, nil
}

func scoreModel(model Model, X [][]float64, y []float64, scorer Scorer) (float64, error) {
	if scorer == nil {
		sm, ok := model.(scoringModel)
		if !ok {
			return 0, fmt.Errorf("%w: scorer is nil and %T has no Score method", ErrInvalidParams, model)
		}
		return sm.Score(X, y)
	}
	pred, err := model.Predict(X)
	if err != nil {
		return 0, err
	}
	return scorer(y, pred)
}

// takeRows returns deep copies of the selected rows of X and entries of y, so a
// model that modifies its inputs cannot disturb the other folds.
func takeRows(X [][]float64, y []float64, idx []int) ([][]float64, []float64) {
	Xs := make([][]float64, len(idx))
	ys := make([]float64, len(idx))
	for i, j := range idx {
		Xs[i] = append([]float64(nil), X[j]...)
		ys[i] = y[j]
	}
	return Xs, ys
}
