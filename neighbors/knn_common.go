package neighbors

import (
	"encoding/gob"
	"errors"
	"fmt"
	"os"
	"sort"
)

// ErrInvalidKNN is returned when hyperparameters fail validation.
var ErrInvalidKNN = errors.New("invalid neighbor hyperparameters")

// knnGob is the versioned on-disk payload shared by both estimators.
type knnGob struct {
	Version    int
	Kind       string // "regressor" or "classifier"
	NNeighbors int
	Weights    string
	P          float64
	NFeatures  int
	X          [][]float64
	Y          []float64
	Classes    []float64
}

const knnFormatVersion = 1

// knnParamsFrom validates the hyperparameters against the training set size:
// asking for more neighbors than there are training samples is rejected here
// rather than surfacing as an index panic inside Predict.
func knnParamsFrom(nNeighbors int, weights string, p float64, nSamples int) (knnParams, error) {
	if nNeighbors < 1 {
		return knnParams{}, fmt.Errorf("%w: n_neighbors must be >= 1, got %d", ErrInvalidKNN, nNeighbors)
	}
	if nNeighbors > nSamples {
		return knnParams{}, fmt.Errorf("%w: n_neighbors=%d exceeds n_samples=%d", ErrInvalidKNN, nNeighbors, nSamples)
	}
	if weights != "uniform" && weights != "distance" {
		return knnParams{}, fmt.Errorf("%w: unsupported weights %q", ErrInvalidKNN, weights)
	}
	if p < 1 {
		return knnParams{}, fmt.Errorf("%w: p must be >= 1, got %v", ErrInvalidKNN, p)
	}
	return knnParams{nNeighbors: nNeighbors, weights: weights, p: p}, nil
}

// classEncoding maps the raw target values to dense class indices. sklearn sorts
// classes in ascending order; the returned mapping is index-invariant for
// prediction but the class labels themselves are stored for output.
func classEncoding(y []float64) (classes []float64, encoded []int) {
	uniq := make(map[float64]bool)
	for _, v := range y {
		uniq[v] = true
	}
	classes = make([]float64, 0, len(uniq))
	for v := range uniq {
		classes = append(classes, v)
	}
	sort.Float64s(classes)
	index := make(map[float64]int, len(classes))
	for i, v := range classes {
		index[v] = i
	}
	encoded = make([]int, len(y))
	for i, v := range y {
		encoded[i] = index[v]
	}
	return classes, encoded
}

// argmaxFirst returns the index of the maximum value, choosing the smallest
// index on ties, mirroring sklearn's mode tie-breaking (lowest class wins).
func argmaxFirst(counts []float64) int {
	best := 0
	for i := 1; i < len(counts); i++ {
		if counts[i] > counts[best] {
			best = i
		}
	}
	return best
}

func loadKNN(path, name, kind string) (*knnModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("Load%s: %w", name, err)
	}
	defer f.Close()

	var payload knnGob
	if err := gob.NewDecoder(f).Decode(&payload); err != nil {
		return nil, fmt.Errorf("Load%s: decode failed: %w", name, err)
	}
	if payload.Version != knnFormatVersion {
		return nil, fmt.Errorf("Load%s: unsupported format version %d (expected %d)", name, payload.Version, knnFormatVersion)
	}
	if payload.Kind != kind {
		return nil, fmt.Errorf("Load%s: file contains a %s, not a %s", name, payload.Kind, kind)
	}
	if payload.NFeatures < 1 || len(payload.X) < 1 || len(payload.Y) != len(payload.X) {
		return nil, fmt.Errorf("Load%s: corrupt payload: %d samples, %d targets, %d features",
			name, len(payload.X), len(payload.Y), payload.NFeatures)
	}
	for i, row := range payload.X {
		if len(row) != payload.NFeatures {
			return nil, fmt.Errorf("Load%s: corrupt payload: row %d has %d features, want %d",
				name, i, len(row), payload.NFeatures)
		}
	}
	params, err := knnParamsFrom(payload.NNeighbors, payload.Weights, payload.P, len(payload.X))
	if err != nil {
		return nil, fmt.Errorf("Load%s: %w", name, err)
	}
	return &knnModel{
		X:          payload.X,
		y:          payload.Y,
		nFeatures:  payload.NFeatures,
		nNeighbors: params.nNeighbors,
		weights:    params.weights,
		p:          params.p,
		fitted:     true,
	}, nil
}
