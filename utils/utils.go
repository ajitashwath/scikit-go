// Package utils provides generic helpers shared across estimators: consistent-length
// validation, seeded shuffling, and train/test splitting mirroring sklearn.model_selection.
package utils

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/ajitashwath/scikit-go/internal/matutil"
)

// ErrInvalidTestSize is returned when TrainTestSplit receives a testSize outside (0, 1).
var ErrInvalidTestSize = errors.New("utils: testSize must be in (0, 1)")

// CheckConsistentLength validates that X and y are non-empty, non-ragged, have matching
// sample counts, and contain no NaN/Inf values. It wraps matutil.ValidateXy so callers
// can match errors with errors.Is against the matutil sentinels.
func CheckConsistentLength(X [][]float64, y []float64) error {
	if err := matutil.ValidateXy(X, y); err != nil {
		return fmt.Errorf("utils.CheckConsistentLength: %w", err)
	}
	return nil
}

// Shuffle returns deep copies of X and y permuted by a shared seeded permutation.
// It intentionally deviates from the in-place plan draft: like sklearn.utils.shuffle,
// the inputs are left untouched and shuffled copies are returned. Both slices must
// satisfy CheckConsistentLength.
func Shuffle(X [][]float64, y []float64, seed int64) ([][]float64, []float64, error) {
	if err := CheckConsistentLength(X, y); err != nil {
		return nil, nil, err
	}
	r := rand.New(rand.NewSource(seed))
	n := len(X)
	perm := r.Perm(n)

	shuffledX := make([][]float64, n)
	shuffledY := make([]float64, n)
	for i, p := range perm {
		row := make([]float64, len(X[p]))
		copy(row, X[p])
		shuffledX[i] = row
		shuffledY[i] = y[p]
	}
	return shuffledX, shuffledY, nil
}

// TrainTestSplit splits X and y into deterministic train/test sets using sklearn's
// semantics: shuffle indices with a seeded permutation, then take the first
// nTest = ceil(testSize*n) as the test set and the rest as the train set. Both sides
// are non-empty; testSize must be in (0, 1).
func TrainTestSplit(X [][]float64, y []float64, testSize float64, seed int64) ([][]float64, [][]float64, []float64, []float64, error) {
	if err := CheckConsistentLength(X, y); err != nil {
		return nil, nil, nil, nil, err
	}
	if testSize <= 0 || testSize >= 1 {
		return nil, nil, nil, nil, fmt.Errorf("%w: got %v", ErrInvalidTestSize, testSize)
	}
	n := len(X)
	nTest := int(math.Ceil(testSize * float64(n)))
	if nTest == 0 || nTest == n {
		return nil, nil, nil, nil, fmt.Errorf("%w: cannot split %d samples", ErrInvalidTestSize, n)
	}

	r := rand.New(rand.NewSource(seed))
	perm := r.Perm(n)

	XTr := make([][]float64, 0, n-nTest)
	XTe := make([][]float64, 0, nTest)
	yTr := make([]float64, 0, n-nTest)
	yTe := make([]float64, 0, nTest)

	// Test set = first nTest permuted indices (sklearn order).
	for i := 0; i < nTest; i++ {
		p := perm[i]
		row := make([]float64, len(X[p]))
		copy(row, X[p])
		XTe = append(XTe, row)
		yTe = append(yTe, y[p])
	}
	for i := nTest; i < n; i++ {
		p := perm[i]
		row := make([]float64, len(X[p]))
		copy(row, X[p])
		XTr = append(XTr, row)
		yTr = append(yTr, y[p])
	}
	return XTr, XTe, yTr, yTe, nil
}
