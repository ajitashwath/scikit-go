// Package datasets provides synthetic data generators and bundled real-world
// datasets, mirroring sklearn.datasets.
//
// The synthetic generators reproduce the *structure* and parameter semantics of
// their scikit-learn counterparts (same shapes, distributions, class balance,
// and cluster geometry), but they use Go's seeded math/rand rather than numpy's
// RNG, so they are not bit-for-bit identical to sklearn's output. All generators
// are deterministic for a given seed.
package datasets

import "errors"

// ErrInvalidParams is returned when a generator is called with invalid parameters
// (e.g. non-positive sample/feature counts or an out-of-range noise level).
var ErrInvalidParams = errors.New("datasets: invalid parameters")

var (
	errRaggedCSV      = errors.New("datasets: CSV rows have inconsistent lengths")
	errEmptyCSV       = errors.New("datasets: CSV is empty")
	errDimMismatchCSV = errors.New("datasets: CSV length mismatch")
)
