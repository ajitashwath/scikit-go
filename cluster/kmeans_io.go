package cluster

import (
	"encoding/gob"
	"fmt"
	"os"

	"scikit-go/core"
	"scikit-go/internal/matutil"
)

// Compile-time checks that KMeans satisfies the core interfaces.
var (
	_ core.Estimator  = (*KMeans)(nil)
	_ core.Predictor  = (*KMeans)(nil)
	_ core.Clusterer  = (*KMeans)(nil)
	_ core.Saver      = (*KMeans)(nil)
)

// kmeansGob is the versioned on-disk payload.
type kmeansGob struct {
	Version    int
	NClusters  int
	MaxIter    int
	NInit      int
	Tol        float64
	Seed       int64
	Centers    [][]float64
	Labels     []float64
	Inertia    float64
}

const kmeansFormatVersion = 1

// Save writes the fitted model to path in the versioned gob format.
func (k *KMeans) Save(path string) error {
	if !k.fitted {
		return matutil.ErrNotFitted
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("KMeans.Save: %w", err)
	}
	defer f.Close()
	payload := kmeansGob{
		Version:   kmeansFormatVersion,
		NClusters: k.NClusters,
		MaxIter:   k.MaxIter,
		NInit:     k.NInit,
		Tol:       k.Tol,
		Seed:      k.Seed,
		Centers:   k.centers,
		Labels:    k.labels,
		Inertia:   k.inertia,
	}
	if err := gob.NewEncoder(f).Encode(payload); err != nil {
		return fmt.Errorf("KMeans.Save: encode failed: %w", err)
	}
	return nil
}

// LoadKMeans reads a fitted model previously written by Save.
func LoadKMeans(path string) (*KMeans, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("LoadKMeans: %w", err)
	}
	defer f.Close()

	var payload kmeansGob
	if err := gob.NewDecoder(f).Decode(&payload); err != nil {
		return nil, fmt.Errorf("LoadKMeans: decode failed: %w", err)
	}
	if payload.Version != kmeansFormatVersion {
		return nil, fmt.Errorf("LoadKMeans: unsupported format version %d (expected %d)", payload.Version, kmeansFormatVersion)
	}
	if len(payload.Centers) < 1 || len(payload.Centers) != payload.NClusters {
		return nil, fmt.Errorf("LoadKMeans: corrupt payload: %d centers for n_clusters=%d", len(payload.Centers), payload.NClusters)
	}
	width := len(payload.Centers[0])
	for i, c := range payload.Centers {
		if width < 1 || len(c) != width {
			return nil, fmt.Errorf("LoadKMeans: corrupt payload: center %d has %d entries, want %d", i, len(c), width)
		}
	}
	return &KMeans{
		NClusters: payload.NClusters,
		MaxIter:   payload.MaxIter,
		NInit:     payload.NInit,
		Tol:       payload.Tol,
		Seed:      payload.Seed,
		centers:   payload.Centers,
		labels:    payload.Labels,
		inertia:   payload.Inertia,
		fitted:    true,
	}, nil
}