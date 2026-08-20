package decomposition

import (
	"encoding/gob"
	"fmt"
	"os"

	"scikit-go/core"
	"scikit-go/internal/matutil"
)

// Compile-time checks that PCA satisfies the core interfaces.
var (
	_ core.Estimator   = (*PCA)(nil)
	_ core.Transformer = (*PCA)(nil)
	_ core.Saver       = (*PCA)(nil)
)

// pcaGob is the versioned on-disk payload.
type pcaGob struct {
	Version                  int
	NComponents              int
	Components               [][]float64
	ExplainedVariance        []float64
	ExplainedVarianceRatio   []float64
	SingularValues           []float64
	Mean                     []float64
}

const pcaFormatVersion = 1

// Save writes the fitted model to path in the versioned gob format.
func (p *PCA) Save(path string) error {
	if !p.fitted {
		return matutil.ErrNotFitted
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("PCA.Save: %w", err)
	}
	defer f.Close()
	payload := pcaGob{
		Version:                pcaFormatVersion,
		NComponents:            p.NComponents,
		Components:             p.components,
		ExplainedVariance:      p.explainedVariance,
		ExplainedVarianceRatio: p.explainedVarianceRatio,
		SingularValues:         p.singularValues,
		Mean:                   p.mean,
	}
	if err := gob.NewEncoder(f).Encode(payload); err != nil {
		return fmt.Errorf("PCA.Save: encode failed: %w", err)
	}
	return nil
}

// LoadPCA reads a fitted model previously written by Save.
func LoadPCA(path string) (*PCA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("LoadPCA: %w", err)
	}
	defer f.Close()

	var payload pcaGob
	if err := gob.NewDecoder(f).Decode(&payload); err != nil {
		return nil, fmt.Errorf("LoadPCA: decode failed: %w", err)
	}
	if payload.Version != pcaFormatVersion {
		return nil, fmt.Errorf("LoadPCA: unsupported format version %d (expected %d)", payload.Version, pcaFormatVersion)
	}
	return &PCA{
		NComponents:            payload.NComponents,
		components:             payload.Components,
		explainedVariance:      payload.ExplainedVariance,
		explainedVarianceRatio: payload.ExplainedVarianceRatio,
		singularValues:         payload.SingularValues,
		mean:                   payload.Mean,
		fitted:                 true,
	}, nil
}