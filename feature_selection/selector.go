// Package feature_selection implements feature selectors mirroring
// sklearn.feature_selection: VarianceThreshold, SelectKBest (with the
// f_classif and f_regression scorers) and RFE. Every selector reduces to a
// boolean support mask over the input columns, exposed through Support and
// applied by Transform.
package feature_selection

import (
	"encoding/gob"
	"errors"
	"fmt"
	"os"

	"github.com/ajitashwath/scikit-go/internal/matutil"
)

// ErrInvalidSelector is returned when selector hyperparameters or inputs fail
// validation (for example K larger than the number of features).
var ErrInvalidSelector = errors.New("invalid feature selector configuration")

// support is the fitted state shared by all selectors: which columns to keep.
type support struct {
	mask      []bool
	nFeatures int
	fitted    bool
}

func (s *support) set(mask []bool) {
	s.mask = mask
	s.nFeatures = len(mask)
	s.fitted = true
}

// Support returns the boolean mask of selected features, or nil before Fit.
func (s *support) Support() []bool {
	if !s.fitted {
		return nil
	}
	return append([]bool(nil), s.mask...)
}

// SupportIndices returns the column indices of the selected features in
// ascending order, or nil before Fit.
func (s *support) SupportIndices() []int {
	if !s.fitted {
		return nil
	}
	idx := make([]int, 0, len(s.mask))
	for j, keep := range s.mask {
		if keep {
			idx = append(idx, j)
		}
	}
	return idx
}

// transform keeps the selected columns of X.
func (s *support) transform(name string, X [][]float64) ([][]float64, error) {
	if !s.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, s.nFeatures); err != nil {
		return nil, fmt.Errorf("%s.Transform: %w", name, err)
	}
	keep := s.SupportIndices()
	out := make([][]float64, len(X))
	for i, row := range X {
		sel := make([]float64, len(keep))
		for k, j := range keep {
			sel[k] = row[j]
		}
		out[i] = sel
	}
	return out, nil
}

// selectorGob is the versioned on-disk payload shared by every selector. Only
// the fields relevant to Kind are populated.
type selectorGob struct {
	Version           int
	Kind              string // "variance_threshold", "select_k_best" or "rfe"
	NFeatures         int
	Support           []bool
	Threshold         float64
	Variances         []float64
	K                 int
	Score             string
	Scores            []float64
	PValues           []float64
	NFeaturesToSelect int
	Step              int
	Ranking           []int
}

const selectorFormatVersion = 1

func saveSelector(path, name string, payload selectorGob) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("%s.Save: %w", name, err)
	}
	defer f.Close()
	payload.Version = selectorFormatVersion
	if err := gob.NewEncoder(f).Encode(payload); err != nil {
		return fmt.Errorf("%s.Save: encode failed: %w", name, err)
	}
	return nil
}

func loadSelector(path, name, kind string) (*selectorGob, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("Load%s: %w", name, err)
	}
	defer f.Close()

	var payload selectorGob
	if err := gob.NewDecoder(f).Decode(&payload); err != nil {
		return nil, fmt.Errorf("Load%s: decode failed: %w", name, err)
	}
	if payload.Version != selectorFormatVersion {
		return nil, fmt.Errorf("Load%s: unsupported format version %d (expected %d)", name, payload.Version, selectorFormatVersion)
	}
	if payload.Kind != kind {
		return nil, fmt.Errorf("Load%s: file contains a %s, not a %s", name, payload.Kind, kind)
	}
	if len(payload.Support) != payload.NFeatures {
		return nil, fmt.Errorf("Load%s: support mask has %d entries for %d features", name, len(payload.Support), payload.NFeatures)
	}
	return &payload, nil
}
