package ensemble

import (
	"encoding/gob"
	"fmt"
	"os"

	"scikit-go/internal/matutil"
	"scikit-go/tree"
)

// forestGob is the versioned on-disk payload shared by both forests. Each tree
// is stored as the tree package's own gob payload.
type forestGob struct {
	Version         int
	Kind            string // "regressor" or "classifier"
	NTrees          int
	Criterion       string
	MaxDepth        int
	MinSamplesSplit int
	MinSamplesLeaf  int
	MaxFeatures     int
	Bootstrap       bool
	Seed            int64
	NJobs           int
	NFeatures       int
	Classes         []float64
	Trees           [][]byte
}

const forestFormatVersion = 1

func writeForest(path, name string, payload forestGob) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("%s.Save: %w", name, err)
	}
	defer f.Close()
	if err := gob.NewEncoder(f).Encode(payload); err != nil {
		return fmt.Errorf("%s.Save: encode failed: %w", name, err)
	}
	return nil
}

func readForest(path, name, kind string) (*forestGob, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("Load%s: %w", name, err)
	}
	defer f.Close()

	var payload forestGob
	if err := gob.NewDecoder(f).Decode(&payload); err != nil {
		return nil, fmt.Errorf("Load%s: decode failed: %w", name, err)
	}
	if payload.Version != forestFormatVersion {
		return nil, fmt.Errorf("Load%s: unsupported format version %d (expected %d)", name, payload.Version, forestFormatVersion)
	}
	if payload.Kind != kind {
		return nil, fmt.Errorf("Load%s: file contains a %s, not a %s", name, payload.Kind, kind)
	}
	if len(payload.Trees) == 0 {
		return nil, fmt.Errorf("Load%s: file contains no trees", name)
	}
	return &payload, nil
}

// Save writes the fitted forest to path in the versioned gob format.
func (f *RandomForestRegressor) Save(path string) error {
	if !f.fitted {
		return matutil.ErrNotFitted
	}
	blobs := make([][]byte, len(f.trees))
	for i, t := range f.trees {
		data, err := t.MarshalBinary()
		if err != nil {
			return fmt.Errorf("RandomForestRegressor.Save: tree %d: %w", i, err)
		}
		blobs[i] = data
	}
	return writeForest(path, "RandomForestRegressor", forestGob{
		Version:         forestFormatVersion,
		Kind:            "regressor",
		NTrees:          f.NTrees,
		Criterion:       f.Criterion,
		MaxDepth:        f.MaxDepth,
		MinSamplesSplit: f.MinSamplesSplit,
		MinSamplesLeaf:  f.MinSamplesLeaf,
		MaxFeatures:     f.MaxFeatures,
		Bootstrap:       f.Bootstrap,
		Seed:            f.Seed,
		NJobs:           f.NJobs,
		NFeatures:       f.nFeatures,
		Trees:           blobs,
	})
}

// LoadRandomForestRegressor reads a fitted forest previously written by Save.
func LoadRandomForestRegressor(path string) (*RandomForestRegressor, error) {
	payload, err := readForest(path, "RandomForestRegressor", "regressor")
	if err != nil {
		return nil, err
	}
	trees := make([]*tree.DecisionTreeRegressor, len(payload.Trees))
	for i, data := range payload.Trees {
		t := &tree.DecisionTreeRegressor{}
		if err := t.UnmarshalBinary(data); err != nil {
			return nil, fmt.Errorf("LoadRandomForestRegressor: tree %d: %w", i, err)
		}
		trees[i] = t
	}
	return &RandomForestRegressor{
		NTrees:          payload.NTrees,
		Criterion:       payload.Criterion,
		MaxDepth:        payload.MaxDepth,
		MinSamplesSplit: payload.MinSamplesSplit,
		MinSamplesLeaf:  payload.MinSamplesLeaf,
		MaxFeatures:     payload.MaxFeatures,
		Bootstrap:       payload.Bootstrap,
		Seed:            payload.Seed,
		NJobs:           payload.NJobs,
		trees:           trees,
		nFeatures:       payload.NFeatures,
		fitted:          true,
	}, nil
}

// Save writes the fitted forest to path in the versioned gob format.
func (f *RandomForestClassifier) Save(path string) error {
	if !f.fitted {
		return matutil.ErrNotFitted
	}
	blobs := make([][]byte, len(f.trees))
	for i, t := range f.trees {
		data, err := t.MarshalBinary()
		if err != nil {
			return fmt.Errorf("RandomForestClassifier.Save: tree %d: %w", i, err)
		}
		blobs[i] = data
	}
	return writeForest(path, "RandomForestClassifier", forestGob{
		Version:         forestFormatVersion,
		Kind:            "classifier",
		NTrees:          f.NTrees,
		Criterion:       f.Criterion,
		MaxDepth:        f.MaxDepth,
		MinSamplesSplit: f.MinSamplesSplit,
		MinSamplesLeaf:  f.MinSamplesLeaf,
		MaxFeatures:     f.MaxFeatures,
		Bootstrap:       f.Bootstrap,
		Seed:            f.Seed,
		NJobs:           f.NJobs,
		NFeatures:       f.nFeatures,
		Classes:         f.classes,
		Trees:           blobs,
	})
}

// LoadRandomForestClassifier reads a fitted forest previously written by Save.
func LoadRandomForestClassifier(path string) (*RandomForestClassifier, error) {
	payload, err := readForest(path, "RandomForestClassifier", "classifier")
	if err != nil {
		return nil, err
	}
	trees := make([]*tree.DecisionTreeClassifier, len(payload.Trees))
	for i, data := range payload.Trees {
		t := &tree.DecisionTreeClassifier{}
		if err := t.UnmarshalBinary(data); err != nil {
			return nil, fmt.Errorf("LoadRandomForestClassifier: tree %d: %w", i, err)
		}
		trees[i] = t
	}
	return &RandomForestClassifier{
		NTrees:          payload.NTrees,
		Criterion:       payload.Criterion,
		MaxDepth:        payload.MaxDepth,
		MinSamplesSplit: payload.MinSamplesSplit,
		MinSamplesLeaf:  payload.MinSamplesLeaf,
		MaxFeatures:     payload.MaxFeatures,
		Bootstrap:       payload.Bootstrap,
		Seed:            payload.Seed,
		NJobs:           payload.NJobs,
		trees:           trees,
		classes:         payload.Classes,
		classIdx:        mapClasses(trees, payload.Classes),
		nFeatures:       payload.NFeatures,
		fitted:          true,
	}, nil
}
