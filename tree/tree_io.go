package tree

import (
	"encoding/gob"
	"fmt"
	"os"

	"scikit-go/internal/matutil"
)

// treeNodeGob is the on-disk node representation.
type treeNodeGob struct {
	Left      int
	Right     int
	Feature   int
	Threshold float64
	Value     []float64
	Impurity  float64
	NSamples  int
}

// treeGob is the versioned on-disk payload shared by both estimators.
type treeGob struct {
	Version         int
	Kind            string // "regressor" or "classifier"
	Criterion       string
	MaxDepth        int
	MinSamplesSplit int
	MinSamplesLeaf  int
	MaxFeatures     int
	Seed            int64
	NFeatures       int
	Nodes           []treeNodeGob
	Classes         []float64
}

const treeFormatVersion = 1

func nodeToGob(n treeNode) treeNodeGob {
	return treeNodeGob{
		Left: n.Left, Right: n.Right, Feature: n.Feature,
		Threshold: n.Threshold, Value: n.Value,
		Impurity: n.Impurity, NSamples: n.NSamples,
	}
}

func nodeFromGob(n treeNodeGob) treeNode {
	return treeNode{
		Left: n.Left, Right: n.Right, Feature: n.Feature,
		Threshold: n.Threshold, Value: n.Value,
		Impurity: n.Impurity, NSamples: n.NSamples,
	}
}

// saveTree writes a fitted tree to path in the versioned gob format.
// name is the display name used in error messages ("DecisionTreeRegressor"),
// kind is the lowercase payload kind ("regressor").
func saveTree(path, name, kind, criterion string, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures int, seed int64, t *treeImpl, classes []float64) error {
	if t == nil {
		return matutil.ErrNotFitted
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("%s.Save: %w", name, err)
	}
	defer f.Close()

	nodes := make([]treeNodeGob, len(t.nodes))
	for i, n := range t.nodes {
		nodes[i] = nodeToGob(n)
	}
	payload := treeGob{
		Version:         treeFormatVersion,
		Kind:            kind,
		Criterion:       criterion,
		MaxDepth:        maxDepth,
		MinSamplesSplit: minSamplesSplit,
		MinSamplesLeaf:  minSamplesLeaf,
		MaxFeatures:     maxFeatures,
		Seed:            seed,
		NFeatures:       t.nFeatures,
		Nodes:           nodes,
		Classes:         classes,
	}
	if err := gob.NewEncoder(f).Encode(payload); err != nil {
		return fmt.Errorf("%s.Save: encode failed: %w", name, err)
	}
	return nil
}

// loadTree reads a versioned tree payload and reconstructs the treeImpl.
// name is the display name used in error messages, kind the expected payload kind.
func loadTree(path, name, kind string) (*treeGob, *treeImpl, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("Load%s: %w", name, err)
	}
	defer f.Close()

	var payload treeGob
	if err := gob.NewDecoder(f).Decode(&payload); err != nil {
		return nil, nil, fmt.Errorf("Load%s: decode failed: %w", name, err)
	}
	if payload.Version != treeFormatVersion {
		return nil, nil, fmt.Errorf("Load%s: unsupported format version %d (expected %d)", name, payload.Version, treeFormatVersion)
	}
	if payload.Kind != kind {
		return nil, nil, fmt.Errorf("Load%s: file contains a %s, not a %s", name, payload.Kind, kind)
	}

	nodes := make([]treeNode, len(payload.Nodes))
	for i, n := range payload.Nodes {
		nodes[i] = nodeFromGob(n)
	}
	return &payload, &treeImpl{
		nodes:     nodes,
		root:      0,
		nFeatures: payload.NFeatures,
	}, nil
}