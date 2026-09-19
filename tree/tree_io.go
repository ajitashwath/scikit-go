package tree

import (
	"bytes"
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

// marshalTree encodes a fitted tree into the versioned gob payload.
// name is the display name used in error messages ("DecisionTreeRegressor"),
// kind is the lowercase payload kind ("regressor").
func marshalTree(name, kind, criterion string, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures int, seed int64, t *treeImpl, classes []float64) ([]byte, error) {
	if t == nil {
		return nil, matutil.ErrNotFitted
	}
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
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(payload); err != nil {
		return nil, fmt.Errorf("%s.Save: encode failed: %w", name, err)
	}
	return buf.Bytes(), nil
}

// saveTree writes a fitted tree to path in the versioned gob format.
func saveTree(path, name, kind, criterion string, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures int, seed int64, t *treeImpl, classes []float64) error {
	data, err := marshalTree(name, kind, criterion, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures, seed, t, classes)
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("%s.Save: %w", name, err)
	}
	return nil
}

// unmarshalTree decodes a versioned tree payload and reconstructs the treeImpl.
// name is the display name used in error messages, kind the expected payload kind.
func unmarshalTree(data []byte, name, kind string) (*treeGob, *treeImpl, error) {
	var payload treeGob
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&payload); err != nil {
		return nil, nil, fmt.Errorf("Load%s: decode failed: %w", name, err)
	}
	if payload.Version != treeFormatVersion {
		return nil, nil, fmt.Errorf("Load%s: unsupported format version %d (expected %d)", name, payload.Version, treeFormatVersion)
	}
	if payload.Kind != kind {
		return nil, nil, fmt.Errorf("Load%s: file contains a %s, not a %s", name, payload.Kind, kind)
	}

	if err := validateNodes(&payload, kind); err != nil {
		return nil, nil, fmt.Errorf("Load%s: corrupt payload: %w", name, err)
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

// validateNodes checks the decoded node array: children must point forward (the
// tree is stored in preorder, so this also rules out cycles), split features must
// exist, and every node's value vector must have the length prediction expects.
func validateNodes(p *treeGob, kind string) error {
	if p.NFeatures < 1 || len(p.Nodes) < 1 {
		return fmt.Errorf("%d nodes over %d features", len(p.Nodes), p.NFeatures)
	}
	valueLen := 1
	if kind == "classifier" {
		valueLen = len(p.Classes)
		if valueLen < 1 {
			return fmt.Errorf("classifier without classes")
		}
	}
	for i, n := range p.Nodes {
		if len(n.Value) != valueLen {
			return fmt.Errorf("node %d has %d values, want %d", i, len(n.Value), valueLen)
		}
		if n.Feature < 0 {
			continue // leaf
		}
		if n.Feature >= p.NFeatures {
			return fmt.Errorf("node %d splits on feature %d of %d", i, n.Feature, p.NFeatures)
		}
		if n.Left <= i || n.Left >= len(p.Nodes) || n.Right <= i || n.Right >= len(p.Nodes) {
			return fmt.Errorf("node %d has children (%d, %d) outside (%d, %d)", i, n.Left, n.Right, i, len(p.Nodes))
		}
	}
	return nil
}

// loadTree reads a versioned tree payload from path.
func loadTree(path, name, kind string) (*treeGob, *treeImpl, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("Load%s: %w", name, err)
	}
	return unmarshalTree(data, name, kind)
}
