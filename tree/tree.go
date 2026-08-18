// Package tree implements CART decision trees mirroring sklearn.tree:
// DecisionTreeRegressor (criterion "mse" or "mae") and
// DecisionTreeClassifier (criterion "gini" or "entropy").
package tree

import (
	"errors"
	"sort"
)

// criterion enumerates the supported impurity criteria.
type criterion int

const (
	criterionMSE criterion = iota
	criterionMAE
	criterionGini
	criterionEntropy
)

// treeNode is a single node of the fitted tree. Internal nodes have Left/Right >= 0 and
// Feature/Threshold describing the split; leaves have Left = Right = Feature = -1.
type treeNode struct {
	Left      int
	Right     int
	Feature   int
	Threshold float64
	Value     []float64
	Impurity  float64
	NSamples  int
}

// treeParams carries the hyperparameters shared by both estimators.
type treeParams struct {
	criterion       criterion
	maxDepth        int
	minSamplesSplit int
	minSamplesLeaf  int
	maxFeatures     int
	seed            int64
}

// treeImpl is the fitted tree: a preorder node array plus build-time parameters.
type treeImpl struct {
	nodes     []treeNode
	root      int
	nFeatures int
	params    treeParams
}

// treeBuilder grows the tree using the splitter, mirroring sklearn's builder:
// left-first processing order, impurity from the parent split, and propagation of
// constant-feature counts.
type treeBuilder struct {
	t       *treeImpl
	sp      *splitter
	y       []float64
	classes []float64
	yClass  []int
}

// buildTree fits CART to X/y. classes is the sorted unique target values for
// classification (nil for regression); yClass maps each sample to a class index.
func buildTree(X [][]float64, y []float64, classes []float64, yClass []int, params treeParams) *treeImpl {
	t := &treeImpl{
		nFeatures: len(X[0]),
		params:    params,
	}
	n := len(X)
	samples := make([]int, n)
	for i := range samples {
		samples[i] = i
	}
	b := &treeBuilder{
		t:       t,
		sp:      newSplitter(X, y, classes, yClass, params),
		y:       y,
		classes: classes,
		yClass:  yClass,
	}
	rootImpurity := b.nodeImpurity(samples)
	t.root = b.build(samples, 0, rootImpurity, 0)
	return t
}

// build recursively grows the tree over the given sample indices and returns the
// node index. impurity is the node's impurity (root: node_impurity; children: the
// parent's split children impurity), matching sklearn's parent_record.impurity.
func (b *treeBuilder) build(samples []int, depth int, impurity float64, nConstants int) int {
	n := len(samples)
	node := treeNode{
		Left:     -1,
		Right:    -1,
		Feature:  -1,
		Value:    b.nodeValue(samples),
		Impurity: impurity,
		NSamples: n,
	}
	nodeIdx := len(b.t.nodes)
	b.t.nodes = append(b.t.nodes, node)

	if (b.t.params.maxDepth >= 0 && depth >= b.t.params.maxDepth) ||
		n < b.t.params.minSamplesSplit ||
		n < 2*b.t.params.minSamplesLeaf ||
		impurity <= epsilon {
		return nodeIdx
	}

	rec := b.sp.nodeSplit(samples, impurity, nConstants)
	if !rec.valid || rec.pos >= n || rec.improvement+epsilon < 0 {
		return nodeIdx
	}

	left := make([]int, 0, rec.pos)
	right := make([]int, 0, n-rec.pos)
	fs := b.sp.sortByFeature(samples, rec.feature)
	for i, si := range fs.indices {
		if i < rec.pos {
			left = append(left, si)
		} else {
			right = append(right, si)
		}
	}

	b.t.nodes[nodeIdx].Left = b.build(left, depth+1, rec.impurityLeft, rec.nConstants)
	b.t.nodes[nodeIdx].Right = b.build(right, depth+1, rec.impurityRight, rec.nConstants)
	b.t.nodes[nodeIdx].Feature = rec.feature
	b.t.nodes[nodeIdx].Threshold = rec.threshold
	return nodeIdx
}

// nodeValue returns the leaf prediction (node value) of a set of samples,
// mirroring sklearn's criterion.node_value: mean for MSE, median for MAE, and the
// normalized class distribution for gini/entropy.
func (b *treeBuilder) nodeValue(samples []int) []float64 {
	switch b.t.params.criterion {
	case criterionMSE:
		var sum float64
		for _, si := range samples {
			sum += b.y[si]
		}
		return []float64{sum / float64(len(samples))}
	case criterionMAE:
		vals := make([]float64, len(samples))
		for i, si := range samples {
			vals[i] = b.y[si]
		}
		return []float64{medianStatistical(vals)}
	default:
		counts := make([]float64, len(b.classes))
		for _, si := range samples {
			counts[b.yClass[si]]++
		}
		nn := float64(len(samples))
		out := make([]float64, len(counts))
		for i, c := range counts {
			out[i] = c / nn
		}
		return out
	}
}

// nodeImpurity mirrors sklearn's criterion.node_impurity for the root.
func (b *treeBuilder) nodeImpurity(samples []int) float64 {
	switch b.t.params.criterion {
	case criterionMSE:
		var sum, sqSum float64
		for _, si := range samples {
			v := b.y[si]
			sum += v
			sqSum += v * v
		}
		nf := float64(len(samples))
		return sqSum/nf - (sum/nf)*(sum/nf)
	case criterionMAE:
		out := make([]float64, len(samples))
		b.sp.computeSuffixAE(samples, out)
		return out[0] / float64(len(samples))
	default:
		counts := make([]float64, len(b.classes))
		for _, si := range samples {
			counts[b.yClass[si]]++
		}
		return b.sp.classImpurity(counts, float64(len(samples)))
	}
}

// medianStatistical returns the statistical median of vals (average of the two
// middle elements for even counts), matching sklearn's MAE weighted median.
func medianStatistical(vals []float64) float64 {
	sorted := append([]float64(nil), vals...)
	sort.Float64s(sorted)
	n := len(sorted)
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}

// maxHeap is a max-heap of float64 (implemented as a negated min-heap).
type maxHeap []float64

func (h maxHeap) Len() int            { return len(h) }
func (h maxHeap) Less(i, j int) bool  { return h[i] > h[j] }
func (h maxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x interface{}) { *h = append(*h, x.(float64)) }
func (h *maxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	v := old[n-1]
	*h = old[:n-1]
	return v
}
func (h maxHeap) Top() float64 { return h[0] }

// minHeap is a min-heap of float64.
type minHeap []float64

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i] < h[j] }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{}) { *h = append(*h, x.(float64)) }
func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	v := old[n-1]
	*h = old[:n-1]
	return v
}
func (h minHeap) Top() float64 { return h[0] }

// ErrInvalidCriterion is returned for unsupported criterion strings.
var ErrInvalidCriterion = errors.New("tree: unsupported criterion")

// predictValue traverses the tree iteratively and returns the leaf's value vector.
func (t *treeImpl) predictValue(row []float64) []float64 {
	idx := t.root
	for t.nodes[idx].Feature >= 0 {
		if row[t.nodes[idx].Feature] <= t.nodes[idx].Threshold {
			idx = t.nodes[idx].Left
		} else {
			idx = t.nodes[idx].Right
		}
	}
	return t.nodes[idx].Value
}

// featureImportances returns the normalized impurity-based feature importances,
// matching sklearn: gains weighted by node sample fraction, normalized to sum 1.
// A single-node tree yields all zeros.
func (t *treeImpl) featureImportances() []float64 {
	imp := make([]float64, t.nFeatures)
	total := float64(t.nodes[t.root].NSamples)
	if total == 0 {
		return imp
	}
	for i := range t.nodes {
		node := &t.nodes[i]
		if node.Feature < 0 {
			continue
		}
		left := &t.nodes[node.Left]
		right := &t.nodes[node.Right]
		gain := (float64(node.NSamples)/total)*node.Impurity -
			(float64(left.NSamples)/total)*left.Impurity -
			(float64(right.NSamples)/total)*right.Impurity
		imp[node.Feature] += gain
	}
	var sum float64
	for _, v := range imp {
		sum += v
	}
	if sum > 0 {
		for j := range imp {
			imp[j] /= sum
		}
	}
	return imp
}