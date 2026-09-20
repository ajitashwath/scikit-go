// Package tree implements CART decision trees mirroring sklearn.tree:
// DecisionTreeRegressor (criterion "mse" or "mae") and
// DecisionTreeClassifier (criterion "gini" or "entropy").
package tree

import (
	"errors"
	"fmt"
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
	t.root = b.build(samples, 0, rootImpurity, 0, b.nodeSumsMSE(samples))
	return t
}

// build recursively grows the tree over the given sample indices and returns the
// node index. impurity is the node's stored impurity: node_impurity for the root,
// and the parent split's children impurity for children, matching sklearn's
// parent_record.impurity. nodeSums carries the per-node sum_total/sq_sum_total
// (sklearn computes them once per node over the node's initial sample order and
// shares them across all feature scans).
func (b *treeBuilder) build(samples []int, depth int, impurity float64, nConstants int, sums nodeSums) int {
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

	rec := b.sp.nodeSplit(samples, impurity, nConstants, sums)
	if !rec.valid || rec.pos >= n || rec.improvement+epsilon < 0 {
		return nodeIdx
	}

	// nodeSplit has already partitioned samples in place; the children are the
	// two halves, in the order the partition left them (sklearn's behavior).
	left, right := samples[:rec.pos], samples[rec.pos:]

	b.t.nodes[nodeIdx].Left = b.build(left, depth+1, rec.impurityLeft, rec.nConstants, b.childSums(left))
	b.t.nodes[nodeIdx].Right = b.build(right, depth+1, rec.impurityRight, rec.nConstants, b.childSums(right))
	b.t.nodes[nodeIdx].Feature = rec.feature
	b.t.nodes[nodeIdx].Threshold = rec.threshold
	return nodeIdx
}

// nodeSums holds the per-node sum_total and sq_sum_total for the MSE criterion,
// computed once over the node's initial sample order.
type nodeSums struct {
	sumTotal    float64
	sqSumTotal  float64
	initialized bool
}

// childSums returns the per-node MSE sums for the given child samples, or an
// uninitialized value for non-MSE criteria (which ignore them).
func (b *treeBuilder) childSums(samples []int) nodeSums {
	if b.t.params.criterion != criterionMSE {
		return nodeSums{}
	}
	return b.nodeSumsMSE(samples)
}

// nodeSumsMSE accumulates sum_total/sq_sum_total over samples in the given order,
// mirroring sklearn's RegressionCriterion.init.
func (b *treeBuilder) nodeSumsMSE(samples []int) nodeSums {
	var sumTotal, sqSumTotal float64
	for _, si := range samples {
		v := b.y[si]
		sumTotal += v
		sqSumTotal += v * v
	}
	return nodeSums{sumTotal: sumTotal, sqSumTotal: sqSumTotal, initialized: true}
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
		ys := make([]float64, len(samples))
		for i, si := range samples {
			ys[i] = b.y[si]
		}
		sortedY, ranks := computeRanks(ys)
		tree := newWeightedFenwick(len(samples))
		out := make([]float64, len(samples))
		precomputeAbsoluteErrors(sortedY, ranks, samples, tree, len(samples)-1, -1, out)
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

// ErrInvalidCriterion is returned for unsupported criterion strings.
var ErrInvalidCriterion = errors.New("tree: unsupported criterion")

// ErrInvalidParams is returned when tree hyperparameters fail validation.
var ErrInvalidParams = errors.New("tree: invalid hyperparameters")

// validateParams checks the integer hyperparameters shared by both estimators,
// using sklearn's bounds: min_samples_split >= 2, min_samples_leaf >= 1 and
// max_features >= 0 (0 meaning all features). MaxDepth < 0 already means
// unlimited, so any value is accepted.
func validateParams(minSamplesSplit, minSamplesLeaf, maxFeatures int) error {
	if minSamplesSplit < 2 {
		return fmt.Errorf("%w: min_samples_split must be >= 2, got %d", ErrInvalidParams, minSamplesSplit)
	}
	if minSamplesLeaf < 1 {
		return fmt.Errorf("%w: min_samples_leaf must be >= 1, got %d", ErrInvalidParams, minSamplesLeaf)
	}
	if maxFeatures < 0 {
		return fmt.Errorf("%w: max_features must be >= 0, got %d", ErrInvalidParams, maxFeatures)
	}
	return nil
}

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
	rootN := float64(t.nodes[t.root].NSamples)
	if rootN == 0 {
		return imp
	}
	for i := range t.nodes {
		node := &t.nodes[i]
		if node.Feature < 0 {
			continue
		}
		left := &t.nodes[node.Left]
		right := &t.nodes[node.Right]
		// Mirror sklearn's accumulation order: n*impurity summed without dividing
		// by the root sample count until the end (matches float rounding).
		gain := float64(node.NSamples)*node.Impurity -
			float64(left.NSamples)*left.Impurity -
			float64(right.NSamples)*right.Impurity
		imp[node.Feature] += gain
	}
	for j := range imp {
		imp[j] /= rootN
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
