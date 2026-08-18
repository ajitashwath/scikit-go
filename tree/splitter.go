package tree

import (
	"container/heap"
	"fmt"
	"math"
	"sort"
)

const (
	// featureThreshold mirrors sklearn.tree._partitioner.FEATURE_THRESHOLD.
	featureThreshold float32 = 1e-7
	// epsilon mirrors sklearn.tree._tree.EPSILON = np.finfo('double').eps.
	epsilon = 2.220446049250313e-16
)

// splitter mirrors sklearn's BestSplitter for dense data without missing values:
// seeded random feature draw order, float32 feature values, FEATURE_THRESHOLD tie
// skipping, and first-best-wins tie-breaking via strict ">" proxy comparisons.
type splitter struct {
	randState        uint32
	features         []int
	constantFeatures []int
	nFeatures        int
	nTotal           int
	criterion        criterion
	minSamplesLeaf   int
	maxFeatures      int
	y                []float64
	yClass           []int
	classes          []float64
	Xf32             [][]float32 // Xf32[f] = float32 column values by sample index
}

// splitRecord is the result of a node split, mirroring sklearn's SplitRecord plus
// the child impurities and normalized impurity improvement.
type splitRecord struct {
	feature       int
	threshold     float64
	pos           int
	proxy         float64
	improvement   float64
	impurityLeft  float64
	impurityRight float64
	nConstants    int
	valid         bool
}

func newSplitter(X [][]float64, y []float64, classes []float64, yClass []int, params treeParams) *splitter {
	nFeatures := len(X[0])
	n := len(X)
	sp := &splitter{
		features:         make([]int, nFeatures),
		constantFeatures: make([]int, nFeatures),
		nFeatures:        nFeatures,
		nTotal:           n,
		criterion:        params.criterion,
		minSamplesLeaf:   params.minSamplesLeaf,
		maxFeatures:      params.maxFeatures,
		y:                y,
		yClass:           yClass,
		classes:          classes,
		Xf32:             make([][]float32, nFeatures),
	}
	for i := range sp.features {
		sp.features[i] = i
	}
	for f := 0; f < nFeatures; f++ {
		col := make([]float32, n)
		for i := 0; i < n; i++ {
			col[i] = float32(X[i][f])
		}
		sp.Xf32[f] = col
	}
	sp.randState = initRandState(uint32(params.seed))
	return sp
}

// featureSorted holds a node's samples sorted ascending by one feature's float32 values.
type featureSorted struct {
	values  []float32
	indices []int
}

func (sp *splitter) sortByFeature(samples []int, feature int) featureSorted {
	n := len(samples)
	fs := featureSorted{values: make([]float32, n), indices: make([]int, n)}
	for i, si := range samples {
		fs.values[i] = sp.Xf32[feature][si]
		fs.indices[i] = si
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return fs.values[a] < fs.values[b] })
	vals := make([]float32, n)
	idx := make([]int, n)
	for i, o := range order {
		vals[i] = fs.values[o]
		idx[i] = fs.indices[o]
	}
	fs.values = vals
	fs.indices = idx
	return fs
}

// nodeSplit finds the best split over the node's samples, mirroring sklearn's
// best_split loop including the constant-feature handling and the persistent
// features array.
func (sp *splitter) nodeSplit(samples []int, parentImpurity float64, nKnownConstants int) splitRecord {
	n := len(samples)
	features := sp.features
	constantFeatures := sp.constantFeatures
	nTotalConstants := nKnownConstants
	nDrawnConstants := 0
	nFoundConstants := 0
	nVisitedFeatures := 0
	maxFeatures := sp.maxFeatures
	if maxFeatures <= 0 {
		maxFeatures = sp.nFeatures
	}
	fi := sp.nFeatures
	bestProxy := math.Inf(-1)
	var best splitRecord

	for fi > nTotalConstants && (nVisitedFeatures < maxFeatures || nVisitedFeatures <= nFoundConstants+nDrawnConstants) {
		nVisitedFeatures++
		fj := randInt(nDrawnConstants, fi-nFoundConstants, &sp.randState)
		if fj < nKnownConstants {
			features[nDrawnConstants], features[fj] = features[fj], features[nDrawnConstants]
			nDrawnConstants++
			continue
		}
		fj += nFoundConstants
		feature := features[fj]

		fs := sp.sortByFeature(samples, feature)

		// Constant feature detection: max - min <= FEATURE_THRESHOLD.
		if fs.values[n-1] <= fs.values[0]+featureThreshold {
			features[fj], features[nTotalConstants] = features[nTotalConstants], features[fj]
			nFoundConstants++
			nTotalConstants++
			continue
		}

		fi--
		features[fi], features[fj] = features[fj], features[fi]

		cand := sp.scanFeature(fs, feature, parentImpurity)
		if cand.valid && cand.proxy > bestProxy {
			bestProxy = cand.proxy
			best = cand
		}
		if cand.valid && n == sp.nTotal {
			fmt.Printf("  debug: feature %d bestProxy=%.12f pos=%d threshold=%.8f\n", feature, cand.proxy, cand.pos, cand.threshold)
		}
	}

	// Restore invariants for constant features (vacuous when none exist).
	copy(features[:nKnownConstants], constantFeatures[:nKnownConstants])
	copy(constantFeatures[nKnownConstants:nTotalConstants], features[nKnownConstants:nTotalConstants])
	best.nConstants = nTotalConstants
	return best
}

// scanFeature evaluates all valid split positions of one feature and returns the
// best candidate together with its children impurities and improvement.
func (sp *splitter) scanFeature(fs featureSorted, feature int, parentImpurity float64) splitRecord {
	switch sp.criterion {
	case criterionMSE:
		return sp.scanMSE(fs, feature, parentImpurity)
	case criterionMAE:
		return sp.scanMAE(fs, feature, parentImpurity)
	default:
		return sp.scanClass(fs, feature, parentImpurity)
	}
}

// scanMSE mirrors the MSE criterion: proxy = sum_L^2/n_L + sum_R^2/n_R (maximize),
// children impurity = sq_sum/n - (sum/n)^2, with sklearn's update direction logic.
func (sp *splitter) scanMSE(fs featureSorted, feature int, parentImpurity float64) splitRecord {
	n := len(fs.values)
	var sumTotal, sqSumTotal float64
	for _, si := range fs.indices {
		v := sp.y[si]
		sumTotal += v
		sqSumTotal += v * v
	}
	bestProxy := math.Inf(-1)
	bestPos := -1
	pos := 0
	var sumL float64
	for i := 0; i < n-1; i++ {
		if fs.values[i+1] <= fs.values[i]+featureThreshold {
			continue
		}
		p := i + 1
		nL, nR := p, n-p
		if nL < sp.minSamplesLeaf || nR < sp.minSamplesLeaf {
			continue
		}
		sumL = sp.updateMSE(fs.indices, n, pos, p, sumL, sumTotal)
		pos = p
		sumR := sumTotal - sumL
		proxy := sumL*sumL/float64(nL) + sumR*sumR/float64(nR)
		if proxy > bestProxy {
			bestProxy = proxy
			bestPos = p
		}
	}
	if bestPos < 0 {
		return splitRecord{}
	}
	return sp.finishMSE(fs, feature, parentImpurity, bestPos, bestProxy, sumTotal, sqSumTotal)
}

// finishMSE reconstructs the criterion state at bestPos from pos=0 (as sklearn's
// final partition + children_impurity does) and returns the completed record.
func (sp *splitter) finishMSE(fs featureSorted, feature int, parentImpurity float64, bestPos int, bestProxy, sumTotal, sqSumTotal float64) splitRecord {
	n := len(fs.values)
	nL, nR := bestPos, n-bestPos
	sumL := sp.updateMSE(fs.indices, n, 0, bestPos, 0, sumTotal)
	sumR := sumTotal - sumL
	var sqSumL float64
	for q := 0; q < bestPos; q++ {
		v := sp.y[fs.indices[q]]
		sqSumL += v * v
	}
	sqSumR := sqSumTotal - sqSumL
	impL := sqSumL/float64(nL) - (sumL/float64(nL))*(sumL/float64(nL))
	impR := sqSumR/float64(nR) - (sumR/float64(nR))*(sumR/float64(nR))
	return sp.makeRecord(fs, feature, parentImpurity, bestPos, bestProxy, impL, impR)
}

// updateMSE mirrors sklearn's MSE criterion.update direction logic.
func (sp *splitter) updateMSE(indices []int, n, pos, newPos int, sumL, sumTotal float64) float64 {
	if (newPos-pos) <= (n-newPos) {
		for q := pos; q < newPos; q++ {
			sumL += sp.y[indices[q]]
		}
		return sumL
	}
	sumL = sumTotal
	for q := n - 1; q >= newPos; q-- {
		sumL -= sp.y[indices[q]]
	}
	return sumL
}

// scanClass mirrors the gini/entropy criteria: proxy = -(n_R*imp_R + n_L*imp_L)
// with children impurity from integer class counts (exact, direction-free).
func (sp *splitter) scanClass(fs featureSorted, feature int, parentImpurity float64) splitRecord {
	n := len(fs.values)
	nClasses := len(sp.classes)
	countsL := make([]float64, nClasses)
	countsR := make([]float64, nClasses)
	for _, si := range fs.indices {
		countsR[sp.yClass[si]]++
	}
	bestProxy := math.Inf(-1)
	bestPos := -1
	pos := 0
	for i := 0; i < n-1; i++ {
		if fs.values[i+1] <= fs.values[i]+featureThreshold {
			continue
		}
		p := i + 1
		nL, nR := p, n-p
		if nL < sp.minSamplesLeaf || nR < sp.minSamplesLeaf {
			continue
		}
		for q := pos; q < p; q++ {
			c := sp.yClass[fs.indices[q]]
			countsL[c]++
			countsR[c]--
		}
		pos = p
		impL := sp.classImpurity(countsL, float64(nL))
		impR := sp.classImpurity(countsR, float64(nR))
		proxy := -float64(nR)*impR - float64(nL)*impL
		if proxy > bestProxy {
			bestProxy = proxy
			bestPos = p
		}
	}
	if bestPos < 0 {
		return splitRecord{}
	}
	// Recompute counts at bestPos from scratch (equal to the scan-time counts).
	countsL = make([]float64, nClasses)
	countsR = make([]float64, nClasses)
	for _, si := range fs.indices {
		countsR[sp.yClass[si]]++
	}
	for q := 0; q < bestPos; q++ {
		c := sp.yClass[fs.indices[q]]
		countsL[c]++
		countsR[c]--
	}
	impL := sp.classImpurity(countsL, float64(bestPos))
	impR := sp.classImpurity(countsR, float64(n-bestPos))
	return sp.makeRecord(fs, feature, parentImpurity, bestPos, bestProxy, impL, impR)
}

// classImpurity mirrors sklearn's gini/entropy children_impurity with base-2 log.
func (sp *splitter) classImpurity(counts []float64, nn float64) float64 {
	switch sp.criterion {
	case criterionGini:
		var sq float64
		for _, c := range counts {
			sq += c * c
		}
		return 1.0 - sq/(nn*nn)
	default: // criterionEntropy
		ent := 0.0
		for _, c := range counts {
			if c > 0 {
				p := c / nn
				ent -= p * (math.Log(p) / math.Log(2))
			}
		}
		return ent
	}
}

// scanMAE mirrors the MAE criterion: proxy = -(AE_left + AE_right) with absolute
// errors computed via the statistical (weighted) median.
func (sp *splitter) scanMAE(fs featureSorted, feature int, parentImpurity float64) splitRecord {
	n := len(fs.values)
	leftAE := make([]float64, n)
	rightAE := make([]float64, n)
	sp.computePrefixAE(fs.indices, leftAE)
	sp.computeSuffixAE(fs.indices, rightAE)
	bestProxy := math.Inf(-1)
	bestPos := -1
	for i := 0; i < n-1; i++ {
		if fs.values[i+1] <= fs.values[i]+featureThreshold {
			continue
		}
		p := i + 1
		nL, nR := p, n-p
		if nL < sp.minSamplesLeaf || nR < sp.minSamplesLeaf {
			continue
		}
		proxy := -(leftAE[i] + rightAE[p])
		if proxy > bestProxy {
			bestProxy = proxy
			bestPos = p
		}
	}
	if bestPos < 0 {
		return splitRecord{}
	}
	impL := leftAE[bestPos-1] / float64(bestPos)
	impR := rightAE[bestPos] / float64(n-bestPos)
	return sp.makeRecord(fs, feature, parentImpurity, bestPos, bestProxy, impL, impR)
}

// makeRecord builds the final split record with the sklearn impurity improvement
// formula: (n_node/n_total) * (parent - (n_R/n)*imp_R - (n_L/n)*imp_L).
func (sp *splitter) makeRecord(fs featureSorted, feature int, parentImpurity float64, pos int, proxy, impL, impR float64) splitRecord {
	n := len(fs.values)
	nL, nR := pos, n-pos
	nf, nLf, nRf := float64(n), float64(nL), float64(nR)
	improvement := (nf / float64(sp.nTotal)) * (parentImpurity - (nRf/nf)*impR - (nLf/nf)*impL)
	threshold := float64(fs.values[pos-1])/2 + float64(fs.values[pos])/2
	return splitRecord{
		feature:       feature,
		threshold:     threshold,
		pos:           pos,
		proxy:         proxy,
		improvement:   improvement,
		impurityLeft:  impL,
		impurityRight: impR,
		valid:         true,
	}
}

// computePrefixAE fills out[i] with the mean-absolute-error sum (|y - median|)
// of fs.indices[0:i+1] using sklearn's weighted-median convention.
func (sp *splitter) computePrefixAE(indices []int, out []float64) {
	var left maxHeap
	var right minHeap
	heap.Init(&left)
	heap.Init(&right)
	var sumL, sumR float64
	for i, si := range indices {
		v := sp.y[si]
		if left.Len() == 0 || v <= left.Top() {
			heap.Push(&left, v)
			sumL += v
		} else {
			heap.Push(&right, v)
			sumR += v
		}
		if left.Len() > right.Len()+1 {
			m := heap.Pop(&left).(float64)
			heap.Push(&right, m)
			sumL -= m
			sumR += m
		} else if right.Len() > left.Len() {
			m := heap.Pop(&right).(float64)
			heap.Push(&left, m)
			sumR -= m
			sumL += m
		}
		out[i] = absError(left, right, sumL, sumR)
	}
}

// computeSuffixAE fills out[i] with the mean-absolute-error sum of fs.indices[i:n].
func (sp *splitter) computeSuffixAE(indices []int, out []float64) {
	n := len(indices)
	var left maxHeap
	var right minHeap
	heap.Init(&left)
	heap.Init(&right)
	var sumL, sumR float64
	for i := n - 1; i >= 0; i-- {
		v := sp.y[indices[i]]
		if left.Len() == 0 || v <= left.Top() {
			heap.Push(&left, v)
			sumL += v
		} else {
			heap.Push(&right, v)
			sumR += v
		}
		if left.Len() > right.Len()+1 {
			m := heap.Pop(&left).(float64)
			heap.Push(&right, m)
			sumL -= m
			sumR += m
		} else if right.Len() > left.Len() {
			m := heap.Pop(&right).(float64)
			heap.Push(&left, m)
			sumR -= m
			sumL += m
		}
		out[i] = absError(left, right, sumL, sumR)
	}
}

// absError mirrors sklearn's pinball-loss identity for the absolute error of the
// current median set: AE = (wy_R - med*w_R) + (med*w_L - wy_L).
func absError(left maxHeap, right minHeap, sumL, sumR float64) float64 {
	k := left.Len() + right.Len()
	var med, wL, wyL float64
	if right.Len() == 0 {
		med = left.Top()
		wL = 0
		wyL = 0
	} else if left.Len() > right.Len() {
		med = left.Top()
		wL = float64(left.Len() - 1)
		wyL = sumL - med
	} else {
		med = (left.Top() + right.Top()) / 2
		wL = float64(left.Len())
		wyL = sumL
	}
	wR := float64(k) - wL
	wyR := (sumL + sumR) - wyL
	return (wyR - med*wR) + (med*wL - wyL)
}