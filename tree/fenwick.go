package tree

// weightedFenwick mirrors sklearn.tree._utils.WeightedFenwickTree: a Fenwick
// tree (binary indexed tree) maintaining prefix sums of weights and of
// weight * y, with a search that locates the smallest activated rank whose
// cumulative weight exceeds a target. The exact float64 accumulation order
// matters because the tree split search ties on last-ulp differences.
type weightedFenwick struct {
	size    int
	treeW   []float64
	treeWy  []float64
	totalW  float64
	totalWy float64
	maxPow2 int
}

func newWeightedFenwick(capacity int) *weightedFenwick {
	return &weightedFenwick{
		treeW:  make([]float64, capacity+1),
		treeWy: make([]float64, capacity+1),
	}
}

// reset clears the aggregates and sets maxPow2 to the highest power of two <= size.
func (t *weightedFenwick) reset(size int) {
	t.size = size
	for i := range t.treeW {
		t.treeW[i] = 0
		t.treeWy[i] = 0
	}
	t.totalW = 0
	t.totalWy = 0
	p := 1
	for p <= size {
		p <<= 1
	}
	t.maxPow2 = p >> 1
}

// add inserts a weighted observation at 0-based rank idx, mirroring
// WeightedFenwickTree.add including the float64 accumulation order.
func (t *weightedFenwick) add(idx int, yValue, weight float64) {
	weightedY := weight * yValue
	fenwickIdx := idx + 1
	for fenwickIdx <= t.size {
		t.treeW[fenwickIdx] += weight
		t.treeWy[fenwickIdx] += weightedY
		fenwickIdx += fenwickIdx & -fenwickIdx
	}
	t.totalW += weight
	t.totalWy += weightedY
}

// search mirrors WeightedFenwickTree.search: it finds the smallest activated
// rank whose cumulative weight (inclusive) first reaches or exceeds targetWeight
// and returns that rank together with the cumulative weight and weighted-y sum
// strictly below it (exclusive) and the previous activated rank.
func (t *weightedFenwick) search(targetWeight float64) (medianRank int, cumulWeight, cumulWeightedY float64, prevRank int) {
	currentIdx := 0
	var equalTarget float64
	equalBit := 0
	searchBit := t.maxPow2

	// Phase 1: standard Fenwick binary search with prefix accumulation.
	for searchBit != 0 {
		nextIdx := currentIdx + searchBit
		if nextIdx <= t.size {
			nodeWeight := t.treeW[nextIdx]
			if targetWeight == nodeWeight {
				equalTarget = targetWeight
				equalBit = searchBit
				break
			} else if targetWeight > nodeWeight {
				targetWeight -= nodeWeight
				currentIdx = nextIdx
				cumulWeight += nodeWeight
				cumulWeightedY += t.treeWy[nextIdx]
			}
		}
		searchBit >>= 1
	}

	if searchBit == 0 {
		// No exact match: standard search done.
		return currentIdx, cumulWeight, cumulWeightedY, currentIdx
	}

	// Phase 2: exact match - find prevIdx, the largest index with
	// cumulative weight < original target.
	prevIdx := currentIdx
	for searchBit != 0 {
		nextIdx := prevIdx + searchBit
		if nextIdx <= t.size {
			nodeWeight := t.treeW[nextIdx]
			if targetWeight > nodeWeight {
				targetWeight -= nodeWeight
				prevIdx = nextIdx
			}
		}
		searchBit >>= 1
	}

	// Phase 3: restore state and complete the exact-match search.
	searchBit = equalBit
	targetWeight = equalTarget
	for searchBit != 0 {
		nextIdx := currentIdx + searchBit
		if nextIdx <= t.size {
			nodeWeight := t.treeW[nextIdx]
			if targetWeight >= nodeWeight {
				targetWeight -= nodeWeight
				currentIdx = nextIdx
				cumulWeight += nodeWeight
				cumulWeightedY += t.treeWy[nextIdx]
			}
		}
		searchBit >>= 1
	}
	return currentIdx, cumulWeight, cumulWeightedY, prevIdx
}

// computeRanks sorts ys in place, filling sortedY with the sorted values and
// returning ranks such that ranks[j] is the rank of the j-th input value
// (sortedY[ranks[j]] == ys[j]). This mirrors sklearn's compute_ranks where the
// ordering among equal values is irrelevant for the Fenwick float sums.
func computeRanks(ys []float64) (sortedY []float64, ranks []int) {
	n := len(ys)
	sortedIndices := make([]int, n)
	for i := range sortedIndices {
		sortedIndices[i] = i
	}
	sortIntsByValues(sortedIndices, ys)
	sortedY = make([]float64, n)
	ranks = make([]int, n)
	for i, j := range sortedIndices {
		sortedY[i] = ys[j]
		ranks[j] = i
	}
	return sortedY, ranks
}

// sortIntsByValues stably sorts the index slice by the referenced values,
// breaking ties by index to be fully deterministic.
func sortIntsByValues(indices []int, values []float64) {
	// Insertion sort: stable, deterministic, and fine for the small node sizes
	// used in the Fenwick MAE criterion.
	for i := 1; i < len(indices); i++ {
		key := indices[i]
		j := i - 1
		for j >= 0 && (values[indices[j]] > values[key] ||
			(values[indices[j]] == values[key] && indices[j] > key)) {
			indices[j+1] = indices[j]
			j--
		}
		indices[j+1] = key
	}
}

// precomputeAbsoluteErrors fills out[p] with the absolute error of the set
// sampleIndices[start:p] (forward pass, step +1) or sampleIndices[p+1:start+1]
// (backward pass, step -1), exactly as sklearn's precompute_absolute_errors.
// out values are incremented (for multi-output parity) into a zeroed buffer.
func precomputeAbsoluteErrors(sortedY []float64, ranks []int, sampleIndices []int, tree *weightedFenwick, start, end int, out []float64) {
	var step int
	var n int
	if start < end {
		step = 1
		n = end - start
	} else {
		step = -1
		n = start - end
	}
	tree.reset(n)
	p := start
	for i := 0; i < n; i++ {
		rank := ranks[p]
		tree.add(rank, sortedY[rank], 1.0)

		halfWeight := 0.5 * tree.totalW
		medianRank, wLeft, wyLeft, medianPrevRank := tree.search(halfWeight)

		var median float64
		if medianRank != medianPrevRank {
			median = (sortedY[medianPrevRank] + sortedY[medianRank]) / 2
		} else {
			median = sortedY[medianRank]
		}

		wRight := tree.totalW - wLeft
		wyRight := tree.totalWy - wyLeft
		out[p] += (wyRight - median*wRight) + (median*wLeft - wyLeft)
		p += step
	}
}
