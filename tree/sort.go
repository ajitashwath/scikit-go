package tree

import "math"

// simultaneousSort sorts values ascending and applies the same permutation to
// indices. It is a port of sklearn.utils._sorting.simultaneous_sort with
// use_three_way_partition=True (introsort: median-of-3 quicksort with 3-way
// partitioning, heapsort past the depth limit, insertion sort for short runs).
//
// The sort is not stable, and that is deliberate: sklearn's tree builder keeps
// its sample array sorted in place, and the order in which samples with equal
// feature values (and, downstream, the float rounding of node sums) come out
// depends on this exact algorithm. Matching it keeps split choices identical
// to sklearn's even when candidate splits tie to the last bit.
func simultaneousSort(values []float32, indices []int) {
	n := len(values)
	if n == 0 {
		return
	}
	introsort3Way(values, indices, 2*int(math.Log2(float64(n))))
}

func swapVI(values []float32, indices []int, i, j int) {
	values[i], values[j] = values[j], values[i]
	indices[i], indices[j] = indices[j], indices[i]
}

func introsort3Way(values []float32, indices []int, maxd int) {
	for len(values) > 15 {
		if maxd <= 0 { // depth limit exceeded ("gone quadratic")
			heapSort(values, indices)
			return
		}
		maxd--
		pivot := median3(values)
		i, l, r := 0, 0, len(values)
		for i < r {
			switch {
			case values[i] < pivot:
				swapVI(values, indices, i, l)
				i++
				l++
			case values[i] > pivot:
				r--
				swapVI(values, indices, i, r)
			default:
				i++
			}
		}
		introsort3Way(values[:l], indices[:l], maxd)
		values, indices = values[r:], indices[r:]
	}
	insertionSort(values, indices)
}

func median3(v []float32) float32 {
	n := len(v)
	a, b, c := v[0], v[n/2], v[n-1]
	if a < b {
		if b < c {
			return b
		} else if a < c {
			return c
		}
		return a
	}
	if b < c {
		if a < c {
			return a
		}
		return c
	}
	return b
}

func heapSort(values []float32, indices []int) {
	n := len(values)
	start := (n - 2) / 2
	for {
		siftDown(values, indices, start, n)
		if start == 0 {
			break
		}
		start--
	}
	for end := n - 1; end > 0; end-- {
		swapVI(values, indices, 0, end)
		siftDown(values, indices, 0, end)
	}
}

func siftDown(values []float32, indices []int, start, end int) {
	root := start
	for {
		child := root*2 + 1
		maxind := root
		if child < end && values[maxind] < values[child] {
			maxind = child
		}
		if child+1 < end && values[maxind] < values[child+1] {
			maxind = child + 1
		}
		if maxind == root {
			return
		}
		swapVI(values, indices, root, maxind)
		root = maxind
	}
}

func insertionSort(values []float32, indices []int) {
	for i := 1; i < len(values); i++ {
		tv, ti := values[i], indices[i]
		j := i
		for j > 0 && values[j-1] > tv {
			values[j] = values[j-1]
			indices[j] = indices[j-1]
			j--
		}
		values[j] = tv
		indices[j] = ti
	}
}
