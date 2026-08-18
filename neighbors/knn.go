// Package neighbors implements k-nearest-neighbors estimators mirroring
// sklearn.neighbors: KNeighborsRegressor and KNeighborsClassifier with brute-force
// neighbor search, Minkowski distances, and uniform/distance weighting.
package neighbors

import (
	"math"
	"sort"
)

// MinkowskiDistance returns the Minkowski distance between vectors a and b for
// the given exponent p (p=2 Euclidean, p=1 Manhattan). Computed in float64,
// matching sklearn's pairwise_distances for dense arrays.
func MinkowskiDistance(a, b []float64, p float64) float64 {
	if p == 2 {
		var sum float64
		for i := range a {
			d := a[i] - b[i]
			sum += d * d
		}
		return math.Sqrt(sum)
	}
	if p == 1 {
		var sum float64
		for i := range a {
			sum += math.Abs(a[i] - b[i])
		}
		return sum
	}
	var sum float64
	for i := range a {
		d := math.Abs(a[i] - b[i])
		sum += math.Pow(d, p)
	}
	return math.Pow(sum, 1.0/p)
}

// neighbor is a single training sample identified by distance and index.
type neighbor struct {
	index    int
	distance float64
}

// knnModel holds the fitted training data and hyperparameters shared by both
// estimators. X and y are kept by reference at fit time (no deep copy), matching
// sklearn's lazy storage contract.
type knnModel struct {
	X          [][]float64
	y          []float64
	yClass     []int
	classes    []float64
	nFeatures  int
	nNeighbors int
	weights    string // "uniform" or "distance"
	p          float64
	fitted     bool
}

// knnParams carries the hyperparameters used to configure a knnModel.
type knnParams struct {
	nNeighbors int
	weights    string
	p          float64
}

// kneighbors returns the indices of the k nearest training rows to each row of
// X, ordered by ascending distance. Ties are broken by the original training
// index, mirroring sklearn's brute-force argpartition+argsort behavior for
// distinct distances.
func (m *knnModel) kneighbors(X [][]float64) [][]int {
	out := make([][]int, len(X))
	for i, row := range X {
		ns := make([]neighbor, len(m.X))
		for j, trainRow := range m.X {
			ns[j] = neighbor{index: j, distance: MinkowskiDistance(row, trainRow, m.p)}
		}
		sort.Slice(ns, func(a, b int) bool {
			if ns[a].distance != ns[b].distance {
				return ns[a].distance < ns[b].distance
			}
			return ns[a].index < ns[b].index
		})
		indices := make([]int, m.nNeighbors)
		for k := 0; k < m.nNeighbors; k++ {
			indices[k] = ns[k].index
		}
		out[i] = indices
	}
	return out
}

// kneighborsDistances returns the k nearest training row indices and their
// distances to each row of X, both ordered by ascending distance.
func (m *knnModel) kneighborsDistances(X [][]float64) (distances [][]float64, indices [][]int) {
	distances = make([][]float64, len(X))
	indices = make([][]int, len(X))
	for i, row := range X {
		ns := make([]neighbor, len(m.X))
		for j, trainRow := range m.X {
			ns[j] = neighbor{index: j, distance: MinkowskiDistance(row, trainRow, m.p)}
		}
		sort.Slice(ns, func(a, b int) bool {
			if ns[a].distance != ns[b].distance {
				return ns[a].distance < ns[b].distance
			}
			return ns[a].index < ns[b].index
		})
		dist := make([]float64, m.nNeighbors)
		idx := make([]int, m.nNeighbors)
		for k := 0; k < m.nNeighbors; k++ {
			dist[k] = ns[k].distance
			idx[k] = ns[k].index
		}
		distances[i] = dist
		indices[i] = idx
	}
	return distances, indices
}

// neighborWeights mirrors sklearn's _get_weights: nil for uniform, and 1/d for
// distance weights with zero-distance neighbors set to 1.0 and the rest 0.0.
func (m *knnModel) neighborWeights(distances [][]float64) [][]float64 {
	if m.weights == "uniform" {
		return nil
	}
	weights := make([][]float64, len(distances))
	for i, dist := range distances {
		row := make([]float64, len(dist))
		hasZero := false
		for _, d := range dist {
			if d == 0 {
				hasZero = true
				break
			}
		}
		if hasZero {
			for j, d := range dist {
				if d == 0 {
					row[j] = 1.0
				} else {
					row[j] = 0.0
				}
			}
		} else {
			for j, d := range dist {
				row[j] = 1.0 / d
			}
		}
		weights[i] = row
	}
	return weights
}