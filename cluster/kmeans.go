// Package cluster implements clustering estimators mirroring sklearn.cluster:
// KMeans with k-means++ seeding and Lloyd iterations.
package cluster

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/ajitashwath/scikit-go/internal/matutil"
)

// ErrInvalidKMeans is returned when KMeans hyperparameters fail validation.
var ErrInvalidKMeans = errors.New("invalid KMeans parameters")

// KMeans fits k clusters via the Lloyd algorithm with k-means++ initialization,
// mirroring sklearn.cluster.KMeans. Multiple restarts (NInit) are run and the
// configuration with the lowest inertia is kept.
type KMeans struct {
	NClusters int
	MaxIter   int
	NInit     int
	Tol       float64
	Seed      int64

	centers [][]float64
	labels  []float64
	inertia float64
	fitted  bool
}

// NewKMeans returns an unfitted KMeans with sklearn's default hyperparameters.
func NewKMeans() *KMeans {
	return &KMeans{
		NClusters: 8,
		MaxIter:   300,
		NInit:     10,
		Tol:       1e-4,
		Seed:      0,
	}
}

// Fit runs k-means++ seeding followed by Lloyd iterations over NInit restarts,
// keeping the assignment with the smallest inertia. The y argument is ignored,
// matching sklearn's fit(X, y=None) signature.
func (k *KMeans) Fit(X [][]float64, y []float64) error {
	if err := matutil.ValidateXMatrix(X); err != nil {
		return fmt.Errorf("KMeans.Fit: %w", err)
	}
	if k.NClusters < 1 {
		return fmt.Errorf("KMeans.Fit: %w: n_clusters must be >= 1, got %d", ErrInvalidKMeans, k.NClusters)
	}
	if k.NClusters > len(X) {
		return fmt.Errorf("KMeans.Fit: %w: n_clusters=%d exceeds n_samples=%d", ErrInvalidKMeans, k.NClusters, len(X))
	}
	if k.MaxIter < 1 {
		return fmt.Errorf("KMeans.Fit: %w: max_iter must be >= 1, got %d", ErrInvalidKMeans, k.MaxIter)
	}
	if k.NInit < 1 {
		return fmt.Errorf("KMeans.Fit: %w: n_init must be >= 1, got %d", ErrInvalidKMeans, k.NInit)
	}
	if k.Tol < 0 {
		return fmt.Errorf("KMeans.Fit: %w: tol must be >= 0, got %v", ErrInvalidKMeans, k.Tol)
	}

	var bestCenters [][]float64
	var bestLabels []float64
	bestInertia := math.Inf(1)
	for init := 0; init < k.NInit; init++ {
		rng := rand.New(rand.NewSource(k.Seed + int64(init)))
		centers := kmeansPlusPlus(X, k.NClusters, rng)
		centers, labels, inertia := lloyd(X, centers, k.MaxIter, k.Tol)
		if inertia < bestInertia {
			bestInertia = inertia
			bestCenters = centers
			bestLabels = labels
		}
	}

	k.centers = bestCenters
	k.labels = bestLabels
	k.inertia = bestInertia
	k.fitted = true
	return nil
}

// Predict assigns each row of X to the nearest cluster center.
func (k *KMeans) Predict(X [][]float64) ([]float64, error) {
	if !k.fitted {
		return nil, matutil.ErrNotFitted
	}
	if err := matutil.ValidateX(X, len(k.centers[0])); err != nil {
		return nil, fmt.Errorf("KMeans.Predict: %w", err)
	}
	preds := make([]float64, len(X))
	for i, row := range X {
		preds[i] = float64(nearestCenter(row, k.centers))
	}
	return preds, nil
}

// FitPredict fits the model and returns the training labels.
func (k *KMeans) FitPredict(X [][]float64) ([]float64, error) {
	if err := k.Fit(X, nil); err != nil {
		return nil, err
	}
	return k.Labels(), nil
}

// Labels returns the cluster index assigned to each training sample during Fit.
func (k *KMeans) Labels() []float64 {
	if !k.fitted {
		return nil
	}
	out := make([]float64, len(k.labels))
	copy(out, k.labels)
	return out
}

// ClusterCenters returns a copy of the fitted cluster centroids.
func (k *KMeans) ClusterCenters() [][]float64 {
	if !k.fitted {
		return nil
	}
	out := make([][]float64, len(k.centers))
	for i, c := range k.centers {
		out[i] = append([]float64(nil), c...)
	}
	return out
}

// Inertia returns the sum of squared distances of samples to their nearest
// cluster center.
func (k *KMeans) Inertia() float64 {
	if !k.fitted {
		return 0
	}
	return k.inertia
}

// Score returns the negative inertia on X, matching sklearn's KMeans.score.
func (k *KMeans) Score(X [][]float64, y []float64) (float64, error) {
	preds, err := k.Predict(X)
	if err != nil {
		return 0, err
	}
	var sum float64
	for i, row := range X {
		sum += sqDist(row, k.centers[int(preds[i])])
	}
	return -sum, nil
}

// kmeansPlusPlus implements sklearn's _kmeans_plusplus seeding: the first center
// is a uniform random sample; each subsequent center is drawn among samples with
// probability proportional to the squared distance to the nearest existing
// center, greedily keeping the best of a logarithmic number of local trials.
func kmeansPlusPlus(X [][]float64, nClusters int, rng *rand.Rand) [][]float64 {
	nSamples := len(X)
	centers := make([][]float64, nClusters)
	centers[0] = append([]float64(nil), X[rng.Intn(nSamples)]...)

	nLocalTrials := 2 + int(math.Log(float64(nClusters)))
	closestDistSq := make([]float64, nSamples)
	for i := range X {
		closestDistSq[i] = sqDist(X[i], centers[0])
	}

	for c := 1; c < nClusters; c++ {
		// cumulative distribution over samples
		cumSum := make([]float64, nSamples)
		var total float64
		for i, d := range closestDistSq {
			total += d
			cumSum[i] = total
		}

		bestCandidate := -1
		bestPotential := math.Inf(1)
		var bestClosest []float64
		for trial := 0; trial < nLocalTrials; trial++ {
			candidate := sampleFromCumulative(cumSum, total, rng)
			// candidate potentials: for each sample, min(closest, dist to candidate)
			var potential float64
			closest := make([]float64, nSamples)
			for i := range X {
				d := sqDist(X[i], X[candidate])
				if d < closestDistSq[i] {
					closest[i] = d
				} else {
					closest[i] = closestDistSq[i]
				}
				potential += closest[i]
			}
			if potential < bestPotential {
				bestPotential = potential
				bestCandidate = candidate
				bestClosest = closest
			}
		}
		centers[c] = append([]float64(nil), X[bestCandidate]...)
		closestDistSq = bestClosest
	}
	return centers
}

// sampleFromCumulative picks an index proportional to the weights encoded in a
// cumulative sum, mirroring np.searchsorted(np.cumsum(weights), rand * total).
func sampleFromCumulative(cumSum []float64, total float64, rng *rand.Rand) int {
	target := rng.Float64() * total
	// binary search for the first entry >= target
	lo, hi := 0, len(cumSum)
	for lo < hi {
		mid := (lo + hi) / 2
		if cumSum[mid] < target {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo >= len(cumSum) {
		lo = len(cumSum) - 1
	}
	return lo
}

// lloyd runs the Lloyd iterations from the given centers until convergence
// (labels unchanged or squared center shift <= tolerance) or MaxIter. Empty
// clusters are re-seeded to the sample farthest from its nearest center, and
// returns the final centers, labels, and inertia.
func lloyd(X [][]float64, centers [][]float64, maxIter int, tol float64) ([][]float64, []float64, float64) {
	nSamples := len(X)
	nClusters := len(centers)
	labels := make([]float64, nSamples)

	// sklearn scales the tol by the mean per-feature variance.
	effectiveTol := tol
	if tol > 0 {
		var varSum float64
		nFeatures := len(X[0])
		for j := 0; j < nFeatures; j++ {
			var mean float64
			for i := 0; i < nSamples; i++ {
				mean += X[i][j]
			}
			mean /= float64(nSamples)
			var v float64
			for i := 0; i < nSamples; i++ {
				d := X[i][j] - mean
				v += d * d
			}
			varSum += v / float64(nSamples)
		}
		effectiveTol = (varSum / float64(nFeatures)) * tol
	}

	for iter := 0; iter < maxIter; iter++ {
		newLabels := assignLabels(X, centers)
		centersNew, counts := recomputeCenters(X, newLabels, nClusters)
		// re-seed empty clusters to the farthest point
		var shift float64
		if reseedEmptyClusters(X, centersNew, counts) {
			shift = math.Inf(1)
		} else {
			for c := 0; c < nClusters; c++ {
				shift += sqDist(centers[c], centersNew[c])
			}
		}
		centers = centersNew

		converged := false
		if labelsEqual(newLabels, labels) {
			converged = true
		} else if shift <= effectiveTol {
			converged = true
		}
		labels = newLabels
		if converged {
			break
		}
	}

	// final E-step so labels match the returned centers
	labels = assignLabels(X, centers)
	inertia := computeInertia(X, centers, labels)
	return centers, labels, inertia
}

func assignLabels(X [][]float64, centers [][]float64) []float64 {
	labels := make([]float64, len(X))
	for i, row := range X {
		labels[i] = float64(nearestCenter(row, centers))
	}
	return labels
}

func recomputeCenters(X [][]float64, labels []float64, nClusters int) ([][]float64, []int) {
	nFeatures := len(X[0])
	centers := make([][]float64, nClusters)
	counts := make([]int, nClusters)
	for c := 0; c < nClusters; c++ {
		centers[c] = make([]float64, nFeatures)
	}
	for i, row := range X {
		c := int(labels[i])
		counts[c]++
		for j, v := range row {
			centers[c][j] += v
		}
	}
	for c := 0; c < nClusters; c++ {
		if counts[c] > 0 {
			for j := range centers[c] {
				centers[c][j] /= float64(counts[c])
			}
		}
	}
	return centers, counts
}

// reseedEmptyClusters moves every empty cluster's center onto the sample
// farthest from its nearest center, and reports whether any cluster was empty.
// Distances are measured against the non-empty centers only (the zero vectors
// recomputeCenters leaves for empty clusters are not real centers), and each
// reseeded center joins that set so that several empty clusters land on
// distinct points rather than all picking the same one.
func reseedEmptyClusters(X [][]float64, centers [][]float64, counts []int) bool {
	live := make([][]float64, 0, len(centers))
	for c := range centers {
		if counts[c] > 0 {
			live = append(live, centers[c])
		}
	}
	reseeded := false
	for c := range centers {
		if counts[c] != 0 {
			continue
		}
		centers[c], _ = farthestPoint(X, live)
		live = append(live, centers[c])
		reseeded = true
	}
	return reseeded
}

// farthestPoint returns the sample whose distance to its nearest center is
// largest, mirroring sklearn's empty-cluster reassignment.
func farthestPoint(X [][]float64, centers [][]float64) ([]float64, int) {
	bestIdx, bestDist := -1, -1.0
	for i, row := range X {
		d := nearestDistSq(row, centers)
		if d > bestDist {
			bestDist = d
			bestIdx = i
		}
	}
	return append([]float64(nil), X[bestIdx]...), bestIdx
}

func computeInertia(X [][]float64, centers [][]float64, labels []float64) float64 {
	var sum float64
	for i, row := range X {
		sum += sqDist(row, centers[int(labels[i])])
	}
	return sum
}

func labelsEqual(a, b []float64) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func nearestCenter(row []float64, centers [][]float64) int {
	best, bestDist := 0, sqDist(row, centers[0])
	for i := 1; i < len(centers); i++ {
		d := sqDist(row, centers[i])
		if d < bestDist {
			bestDist = d
			best = i
		}
	}
	return best
}

func nearestDistSq(row []float64, centers [][]float64) float64 {
	best := sqDist(row, centers[0])
	for i := 1; i < len(centers); i++ {
		d := sqDist(row, centers[i])
		if d < best {
			best = d
		}
	}
	return best
}

func sqDist(a, b []float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}
