package metrics

import (
	"errors"
	"fmt"
	"math"

	"github.com/ajitashwath/scikit-go/internal/matutil"
)

// AdjustedRandIndex returns the Rand index adjusted for chance, matching
// sklearn.metrics.adjusted_rand_score. It ranges from -0.5 to 1.0, with 1.0 for
// identical labelings and ~0.0 for random labelings.
func AdjustedRandIndex(labelsTrue, labelsPred []float64) (float64, error) {
	if err := validateLabels(labelsTrue, labelsPred); err != nil {
		return 0, err
	}
	n := len(labelsTrue)
	cont, err := contingencyMatrix(labelsTrue, labelsPred)
	if err != nil {
		return 0, err
	}

	var sumSquares, sumA2, sumB2 int64
	rowSums := make([]int64, len(cont))
	colSums := make([]int64, len(cont[0]))
	for i, row := range cont {
		for j, v := range row {
			v64 := int64(v)
			sumSquares += v64 * v64
			rowSums[i] += v64
			colSums[j] += v64
		}
	}
	for _, s := range rowSums {
		sumA2 += s * s
	}
	for _, s := range colSums {
		sumB2 += s * s
	}

	// sklearn's pair_confusion_matrix returns ordered-pair counts (2x the
	// usual unordered counts); reproduce it exactly, including its formula.
	tp := sumSquares - int64(n)
	fp := sumB2 - sumSquares
	fn := sumA2 - sumSquares
	tn := int64(n)*int64(n) - fp - fn - sumSquares

	// Special case: full agreement.
	if fn == 0 && fp == 0 {
		return 1.0, nil
	}

	numerator := 2.0 * (float64(tp)*float64(tn) - float64(fn)*float64(fp))
	denominator := float64(tp+fn)*float64(fn+tn) + float64(tp+fp)*float64(fp+tn)
	return numerator / denominator, nil
}

// HomogeneityScore returns the homogeneity of a clustering against a ground truth,
// matching sklearn.metrics.homogeneity_score: MI / H(C), or 1.0 when the true
// labeling has zero entropy.
func HomogeneityScore(labelsTrue, labelsPred []float64) (float64, error) {
	if err := validateLabels(labelsTrue, labelsPred); err != nil {
		return 0, err
	}
	entropyC := entropy(labelsTrue)
	cont, err := contingencyMatrix(labelsTrue, labelsPred)
	if err != nil {
		return 0, err
	}
	mi := mutualInfo(cont, len(labelsTrue))
	if entropyC == 0 {
		return 1.0, nil
	}
	return mi / entropyC, nil
}

// CompletenessScore returns the completeness of a clustering against a ground truth,
// matching sklearn.metrics.completeness_score: MI / H(K), or 1.0 when the predicted
// labeling has zero entropy.
func CompletenessScore(labelsTrue, labelsPred []float64) (float64, error) {
	if err := validateLabels(labelsTrue, labelsPred); err != nil {
		return 0, err
	}
	entropyK := entropy(labelsPred)
	cont, err := contingencyMatrix(labelsTrue, labelsPred)
	if err != nil {
		return 0, err
	}
	mi := mutualInfo(cont, len(labelsTrue))
	if entropyK == 0 {
		return 1.0, nil
	}
	return mi / entropyK, nil
}

// VMeasure returns the V-measure (beta = 1), matching sklearn.metrics.v_measure_score:
// the harmonic mean of HomogeneityScore and CompletenessScore.
func VMeasure(labelsTrue, labelsPred []float64) (float64, error) {
	h, err := HomogeneityScore(labelsTrue, labelsPred)
	if err != nil {
		return 0, err
	}
	c, err := CompletenessScore(labelsTrue, labelsPred)
	if err != nil {
		return 0, err
	}
	if h+c == 0 {
		return 0.0, nil
	}
	return 2 * h * c / (h + c), nil
}

// AdjustedMutualInfo returns the mutual information between two clusterings
// adjusted for chance, matching sklearn.metrics.adjusted_mutual_info_score with
// its default average_method="arithmetic" and the "allen institute" expected-MI
// approximation.
func AdjustedMutualInfo(labelsTrue, labelsPred []float64) (float64, error) {
	if err := validateLabels(labelsTrue, labelsPred); err != nil {
		return 0, err
	}
	n := len(labelsTrue)
	classes := sortedUnique(labelsTrue)
	clusters := sortedUnique(labelsPred)

	// Special limit cases, mirroring sklearn: no split on either side is a perfect
	// match; a split on exactly one side has zero AMI.
	if len(classes) == len(clusters) && (len(classes) == 1 || len(classes) == 0) {
		return 1.0, nil
	}
	if len(classes) == 1 || len(clusters) == 1 {
		return 0.0, nil
	}

	cont, err := contingencyMatrix(labelsTrue, labelsPred)
	if err != nil {
		return 0, err
	}
	mi := mutualInfo(cont, n)
	emi := expectedMutualInfo(cont, n)
	hTrue := entropy(labelsTrue)
	hPred := entropy(labelsPred)
	normalizer := (hTrue + hPred) / 2

	denominator := normalizer - emi
	if denominator < 0 {
		denominator = math.Min(denominator, -float64Eps)
	} else {
		denominator = math.Max(denominator, float64Eps)
	}
	numerator := mi - emi
	if numerator < 0 {
		numerator = math.Min(numerator, -float64Eps)
	} else {
		numerator = math.Max(numerator, float64Eps)
	}
	return numerator / denominator, nil
}

// ErrInvalidClusterCount is returned by SilhouetteScore when the labeling does not
// have between 2 and n_samples-1 distinct clusters, where the score is undefined.
var ErrInvalidClusterCount = errors.New("metrics: invalid number of clusters for the silhouette score")

// SilhouetteScore returns the mean silhouette coefficient over all samples,
// matching sklearn.metrics.silhouette_score with its default Euclidean metric.
// Samples in singleton clusters score 0, as in scikit-learn. Like scikit-learn it
// requires 2 <= n_labels <= n_samples-1 and otherwise returns ErrInvalidClusterCount.
func SilhouetteScore(X [][]float64, labels []float64) (float64, error) {
	if err := matutil.ValidateXy(X, labels); err != nil {
		return 0, fmt.Errorf("metrics: %w", err)
	}
	n := len(X)

	clusterMembers := make(map[float64][]int)
	for i, l := range labels {
		clusterMembers[l] = append(clusterMembers[l], i)
	}
	if k := len(clusterMembers); k < 2 || k > n-1 {
		return 0, fmt.Errorf("%w: got %d labels for %d samples, valid values are 2 to %d",
			ErrInvalidClusterCount, k, n, n-1)
	}

	dist := make([][]float64, n)
	for i := range dist {
		dist[i] = make([]float64, n)
		for j := range dist[i] {
			dist[i][j] = euclidean(X[i], X[j])
		}
	}

	var total float64
	for i := 0; i < n; i++ {
		own := labels[i]
		if len(clusterMembers[own]) == 1 {
			continue // sklearn scores singleton clusters 0
		}
		a := meanIntraDist(i, own, dist, clusterMembers)
		b := minInterDist(i, own, dist, clusterMembers)
		m := math.Max(a, b)
		if m > 0 {
			total += (b - a) / m
		}
	}
	return total / float64(n), nil
}

// meanIntraDist is the mean distance from sample i to the other members of its cluster.
func meanIntraDist(i int, label float64, dist [][]float64, members map[float64][]int) float64 {
	var sum float64
	var count int
	for _, j := range members[label] {
		if j == i {
			continue
		}
		sum += dist[i][j]
		count++
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

// minInterDist is the minimum over other clusters of the mean distance from sample i
// to the members of that cluster.
func minInterDist(i int, label float64, dist [][]float64, members map[float64][]int) float64 {
	minD := math.Inf(1)
	for l, idx := range members {
		if l == label {
			continue
		}
		var sum float64
		for _, j := range idx {
			sum += dist[i][j]
		}
		if m := sum / float64(len(idx)); m < minD {
			minD = m
		}
	}
	return minD
}
