// Package metrics provides scoring functions mirroring scikit-learn's metrics module.
//
// Every function validates its inputs (non-empty, equally long vectors) and wraps
// validation errors from internal/matutil, following the same conventions as the
// linear and preprocessing estimators. Scores are returned as plain float64; callers
// should treat NaN as an indicator of non-finite inputs.
package metrics

import (
	"fmt"
	"math"
	"sort"

	"scikit-go/internal/matutil"
)

// float64Eps mirrors numpy.finfo("float64").eps, used by AdjustedMutualInfo to
// match scikit-learn's denominator/numerator clamping.
const float64Eps = 2.220446049250313e-16

// validateVectors checks that two numeric vectors are non-empty and equally long.
func validateVectors(yTrue, yPred []float64) error {
	if len(yTrue) == 0 || len(yPred) == 0 {
		return fmt.Errorf("metrics: %w", matutil.ErrEmptyInput)
	}
	if len(yTrue) != len(yPred) {
		return fmt.Errorf("metrics: %w: y_true has %d samples, y_pred has %d",
			matutil.ErrLengthMismatch, len(yTrue), len(yPred))
	}
	return nil
}

// checkFiniteLabels rejects NaN/Inf label values. Labels become map keys when
// building contingency matrices, so non-finite values cannot be handled safely.
func checkFiniteLabels(vals []float64, name string) error {
	for i, v := range vals {
		if math.IsNaN(v) {
			return fmt.Errorf("metrics: %w at %s[%d]", matutil.ErrContainsNaN, name, i)
		}
		if math.IsInf(v, 0) {
			return fmt.Errorf("metrics: %w at %s[%d]", matutil.ErrContainsInf, name, i)
		}
	}
	return nil
}

// validateLabels checks that two label vectors are non-empty, equally long, and finite.
func validateLabels(yTrue, yPred []float64) error {
	if err := validateVectors(yTrue, yPred); err != nil {
		return err
	}
	if err := checkFiniteLabels(yTrue, "y_true"); err != nil {
		return err
	}
	return checkFiniteLabels(yPred, "y_pred")
}

// sortedUnique returns the sorted distinct values of vals.
func sortedUnique(vals []float64) []float64 {
	set := make(map[float64]struct{}, len(vals))
	for _, v := range vals {
		set[v] = struct{}{}
	}
	out := make([]float64, 0, len(set))
	for v := range set {
		out = append(out, v)
	}
	sort.Float64s(out)
	return out
}

// contingencyMatrix returns the joint count matrix whose rows are the sorted
// unique labels of labelsRows and whose columns are the sorted unique labels of
// labelsCols, matching scikit-learn's contingency_matrix.
func contingencyMatrix(labelsRows, labelsCols []float64) ([][]int, error) {
	rows := sortedUnique(labelsRows)
	cols := sortedUnique(labelsCols)
	rowIdx := make(map[float64]int, len(rows))
	for i, l := range rows {
		rowIdx[l] = i
	}
	colIdx := make(map[float64]int, len(cols))
	for j, l := range cols {
		colIdx[l] = j
	}
	out := make([][]int, len(rows))
	for i := range out {
		out[i] = make([]int, len(cols))
	}
	for i := range labelsRows {
		out[rowIdx[labelsRows[i]]][colIdx[labelsCols[i]]]++
	}
	return out, nil
}

// entropy computes the natural-log entropy of a label vector, matching
// sklearn.metrics.cluster._supervised._entropy: zero for a single distinct label.
func entropy(labels []float64) float64 {
	counts := make(map[float64]int, len(labels))
	for _, l := range labels {
		counts[l]++
	}
	if len(counts) == 1 {
		return 0.0
	}
	n := len(labels)
	var h float64
	for _, c := range counts {
		p := float64(c) / float64(n)
		h -= p * (math.Log(float64(c)) - math.Log(float64(n)))
	}
	return h
}

// mutualInfo computes mutual information from a contingency matrix, matching
// sklearn.metrics.mutual_info_score: sum over cells with n_ij > 0 of
// (n_ij / n) * log(n_ij * n / (a_i * b_j)).
func mutualInfo(contingency [][]int, n int) float64 {
	rowSums := make([]int, len(contingency))
	colSums := make([]int, len(contingency[0]))
	for i, row := range contingency {
		for j, v := range row {
			rowSums[i] += v
			colSums[j] += v
		}
	}
	var mi float64
	for i, row := range contingency {
		for j, v := range row {
			if v == 0 {
				continue
			}
			// (n_ij / n) * log(n_ij * n / (a_i * b_j)), computed in log space
			// to mirror sklearn's mutual_info_score precision handling.
			mi += (float64(v) / float64(n)) * (math.Log(float64(v)) -
				math.Log(float64(rowSums[i])) -
				math.Log(float64(colSums[j])) +
				math.Log(float64(n)))
		}
	}
	return mi
}

// logGamma is the natural log of |Gamma(x)| for x > 0 (sign is always +1 here).
func logGamma(x float64) float64 {
	l, _ := math.Lgamma(x)
	return l
}

// expectedMutualInfo computes the expected mutual information for two labelings,
// translating sklearn.metrics.cluster._expected_mutual_info_fast.expected_mutual_information
// exactly (the "allen institute" EMI approximation).
func expectedMutualInfo(contingency [][]int, nSamples int) float64 {
	nRows := len(contingency)
	nCols := len(contingency[0])

	a := make([]int, nRows) // row sums
	b := make([]int, nCols) // column sums
	maxRow, maxCol := 0, 0
	for i, row := range contingency {
		for j, v := range row {
			a[i] += v
			b[j] += v
		}
		if a[i] > maxRow {
			maxRow = a[i]
		}
	}
	for j := range b {
		if b[j] > maxCol {
			maxCol = b[j]
		}
	}

	// Any labelling with zero entropy implies EMI = 0.
	if nRows == 1 || nCols == 1 {
		return 0.0
	}

	// nijs[k] for k in 0..max(a or b); nijs[0] is set to 1 and never used.
	maxVal := maxRow
	if maxCol > maxVal {
		maxVal = maxCol
	}
	nijs := make([]float64, maxVal+1)
	for k := 1; k <= maxVal; k++ {
		nijs[k] = float64(k)
	}
	nijs[0] = 1

	term1 := make([]float64, maxVal+1)
	logNnij := make([]float64, maxVal+1)
	glnNnij := make([]float64, maxVal+1)
	for k := 1; k <= maxVal; k++ {
		term1[k] = nijs[k] / float64(nSamples)
		logNnij[k] = math.Log(float64(nSamples)) + math.Log(nijs[k])
		glnNnij[k] = logGamma(nijs[k]+1) + logGamma(float64(nSamples)+1)
	}

	logA := make([]float64, nRows)
	glnA := make([]float64, nRows)
	glnNA := make([]float64, nRows)
	for i := range a {
		logA[i] = math.Log(float64(a[i]))
		glnA[i] = logGamma(float64(a[i]) + 1)
		glnNA[i] = logGamma(float64(nSamples-a[i]) + 1)
	}
	logB := make([]float64, nCols)
	glnB := make([]float64, nCols)
	glnNB := make([]float64, nCols)
	for j := range b {
		logB[j] = math.Log(float64(b[j]))
		glnB[j] = logGamma(float64(b[j]) + 1)
		glnNB[j] = logGamma(float64(nSamples-b[j]) + 1)
	}

	var emi float64
	for i := 0; i < nRows; i++ {
		for j := 0; j < nCols; j++ {
			start := max(1, a[i]-nSamples+b[j])
			end := min(a[i], b[j]) + 1
			for nij := start; nij < end; nij++ {
				term2 := logNnij[nij] - logA[i] - logB[j]
				gln := glnA[i] + glnB[j] + glnNA[i] + glnNB[j] -
					glnNnij[nij] -
					logGamma(float64(a[i]-nij)+1) -
					logGamma(float64(b[j]-nij)+1) -
					logGamma(float64(nSamples-a[i]-b[j]+nij)+1)
				emi += term1[nij] * term2 * math.Exp(gln)
			}
		}
	}
	return emi
}

// euclidean returns the Euclidean distance between two equally long vectors.
func euclidean(a, b []float64) float64 {
	var sum float64
	for j := range a {
		d := a[j] - b[j]
		sum += d * d
	}
	return math.Sqrt(sum)
}
