package metrics

import "fmt"

// ClassMetrics reports the per-class classification scores for a single label,
// mirroring one row of sklearn.metrics.classification_report.
type ClassMetrics struct {
	Label     float64
	Precision float64
	Recall    float64
	F1        float64
	Support   int
}

// Report is the result of ClassificationReport, mirroring
// sklearn.metrics.classification_report: per-class rows plus macro/weighted
// averages and overall accuracy.
type Report struct {
	Classes           []ClassMetrics
	Accuracy          float64
	MacroPrecision    float64
	MacroRecall       float64
	MacroF1           float64
	WeightedPrecision float64
	WeightedRecall    float64
	WeightedF1        float64
}

// AccuracyScore returns the fraction of correctly classified samples, matching
// sklearn.metrics.accuracy_score.
func AccuracyScore(yTrue, yPred []float64) (float64, error) {
	if err := validateLabels(yTrue, yPred); err != nil {
		return 0, err
	}
	correct := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(yTrue)), nil
}

// ConfusionMatrix returns the square confusion matrix whose rows and columns are
// indexed by the sorted distinct labels of yTrue ∪ yPred, matching
// sklearn.metrics.confusion_matrix.
func ConfusionMatrix(yTrue, yPred []float64) ([][]int, error) {
	if err := validateLabels(yTrue, yPred); err != nil {
		return nil, err
	}
	union := make([]float64, 0, len(yTrue)+len(yPred))
	union = append(union, yTrue...)
	union = append(union, yPred...)
	labels := sortedUnique(union)
	idx := make(map[float64]int, len(labels))
	for i, l := range labels {
		idx[l] = i
	}
	out := make([][]int, len(labels))
	for i := range out {
		out[i] = make([]int, len(labels))
	}
	for i := range yTrue {
		out[idx[yTrue[i]]][idx[yPred[i]]]++
	}
	return out, nil
}

// binaryConfusionCounts counts true/false positives and negatives for the positive
// class posLabel over an at-most-binary labeling. It returns an error when more
// than two distinct labels are present (mirroring sklearn's average="binary" rule)
// or when posLabel is not one of the observed labels.
func binaryConfusionCounts(yTrue, yPred []float64, posLabel float64) (tp, fp, fn, tn int, err error) {
	if err = validateLabels(yTrue, yPred); err != nil {
		return
	}
	union := make([]float64, 0, len(yTrue)+len(yPred))
	union = append(union, yTrue...)
	union = append(union, yPred...)
	labels := sortedUnique(union)
	if len(labels) > 2 {
		return 0, 0, 0, 0, fmt.Errorf("metrics: found %d classes but %s is only defined for binary targets",
			len(labels), "PrecisionScore/RecallScore/F1Score")
	}
	found := false
	for _, l := range labels {
		if l == posLabel {
			found = true
			break
		}
	}
	if !found {
		return 0, 0, 0, 0, fmt.Errorf("metrics: pos_label %v is not a valid label", posLabel)
	}
	for i := range yTrue {
		posTrue := yTrue[i] == posLabel
		posPred := yPred[i] == posLabel
		switch {
		case posTrue && posPred:
			tp++
		case !posTrue && posPred:
			fp++
		case posTrue && !posPred:
			fn++
		default:
			tn++
		}
	}
	return
}

// PrecisionScore returns the binary precision TP/(TP+FP) for the positive class
// posLabel (default 1.0), matching sklearn.metrics.precision_score with its
// default average="binary". Undefined cases score 0.0, matching sklearn's
// zero_division=0 behavior.
func PrecisionScore(yTrue, yPred []float64, posLabel ...float64) (float64, error) {
	pos := 1.0
	if len(posLabel) > 0 {
		pos = posLabel[0]
	}
	tp, fp, _, _, err := binaryConfusionCounts(yTrue, yPred, pos)
	if err != nil {
		return 0, err
	}
	if tp+fp == 0 {
		return 0.0, nil
	}
	return float64(tp) / float64(tp+fp), nil
}

// RecallScore returns the binary recall TP/(TP+FN) for the positive class
// posLabel (default 1.0), matching sklearn.metrics.recall_score with its default
// average="binary". Undefined cases score 0.0.
func RecallScore(yTrue, yPred []float64, posLabel ...float64) (float64, error) {
	pos := 1.0
	if len(posLabel) > 0 {
		pos = posLabel[0]
	}
	tp, _, fn, _, err := binaryConfusionCounts(yTrue, yPred, pos)
	if err != nil {
		return 0, err
	}
	if tp+fn == 0 {
		return 0.0, nil
	}
	return float64(tp) / float64(tp+fn), nil
}

// F1Score returns the binary F1 score 2*TP/(2*TP+FP+FN) for the positive class
// posLabel (default 1.0), matching sklearn.metrics.f1_score with its default
// average="binary". Undefined cases score 0.0.
func F1Score(yTrue, yPred []float64, posLabel ...float64) (float64, error) {
	pos := 1.0
	if len(posLabel) > 0 {
		pos = posLabel[0]
	}
	tp, fp, fn, _, err := binaryConfusionCounts(yTrue, yPred, pos)
	if err != nil {
		return 0, err
	}
	if 2*tp+fp+fn == 0 {
		return 0.0, nil
	}
	return 2 * float64(tp) / float64(2*tp+fp+fn), nil
}

// ClassificationReport computes per-class precision, recall, f1, and support plus
// macro/weighted averages and overall accuracy, mirroring
// sklearn.metrics.classification_report.
func ClassificationReport(yTrue, yPred []float64) (*Report, error) {
	if err := validateLabels(yTrue, yPred); err != nil {
		return nil, err
	}
	cm, err := ConfusionMatrix(yTrue, yPred)
	if err != nil {
		return nil, err
	}
	union := make([]float64, 0, len(yTrue)+len(yPred))
	union = append(union, yTrue...)
	union = append(union, yPred...)
	labels := sortedUnique(union)

	rowSums := make([]int, len(labels))
	colSums := make([]int, len(labels))
	for i := range labels {
		for j := range labels {
			rowSums[i] += cm[i][j]
			colSums[j] += cm[i][j]
		}
	}

	report := &Report{
		Classes:  make([]ClassMetrics, 0, len(labels)),
		Accuracy: 0,
	}
	var totalSupport int
	for i, l := range labels {
		tp := cm[i][i]
		fp := colSums[i] - tp
		fn := rowSums[i] - tp
		support := rowSums[i]
		var precision, recall, f1 float64
		if tp+fp > 0 {
			precision = float64(tp) / float64(tp+fp)
		}
		if tp+fn > 0 {
			recall = float64(tp) / float64(tp+fn)
		}
		if 2*tp+fp+fn > 0 {
			f1 = 2 * float64(tp) / float64(2*tp+fp+fn)
		}
		report.Classes = append(report.Classes, ClassMetrics{
			Label:     l,
			Precision: precision,
			Recall:    recall,
			F1:        f1,
			Support:   support,
		})
		report.MacroPrecision += precision
		report.MacroRecall += recall
		report.MacroF1 += f1
		report.WeightedPrecision += precision * float64(support)
		report.WeightedRecall += recall * float64(support)
		report.WeightedF1 += f1 * float64(support)
		totalSupport += support
	}

	if len(labels) > 0 {
		report.MacroPrecision /= float64(len(labels))
		report.MacroRecall /= float64(len(labels))
		report.MacroF1 /= float64(len(labels))
	}
	if totalSupport > 0 {
		report.WeightedPrecision /= float64(totalSupport)
		report.WeightedRecall /= float64(totalSupport)
		report.WeightedF1 /= float64(totalSupport)
	}

	correct := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			correct++
		}
	}
	report.Accuracy = float64(correct) / float64(len(yTrue))
	return report, nil
}
