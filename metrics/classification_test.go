package metrics

import (
	"reflect"
	"testing"
)

func TestBinaryClassification_AgainstSklearn(t *testing.T) {
	fx := exampleFixture(t, "binary_classification")
	yTrue := fixtureSlice(t, fx, "y_true")
	yPred := fixtureSlice(t, fx, "y_pred")

	cm, err := ConfusionMatrix(yTrue, yPred)
	if err != nil {
		t.Fatalf("ConfusionMatrix: %v", err)
	}
	assertMatrixClose(t, "confusion_matrix", cm, fixtureIntMatrix(t, fx, "confusion_matrix"))

	acc, err := AccuracyScore(yTrue, yPred)
	if err != nil {
		t.Fatalf("AccuracyScore: %v", err)
	}
	assertClose(t, "accuracy", acc, fixtureFloat(t, fx, "accuracy"))

	prec, err := PrecisionScore(yTrue, yPred)
	if err != nil {
		t.Fatalf("PrecisionScore: %v", err)
	}
	assertClose(t, "precision", prec, fixtureFloat(t, fx, "precision"))

	rec, err := RecallScore(yTrue, yPred)
	if err != nil {
		t.Fatalf("RecallScore: %v", err)
	}
	assertClose(t, "recall", rec, fixtureFloat(t, fx, "recall"))

	f1, err := F1Score(yTrue, yPred)
	if err != nil {
		t.Fatalf("F1Score: %v", err)
	}
	assertClose(t, "f1", f1, fixtureFloat(t, fx, "f1"))

	// pos_label=0 variants.
	prec0, err := PrecisionScore(yTrue, yPred, 0)
	if err != nil {
		t.Fatalf("PrecisionScore(0): %v", err)
	}
	assertClose(t, "precision pos0", prec0, fixtureFloat(t, fx, "precision_pos0"))

	rec0, err := RecallScore(yTrue, yPred, 0)
	if err != nil {
		t.Fatalf("RecallScore(0): %v", err)
	}
	assertClose(t, "recall pos0", rec0, fixtureFloat(t, fx, "recall_pos0"))

	f10, err := F1Score(yTrue, yPred, 0)
	if err != nil {
		t.Fatalf("F1Score(0): %v", err)
	}
	assertClose(t, "f1 pos0", f10, fixtureFloat(t, fx, "f1_pos0"))
}

func TestMulticlassClassification_AgainstSklearn(t *testing.T) {
	fx := exampleFixture(t, "multiclass_classification")
	yTrue := fixtureSlice(t, fx, "y_true")
	yPred := fixtureSlice(t, fx, "y_pred")

	cm, err := ConfusionMatrix(yTrue, yPred)
	if err != nil {
		t.Fatalf("ConfusionMatrix: %v", err)
	}
	assertMatrixClose(t, "confusion_matrix", cm, fixtureIntMatrix(t, fx, "confusion_matrix"))

	acc, err := AccuracyScore(yTrue, yPred)
	if err != nil {
		t.Fatalf("AccuracyScore: %v", err)
	}
	assertClose(t, "accuracy", acc, fixtureFloat(t, fx, "accuracy"))

	report, err := ClassificationReport(yTrue, yPred)
	if err != nil {
		t.Fatalf("ClassificationReport: %v", err)
	}
	assertClose(t, "macro_precision", report.MacroPrecision, fixtureFloat(t, fx, "macro_precision"))
	assertClose(t, "macro_recall", report.MacroRecall, fixtureFloat(t, fx, "macro_recall"))
	assertClose(t, "macro_f1", report.MacroF1, fixtureFloat(t, fx, "macro_f1"))
	assertClose(t, "weighted_precision", report.WeightedPrecision, fixtureFloat(t, fx, "weighted_precision"))
	assertClose(t, "weighted_recall", report.WeightedRecall, fixtureFloat(t, fx, "weighted_recall"))
	assertClose(t, "weighted_f1", report.WeightedF1, fixtureFloat(t, fx, "weighted_f1"))
	assertClose(t, "accuracy in report", report.Accuracy, fixtureFloat(t, fx, "accuracy"))
}

func TestMulticlassClassification_Classes(t *testing.T) {
	fx := exampleFixture(t, "multiclass_classification")
	yTrue := fixtureSlice(t, fx, "y_true")
	yPred := fixtureSlice(t, fx, "y_pred")
	report, err := ClassificationReport(yTrue, yPred)
	if err != nil {
		t.Fatalf("ClassificationReport: %v", err)
	}

	raw, ok := fx["classes"].([]interface{})
	if !ok {
		t.Fatalf("fixture key %q is not an array", "classes")
	}
	if len(report.Classes) != len(raw) {
		t.Fatalf("class count mismatch: got %d, want %d", len(report.Classes), len(raw))
	}
	for i, c := range raw {
		cm := c.(map[string]interface{})
		want := ClassMetrics{
			Label:     cm["label"].(float64),
			Precision: cm["precision"].(float64),
			Recall:    cm["recall"].(float64),
			F1:        cm["f1"].(float64),
			Support:   int(cm["support"].(float64)),
		}
		if !reflect.DeepEqual(report.Classes[i], want) {
			t.Errorf("class %d: got %+v, want %+v", i, report.Classes[i], want)
		}
	}
}

func TestBinary_MulticlassRejected(t *testing.T) {
	yTrue := []float64{0, 1, 2}
	yPred := []float64{0, 1, 2}
	if _, err := PrecisionScore(yTrue, yPred); err == nil {
		t.Error("expected error for multiclass input with binary scorer, got nil")
	}
	if _, err := RecallScore(yTrue, yPred); err == nil {
		t.Error("expected error for multiclass input with binary scorer, got nil")
	}
	if _, err := F1Score(yTrue, yPred); err == nil {
		t.Error("expected error for multiclass input with binary scorer, got nil")
	}
}

func TestBinary_InvalidPosLabel(t *testing.T) {
	yTrue := []float64{0, 0, 1, 1}
	yPred := []float64{0, 1, 0, 1}
	if _, err := PrecisionScore(yTrue, yPred, 7); err == nil {
		t.Error("expected error for invalid pos_label, got nil")
	}
}

func TestBinary_UndefinedDivision(t *testing.T) {
	// All-negative predictions: precision is undefined and must score 0.0,
	// matching sklearn's zero_division=0 behavior.
	yTrue := []float64{1, 1, 1}
	yPred := []float64{0, 0, 0}
	prec, err := PrecisionScore(yTrue, yPred)
	if err != nil {
		t.Fatalf("PrecisionScore: %v", err)
	}
	assertClose(t, "precision undefined", prec, 0.0)
	rec, err := RecallScore(yTrue, yPred)
	if err != nil {
		t.Fatalf("RecallScore: %v", err)
	}
	assertClose(t, "recall undefined", rec, 0.0)
	f1, err := F1Score(yTrue, yPred)
	if err != nil {
		t.Fatalf("F1Score: %v", err)
	}
	assertClose(t, "f1 undefined", f1, 0.0)
}

func TestConfusionMatrix_SingleClass(t *testing.T) {
	yTrue := []float64{1, 1, 1}
	yPred := []float64{1, 1, 1}
	cm, err := ConfusionMatrix(yTrue, yPred)
	if err != nil {
		t.Fatalf("ConfusionMatrix: %v", err)
	}
	want := [][]int{{3}}
	if !reflect.DeepEqual(cm, want) {
		t.Errorf("got %v, want %v", cm, want)
	}
}

func TestConfusionMatrix_PredOnlyClass(t *testing.T) {
	// A class seen only in predictions must appear in the matrix (union of labels).
	yTrue := []float64{0, 0, 1, 1}
	yPred := []float64{0, 0, 2, 2}
	cm, err := ConfusionMatrix(yTrue, yPred)
	if err != nil {
		t.Fatalf("ConfusionMatrix: %v", err)
	}
	want := [][]int{
		{2, 0, 0},
		{0, 0, 2},
		{0, 0, 0},
	}
	if !reflect.DeepEqual(cm, want) {
		t.Errorf("got %v, want %v", cm, want)
	}
}
