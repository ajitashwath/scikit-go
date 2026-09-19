package metrics

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Differential tests: sklearn's answer (or error) on hundreds of small random and
// deliberately degenerate inputs. They are what catches behavior that hand-picked
// fixtures miss, such as one-class label sets or a constant clustering.

// sk is one sklearn outcome: a value, NaN (JSON null), or an error.
type sk struct {
	isErr bool
	val   float64
	list  []float64
	grid  [][]float64
}

func (s *sk) UnmarshalJSON(b []byte) error {
	var e struct {
		Error bool `json:"error"`
	}
	if json.Unmarshal(b, &e) == nil && e.Error {
		s.isErr = true
		return nil
	}
	if string(b) == "null" {
		s.val = math.NaN()
		return nil
	}
	if err := json.Unmarshal(b, &s.val); err == nil {
		return nil
	}
	if err := json.Unmarshal(b, &s.list); err == nil {
		return nil
	}
	return json.Unmarshal(b, &s.grid)
}

type edgeFile struct {
	Labels []struct {
		YTrue        []float64 `json:"y_true"`
		YPred        []float64 `json:"y_pred"`
		Accuracy     sk        `json:"accuracy"`
		Confusion    sk        `json:"confusion"`
		Precision    sk        `json:"precision"`
		Recall       sk        `json:"recall"`
		F1           sk        `json:"f1"`
		Macro        sk        `json:"macro"`
		Weighted     sk        `json:"weighted"`
		ARI          sk        `json:"ari"`
		Homogeneity  sk        `json:"homogeneity"`
		Completeness sk        `json:"completeness"`
		VMeasure     sk        `json:"v_measure"`
		AMI          sk        `json:"ami"`
	} `json:"labels"`
	Regression []struct {
		YTrue             []float64 `json:"y_true"`
		YPred             []float64 `json:"y_pred"`
		MSE               sk        `json:"mse"`
		RMSE              sk        `json:"rmse"`
		MAE               sk        `json:"mae"`
		R2                sk        `json:"r2"`
		ExplainedVariance sk        `json:"explained_variance"`
		MaxError          sk        `json:"max_error"`
	} `json:"regression"`
	Silhouette []struct {
		X          [][]float64 `json:"X"`
		Labels     []float64   `json:"labels"`
		Silhouette sk          `json:"silhouette"`
	} `json:"silhouette"`
}

// mismatches collects disagreements per metric so one run shows every kind of
// failure rather than stopping at the first.
type mismatches map[string][]string

func (m mismatches) add(metric, detail string) { m[metric] = append(m[metric], detail) }

func (m mismatches) report(t *testing.T) {
	t.Helper()
	for metric, list := range m {
		shown := list
		if len(shown) > 4 {
			shown = shown[:4]
		}
		t.Errorf("%s: %d mismatches with sklearn, e.g.:\n\t%v", metric, len(list), shown)
	}
}

// scalar compares a Go (value, error) result with sklearn's outcome.
func (m mismatches) scalar(metric, input string, got float64, err error, want sk) {
	switch {
	case want.isErr && err == nil:
		m.add(metric, fmt.Sprintf("%s: sklearn raises, got %v", input, got))
	case want.isErr:
		// both fail
	case err != nil:
		m.add(metric, fmt.Sprintf("%s: got error %q, sklearn returns %v", input, err, want.val))
	case math.IsNaN(want.val) && !math.IsNaN(got):
		m.add(metric, fmt.Sprintf("%s: got %v, sklearn NaN", input, got))
	case !math.IsNaN(want.val) && math.Abs(got-want.val) > 1e-9*(1+math.Abs(want.val)):
		m.add(metric, fmt.Sprintf("%s: got %v, sklearn %v", input, got, want.val))
	}
}

func loadEdge(t *testing.T) *edgeFile {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "metrics_edge_fixtures.json"))
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var f edgeFile
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatalf("parse fixtures: %v", err)
	}
	return &f
}

func TestLabelMetrics_AgainstSklearnEdgeCases(t *testing.T) {
	f := loadEdge(t)
	m := mismatches{}
	for _, c := range f.Labels {
		in := fmt.Sprintf("true=%v pred=%v", c.YTrue, c.YPred)

		// sklearn rejects non-integer labels as a continuous target for every
		// classification metric. This library deliberately accepts any float as a
		// class label, so those inputs only exercise the clustering metrics.
		var v float64
		var err error
		if !c.Accuracy.isErr {
			v, err = AccuracyScore(c.YTrue, c.YPred)
			m.scalar("accuracy", in, v, err, c.Accuracy)
			checkClassification(m, in, c.YTrue, c.YPred, c.Confusion, c.Precision, c.Recall, c.F1, c.Macro, c.Weighted)
		}

		v, err = AdjustedRandIndex(c.YTrue, c.YPred)
		m.scalar("ari", in, v, err, c.ARI)
		v, err = HomogeneityScore(c.YTrue, c.YPred)
		m.scalar("homogeneity", in, v, err, c.Homogeneity)
		v, err = CompletenessScore(c.YTrue, c.YPred)
		m.scalar("completeness", in, v, err, c.Completeness)
		v, err = VMeasure(c.YTrue, c.YPred)
		m.scalar("v_measure", in, v, err, c.VMeasure)
		v, err = AdjustedMutualInfo(c.YTrue, c.YPred)
		m.scalar("ami", in, v, err, c.AMI)
	}
	m.report(t)
}

func checkClassification(m mismatches, in string, yTrue, yPred []float64, confusion, precision, recall, f1, macro, weighted sk) {
	cm, err := ConfusionMatrix(yTrue, yPred)
	if err != nil {
		m.add("confusion", fmt.Sprintf("%s: error %v", in, err))
	} else if confusion.isErr {
		m.add("confusion", in+": sklearn raises")
	} else if len(cm) != len(confusion.grid) {
		m.add("confusion", fmt.Sprintf("%s: size %d, sklearn %d", in, len(cm), len(confusion.grid)))
	} else {
		for i := range cm {
			for j := range cm[i] {
				if float64(cm[i][j]) != confusion.grid[i][j] {
					m.add("confusion", fmt.Sprintf("%s: cell (%d,%d) %d, sklearn %v", in, i, j, cm[i][j], confusion.grid[i][j]))
				}
			}
		}
	}

	v, err := PrecisionScore(yTrue, yPred)
	m.scalar("precision", in, v, err, precision)
	v, err = RecallScore(yTrue, yPred)
	m.scalar("recall", in, v, err, recall)
	v, err = F1Score(yTrue, yPred)
	m.scalar("f1", in, v, err, f1)

	rep, err := ClassificationReport(yTrue, yPred)
	if err != nil {
		m.add("report", fmt.Sprintf("%s: error %v", in, err))
	} else {
		for i, got := range []float64{rep.MacroPrecision, rep.MacroRecall, rep.MacroF1} {
			m.scalar("macro", in, got, nil, sk{val: macro.list[i]})
		}
		for i, got := range []float64{rep.WeightedPrecision, rep.WeightedRecall, rep.WeightedF1} {
			m.scalar("weighted", in, got, nil, sk{val: weighted.list[i]})
		}
	}

}

func TestRegressionMetrics_AgainstSklearnEdgeCases(t *testing.T) {
	f := loadEdge(t)
	m := mismatches{}
	for _, c := range f.Regression {
		in := fmt.Sprintf("true=%v pred=%v", c.YTrue, c.YPred)
		v, err := MeanSquaredError(c.YTrue, c.YPred)
		m.scalar("mse", in, v, err, c.MSE)
		v, err = RootMeanSquaredError(c.YTrue, c.YPred)
		m.scalar("rmse", in, v, err, c.RMSE)
		v, err = MeanAbsoluteError(c.YTrue, c.YPred)
		m.scalar("mae", in, v, err, c.MAE)
		v, err = R2Score(c.YTrue, c.YPred)
		m.scalar("r2", in, v, err, c.R2)
		v, err = ExplainedVarianceScore(c.YTrue, c.YPred)
		m.scalar("explained_variance", in, v, err, c.ExplainedVariance)
		v, err = MaxError(c.YTrue, c.YPred)
		m.scalar("max_error", in, v, err, c.MaxError)
	}
	m.report(t)
}

func TestSilhouette_AgainstSklearnEdgeCases(t *testing.T) {
	f := loadEdge(t)
	m := mismatches{}
	for _, c := range f.Silhouette {
		v, err := SilhouetteScore(c.X, c.Labels)
		m.scalar("silhouette", fmt.Sprintf("n=%d labels=%v", len(c.X), c.Labels), v, err, c.Silhouette)
	}
	m.report(t)
}
