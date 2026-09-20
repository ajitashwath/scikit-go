package linear

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ajitashwath/scikit-go/internal/matutil"
)

type regressionFixture struct {
	Params struct {
		Alpha        float64 `json:"alpha"`
		L1Ratio      float64 `json:"l1_ratio"`
		FitIntercept bool    `json:"fit_intercept"`
		MaxIter      int     `json:"max_iter"`
		Tol          float64 `json:"tol"`
	} `json:"params"`
	X           [][]float64 `json:"X"`
	Y           []float64   `json:"y"`
	XTest       [][]float64 `json:"X_test"`
	YTest       []float64   `json:"y_test"`
	Coef        []float64   `json:"coef"`
	Intercept   float64     `json:"intercept"`
	PredictTest []float64   `json:"predict_test"`
	ScoreTest   float64     `json:"score_test"`
	NIter       *int        `json:"n_iter"`
}

type logisticFixture struct {
	Params struct {
		C            float64 `json:"C"`
		FitIntercept bool    `json:"fit_intercept"`
	} `json:"params"`
	X            [][]float64 `json:"X"`
	Y            []float64   `json:"y"`
	XTest        [][]float64 `json:"X_test"`
	Classes      []float64   `json:"classes"`
	Coef         [][]float64 `json:"coef"`
	Intercept    []float64   `json:"intercept"`
	PredictTest  []float64   `json:"predict_test"`
	ProbaTest    [][]float64 `json:"proba_test"`
	DecisionTest []any       `json:"decision_test"` // a vector for binary problems, a matrix otherwise
	ScoreTrain   float64     `json:"score_train"`
}

type linearModelsFixtures struct {
	Ridge      map[string]regressionFixture `json:"ridge"`
	Lasso      map[string]regressionFixture `json:"lasso"`
	ElasticNet map[string]regressionFixture `json:"elastic_net"`
	Logistic   map[string]logisticFixture   `json:"logistic"`
}

func loadLinearModelsFixtures(t testing.TB) linearModelsFixtures {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "linear_models_fixtures.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f linearModelsFixtures
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func maxAbsDiff(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var d float64
	for i := range a {
		d = math.Max(d, math.Abs(a[i]-b[i]))
	}
	return d
}

// regressor is what the three regression estimators share.
type regressor interface {
	Fit(X [][]float64, y []float64) error
	Predict(X [][]float64) ([]float64, error)
	Score(X [][]float64, y []float64) (float64, error)
}

func checkRegression(t *testing.T, name string, m regressor, coef []float64, intercept float64, fx regressionFixture, tol float64) {
	t.Helper()
	if err := m.Fit(fx.X, fx.Y); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if d := maxAbsDiff(coef, fx.Coef); d > tol {
		t.Errorf("%s: coef differs from sklearn by %.3e (tolerance %.0e)\n got  %v\n want %v", name, d, tol, coef, fx.Coef)
	}
	if d := math.Abs(intercept - fx.Intercept); d > tol {
		t.Errorf("%s: intercept %.10f, want %.10f (diff %.3e)", name, intercept, fx.Intercept, d)
	}
	pred, err := m.Predict(fx.XTest)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if d := maxAbsDiff(pred, fx.PredictTest); d > 10*tol {
		t.Errorf("%s: predictions differ from sklearn by %.3e", name, d)
	}
	score, err := m.Score(fx.XTest, fx.YTest)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if d := math.Abs(score - fx.ScoreTest); d > 10*tol {
		t.Errorf("%s: R^2 %.10f, want %.10f", name, score, fx.ScoreTest)
	}
}

func TestRidge_AgainstSklearn(t *testing.T) {
	for name, fx := range loadLinearModelsFixtures(t).Ridge {
		m := NewRidge()
		m.Alpha, m.FitIntercept = fx.Params.Alpha, fx.Params.FitIntercept
		if err := m.Fit(fx.X, fx.Y); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		checkRegression(t, "ridge/"+name, m, m.Coef, m.Intercept, fx, 1e-7)
	}
}

func TestLasso_AgainstSklearn(t *testing.T) {
	for name, fx := range loadLinearModelsFixtures(t).Lasso {
		m := NewLasso()
		m.Alpha, m.FitIntercept, m.MaxIter, m.Tol = fx.Params.Alpha, fx.Params.FitIntercept, fx.Params.MaxIter, fx.Params.Tol
		if err := m.Fit(fx.X, fx.Y); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		checkRegression(t, "lasso/"+name, m, m.Coef, m.Intercept, fx, 1e-7)
		if fx.NIter != nil && m.NIter() != *fx.NIter {
			t.Errorf("lasso/%s: %d iterations, sklearn used %d", name, m.NIter(), *fx.NIter)
		}
	}
}

func TestElasticNet_AgainstSklearn(t *testing.T) {
	for name, fx := range loadLinearModelsFixtures(t).ElasticNet {
		m := NewElasticNet()
		m.Alpha, m.L1Ratio, m.FitIntercept = fx.Params.Alpha, fx.Params.L1Ratio, fx.Params.FitIntercept
		m.MaxIter, m.Tol = fx.Params.MaxIter, fx.Params.Tol
		if err := m.Fit(fx.X, fx.Y); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		checkRegression(t, "elastic_net/"+name, m, m.Coef, m.Intercept, fx, 1e-7)
		if fx.NIter != nil && m.NIter() != *fx.NIter {
			t.Errorf("elastic_net/%s: %d iterations, sklearn used %d", name, m.NIter(), *fx.NIter)
		}
	}
}

var _ = errors.Is
var _ = matutil.ErrNotFitted

func TestLogisticRegression_AgainstSklearn(t *testing.T) {
	for name, fx := range loadLinearModelsFixtures(t).Logistic {
		m := NewLogisticRegression()
		m.C, m.FitIntercept = fx.Params.C, fx.Params.FitIntercept
		m.Tol, m.MaxIter = 1e-10, 10000
		if err := m.Fit(fx.X, fx.Y); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !m.Converged() {
			t.Errorf("%s: did not converge in %d iterations", name, m.NIter())
		}
		if d := maxAbsDiff(m.Classes(), fx.Classes); d != 0 {
			t.Errorf("%s: classes %v, want %v", name, m.Classes(), fx.Classes)
		}
		coef := m.Coef()
		if len(coef) != len(fx.Coef) {
			t.Fatalf("%s: %d coefficient rows, want %d", name, len(coef), len(fx.Coef))
		}
		var worst float64
		for i := range coef {
			worst = math.Max(worst, maxAbsDiff(coef[i], fx.Coef[i]))
		}
		t.Logf("%s: iterations %d, max coef diff %.3e, intercept diff %.3e", name, m.NIter(), worst, maxAbsDiff(m.Intercept(), fx.Intercept))
		if worst > 1e-5 {
			t.Errorf("%s: coefficients differ from sklearn by %.3e", name, worst)
		}
		if d := maxAbsDiff(m.Intercept(), fx.Intercept); d > 1e-5 {
			t.Errorf("%s: intercepts differ by %.3e: got %v want %v", name, d, m.Intercept(), fx.Intercept)
		}
		pred, err := m.Predict(fx.XTest)
		if err != nil {
			t.Fatal(err)
		}
		if d := maxAbsDiff(pred, fx.PredictTest); d != 0 {
			t.Errorf("%s: predicted labels differ from sklearn", name)
		}
		proba, err := m.PredictProba(fx.XTest)
		if err != nil {
			t.Fatal(err)
		}
		for i := range proba {
			if d := maxAbsDiff(proba[i], fx.ProbaTest[i]); d > 1e-5 {
				t.Errorf("%s: probabilities row %d differ by %.3e", name, i, d)
				break
			}
		}
		if acc, _ := m.Score(fx.X, fx.Y); math.Abs(acc-fx.ScoreTrain) > 1e-12 {
			t.Errorf("%s: training accuracy %v, want %v", name, acc, fx.ScoreTrain)
		}
	}
}
