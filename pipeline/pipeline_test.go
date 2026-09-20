package pipeline

import (
	"encoding/gob"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/decomposition"
	"github.com/ajitashwath/scikit-go/ensemble"
	"github.com/ajitashwath/scikit-go/feature_selection"
	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/linear"
	"github.com/ajitashwath/scikit-go/neighbors"
	"github.com/ajitashwath/scikit-go/preprocessing"
	"github.com/ajitashwath/scikit-go/svm"
)

type fixture struct {
	X            [][]float64 `json:"X"`
	Y            []float64   `json:"y"`
	XTest        [][]float64 `json:"X_test"`
	YTest        []float64   `json:"y_test"`
	PredictTest  []float64   `json:"predict_test"`
	ProbaTest    [][]float64 `json:"proba_test"`
	DecisionTest []float64   `json:"decision_test"`
	TransformTst [][]float64 `json:"transform_test"`
	ScoreTest    float64     `json:"score_test"`
	Support      []bool      `json:"support"`
}

func loadFixtures(t *testing.T) map[string]fixture {
	t.Helper()
	path := filepath.Join("testdata", "pipeline_fixtures.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read fixtures at %s: %v", path, err)
	}
	var fx map[string]fixture
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("failed to parse fixtures: %v", err)
	}
	return fx
}

func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol*(1+math.Abs(b)) }

func assertSlice(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if !near(got[i], want[i], tol) {
			t.Errorf("%s[%d]: got %v, want %v", name, i, got[i], want[i])
		}
	}
}

func assertMatrix(t *testing.T, name string, got, want [][]float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d rows, want %d", name, len(got), len(want))
	}
	for i := range want {
		assertSlice(t, name, got[i], want[i], tol)
	}
}

func mustPipeline(t *testing.T, steps ...Step) *Pipeline {
	t.Helper()
	p, err := NewPipeline(steps...)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	return p
}

func TestPipeline_ScalerLinear_AgainstSklearn(t *testing.T) {
	fx := loadFixtures(t)["scaler_linear"]
	p := mustPipeline(t,
		Step{"scaler", preprocessing.NewStandardScaler()},
		Step{"lr", linear.NewLinearRegression()})
	if err := p.Fit(fx.X, fx.Y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	pred, err := p.Predict(fx.XTest)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	assertSlice(t, "predict", pred, fx.PredictTest, 1e-8)
	score, err := p.Score(fx.XTest, fx.YTest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if !near(score, fx.ScoreTest, 1e-8) {
		t.Errorf("score: got %v, want %v", score, fx.ScoreTest)
	}
}

func TestPipeline_ScalerPCAKNN_AgainstSklearn(t *testing.T) {
	fx := loadFixtures(t)["scaler_pca_knn"]
	pca := decomposition.NewPCA()
	pca.NComponents = 2
	knn := neighbors.NewKNeighborsClassifier()
	p := mustPipeline(t,
		Step{"scaler", preprocessing.NewStandardScaler()},
		Step{"pca", pca},
		Step{"knn", knn})
	if err := p.Fit(fx.X, fx.Y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	pred, err := p.Predict(fx.XTest)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	assertSlice(t, "predict", pred, fx.PredictTest, 0)
	proba, err := p.PredictProba(fx.XTest)
	if err != nil {
		t.Fatalf("PredictProba: %v", err)
	}
	assertMatrix(t, "proba", proba, fx.ProbaTest, 1e-9)
	score, err := p.Score(fx.XTest, fx.YTest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if !near(score, fx.ScoreTest, 1e-9) {
		t.Errorf("score: got %v, want %v", score, fx.ScoreTest)
	}
}

func TestPipeline_SupervisedSelectorReceivesTarget_AgainstSklearn(t *testing.T) {
	fx := loadFixtures(t)["scaler_kbest_linear"]
	sel := feature_selection.NewSelectKBest()
	sel.K = 3
	sel.Score = feature_selection.ScoreFRegression
	p := mustPipeline(t,
		Step{"scaler", preprocessing.NewStandardScaler()},
		Step{"select", sel},
		Step{"lr", linear.NewLinearRegression()})
	if err := p.Fit(fx.X, fx.Y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	for j, want := range fx.Support {
		if sel.Support()[j] != want {
			t.Fatalf("support: got %v, want %v", sel.Support(), fx.Support)
		}
	}
	pred, err := p.Predict(fx.XTest)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	assertSlice(t, "predict", pred, fx.PredictTest, 1e-8)
	score, _ := p.Score(fx.XTest, fx.YTest)
	if !near(score, fx.ScoreTest, 1e-8) {
		t.Errorf("score: got %v, want %v", score, fx.ScoreTest)
	}
}

func TestPipeline_VarianceScalerSVC_AgainstSklearn(t *testing.T) {
	fx := loadFixtures(t)["variance_scaler_svc"]
	svc := svm.NewSVC()
	svc.C = 2
	p := mustPipeline(t,
		Step{"vt", feature_selection.NewVarianceThreshold()},
		Step{"scaler", preprocessing.NewStandardScaler()},
		Step{"svc", svc})
	if err := p.Fit(fx.X, fx.Y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	pred, err := p.Predict(fx.XTest)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	assertSlice(t, "predict", pred, fx.PredictTest, 0)
	dec, err := svc.DecisionFunction(mustTransform(t, p, fx.XTest))
	if err != nil {
		t.Fatalf("DecisionFunction: %v", err)
	}
	for i, want := range fx.DecisionTest {
		if !near(dec[i][0], want, 5e-3) {
			t.Errorf("decision[%d]: got %v, want %v", i, dec[i][0], want)
		}
	}
	score, _ := p.Score(fx.XTest, fx.YTest)
	if !near(score, fx.ScoreTest, 1e-9) {
		t.Errorf("score: got %v, want %v", score, fx.ScoreTest)
	}
}

// mustTransform pushes X through every step but the last.
func mustTransform(t *testing.T, p *Pipeline, X [][]float64) [][]float64 {
	t.Helper()
	out, err := p.transformThrough(X, len(p.steps)-1)
	if err != nil {
		t.Fatalf("transformThrough: %v", err)
	}
	return out
}

func TestPipeline_TransformOnlyChain(t *testing.T) {
	fx := loadFixtures(t)["scaler_linear"]
	p := mustPipeline(t,
		Step{"scaler", preprocessing.NewStandardScaler()},
		Step{"pca", decomposition.NewPCA()})
	out, err := p.FitTransform(fx.X)
	if err != nil {
		t.Fatalf("FitTransform: %v", err)
	}

	// The same chain by hand.
	scaler := preprocessing.NewStandardScaler()
	scaled, _ := scaler.FitTransform(fx.X)
	pca := decomposition.NewPCA()
	want, _ := pca.FitTransform(scaled)
	assertMatrix(t, "FitTransform", out, want, 1e-12)

	got, err := p.Transform(fx.XTest)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}
	scaledTest, _ := scaler.Transform(fx.XTest)
	wantTest, _ := pca.Transform(scaledTest)
	assertMatrix(t, "Transform", got, wantTest, 1e-12)

	if _, err := p.Predict(fx.XTest); !errors.Is(err, ErrInvalidPipeline) {
		t.Errorf("Predict on a transformer-only chain: got %v, want ErrInvalidPipeline", err)
	}
	if _, err := p.Score(fx.XTest, fx.YTest); !errors.Is(err, ErrInvalidPipeline) {
		t.Errorf("Score on a transformer-only chain: got %v, want ErrInvalidPipeline", err)
	}
	if _, err := p.PredictProba(fx.XTest); !errors.Is(err, ErrInvalidPipeline) {
		t.Errorf("PredictProba on a transformer-only chain: got %v, want ErrInvalidPipeline", err)
	}
}

func TestPipeline_EstimatorChainRejectsTransformAndProba(t *testing.T) {
	fx := loadFixtures(t)["scaler_linear"]
	p := mustPipeline(t,
		Step{"scaler", preprocessing.NewStandardScaler()},
		Step{"lr", linear.NewLinearRegression()})
	if err := p.Fit(fx.X, fx.Y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if _, err := p.Transform(fx.XTest); !errors.Is(err, ErrInvalidPipeline) {
		t.Errorf("Transform: got %v, want ErrInvalidPipeline", err)
	}
	if _, err := p.PredictProba(fx.XTest); !errors.Is(err, ErrInvalidPipeline) {
		t.Errorf("PredictProba: got %v, want ErrInvalidPipeline", err)
	}
	if _, err := p.FitTransform(fx.X); !errors.Is(err, ErrInvalidPipeline) {
		t.Errorf("FitTransform: got %v, want ErrInvalidPipeline", err)
	}
}

func TestPipeline_FitTransformCannotSatisfySupervisedStep(t *testing.T) {
	X := [][]float64{{1, 2}, {2, 1}, {3, 5}, {4, 3}}
	sel := feature_selection.NewSelectKBest()
	sel.K = 1
	p := mustPipeline(t, Step{"select", sel})
	if _, err := p.FitTransform(X); !errors.Is(err, matutil.ErrEmptyInput) {
		t.Errorf("got %v, want ErrEmptyInput (SelectKBest needs a target)", err)
	}
}

func TestNewPipeline_Validation(t *testing.T) {
	scaler := func() any { return preprocessing.NewStandardScaler() }
	lr := func() any { return linear.NewLinearRegression() }
	var nilLR *linear.LinearRegression

	cases := []struct {
		name  string
		steps []Step
	}{
		{"no steps", nil},
		{"empty name", []Step{{"", scaler()}, {"lr", lr()}}},
		{"name with separator", []Step{{"a__b", scaler()}, {"lr", lr()}}},
		{"duplicate names", []Step{{"s", scaler()}, {"s", lr()}}},
		{"nil estimator", []Step{{"s", nil}, {"lr", lr()}}},
		{"typed nil pointer", []Step{{"s", scaler()}, {"lr", nilLR}}},
		{"predictor in the middle", []Step{{"lr", lr()}, {"s", scaler()}}},
		{"non-estimator in the middle", []Step{{"x", struct{}{}}, {"lr", lr()}}},
		{"unfittable final step", []Step{{"s", scaler()}, {"x", struct{}{}}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewPipeline(tc.steps...); !errors.Is(err, ErrInvalidPipeline) {
				t.Errorf("got %v, want ErrInvalidPipeline", err)
			}
		})
	}

	// Valid shapes: a lone estimator, and a lone transformer.
	if _, err := NewPipeline(Step{"lr", lr()}); err != nil {
		t.Errorf("single estimator: %v", err)
	}
	if _, err := NewPipeline(Step{"s", scaler()}); err != nil {
		t.Errorf("single transformer: %v", err)
	}
}

func TestMakePipeline_Naming(t *testing.T) {
	p, err := MakePipeline(
		preprocessing.NewStandardScaler(),
		preprocessing.NewStandardScaler(),
		decomposition.NewPCA(),
		linear.NewLinearRegression())
	if err != nil {
		t.Fatalf("MakePipeline: %v", err)
	}
	var names []string
	for _, s := range p.Steps() {
		names = append(names, s.Name)
	}
	want := []string{"standardscaler-1", "standardscaler-2", "pca", "linearregression"}
	if !reflect.DeepEqual(names, want) {
		t.Errorf("names: got %v, want %v", names, want)
	}
	if est, ok := p.Named("pca"); !ok || est == nil {
		t.Error("Named(pca) not found")
	}
	if _, ok := p.Named("missing"); ok {
		t.Error("Named(missing) should not be found")
	}
	if _, err := MakePipeline(nil); !errors.Is(err, ErrInvalidPipeline) {
		t.Errorf("nil estimator: got %v, want ErrInvalidPipeline", err)
	}
	if _, err := MakePipeline(linear.NewLinearRegression(), preprocessing.NewStandardScaler()); !errors.Is(err, ErrInvalidPipeline) {
		t.Errorf("predictor before transformer: got %v, want ErrInvalidPipeline", err)
	}
}

func TestPipeline_StepsReturnsCopy(t *testing.T) {
	p, _ := MakePipeline(preprocessing.NewStandardScaler(), linear.NewLinearRegression())
	steps := p.Steps()
	steps[0].Name = "hacked"
	if p.Steps()[0].Name == "hacked" {
		t.Error("Steps must return a copy")
	}
}

func TestPipeline_BeforeFitAndErrorPropagation(t *testing.T) {
	X := [][]float64{{1, 2}, {3, 4}, {5, 7}, {8, 9}}
	y := []float64{1, 2, 3, 4}
	p, _ := MakePipeline(preprocessing.NewStandardScaler(), linear.NewLinearRegression())
	if _, err := p.Predict(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Predict: got %v", err)
	}
	if _, err := p.PredictProba(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("PredictProba: got %v", err)
	}
	if _, err := p.Transform(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Transform: got %v", err)
	}
	if _, err := p.Score(X, y); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Score: got %v", err)
	}

	// Errors from a step keep their identity and name the failing step.
	bad := [][]float64{{1, 2}, {3}, {5, 7}, {8, 9}}
	err := p.Fit(bad, y)
	if !errors.Is(err, matutil.ErrRaggedInput) {
		t.Fatalf("Fit on ragged input: got %v, want ErrRaggedInput", err)
	}
	if got := err.Error(); !containsAll(got, "Pipeline.Fit", `"standardscaler"`) {
		t.Errorf("error should name the pipeline and step, got %q", got)
	}
	if _, err := p.Predict(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("a failed Fit must leave the pipeline unfitted, Predict got %v", err)
	}

	if err := p.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if _, err := p.Predict([][]float64{{1, 2, 3}}); !errors.Is(err, matutil.ErrDimMismatch) {
		t.Errorf("Predict with wrong feature count: got %v, want ErrDimMismatch", err)
	}
	if _, err := p.Predict(nil); !errors.Is(err, matutil.ErrEmptyInput) {
		t.Errorf("Predict(nil): got %v, want ErrEmptyInput", err)
	}
}

func containsAll(s string, parts ...string) bool {
	for _, p := range parts {
		found := false
		for i := 0; i+len(p) <= len(s); i++ {
			if s[i:i+len(p)] == p {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func TestPipeline_GetParams(t *testing.T) {
	knn := neighbors.NewKNeighborsClassifier()
	rfe := feature_selection.NewRFE(linear.NewLinearRegression())
	p := mustPipeline(t,
		Step{"scaler", preprocessing.NewStandardScaler()},
		Step{"rfe", rfe},
		Step{"knn", knn})
	params := p.GetParams()

	if params["knn"] != knn {
		t.Error("the step itself should be under its name")
	}
	if params["knn__NNeighbors"] != 5 || params["knn__Weights"] != "uniform" || params["knn__P"] != 2.0 {
		t.Errorf("knn params wrong: %v", params)
	}
	if params["rfe__Step"] != 1 || params["rfe__NFeaturesToSelect"] != 0 {
		t.Errorf("rfe params wrong: %v", params)
	}
	if _, ok := params["rfe__Estimator"]; !ok {
		t.Error("nested estimator should be reported under rfe__Estimator")
	}
	if _, ok := params["rfe__Estimator__Intercept"]; !ok {
		t.Errorf("nested estimator fields should be reported, got keys %v", keys(params))
	}
	for k := range params {
		if k == "scaler__Mean" || k == "scaler__Scale" {
			t.Errorf("slice-valued fields must not be reported as params: %s", k)
		}
	}
}

func keys(m map[string]any) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	return out
}

func TestPipeline_SetParams(t *testing.T) {
	fx := loadFixtures(t)["scaler_pca_knn"]
	knn := neighbors.NewKNeighborsClassifier()
	pca := decomposition.NewPCA()
	pca.NComponents = 2
	p := mustPipeline(t,
		Step{"scaler", preprocessing.NewStandardScaler()},
		Step{"pca", pca},
		Step{"knn", knn})
	if err := p.Fit(fx.X, fx.Y); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	if err := p.SetParams(map[string]any{
		"knn__NNeighbors":  3,
		"knn__Weights":     "distance",
		"knn__P":           1,   // an int into a float field
		"pca__NComponents": 2.0, // an integral float into an int field
	}); err != nil {
		t.Fatalf("SetParams: %v", err)
	}
	if knn.NNeighbors != 3 || knn.Weights != "distance" || knn.P != 1 || pca.NComponents != 2 {
		t.Errorf("params not applied: %+v %+v", knn, pca)
	}
	if _, err := p.Predict(fx.XTest); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("SetParams must mark the pipeline unfitted, Predict got %v", err)
	}
	if err := p.Fit(fx.X, fx.Y); err != nil {
		t.Fatalf("refit: %v", err)
	}
	if _, err := p.Predict(fx.XTest); err != nil {
		t.Errorf("Predict after refit: %v", err)
	}
}

func TestPipeline_SetParams_Errors(t *testing.T) {
	knn := neighbors.NewKNeighborsClassifier()
	p := mustPipeline(t,
		Step{"scaler", preprocessing.NewStandardScaler()},
		Step{"knn", knn})
	bad := map[string]map[string]any{
		"unknown step":           {"nope__NNeighbors": 3},
		"unknown field":          {"knn__Nope": 3},
		"unexported field":       {"knn__model": 3},
		"string into int":        {"knn__NNeighbors": "3"},
		"non-integral float":     {"knn__NNeighbors": 2.5},
		"int into string":        {"knn__Weights": 7},
		"bool into int":          {"knn__NNeighbors": true},
		"nil value":              {"knn__NNeighbors": nil},
		"scalar field has depth": {"knn__NNeighbors__x": 3},
		"replace with junk":      {"knn": struct{}{}},
	}
	for name, params := range bad {
		t.Run(name, func(t *testing.T) {
			err := p.SetParams(params)
			if err == nil {
				t.Fatal("expected an error")
			}
			if !errors.Is(err, ErrUnknownParam) && !errors.Is(err, ErrInvalidPipeline) {
				t.Errorf("got %v, want ErrUnknownParam or ErrInvalidPipeline", err)
			}
			if knn.NNeighbors != 5 || knn.Weights != "uniform" {
				t.Errorf("a failed SetParams changed the estimator: %+v", knn)
			}
		})
	}
}

func TestPipeline_SetParams_ReplaceStep(t *testing.T) {
	p := mustPipeline(t,
		Step{"scaler", preprocessing.NewStandardScaler()},
		Step{"model", linear.NewLinearRegression()})
	dst := neighbors.NewKNeighborsRegressor()
	if err := p.SetParams(map[string]any{"model": dst}); err != nil {
		t.Fatalf("SetParams: %v", err)
	}
	if est, _ := p.Named("model"); est != dst {
		t.Error("step was not replaced")
	}
	// Replacing an intermediate step with a predictor invalidates the pipeline.
	if err := p.SetParams(map[string]any{"scaler": linear.NewLinearRegression()}); !errors.Is(err, ErrInvalidPipeline) {
		t.Errorf("got %v, want ErrInvalidPipeline", err)
	}
	if est, _ := p.Named("scaler"); reflect.TypeOf(est) != reflect.TypeOf(preprocessing.NewStandardScaler()) {
		t.Error("a rejected replacement must leave the step alone")
	}
}

func TestPipeline_SetParams_NestedEstimatorField(t *testing.T) {
	rfe := feature_selection.NewRFE(linear.NewLinearRegression())
	p := mustPipeline(t, Step{"rfe", rfe}, Step{"lr", linear.NewLinearRegression()})
	if err := p.SetParams(map[string]any{"rfe__NFeaturesToSelect": 2, "rfe__Estimator__Intercept": 1.5}); err != nil {
		t.Fatalf("SetParams: %v", err)
	}
	if rfe.NFeaturesToSelect != 2 || rfe.Estimator.(*linear.LinearRegression).Intercept != 1.5 {
		t.Errorf("nested params not applied: %+v", rfe)
	}
}

// A step type from outside this module works in a pipeline as long as it has
// Transform and one of the Fit shapes; it just cannot be persisted.
type doubler struct{ fitted bool }

func (d *doubler) FitTransform(X [][]float64) ([][]float64, error) {
	d.fitted = true
	return d.Transform(X)
}

func (d *doubler) Transform(X [][]float64) ([][]float64, error) {
	out := make([][]float64, len(X))
	for i, row := range X {
		out[i] = make([]float64, len(row))
		for j, v := range row {
			out[i][j] = 2 * v
		}
	}
	return out, nil
}

func TestPipeline_CustomTransformerAndPersistence(t *testing.T) {
	X, y, _ := datasets.MakeRegression(40, 3, 0.1, 1)
	d := &doubler{}
	p := mustPipeline(t, Step{"double", d}, Step{"lr", linear.NewLinearRegression()})
	if err := p.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if !d.fitted {
		t.Error("custom transformer was not fitted through FitTransform")
	}
	if _, err := p.Predict(X); err != nil {
		t.Errorf("Predict: %v", err)
	}
	if err := p.Save(filepath.Join(t.TempDir(), "p.gob")); err == nil {
		t.Error("saving a pipeline with an unpersistable step should fail")
	}
}

func TestPipeline_SaveLoadRoundTrip(t *testing.T) {
	X, y, _ := datasets.MakeClassification(120, 6, 3, 9)
	sel := feature_selection.NewSelectKBest()
	sel.K = 4
	forest := ensemble.NewRandomForestClassifier()
	forest.NTrees = 15
	forest.Seed = 3
	p := mustPipeline(t,
		Step{"scaler", preprocessing.NewStandardScaler()},
		Step{"select", sel},
		Step{"forest", forest})
	if err := p.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	path := filepath.Join(t.TempDir(), "pipeline.gob")
	if err := p.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadPipeline(path)
	if err != nil {
		t.Fatalf("LoadPipeline: %v", err)
	}

	if !reflect.DeepEqual(stepNames(loaded), stepNames(p)) {
		t.Errorf("step names: got %v, want %v", stepNames(loaded), stepNames(p))
	}
	want, _ := p.PredictProba(X)
	got, err := loaded.PredictProba(X)
	if err != nil {
		t.Fatalf("PredictProba after load: %v", err)
	}
	assertMatrix(t, "proba", got, want, 0)
	wantPred, _ := p.Predict(X)
	gotPred, _ := loaded.Predict(X)
	assertSlice(t, "predict", gotPred, wantPred, 0)
}

func stepNames(p *Pipeline) []string {
	var out []string
	for _, s := range p.Steps() {
		out = append(out, s.Name)
	}
	return out
}

func TestPipeline_SaveLoadNestedAndEveryKind(t *testing.T) {
	X, y, _ := datasets.MakeClassification(90, 5, 2, 10)
	inner := mustPipeline(t,
		Step{"vt", feature_selection.NewVarianceThreshold()},
		Step{"scaler", preprocessing.NewStandardScaler()})
	svc := svm.NewSVC()
	outer := mustPipeline(t, Step{"prep", inner}, Step{"svc", svc})
	if err := outer.Fit(X, y); err != nil {
		t.Fatalf("Fit nested: %v", err)
	}
	dir := t.TempDir()
	if err := outer.Save(filepath.Join(dir, "nested.gob")); err != nil {
		t.Fatalf("Save nested: %v", err)
	}
	loaded, err := LoadPipeline(filepath.Join(dir, "nested.gob"))
	if err != nil {
		t.Fatalf("Load nested: %v", err)
	}
	want, _ := outer.Predict(X)
	got, _ := loaded.Predict(X)
	assertSlice(t, "nested predict", got, want, 0)

	// One pipeline per remaining estimator kind, so every registry entry round-trips.
	Xr, yr, _ := datasets.MakeRegression(60, 4, 0.5, 2)
	chains := map[string][]Step{
		"pca+knn_regressor": {{"pca", decompositionK(2)}, {"knn", neighbors.NewKNeighborsRegressor()}},
		"linear":            {{"lr", linear.NewLinearRegression()}},
		"svr":               {{"scaler", preprocessing.NewStandardScaler()}, {"svr", svm.NewSVR()}},
		"forest_regressor":  {{"rf", smallForestRegressor()}},
		"rfe+linear":        {{"rfe", rfeFor(2)}, {"lr", linear.NewLinearRegression()}},
	}
	for name, steps := range chains {
		t.Run(name, func(t *testing.T) {
			p := mustPipeline(t, steps...)
			if err := p.Fit(Xr, yr); err != nil {
				t.Fatalf("Fit: %v", err)
			}
			path := filepath.Join(dir, name+".gob")
			if err := p.Save(path); err != nil {
				t.Fatalf("Save: %v", err)
			}
			l, err := LoadPipeline(path)
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			want, _ := p.Predict(Xr)
			got, err := l.Predict(Xr)
			if err != nil {
				t.Fatalf("Predict after load: %v", err)
			}
			assertSlice(t, "predict", got, want, 0)
		})
	}
}

func decompositionK(k int) *decomposition.PCA {
	p := decomposition.NewPCA()
	p.NComponents = k
	return p
}

func smallForestRegressor() *ensemble.RandomForestRegressor {
	rf := ensemble.NewRandomForestRegressor()
	rf.NTrees = 5
	return rf
}

func rfeFor(n int) *feature_selection.RFE {
	r := feature_selection.NewRFE(linear.NewLinearRegression())
	r.NFeaturesToSelect = n
	return r
}

func TestPipeline_SaveBeforeFitAndBadFiles(t *testing.T) {
	dir := t.TempDir()
	p, _ := MakePipeline(preprocessing.NewStandardScaler(), linear.NewLinearRegression())
	if err := p.Save(filepath.Join(dir, "x.gob")); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Save before Fit: got %v", err)
	}
	if _, err := LoadPipeline(filepath.Join(dir, "missing.gob")); err == nil {
		t.Error("loading a nonexistent file should fail")
	}
	junk := filepath.Join(dir, "junk.gob")
	if err := os.WriteFile(junk, []byte("not a gob"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPipeline(junk); err == nil {
		t.Error("loading a corrupt file should fail")
	}
}

func TestLoadPipeline_CorruptPayloads(t *testing.T) {
	dir := t.TempDir()
	scaler := preprocessing.NewStandardScaler()
	if _, err := scaler.FitTransform([][]float64{{1, 2}, {3, 4}, {5, 7}}); err != nil {
		t.Fatal(err)
	}
	blob, err := saveBlob(scaler)
	if err != nil {
		t.Fatalf("saveBlob: %v", err)
	}

	cases := map[string]pipelineGob{
		"bad version":     {Version: 99, Names: []string{"s"}, Kinds: []string{"standard_scaler"}, Blobs: [][]byte{blob}},
		"unknown kind":    {Version: pipelineFormatVersion, Names: []string{"s"}, Kinds: []string{"nonsense"}, Blobs: [][]byte{blob}},
		"length mismatch": {Version: pipelineFormatVersion, Names: []string{"a", "b"}, Kinds: []string{"standard_scaler"}, Blobs: [][]byte{blob}},
		"bad step blob":   {Version: pipelineFormatVersion, Names: []string{"s"}, Kinds: []string{"standard_scaler"}, Blobs: [][]byte{[]byte("junk")}},
		"no steps":        {Version: pipelineFormatVersion},
		"wrong step type": {Version: pipelineFormatVersion, Names: []string{"s", "lr"}, Kinds: []string{"standard_scaler", "standard_scaler"}, Blobs: [][]byte{blob, blob}},
	}
	for name, payload := range cases {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(dir, name+".gob")
			f, err := os.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := gob.NewEncoder(f).Encode(payload); err != nil {
				t.Fatal(err)
			}
			f.Close()
			if name == "wrong step type" {
				// Two valid scalers form a valid chain; this case is the control.
				if _, err := LoadPipeline(path); err != nil {
					t.Fatalf("a valid two-transformer chain should load: %v", err)
				}
				return
			}
			if _, err := LoadPipeline(path); err == nil {
				t.Error("expected an error")
			}
		})
	}

	if _, err := loadBlob("nonsense", nil); err == nil {
		t.Error("an unknown step kind should fail to load")
	}
}
