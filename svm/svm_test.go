package svm

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/internal/matutil"
)

// The solver is a port of libsvm's, so on the same problem it should land on
// the same support vectors; coefficients agree to well within the solver's
// stopping tolerance (1e-3).
const (
	coefTol     = 5e-3
	decisionTol = 5e-3
)

type fixtureParams map[string]any

func (p fixtureParams) float(key string, def float64) float64 {
	if v, ok := p[key].(float64); ok {
		return v
	}
	return def // absent, or a string such as gamma="scale"
}

func (p fixtureParams) str(key, def string) string {
	if v, ok := p[key].(string); ok {
		return v
	}
	return def
}

type svcFixture struct {
	X            [][]float64     `json:"X"`
	Y            []float64       `json:"y"`
	XTest        [][]float64     `json:"X_test"`
	Params       fixtureParams   `json:"params"`
	Classes      []float64       `json:"classes"`
	Support      []int           `json:"support"`
	NSupport     []int           `json:"n_support"`
	DualCoef     [][]float64     `json:"dual_coef"`
	Intercept    []float64       `json:"intercept"`
	Gamma        float64         `json:"gamma"`
	PredictTest  []float64       `json:"predict_test"`
	DecisionTest json.RawMessage `json:"decision_test"`
	ScoreTrain   float64         `json:"score_train"`
	Probability  bool            `json:"probability"`
	ProbaTest    [][]float64     `json:"proba_test"`
}

type svrFixture struct {
	X            [][]float64   `json:"X"`
	Y            []float64     `json:"y"`
	XTest        [][]float64   `json:"X_test"`
	Params       fixtureParams `json:"params"`
	Support      []int         `json:"support"`
	DualCoef     []float64     `json:"dual_coef"`
	Intercept    float64       `json:"intercept"`
	Gamma        float64       `json:"gamma"`
	PredictTest  []float64     `json:"predict_test"`
	PredictTrain []float64     `json:"predict_train"`
	ScoreTrain   float64       `json:"score_train"`
}

func loadRaw(t *testing.T) map[string]json.RawMessage {
	t.Helper()
	path := filepath.Join("testdata", "svm_fixtures.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read fixtures at %s: %v", path, err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("failed to parse fixtures: %v", err)
	}
	return raw
}

func loadSVC(t *testing.T) map[string]svcFixture {
	t.Helper()
	out := map[string]svcFixture{}
	for name, raw := range loadRaw(t) {
		var probe struct {
			Classes json.RawMessage `json:"classes"`
		}
		_ = json.Unmarshal(raw, &probe)
		if probe.Classes == nil {
			continue // an SVR fixture
		}
		var fx svcFixture
		if err := json.Unmarshal(raw, &fx); err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		out[name] = fx
	}
	return out
}

func loadSVR(t *testing.T) map[string]svrFixture {
	t.Helper()
	out := map[string]svrFixture{}
	for name, raw := range loadRaw(t) {
		var probe struct {
			Classes json.RawMessage `json:"classes"`
		}
		_ = json.Unmarshal(raw, &probe)
		if probe.Classes != nil {
			continue
		}
		var fx svrFixture
		if err := json.Unmarshal(raw, &fx); err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		out[name] = fx
	}
	return out
}

// decisionRows accepts sklearn's 1-D (binary) or 2-D (multiclass) decision_function output.
func decisionRows(t *testing.T, raw json.RawMessage) [][]float64 {
	t.Helper()
	var two [][]float64
	if err := json.Unmarshal(raw, &two); err == nil {
		return two
	}
	var one []float64
	if err := json.Unmarshal(raw, &one); err != nil {
		t.Fatalf("decision_test is neither 1-D nor 2-D: %v", err)
	}
	rows := make([][]float64, len(one))
	for i, v := range one {
		rows[i] = []float64{v}
	}
	return rows
}

func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol*(1+math.Abs(b)) }

func newSVCFromParams(p fixtureParams) *SVC {
	s := NewSVC()
	s.Kernel = p.str("kernel", KernelRBF)
	s.C = p.float("C", 1)
	s.Degree = int(p.float("degree", 3))
	s.Gamma = p.float("gamma", 0)
	s.Coef0 = p.float("coef0", 0)
	return s
}

func TestSVC_AgainstSklearn(t *testing.T) {
	for name, fx := range loadSVC(t) {
		fx := fx
		t.Run(name, func(t *testing.T) {
			s := newSVCFromParams(fx.Params)
			s.Probability = fx.Probability
			if err := s.Fit(fx.X, fx.Y); err != nil {
				t.Fatalf("Fit: %v", err)
			}

			if len(s.Classes()) != len(fx.Classes) {
				t.Fatalf("classes: got %v, want %v", s.Classes(), fx.Classes)
			}
			if !near(s.GammaValue(), fx.Gamma, 1e-9) {
				t.Errorf("gamma: got %v, want %v", s.GammaValue(), fx.Gamma)
			}
			if fx.Probability {
				return // probability fixtures are checked loosely in TestSVC_ProbabilityAgainstSklearn
			}

			if !equalInts(s.Support(), fx.Support) {
				t.Errorf("support: got %v, want %v", s.Support(), fx.Support)
			}
			if !equalInts(s.NSupport(), fx.NSupport) {
				t.Errorf("n_support: got %v, want %v", s.NSupport(), fx.NSupport)
			}
			dual := s.DualCoef()
			for r := range fx.DualCoef {
				for c := range fx.DualCoef[r] {
					if !near(dual[r][c], fx.DualCoef[r][c], coefTol) {
						t.Fatalf("dual_coef[%d][%d]: got %v, want %v", r, c, dual[r][c], fx.DualCoef[r][c])
					}
				}
			}
			for i, want := range fx.Intercept {
				if !near(s.Intercept()[i], want, coefTol) {
					t.Errorf("intercept[%d]: got %v, want %v", i, s.Intercept()[i], want)
				}
			}

			pred, err := s.Predict(fx.XTest)
			if err != nil {
				t.Fatalf("Predict: %v", err)
			}
			for i, want := range fx.PredictTest {
				if pred[i] != want {
					t.Errorf("predict[%d]: got %v, want %v", i, pred[i], want)
				}
			}
			dec, err := s.DecisionFunction(fx.XTest)
			if err != nil {
				t.Fatalf("DecisionFunction: %v", err)
			}
			want := decisionRows(t, fx.DecisionTest)
			for i := range want {
				if len(dec[i]) != len(want[i]) {
					t.Fatalf("decision row %d has %d columns, want %d", i, len(dec[i]), len(want[i]))
				}
				for j := range want[i] {
					if !near(dec[i][j], want[i][j], decisionTol) {
						t.Errorf("decision[%d][%d]: got %v, want %v", i, j, dec[i][j], want[i][j])
					}
				}
			}
			score, err := s.Score(fx.X, fx.Y)
			if err != nil {
				t.Fatalf("Score: %v", err)
			}
			if !near(score, fx.ScoreTrain, 1e-9) {
				t.Errorf("train accuracy: got %v, want %v", score, fx.ScoreTrain)
			}
		})
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestSVR_AgainstSklearn(t *testing.T) {
	for name, fx := range loadSVR(t) {
		fx := fx
		t.Run(name, func(t *testing.T) {
			r := NewSVR()
			r.Kernel = fx.Params.str("kernel", KernelRBF)
			r.C = fx.Params.float("C", 1)
			r.Epsilon = fx.Params.float("epsilon", 0.1)
			r.Degree = int(fx.Params.float("degree", 3))
			r.Gamma = fx.Params.float("gamma", 0)
			r.Coef0 = fx.Params.float("coef0", 0)
			if err := r.Fit(fx.X, fx.Y); err != nil {
				t.Fatalf("Fit: %v", err)
			}
			if !near(r.GammaValue(), fx.Gamma, 1e-9) {
				t.Errorf("gamma: got %v, want %v", r.GammaValue(), fx.Gamma)
			}
			if !equalInts(r.Support(), fx.Support) {
				t.Errorf("support: got %v, want %v", r.Support(), fx.Support)
			}
			// With a linear kernel the dual solution is unique and must match. For
			// rbf/poly the Gram matrix is near-singular, so different valid
			// solutions within the solver tolerance (libsvm also shrinks its
			// active set, changing the path) differ in individual coefficients
			// while defining the same function: check dual feasibility (the
			// coefficients sum to zero and respect the box) and, below, the
			// predictions.
			if r.Kernel == KernelLinear {
				for i, want := range fx.DualCoef {
					if !near(r.DualCoef()[i], want, coefTol) {
						t.Errorf("dual_coef[%d]: got %v, want %v", i, r.DualCoef()[i], want)
					}
				}
			} else {
				var sum float64
				for i, c := range r.DualCoef() {
					sum += c
					if math.Abs(c) > r.C+1e-12 {
						t.Errorf("dual_coef[%d]=%v exceeds C=%v", i, c, r.C)
					}
				}
				if math.Abs(sum) > 1e-9 {
					t.Errorf("dual coefficients sum to %v, want 0", sum)
				}
			}
			if !near(r.Intercept(), fx.Intercept, coefTol) {
				t.Errorf("intercept: got %v, want %v", r.Intercept(), fx.Intercept)
			}
			pred, err := r.Predict(fx.XTest)
			if err != nil {
				t.Fatalf("Predict: %v", err)
			}
			for i, want := range fx.PredictTest {
				if !near(pred[i], want, decisionTol) {
					t.Errorf("predict_test[%d]: got %v, want %v", i, pred[i], want)
				}
			}
			score, err := r.Score(fx.X, fx.Y)
			if err != nil {
				t.Fatalf("Score: %v", err)
			}
			if !near(score, fx.ScoreTrain, 1e-3) {
				t.Errorf("train R2: got %v, want %v", score, fx.ScoreTrain)
			}
		})
	}
}

// Probabilities come from cross-validated Platt scaling, whose folds cannot
// match libsvm's C rand(); check the structure and rough agreement instead.
func TestSVC_ProbabilityAgainstSklearn(t *testing.T) {
	for name, fx := range loadSVC(t) {
		if !fx.Probability {
			continue
		}
		fx := fx
		t.Run(name, func(t *testing.T) {
			s := newSVCFromParams(fx.Params)
			s.Probability = true
			s.Seed = 1
			if err := s.Fit(fx.X, fx.Y); err != nil {
				t.Fatalf("Fit: %v", err)
			}
			proba, err := s.PredictProba(fx.XTest)
			if err != nil {
				t.Fatalf("PredictProba: %v", err)
			}
			var maxDiff float64
			agree := 0
			for i, row := range proba {
				var sum float64
				for k, p := range row {
					if p < 0 || p > 1 {
						t.Fatalf("proba[%d][%d]=%v outside [0,1]", i, k, p)
					}
					sum += p
					maxDiff = math.Max(maxDiff, math.Abs(p-fx.ProbaTest[i][k]))
				}
				if math.Abs(sum-1) > 1e-6 {
					t.Errorf("row %d sums to %v", i, sum)
				}
				if argmax(row) == argmax(fx.ProbaTest[i]) {
					agree++
				}
			}
			if maxDiff > 0.25 {
				t.Errorf("max |proba - sklearn| = %.3f, want <= 0.25", maxDiff)
			}
			if frac := float64(agree) / float64(len(proba)); frac < 0.85 {
				t.Errorf("most probable class agrees on %.0f%% of rows, want >= 85%%", 100*frac)
			}
		})
	}
}

func argmax(v []float64) int {
	best := 0
	for i := range v {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func TestSVC_PredictProbaRequiresProbability(t *testing.T) {
	X := [][]float64{{0, 0}, {1, 1}, {0, 1}, {1, 0}, {5, 5}, {6, 5}, {5, 6}, {6, 6}}
	y := []float64{0, 0, 0, 0, 1, 1, 1, 1}
	s := NewSVC()
	if err := s.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if _, err := s.PredictProba(X); !errors.Is(err, ErrInvalidSVM) {
		t.Errorf("got %v, want ErrInvalidSVM", err)
	}
}

func TestSVC_RecoversLinearlySeparableData(t *testing.T) {
	X, y, err := datasets.MakeBlobs(200, 2, 2, 3)
	if err != nil {
		t.Fatalf("MakeBlobs: %v", err)
	}
	for _, kernel := range []string{KernelLinear, KernelRBF, KernelPoly} {
		s := NewSVC()
		s.Kernel = kernel
		if err := s.Fit(X, y); err != nil {
			t.Fatalf("%s Fit: %v", kernel, err)
		}
		acc, err := s.Score(X, y)
		if err != nil {
			t.Fatalf("%s Score: %v", kernel, err)
		}
		if acc < 0.95 {
			t.Errorf("%s kernel: training accuracy %.3f on well-separated blobs", kernel, acc)
		}
	}
}

func TestSVC_SingleSamplePerClass(t *testing.T) {
	X := [][]float64{{0, 0}, {4, 4}}
	y := []float64{-1, 1}
	s := NewSVC()
	s.Kernel = KernelLinear
	if err := s.Fit(X, y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	pred, err := s.Predict([][]float64{{0.5, 0.5}, {3.5, 3.5}})
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if pred[0] != -1 || pred[1] != 1 {
		t.Errorf("got %v, want [-1 1]", pred)
	}
}

func TestSVC_MaxIterStopsEarly(t *testing.T) {
	X, y, _ := datasets.MakeBlobs(100, 2, 2, 4)
	s := NewSVC()
	s.MaxIter = 2
	if err := s.Fit(X, y); err != nil {
		t.Fatalf("Fit with a tiny iteration budget should still return a model: %v", err)
	}
	if _, err := s.Predict(X); err != nil {
		t.Errorf("Predict: %v", err)
	}
}

func TestSVC_SmallCacheGivesSameModel(t *testing.T) {
	X, y, _ := datasets.MakeClassification(150, 4, 3, 5)
	fit := func(cacheMB float64) []float64 {
		s := NewSVC()
		s.CacheSize = cacheMB
		if err := s.Fit(X, y); err != nil {
			t.Fatalf("Fit: %v", err)
		}
		out, _ := s.Predict(X)
		return append(out, s.Intercept()...)
	}
	big, tiny := fit(200), fit(0.001) // 0.001MB holds only the minimum two rows
	for i := range big {
		if big[i] != tiny[i] {
			t.Fatalf("cache size changed the model at %d: %v vs %v", i, big[i], tiny[i])
		}
	}
}

func TestSVC_InvalidHyperparameters(t *testing.T) {
	X := [][]float64{{0, 0}, {1, 1}, {5, 5}, {6, 6}}
	y := []float64{0, 0, 1, 1}
	mutations := map[string]func(*SVC){
		"C=0":            func(s *SVC) { s.C = 0 },
		"C<0":            func(s *SVC) { s.C = -1 },
		"C=NaN":          func(s *SVC) { s.C = math.NaN() },
		"tol=0":          func(s *SVC) { s.Tol = 0 },
		"gamma<0":        func(s *SVC) { s.Gamma = -1 },
		"degree=0 poly":  func(s *SVC) { s.Kernel = KernelPoly; s.Degree = 0 },
		"max_iter=0":     func(s *SVC) { s.MaxIter = 0 },
		"cache_size<=0":  func(s *SVC) { s.CacheSize = 0 },
		"unknown kernel": func(s *SVC) { s.Kernel = "chi2" },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			s := NewSVC()
			mutate(s)
			if err := s.Fit(X, y); !errors.Is(err, ErrInvalidSVM) {
				t.Errorf("got %v, want ErrInvalidSVM", err)
			}
		})
	}
	if err := NewSVC().Fit(X, []float64{1, 1, 1, 1}); !errors.Is(err, ErrInvalidSVM) {
		t.Errorf("single class: got %v, want ErrInvalidSVM", err)
	}
}

func TestSVR_InvalidHyperparameters(t *testing.T) {
	X := [][]float64{{0}, {1}, {2}, {3}}
	y := []float64{0, 1, 2, 3}
	mutations := map[string]func(*SVR){
		"C=0":            func(r *SVR) { r.C = 0 },
		"epsilon<0":      func(r *SVR) { r.Epsilon = -0.1 },
		"epsilon=NaN":    func(r *SVR) { r.Epsilon = math.NaN() },
		"tol=0":          func(r *SVR) { r.Tol = 0 },
		"unknown kernel": func(r *SVR) { r.Kernel = "chi2" },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			r := NewSVR()
			mutate(r)
			if err := r.Fit(X, y); !errors.Is(err, ErrInvalidSVM) {
				t.Errorf("got %v, want ErrInvalidSVM", err)
			}
		})
	}
}

func TestSVM_BeforeFit(t *testing.T) {
	X := [][]float64{{1, 2}}
	s, r := NewSVC(), NewSVR()
	if _, err := s.Predict(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("SVC.Predict: got %v", err)
	}
	if _, err := s.DecisionFunction(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("SVC.DecisionFunction: got %v", err)
	}
	if _, err := s.PredictProba(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("SVC.PredictProba: got %v", err)
	}
	if _, err := r.Predict(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("SVR.Predict: got %v", err)
	}
	if s.SupportVectors() != nil || s.Support() != nil || s.DualCoef() != nil || s.Classes() != nil ||
		r.SupportVectors() != nil || r.Support() != nil || r.DualCoef() != nil {
		t.Error("accessors should return nil before Fit")
	}
}

func TestSVM_InvalidInput(t *testing.T) {
	good := [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
	y := []float64{0, 1, 0, 1}
	cases := []struct {
		name string
		X    [][]float64
		y    []float64
		want error
	}{
		{"empty", nil, nil, matutil.ErrEmptyInput},
		{"mismatched", good, y[:3], matutil.ErrDimMismatch},
		{"ragged", [][]float64{{1, 2}, {3}, {5, 6}, {7, 8}}, y, matutil.ErrRaggedInput},
		{"nan", [][]float64{{1, math.NaN()}, {3, 4}, {5, 6}, {7, 8}}, y, matutil.ErrContainsNaN},
		{"inf", [][]float64{{1, math.Inf(1)}, {3, 4}, {5, 6}, {7, 8}}, y, matutil.ErrContainsInf},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := NewSVC().Fit(tc.X, tc.y); !errors.Is(err, tc.want) {
				t.Errorf("SVC: got %v, want %v", err, tc.want)
			}
			if err := NewSVR().Fit(tc.X, tc.y); !errors.Is(err, tc.want) {
				t.Errorf("SVR: got %v, want %v", err, tc.want)
			}
		})
	}
}

func TestSVM_PredictWrongFeatureCount(t *testing.T) {
	X, y, _ := datasets.MakeBlobs(40, 3, 2, 6)
	s := NewSVC()
	if err := s.Fit(X, y); err != nil {
		t.Fatalf("SVC.Fit: %v", err)
	}
	r := NewSVR()
	if err := r.Fit(X, y); err != nil {
		t.Fatalf("SVR.Fit: %v", err)
	}
	bad := [][]float64{{1, 2}}
	if _, err := s.Predict(bad); !errors.Is(err, matutil.ErrDimMismatch) {
		t.Errorf("SVC.Predict: got %v", err)
	}
	if _, err := s.DecisionFunction(bad); !errors.Is(err, matutil.ErrDimMismatch) {
		t.Errorf("SVC.DecisionFunction: got %v", err)
	}
	if _, err := r.Predict(bad); !errors.Is(err, matutil.ErrDimMismatch) {
		t.Errorf("SVR.Predict: got %v", err)
	}
	if _, err := r.Predict(nil); !errors.Is(err, matutil.ErrEmptyInput) {
		t.Errorf("SVR.Predict(nil): got %v", err)
	}
}

func TestSVC_SaveLoadRoundTrip(t *testing.T) {
	fx := loadSVC(t)["proba_multi"]
	s := newSVCFromParams(fx.Params)
	s.Probability = true
	s.Seed = 2
	if err := s.Fit(fx.X, fx.Y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	path := filepath.Join(t.TempDir(), "svc.gob")
	if err := s.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadSVC(path)
	if err != nil {
		t.Fatalf("LoadSVC: %v", err)
	}
	wantPred, _ := s.Predict(fx.XTest)
	gotPred, err := loaded.Predict(fx.XTest)
	if err != nil {
		t.Fatalf("Predict after load: %v", err)
	}
	for i := range wantPred {
		if gotPred[i] != wantPred[i] {
			t.Errorf("pred[%d]: got %v, want %v", i, gotPred[i], wantPred[i])
		}
	}
	wantP, _ := s.PredictProba(fx.XTest)
	gotP, err := loaded.PredictProba(fx.XTest)
	if err != nil {
		t.Fatalf("PredictProba after load: %v", err)
	}
	wantD, _ := s.DecisionFunction(fx.XTest)
	gotD, _ := loaded.DecisionFunction(fx.XTest)
	for i := range wantP {
		for k := range wantP[i] {
			if gotP[i][k] != wantP[i][k] || gotD[i][k] != wantD[i][k] {
				t.Fatalf("row %d col %d differs after load", i, k)
			}
		}
	}
	if loaded.Kernel != s.Kernel || loaded.C != s.C || !loaded.Probability {
		t.Errorf("hyperparameters not restored: %+v", loaded)
	}
	if _, err := LoadSVR(path); err == nil {
		t.Error("loading an SVC file as an SVR should fail")
	}
}

func TestSVR_SaveLoadRoundTrip(t *testing.T) {
	fx := loadSVR(t)["svr_rbf"]
	r := NewSVR()
	r.C, r.Epsilon = 10, 0.1
	if err := r.Fit(fx.X, fx.Y); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	path := filepath.Join(t.TempDir(), "svr.gob")
	if err := r.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadSVR(path)
	if err != nil {
		t.Fatalf("LoadSVR: %v", err)
	}
	want, _ := r.Predict(fx.XTest)
	got, err := loaded.Predict(fx.XTest)
	if err != nil {
		t.Fatalf("Predict after load: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("pred[%d]: got %v, want %v", i, got[i], want[i])
		}
	}
	if loaded.Epsilon != 0.1 || loaded.C != 10 {
		t.Errorf("hyperparameters not restored: %+v", loaded)
	}
}

func TestSVM_SaveBeforeFitAndBadFiles(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.gob")
	if err := NewSVC().Save(path); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("SVC.Save: got %v", err)
	}
	if err := NewSVR().Save(path); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("SVR.Save: got %v", err)
	}
	if _, err := LoadSVC(filepath.Join(dir, "missing.gob")); err == nil {
		t.Error("loading a nonexistent file should fail")
	}
	junk := filepath.Join(dir, "junk.gob")
	if err := os.WriteFile(junk, []byte("not a gob"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadSVR(junk); err == nil {
		t.Error("loading a corrupt file should fail")
	}
}

func TestMulticlassProbability_BinaryMatchesPairwise(t *testing.T) {
	// With two classes the coupled probability must equal the pairwise one.
	r := [][]float64{{0, 0.8}, {0.2, 0}}
	p := multiclassProbability(2, r)
	if math.Abs(p[0]-0.8) > 1e-3 || math.Abs(p[1]-0.2) > 1e-3 {
		t.Errorf("got %v, want [0.8 0.2]", p)
	}
}

func TestSigmoidTrain_SeparableScoresGiveMonotoneProbabilities(t *testing.T) {
	dec := []float64{-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2}
	labels := []float64{-1, -1, -1, -1, 1, 1, 1, 1}
	a, b := sigmoidTrain(dec, labels)
	if a >= 0 {
		t.Errorf("A=%v should be negative so that P(y=1) grows with the decision value", a)
	}
	last := -1.0
	for _, d := range dec {
		p := sigmoidPredict(d, a, b)
		if p <= last {
			t.Errorf("probability not increasing at decision value %v", d)
		}
		last = p
	}
}

// Feature magnitudes whose kernel products overflow used to send the solver into
// an infinite loop (NaN never satisfies the stopping test). It must fail fast.
func TestSVM_KernelOverflowReturnsError(t *testing.T) {
	X := [][]float64{{1e200, 1}, {-1e200, 2}, {2e200, 3}, {-2e200, 4}, {0, 5}, {1e200, 6}}
	y := []float64{0, 1, 0, 1, 0, 1}

	s := NewSVC()
	s.Kernel = KernelLinear
	if err := s.Fit(X, y); !errors.Is(err, ErrNumerical) {
		t.Errorf("SVC.Fit: got %v, want ErrNumerical", err)
	}
	s = NewSVC()
	s.Kernel, s.Probability = KernelLinear, true
	if err := s.Fit(X, y); !errors.Is(err, ErrNumerical) {
		t.Errorf("SVC.Fit with probability: got %v, want ErrNumerical", err)
	}
	r := NewSVR()
	r.Kernel = KernelLinear
	if err := r.Fit(X, y); !errors.Is(err, ErrNumerical) {
		t.Errorf("SVR.Fit: got %v, want ErrNumerical", err)
	}
}

// At 1e150 the kernel values are finite but the dual problem spans roughly 300
// orders of magnitude, so SMO crawls without ever meeting the tolerance. It used
// to be stopped only by a 10M-iteration backstop that then returned a meaningless
// model; it must now fail with an error.
func TestSVM_IllConditionedProblemFailsInsteadOfHanging(t *testing.T) {
	X := [][]float64{{1e150, 1}, {-1e150, 2}, {2e150, 3}, {-2e150, 4}, {0, 5}, {1e150, 6}}
	y := []float64{0, 1, 0, 1, 0, 1}

	start := time.Now()
	s := NewSVC()
	s.Kernel, s.Probability = KernelLinear, true
	failed := func(err error) bool { return errors.Is(err, ErrNumerical) || errors.Is(err, ErrNotConverged) }
	if err := s.Fit(X, y); !failed(err) {
		t.Errorf("SVC.Fit: got %v, want ErrNumerical or ErrNotConverged", err)
	}
	r := NewSVR()
	r.Kernel = KernelLinear
	if err := r.Fit(X, y); !failed(err) {
		t.Errorf("SVR.Fit: got %v, want ErrNumerical or ErrNotConverged", err)
	}
	if elapsed := time.Since(start); elapsed > 10*time.Second {
		t.Errorf("took %v: the safety limit is too generous for a six-sample problem", elapsed)
	}
}

// An explicit MaxIter is a request to stop early, not an error.
func TestSVM_ExplicitMaxIterStillReturnsModelOnIllConditionedData(t *testing.T) {
	X := [][]float64{{1e150, 1}, {-1e150, 2}, {2e150, 3}, {-2e150, 4}, {0, 5}, {1e150, 6}}
	s := NewSVC()
	s.Kernel, s.MaxIter = KernelLinear, 50
	if err := s.Fit(X, []float64{0, 1, 0, 1, 0, 1}); err != nil {
		t.Errorf("an explicit iteration limit should return the model as it stands: %v", err)
	}
}
