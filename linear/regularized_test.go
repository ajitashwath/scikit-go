package linear

import (
	"errors"
	"math"
	"path/filepath"
	"testing"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/internal/matutil"
)

var (
	_ core.Estimator  = (*Ridge)(nil)
	_ core.Estimator  = (*Lasso)(nil)
	_ core.Estimator  = (*ElasticNet)(nil)
	_ core.Estimator  = (*LogisticRegression)(nil)
	_ core.Predictor  = (*LogisticRegression)(nil)
	_ core.Classifier = (*LogisticRegression)(nil)
	_ core.Saver      = (*Ridge)(nil)
	_ core.Saver      = (*Lasso)(nil)
	_ core.Saver      = (*ElasticNet)(nil)
	_ core.Saver      = (*LogisticRegression)(nil)
)

// sparseData has 6 features of which only the first three matter.
func sparseData() ([][]float64, []float64) {
	X := make([][]float64, 120)
	y := make([]float64, len(X))
	for i := range X {
		row := make([]float64, 6)
		for j := range row {
			row[j] = math.Sin(float64(i*(j+3))*0.7) + 0.3*math.Cos(float64(i*(j+1))*1.3)
		}
		X[i] = row
		y[i] = 4*row[0] - 3*row[1] + 2*row[2] + 0.05*math.Sin(float64(i)*11)
	}
	return X, y
}

func norm2(v []float64) float64 { return math.Sqrt(dot(v, v)) }

func TestRidge_AlphaZeroMatchesLinearRegression(t *testing.T) {
	X, y := sparseData()
	r := NewRidge()
	r.Alpha = 0
	lr := NewLinearRegression()
	if err := r.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	if err := lr.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(r.Coef, lr.Coef); d > 1e-8 {
		t.Errorf("Ridge(alpha=0) coefficients differ from LinearRegression by %.3e", d)
	}
	if math.Abs(r.Intercept-lr.Intercept) > 1e-8 {
		t.Errorf("intercepts differ: %v vs %v", r.Intercept, lr.Intercept)
	}
}

func TestRidge_LargerAlphaShrinksCoefficients(t *testing.T) {
	X, y := sparseData()
	prev := math.Inf(1)
	for _, alpha := range []float64{0, 0.1, 1, 10, 100, 1e4} {
		r := NewRidge()
		r.Alpha = alpha
		if err := r.Fit(X, y); err != nil {
			t.Fatal(err)
		}
		if n := norm2(r.Coef); n > prev+1e-12 {
			t.Errorf("alpha=%v: ||coef|| = %v grew from %v", alpha, n, prev)
		} else {
			prev = n
		}
	}
	// A huge alpha drives the coefficients to zero and the intercept to the mean of y.
	r := NewRidge()
	r.Alpha = 1e15
	if err := r.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	if norm2(r.Coef) > 1e-9 {
		t.Errorf("alpha=1e15: ||coef|| = %v, want about 0", norm2(r.Coef))
	}
}

func TestLasso_FindsTheSparseSolution(t *testing.T) {
	X, y := sparseData()
	l := NewLasso()
	l.Alpha, l.Tol = 0.05, 1e-10
	if err := l.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	if !l.Converged() {
		t.Fatalf("did not converge in %d iterations", l.NIter())
	}
	for j := 3; j < 6; j++ {
		if l.Coef[j] != 0 {
			t.Errorf("irrelevant feature %d got coefficient %v, want exactly 0", j, l.Coef[j])
		}
	}
	for j := 0; j < 3; j++ {
		if l.Coef[j] == 0 {
			t.Errorf("relevant feature %d was zeroed out", j)
		}
	}
	// Lasso shrinks: every kept coefficient is smaller in magnitude than the true one.
	for j, truth := range []float64{4, -3, 2} {
		if math.Abs(l.Coef[j]) >= math.Abs(truth) {
			t.Errorf("coef[%d] = %v not shrunk below %v", j, l.Coef[j], truth)
		}
	}
}

func TestLassoAndElasticNet_L1RatioOneAreIdentical(t *testing.T) {
	X, y := sparseData()
	l := NewLasso()
	l.Alpha = 0.1
	e := NewElasticNet()
	e.Alpha, e.L1Ratio = 0.1, 1
	if err := l.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	if err := e.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	if maxAbsDiff(l.Coef, e.Coef) != 0 || l.Intercept != e.Intercept || l.NIter() != e.NIter() {
		t.Errorf("Lasso and ElasticNet(L1Ratio=1) disagree:\n %v\n %v", l.Coef, e.Coef)
	}
}

func TestLasso_HugeAlphaZeroesEverything(t *testing.T) {
	X, y := sparseData()
	l := NewLasso()
	l.Alpha = 1e6
	if err := l.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	for j, c := range l.Coef {
		if c != 0 {
			t.Errorf("coef[%d] = %v, want 0", j, c)
		}
	}
	var mean float64
	for _, v := range y {
		mean += v / float64(len(y))
	}
	if math.Abs(l.Intercept-mean) > 1e-12 {
		t.Errorf("intercept %v, want the mean of y %v", l.Intercept, mean)
	}
	if l.NIter() != 0 || !l.Converged() {
		t.Errorf("an all-zero solution should be recognized before iterating: %d iterations, converged=%v", l.NIter(), l.Converged())
	}
}

func TestCoordinateDescent_MaxIterIsReportedNotFatal(t *testing.T) {
	X, y := sparseData()
	l := NewLasso()
	l.Alpha, l.MaxIter, l.Tol = 0.001, 1, 1e-12
	if err := l.Fit(X, y); err != nil {
		t.Fatalf("hitting MaxIter must not fail the fit: %v", err)
	}
	if l.Converged() || l.NIter() != 1 {
		t.Errorf("converged=%v after %d iterations, want not converged after exactly 1", l.Converged(), l.NIter())
	}
	if _, err := l.Predict(X); err != nil {
		t.Errorf("the partially converged model should still predict: %v", err)
	}
}

func TestCoordinateDescent_ConstantColumnGetsZero(t *testing.T) {
	X, y := sparseData()
	for i := range X {
		X[i] = append(X[i], 7) // a constant column has zero variance
	}
	l := NewLasso()
	l.Alpha = 0.05
	if err := l.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	if l.Coef[6] != 0 || !allFinite(l.Coef) {
		t.Errorf("constant column coefficient = %v, want 0 (coefs %v)", l.Coef[6], l.Coef)
	}
}

func TestRegularizedRegressors_ParamErrors(t *testing.T) {
	X, y := sparseData()
	nan, inf := math.NaN(), math.Inf(1)
	cases := map[string]func() error{
		"ridge negative alpha":  func() error { r := NewRidge(); r.Alpha = -1; return r.Fit(X, y) },
		"ridge NaN alpha":       func() error { r := NewRidge(); r.Alpha = nan; return r.Fit(X, y) },
		"ridge infinite alpha":  func() error { r := NewRidge(); r.Alpha = inf; return r.Fit(X, y) },
		"lasso negative alpha":  func() error { l := NewLasso(); l.Alpha = -1; return l.Fit(X, y) },
		"lasso zero MaxIter":    func() error { l := NewLasso(); l.MaxIter = 0; return l.Fit(X, y) },
		"lasso negative tol":    func() error { l := NewLasso(); l.Tol = -1; return l.Fit(X, y) },
		"lasso NaN tol":         func() error { l := NewLasso(); l.Tol = nan; return l.Fit(X, y) },
		"enet l1ratio > 1":      func() error { e := NewElasticNet(); e.L1Ratio = 1.5; return e.Fit(X, y) },
		"enet l1ratio < 0":      func() error { e := NewElasticNet(); e.L1Ratio = -0.1; return e.Fit(X, y) },
		"enet NaN l1ratio":      func() error { e := NewElasticNet(); e.L1Ratio = nan; return e.Fit(X, y) },
		"ridge empty":           func() error { return NewRidge().Fit(nil, nil) },
		"lasso length mismatch": func() error { return NewLasso().Fit(X, y[:5]) },
		"enet NaN in X": func() error {
			bad := [][]float64{{1, nan}, {2, 3}, {4, 5}}
			return NewElasticNet().Fit(bad, []float64{1, 2, 3})
		},
	}
	for name, run := range cases {
		if err := run(); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestRegularizedRegressors_NotFittedAndFeatureMismatch(t *testing.T) {
	X, y := sparseData()
	models := map[string]regressor{"ridge": NewRidge(), "lasso": NewLasso(), "elastic_net": NewElasticNet()}
	for name, m := range models {
		if _, err := m.Predict(X); !errors.Is(err, matutil.ErrNotFitted) {
			t.Errorf("%s: Predict before Fit = %v, want ErrNotFitted", name, err)
		}
		if _, err := m.Score(X, y); !errors.Is(err, matutil.ErrNotFitted) {
			t.Errorf("%s: Score before Fit = %v, want ErrNotFitted", name, err)
		}
		if err := m.Fit(X, y); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if _, err := m.Predict([][]float64{{1, 2}}); !errors.Is(err, matutil.ErrDimMismatch) {
			t.Errorf("%s: Predict with 2 features = %v, want ErrDimMismatch", name, err)
		}
	}
}

func TestNoIntercept_ThroughTheOrigin(t *testing.T) {
	X := [][]float64{{1}, {2}, {3}, {4}}
	y := []float64{12, 22, 32, 42} // y = 10x + 2 : no fit through the origin can be exact
	r := NewRidge()
	r.Alpha, r.FitIntercept = 0, false
	l := NewLasso()
	l.Alpha, l.FitIntercept = 0, false
	for _, m := range []regressor{r, l} {
		if err := m.Fit(X, y); err != nil {
			t.Fatal(err)
		}
	}
	if r.Intercept != 0 || l.Intercept != 0 {
		t.Errorf("intercepts %v and %v, want 0 when FitIntercept is false", r.Intercept, l.Intercept)
	}
	// Least squares through the origin: slope = sum(xy)/sum(x^2) = 320/30.
	if want := 320.0 / 30.0; math.Abs(r.Coef[0]-want) > 1e-9 {
		t.Errorf("ridge slope %v, want %v", r.Coef[0], want)
	}
}

// --- LogisticRegression -----------------------------------------------------

// blobs returns two or three well separated 2-D clusters with class labels labels[k].
func blobs(labels ...float64) ([][]float64, []float64) {
	var X [][]float64
	var y []float64
	centers := [][2]float64{{-4, -4}, {4, 4}, {-4, 5}}
	for k, label := range labels {
		for i := 0; i < 25; i++ {
			dx := 0.8 * math.Sin(float64(i*7+k*13))
			dy := 0.8 * math.Cos(float64(i*5+k*3))
			X = append(X, []float64{centers[k][0] + dx, centers[k][1] + dy})
			y = append(y, label)
		}
	}
	return X, y
}

func TestLogisticRegression_SeparableClassesAndOriginalLabels(t *testing.T) {
	X, y := blobs(-3, 8) // odd labels must come back unchanged
	m := NewLogisticRegression()
	if err := m.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	if got := m.Classes(); len(got) != 2 || got[0] != -3 || got[1] != 8 {
		t.Errorf("Classes() = %v, want [-3 8]", got)
	}
	pred, err := m.Predict(X)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(pred, y); d != 0 {
		t.Error("separable training data should be classified perfectly, with the original labels")
	}
	acc, _ := m.Score(X, y)
	if acc != 1 {
		t.Errorf("accuracy %v, want 1", acc)
	}
}

func TestLogisticRegression_OutputShapes(t *testing.T) {
	for _, tc := range []struct {
		name   string
		labels []float64
		cols   int
		rows   int
	}{{"binary", []float64{0, 1}, 1, 1}, {"three classes", []float64{0, 1, 2}, 3, 3}} {
		X, y := blobs(tc.labels...)
		m := NewLogisticRegression()
		if err := m.Fit(X, y); err != nil {
			t.Fatal(err)
		}
		if got := len(m.Coef()); got != tc.rows {
			t.Errorf("%s: %d coefficient rows, want %d", tc.name, got, tc.rows)
		}
		if got := len(m.Intercept()); got != tc.rows {
			t.Errorf("%s: %d intercepts, want %d", tc.name, got, tc.rows)
		}
		dec, err := m.DecisionFunction(X)
		if err != nil {
			t.Fatal(err)
		}
		if len(dec) != len(X) || len(dec[0]) != tc.cols {
			t.Errorf("%s: DecisionFunction is %dx%d, want %dx%d", tc.name, len(dec), len(dec[0]), len(X), tc.cols)
		}
		proba, err := m.PredictProba(X)
		if err != nil {
			t.Fatal(err)
		}
		for i, row := range proba {
			if len(row) != len(tc.labels) {
				t.Fatalf("%s: probability row has %d entries, want %d", tc.name, len(row), len(tc.labels))
			}
			var sum float64
			for _, p := range row {
				if p < 0 || p > 1 {
					t.Errorf("%s: probability %v out of [0, 1]", tc.name, p)
				}
				sum += p
			}
			if math.Abs(sum-1) > 1e-12 {
				t.Errorf("%s: row %d sums to %v", tc.name, i, sum)
			}
		}
	}
}

func TestLogisticRegression_StrongerRegularizationShrinks(t *testing.T) {
	X, y := blobs(0, 1, 2)
	prev := math.Inf(1)
	for _, c := range []float64{100, 1, 0.01, 0.0001} {
		m := NewLogisticRegression()
		m.C = c
		if err := m.Fit(X, y); err != nil {
			t.Fatal(err)
		}
		var n float64
		for _, row := range m.Coef() {
			n += dot(row, row)
		}
		if n > prev {
			t.Errorf("C=%v: ||coef||^2 = %v grew from %v", c, n, prev)
		}
		prev = n
	}
}

func TestLogisticRegression_MaxIterIsReportedNotFatal(t *testing.T) {
	X, y := blobs(0, 1, 2)
	m := NewLogisticRegression()
	m.C, m.MaxIter, m.Tol = 100, 1, 1e-12
	if err := m.Fit(X, y); err != nil {
		t.Fatalf("hitting MaxIter must not fail the fit: %v", err)
	}
	if m.Converged() || m.NIter() != 1 {
		t.Errorf("converged=%v after %d iterations, want not converged after 1", m.Converged(), m.NIter())
	}
	if _, err := m.Predict(X); err != nil {
		t.Error(err)
	}
}

func TestLogisticRegression_Errors(t *testing.T) {
	X, y := blobs(0, 1)
	oneClass := make([]float64, len(y))
	nan := math.NaN()
	cases := map[string]func() error{
		"single class":    func() error { return NewLogisticRegression().Fit(X, oneClass) },
		"zero C":          func() error { m := NewLogisticRegression(); m.C = 0; return m.Fit(X, y) },
		"negative C":      func() error { m := NewLogisticRegression(); m.C = -1; return m.Fit(X, y) },
		"NaN C":           func() error { m := NewLogisticRegression(); m.C = nan; return m.Fit(X, y) },
		"zero MaxIter":    func() error { m := NewLogisticRegression(); m.MaxIter = 0; return m.Fit(X, y) },
		"zero Tol":        func() error { m := NewLogisticRegression(); m.Tol = 0; return m.Fit(X, y) },
		"NaN Tol":         func() error { m := NewLogisticRegression(); m.Tol = nan; return m.Fit(X, y) },
		"empty":           func() error { return NewLogisticRegression().Fit(nil, nil) },
		"length mismatch": func() error { return NewLogisticRegression().Fit(X, y[:3]) },
	}
	for name, run := range cases {
		if err := run(); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}

	m := NewLogisticRegression()
	if _, err := m.Predict(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Predict before Fit = %v, want ErrNotFitted", err)
	}
	if _, err := m.PredictProba(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("PredictProba before Fit = %v, want ErrNotFitted", err)
	}
	if _, err := m.DecisionFunction(X); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("DecisionFunction before Fit = %v, want ErrNotFitted", err)
	}
	if m.Coef() != nil || m.Intercept() != nil || len(m.Classes()) != 0 {
		t.Error("accessors should be empty before Fit")
	}
	if err := m.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Predict([][]float64{{1, 2, 3}}); !errors.Is(err, matutil.ErrDimMismatch) {
		t.Errorf("Predict with 3 features = %v, want ErrDimMismatch", err)
	}
	if _, err := m.Score(X, y[:3]); !errors.Is(err, matutil.ErrDimMismatch) {
		t.Errorf("Score with mismatched y = %v, want ErrDimMismatch", err)
	}
}

func TestLogisticRegression_AccessorsReturnCopies(t *testing.T) {
	X, y := blobs(0, 1)
	m := NewLogisticRegression()
	if err := m.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	m.Coef()[0][0] = 1e9
	m.Intercept()[0] = 1e9
	m.Classes()[0] = 1e9
	if m.Coef()[0][0] == 1e9 || m.Intercept()[0] == 1e9 || m.Classes()[0] == 1e9 {
		t.Error("modifying an accessor's result changed the model")
	}
}

// --- persistence ------------------------------------------------------------

func TestLinearModels_SaveLoadRoundTrip(t *testing.T) {
	X, y := sparseData()
	Xc, yc := blobs(0, 1, 2)
	dir := t.TempDir()

	r := NewRidge()
	r.Alpha = 2.5
	l := NewLasso()
	l.Alpha = 0.05
	e := NewElasticNet()
	e.Alpha, e.L1Ratio = 0.05, 0.3
	for _, m := range []regressor{r, l, e} {
		if err := m.Fit(X, y); err != nil {
			t.Fatal(err)
		}
	}
	lg := NewLogisticRegression()
	lg.C = 0.5
	if err := lg.Fit(Xc, yc); err != nil {
		t.Fatal(err)
	}

	rl := mustLoad(t, dir, "ridge", r.Save, func(p string) (regressor, error) { return LoadRidge(p) })
	ll := mustLoad(t, dir, "lasso", l.Save, func(p string) (regressor, error) { return LoadLasso(p) })
	el := mustLoad(t, dir, "enet", e.Save, func(p string) (regressor, error) { return LoadElasticNet(p) })
	for name, pair := range map[string][2]regressor{"ridge": {r, rl}, "lasso": {l, ll}, "elastic_net": {e, el}} {
		want, _ := pair[0].Predict(X)
		got, err := pair[1].Predict(X)
		if err != nil || maxAbsDiff(want, got) != 0 {
			t.Errorf("%s: loaded model predicts differently (err %v)", name, err)
		}
	}
	if rl.(*Ridge).Alpha != 2.5 || ll.(*Lasso).NIter() != l.NIter() || ll.(*Lasso).Converged() != l.Converged() ||
		el.(*ElasticNet).L1Ratio != 0.3 {
		t.Error("hyperparameters or convergence info were not restored")
	}

	path := filepath.Join(dir, "logistic.gob")
	if err := lg.Save(path); err != nil {
		t.Fatal(err)
	}
	got, err := LoadLogisticRegression(path)
	if err != nil {
		t.Fatal(err)
	}
	wantP, _ := lg.PredictProba(Xc)
	gotP, err := got.PredictProba(Xc)
	if err != nil {
		t.Fatal(err)
	}
	for i := range wantP {
		if maxAbsDiff(wantP[i], gotP[i]) != 0 {
			t.Fatalf("loaded logistic model gives different probabilities at row %d", i)
		}
	}
	if got.C != 0.5 || got.NIter() != lg.NIter() || len(got.Classes()) != 3 {
		t.Error("logistic hyperparameters or metadata were not restored")
	}
}

func mustLoad(t *testing.T, dir, name string, save func(string) error, load func(string) (regressor, error)) regressor {
	t.Helper()
	path := filepath.Join(dir, name+".gob")
	if err := save(path); err != nil {
		t.Fatal(err)
	}
	m, err := load(path)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return m
}

func TestLinearModels_SaveBeforeFit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.gob")
	for name, save := range map[string]func(string) error{
		"ridge": NewRidge().Save, "lasso": NewLasso().Save, "elastic_net": NewElasticNet().Save, "logistic": NewLogisticRegression().Save,
	} {
		if err := save(path); !errors.Is(err, matutil.ErrNotFitted) {
			t.Errorf("%s: Save before Fit = %v, want ErrNotFitted", name, err)
		}
	}
}

func TestLoad_RejectsAFileOfTheWrongKind(t *testing.T) {
	X, y := sparseData()
	dir := t.TempDir()
	l := NewLasso()
	l.Alpha = 0.05
	r := NewRidge()
	for _, m := range []regressor{l, r} {
		if err := m.Fit(X, y); err != nil {
			t.Fatal(err)
		}
	}
	lassoPath, ridgePath := filepath.Join(dir, "lasso.gob"), filepath.Join(dir, "ridge.gob")
	if err := l.Save(lassoPath); err != nil {
		t.Fatal(err)
	}
	if err := r.Save(ridgePath); err != nil {
		t.Fatal(err)
	}
	// Gob matches fields by name, so without the Kind check these would silently load.
	if _, err := LoadElasticNet(lassoPath); err == nil {
		t.Error("a Lasso file loaded as an ElasticNet")
	}
	if _, err := LoadLasso(ridgePath); err == nil {
		t.Error("a Ridge file loaded as a Lasso")
	}
	if _, err := LoadRidge(lassoPath); err == nil {
		t.Error("a Lasso file loaded as a Ridge")
	}
	if _, err := LoadLogisticRegression(ridgePath); err == nil {
		t.Error("a Ridge file loaded as a LogisticRegression")
	}
	if _, err := LoadRidge(filepath.Join(dir, "missing.gob")); err == nil {
		t.Error("loading a missing file should fail")
	}
}

func TestLoadRidgeAndCD_CorruptPayload(t *testing.T) {
	X, y := sparseData()
	dir := t.TempDir()
	r := NewRidge()
	l := NewLasso()
	l.Alpha = 0.05
	for _, m := range []regressor{r, l} {
		if err := m.Fit(X, y); err != nil {
			t.Fatal(err)
		}
	}
	ridgePath, lassoPath := filepath.Join(dir, "r.gob"), filepath.Join(dir, "l.gob")
	if err := r.Save(ridgePath); err != nil {
		t.Fatal(err)
	}
	if err := l.Save(lassoPath); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadRidge(ridgePath); err != nil {
		t.Fatalf("the unmodified file must load: %v", err)
	}
	nan := math.NaN()
	ridgeCases := map[string]func(*ridgeGob){
		"wrong version":      func(g *ridgeGob) { g.Version = 99 },
		"no features":        func(g *ridgeGob) { g.NFeatures = 0 },
		"short coefficients": func(g *ridgeGob) { g.Coef = g.Coef[:2] },
		"NaN coefficient":    func(g *ridgeGob) { g.Coef[0] = nan },
		"NaN intercept":      func(g *ridgeGob) { g.Intercept = nan },
		"negative alpha":     func(g *ridgeGob) { g.Alpha = -1 },
	}
	for name, mutate := range ridgeCases {
		t.Run("ridge/"+name, func(t *testing.T) {
			if _, err := LoadRidge(mutatePayload(t, ridgePath, mutate)); err == nil {
				t.Error("expected an error for a corrupt payload")
			}
		})
	}
	cdCases := map[string]func(*elasticNetGob){
		"wrong version":  func(g *elasticNetGob) { g.Version = 99 },
		"short coef":     func(g *elasticNetGob) { g.Coef = nil },
		"bad l1 ratio":   func(g *elasticNetGob) { g.L1Ratio = 2 },
		"zero max iter":  func(g *elasticNetGob) { g.MaxIter = 0 },
		"negative alpha": func(g *elasticNetGob) { g.Alpha = -3 },
		"Inf coef":       func(g *elasticNetGob) { g.Coef[1] = math.Inf(1) },
	}
	for name, mutate := range cdCases {
		t.Run("lasso/"+name, func(t *testing.T) {
			if _, err := LoadLasso(mutatePayload(t, lassoPath, mutate)); err == nil {
				t.Error("expected an error for a corrupt payload")
			}
		})
	}
}

func TestLoadLogisticRegression_CorruptPayload(t *testing.T) {
	X, y := blobs(0, 1, 2)
	m := NewLogisticRegression()
	if err := m.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	valid := filepath.Join(t.TempDir(), "lg.gob")
	if err := m.Save(valid); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadLogisticRegression(valid); err != nil {
		t.Fatalf("the unmodified file must load: %v", err)
	}
	cases := map[string]func(*logisticGob){
		"wrong version":      func(g *logisticGob) { g.Version = 99 },
		"one class":          func(g *logisticGob) { g.Classes = g.Classes[:1] },
		"unsorted classes":   func(g *logisticGob) { g.Classes[0], g.Classes[1] = g.Classes[1], g.Classes[0] },
		"duplicate classes":  func(g *logisticGob) { g.Classes[1] = g.Classes[0] },
		"missing coef row":   func(g *logisticGob) { g.Coef = g.Coef[:2] },
		"short coef row":     func(g *logisticGob) { g.Coef[0] = g.Coef[0][:1] },
		"missing intercepts": func(g *logisticGob) { g.Intercept = g.Intercept[:1] },
		"NaN coefficient":    func(g *logisticGob) { g.Coef[1][0] = math.NaN() },
		"NaN intercept":      func(g *logisticGob) { g.Intercept[0] = math.NaN() },
		"zero C":             func(g *logisticGob) { g.C = 0 },
		"no features":        func(g *logisticGob) { g.NFeatures = 0 },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := LoadLogisticRegression(mutatePayload(t, valid, mutate)); err == nil {
				t.Error("expected an error for a corrupt payload")
			}
		})
	}
}

// --- lbfgs ------------------------------------------------------------------

func TestLBFGS_IllConditionedQuadratic(t *testing.T) {
	// f(x) = 1/2 sum_i c_i (x_i - i)^2 with condition number 1e4.
	c := []float64{1, 10, 100, 1000, 10000}
	fg := func(x, g []float64) float64 {
		var f float64
		for i := range x {
			d := x[i] - float64(i)
			f += 0.5 * c[i] * d * d
			g[i] = c[i] * d
		}
		return f
	}
	x, iters, ok := lbfgs(fg, make([]float64, 5), 1e-9, 200)
	if !ok {
		t.Fatalf("did not converge in %d iterations", iters)
	}
	for i, v := range x {
		if math.Abs(v-float64(i)) > 1e-9 {
			t.Errorf("x[%d] = %v, want %d", i, v, i)
		}
	}
}

func TestLBFGS_Rosenbrock(t *testing.T) {
	fg := func(x, g []float64) float64 {
		a, b := 1-x[0], x[1]-x[0]*x[0]
		g[0] = -2*a - 400*x[0]*b
		g[1] = 200 * b
		return a*a + 100*b*b
	}
	x, iters, ok := lbfgs(fg, []float64{-1.2, 1}, 1e-8, 500)
	if !ok {
		t.Fatalf("did not converge in %d iterations (x = %v)", iters, x)
	}
	if math.Abs(x[0]-1) > 1e-6 || math.Abs(x[1]-1) > 1e-6 {
		t.Errorf("minimum at %v, want (1, 1)", x)
	}
}

func TestLBFGS_StoppingConditions(t *testing.T) {
	quad := func(x, g []float64) float64 {
		g[0] = 2 * (x[0] - 3)
		return (x[0] - 3) * (x[0] - 3)
	}
	// Already at the minimum: zero iterations, converged.
	if x, iters, ok := lbfgs(quad, []float64{3}, 1e-9, 10); !ok || iters != 0 || x[0] != 3 {
		t.Errorf("at the optimum: x=%v iters=%d converged=%v", x, iters, ok)
	}
	// An iteration cap is honored and reported.
	rosen := func(x, g []float64) float64 {
		a, b := 1-x[0], x[1]-x[0]*x[0]
		g[0] = -2*a - 400*x[0]*b
		g[1] = 200 * b
		return a*a + 100*b*b
	}
	if _, iters, ok := lbfgs(rosen, []float64{-1.2, 1}, 1e-12, 3); ok || iters != 3 {
		t.Errorf("maxIter=3: iters=%d converged=%v, want 3 and false", iters, ok)
	}
	// A non-finite start is reported, not looped on.
	bad := func(x, g []float64) float64 { g[0] = 0; return math.NaN() }
	if _, iters, ok := lbfgs(bad, []float64{0}, 1e-9, 10); ok || iters != 0 {
		t.Errorf("NaN objective: iters=%d converged=%v", iters, ok)
	}
}
