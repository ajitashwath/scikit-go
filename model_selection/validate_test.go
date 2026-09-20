package model_selection

import (
	"errors"
	"math"
	"testing"

	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/linear"
	"github.com/ajitashwath/scikit-go/metrics"
	"github.com/ajitashwath/scikit-go/neighbors"
)

func knnFactory() (Model, error)    { return neighbors.NewKNeighborsClassifier(), nil }
func linearFactory() (Model, error) { return linear.NewLinearRegression(), nil }

func assertClose(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d values, want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > tol {
			t.Errorf("%s[%d] = %.12f, want %.12f", name, i, got[i], want[i])
		}
	}
}

func TestCrossValScore_AgainstSklearn(t *testing.T) {
	factories := map[string]ModelFactory{
		"knn_kfold":      knnFactory,
		"knn_stratified": knnFactory,
		"linear_kfold":   linearFactory,
	}
	for name, tc := range loadFixtures(t).CrossVal {
		got, err := CrossValScore(factories[name], tc.X, tc.Y, tc.CV.splitter(), nil)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		assertClose(t, name, got, tc.Scores, 1e-9)
	}
}

func TestCrossValScore_ExplicitScorerMatchesDefault(t *testing.T) {
	tc := loadFixtures(t).CrossVal["knn_kfold"]
	got, err := CrossValScore(knnFactory, tc.X, tc.Y, tc.CV.splitter(), metrics.AccuracyScore)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "accuracy scorer", got, tc.Scores, 1e-12)
}

func TestCrossValScore_CustomScorer(t *testing.T) {
	tc := loadFixtures(t).CrossVal["linear_kfold"]
	rmse := func(yTrue, yPred []float64) (float64, error) {
		v, err := metrics.RootMeanSquaredError(yTrue, yPred)
		return -v, err // larger is better, so negate the error
	}
	got, err := CrossValScore(linearFactory, tc.X, tc.Y, tc.CV.splitter(), rmse)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v >= 0 || math.IsNaN(v) {
			t.Errorf("fold %d: negated RMSE = %v, want a negative number", i, v)
		}
	}
}

func TestCrossValScore_NilCVIsFiveFold(t *testing.T) {
	tc := loadFixtures(t).CrossVal["knn_kfold"] // fixture is KFold(5)
	got, err := CrossValScore(knnFactory, tc.X, tc.Y, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "default cv", got, tc.Scores, 1e-9)
}

// noScoreModel is a valid Model with no Score method.
type noScoreModel struct{ mean float64 }

func (m *noScoreModel) Fit(X [][]float64, y []float64) error {
	for _, v := range y {
		m.mean += v / float64(len(y))
	}
	return nil
}
func (m *noScoreModel) Predict(X [][]float64) ([]float64, error) {
	out := make([]float64, len(X))
	for i := range out {
		out[i] = m.mean
	}
	return out, nil
}

func TestCrossValScore_NoScorerNoScoreMethod(t *testing.T) {
	X, y := dummyX(20), make([]float64, 20)
	factory := func() (Model, error) { return &noScoreModel{}, nil }

	if _, err := CrossValScore(factory, X, y, NewKFold(4), nil); !errors.Is(err, ErrInvalidParams) {
		t.Errorf("got %v, want ErrInvalidParams when there is neither a scorer nor a Score method", err)
	}
	// With an explicit scorer the same model is fine.
	got, err := CrossValScore(factory, X, y, NewKFold(4), metrics.MeanSquaredError)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 4 {
		t.Errorf("got %d scores, want 4", len(got))
	}
}

func TestCrossValScore_Errors(t *testing.T) {
	X, y := dummyX(20), make([]float64, 20)
	boom := errors.New("boom")

	cases := []struct {
		name string
		run  func() error
		want error
	}{
		{"nil factory", func() error { _, err := CrossValScore(nil, X, y, nil, nil); return err }, ErrInvalidParams},
		{"factory error", func() error {
			_, err := CrossValScore(func() (Model, error) { return nil, boom }, X, y, nil, nil)
			return err
		}, boom},
		{"factory returns nil model", func() error {
			_, err := CrossValScore(func() (Model, error) { return nil, nil }, X, y, nil, nil)
			return err
		}, ErrInvalidParams},
		{"empty data", func() error { _, err := CrossValScore(knnFactory, nil, nil, nil, nil); return err }, matutil.ErrEmptyInput},
		{"length mismatch", func() error { _, err := CrossValScore(knnFactory, X, y[:5], nil, nil); return err }, matutil.ErrDimMismatch},
		{"too many folds", func() error { _, err := CrossValScore(knnFactory, X, y, NewKFold(30), nil); return err }, ErrInvalidParams},
		{"scorer error", func() error {
			_, err := CrossValScore(linearFactory, X, y, NewKFold(2), func(a, b []float64) (float64, error) { return 0, boom })
			return err
		}, boom},
	}
	for _, tc := range cases {
		if err := tc.run(); !errors.Is(err, tc.want) {
			t.Errorf("%s: got %v, want %v", tc.name, err, tc.want)
		}
	}
}

// A model that rewrites its inputs must not corrupt later folds or the caller's data.
type vandalModel struct{ noScoreModel }

func (m *vandalModel) Fit(X [][]float64, y []float64) error {
	for _, row := range X {
		for j := range row {
			row[j] = -999
		}
	}
	for i := range y {
		y[i] = -999
	}
	return m.noScoreModel.Fit(X, y)
}

func TestCrossValScore_FoldsAreIsolatedFromModelMutation(t *testing.T) {
	X := [][]float64{{1}, {2}, {3}, {4}, {5}, {6}}
	y := []float64{1, 2, 3, 4, 5, 6}
	Xcopy := [][]float64{{1}, {2}, {3}, {4}, {5}, {6}}
	ycopy := append([]float64(nil), y...)

	seen := 0.0
	scorer := func(yTrue, yPred []float64) (float64, error) {
		for _, v := range yTrue {
			if v == -999 {
				t.Error("a test fold contained data another fold's model had overwritten")
			}
		}
		seen++
		return 0, nil
	}
	_, err := CrossValScore(func() (Model, error) { return &vandalModel{}, nil }, X, y, NewKFold(3), scorer)
	if err != nil {
		t.Fatal(err)
	}
	if seen != 3 {
		t.Errorf("scorer ran %v times, want 3", seen)
	}
	for i := range X {
		if X[i][0] != Xcopy[i][0] || y[i] != ycopy[i] {
			t.Fatalf("the caller's data was modified at row %d", i)
		}
	}
}
