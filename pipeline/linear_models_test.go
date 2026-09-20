package pipeline

import (
	"path/filepath"
	"testing"

	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/linear"
	"github.com/ajitashwath/scikit-go/preprocessing"
)

// Every regularized linear model can sit in a Pipeline and survive Save / LoadPipeline.
func TestPipeline_RegularizedRegressorsSaveLoad(t *testing.T) {
	X, y, err := datasets.MakeRegression(120, 6, 5.0, 3)
	if err != nil {
		t.Fatal(err)
	}
	lasso := linear.NewLasso()
	lasso.Alpha = 0.5
	enet := linear.NewElasticNet()
	enet.Alpha = 0.5
	for name, est := range map[string]any{"ridge": linear.NewRidge(), "lasso": lasso, "elastic_net": enet} {
		p, err := MakePipeline(preprocessing.NewStandardScaler(), est)
		if err != nil {
			t.Fatal(err)
		}
		if err := p.Fit(X, y); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		want, err := p.Predict(X)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(t.TempDir(), name+".gob")
		if err := p.Save(path); err != nil {
			t.Fatalf("%s: Save: %v", name, err)
		}
		loaded, err := LoadPipeline(path)
		if err != nil {
			t.Fatalf("%s: LoadPipeline: %v", name, err)
		}
		got, err := loaded.Predict(X)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("%s: loaded pipeline predicts %v at row %d, original %v", name, got[i], i, want[i])
			}
		}
	}
}

func TestPipeline_LogisticRegressionSaveLoadAndProba(t *testing.T) {
	X, y, err := datasets.LoadIris()
	if err != nil {
		t.Fatal(err)
	}
	p, err := MakePipeline(preprocessing.NewStandardScaler(), linear.NewLogisticRegression())
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Fit(X, y); err != nil {
		t.Fatal(err)
	}
	if acc, err := p.Score(X, y); err != nil || acc < 0.9 {
		t.Fatalf("training accuracy %v (err %v), want >= 0.9", acc, err)
	}
	wantProba, err := p.PredictProba(X)
	if err != nil {
		t.Fatalf("a pipeline ending in LogisticRegression must expose PredictProba: %v", err)
	}
	path := filepath.Join(t.TempDir(), "logit.gob")
	if err := p.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadPipeline(path)
	if err != nil {
		t.Fatal(err)
	}
	gotProba, err := loaded.PredictProba(X)
	if err != nil {
		t.Fatal(err)
	}
	for i := range wantProba {
		for k := range wantProba[i] {
			if wantProba[i][k] != gotProba[i][k] {
				t.Fatalf("probability [%d][%d] changed across save/load", i, k)
			}
		}
	}
}

// Hyperparameters of the new estimators are reachable through SetParams.
func TestPipeline_SetParamsReachesRegularizedModels(t *testing.T) {
	p, err := MakePipeline(preprocessing.NewStandardScaler(), linear.NewLogisticRegression())
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetParams(map[string]any{"logisticregression__C": 0.25, "logisticregression__FitIntercept": false}); err != nil {
		t.Fatal(err)
	}
	lg, _ := p.Named("logisticregression")
	if m := lg.(*linear.LogisticRegression); m.C != 0.25 || m.FitIntercept {
		t.Errorf("C=%v FitIntercept=%v, want 0.25 and false", m.C, m.FitIntercept)
	}
	r, err := MakePipeline(linear.NewRidge())
	if err != nil {
		t.Fatal(err)
	}
	if err := r.SetParams(map[string]any{"ridge__Alpha": 7}); err != nil { // an int converts to the float field
		t.Fatal(err)
	}
	est, _ := r.Named("ridge")
	if est.(*linear.Ridge).Alpha != 7 {
		t.Errorf("alpha = %v, want 7", est.(*linear.Ridge).Alpha)
	}
}
