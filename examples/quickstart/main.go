// Command quickstart is a runnable tour of scikit-go: it loads the iris and
// diabetes datasets and fits a model from every package, printing a score for
// each. It exits non-zero if anything fails or a model scores implausibly low,
// so it doubles as an end-to-end smoke test:
//
//	go run ./examples/quickstart
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/ajitashwath/scikit-go/cluster"
	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/decomposition"
	"github.com/ajitashwath/scikit-go/ensemble"
	"github.com/ajitashwath/scikit-go/feature_selection"
	"github.com/ajitashwath/scikit-go/linear"
	"github.com/ajitashwath/scikit-go/metrics"
	"github.com/ajitashwath/scikit-go/neighbors"
	"github.com/ajitashwath/scikit-go/pipeline"
	"github.com/ajitashwath/scikit-go/preprocessing"
	"github.com/ajitashwath/scikit-go/svm"
	"github.com/ajitashwath/scikit-go/tree"
	"github.com/ajitashwath/scikit-go/utils"
)

var failed bool

// check prints a labeled score and records a failure when it falls below min.
func check(name string, score, min float64) {
	status := "ok"
	if score < min {
		status = fmt.Sprintf("FAIL (want >= %.2f)", min)
		failed = true
	}
	fmt.Printf("  %-34s %7.4f  %s\n", name, score, status)
}

func must[T any](v T, err error) T {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	return v
}

func ok(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func main() {
	classification()
	regression()
	clustering()
	persistence()

	if failed {
		fmt.Println("\nsome models scored below their expected floor")
		os.Exit(1)
	}
	fmt.Println("\nall models ran and scored as expected")
}

func classification() {
	fmt.Println("classification on iris (accuracy on a held-out 30%):")
	X, y, err := datasets.LoadIris()
	ok(err)
	Xtr, Xte, ytr, yte, err := utils.TrainTestSplit(X, y, 0.3, 42)
	ok(err)

	knn := neighbors.NewKNeighborsClassifier()
	ok(knn.Fit(Xtr, ytr))
	check("KNeighborsClassifier", must(knn.Score(Xte, yte)), 0.9)

	dt := tree.NewDecisionTreeClassifier()
	ok(dt.Fit(Xtr, ytr))
	check("DecisionTreeClassifier", must(dt.Score(Xte, yte)), 0.85)

	rf := ensemble.NewRandomForestClassifier()
	rf.NTrees = 100
	rf.Seed = 1
	ok(rf.Fit(Xtr, ytr))
	check("RandomForestClassifier", must(rf.Score(Xte, yte)), 0.9)

	svc := svm.NewSVC()
	svc.Probability = true
	ok(svc.Fit(Xtr, ytr))
	check("SVC (rbf, probability=true)", must(svc.Score(Xte, yte)), 0.9)

	// A pipeline: scale, keep the two most informative features, then classify.
	sel := feature_selection.NewSelectKBest()
	sel.K = 2
	p := must(pipeline.MakePipeline(preprocessing.NewStandardScaler(), sel, svm.NewSVC()))
	ok(p.Fit(Xtr, ytr))
	check("Pipeline(scaler, kbest, svc)", must(p.Score(Xte, yte)), 0.85)

	pred := must(rf.Predict(Xte))
	report := must(metrics.ClassificationReport(yte, pred))
	fmt.Printf("  random forest macro F1 %.4f, weighted F1 %.4f\n", report.MacroF1, report.WeightedF1)
}

func regression() {
	fmt.Println("regression on diabetes (R^2 on a held-out 30%):")
	X, y, err := datasets.LoadDiabetes()
	ok(err)
	Xtr, Xte, ytr, yte, err := utils.TrainTestSplit(X, y, 0.3, 42)
	ok(err)

	lr := linear.NewLinearRegression()
	ok(lr.Fit(Xtr, ytr))
	check("LinearRegression", must(lr.Score(Xte, yte)), 0.3)

	rf := ensemble.NewRandomForestRegressor()
	rf.NTrees = 100
	rf.Seed = 1
	ok(rf.Fit(Xtr, ytr))
	check("RandomForestRegressor", must(rf.Score(Xte, yte)), 0.3)

	p := must(pipeline.MakePipeline(preprocessing.NewStandardScaler(), svm.NewSVR()))
	ok(p.SetParams(map[string]any{"svr__C": 100.0, "svr__Kernel": "linear"}))
	ok(p.Fit(Xtr, ytr))
	check("Pipeline(scaler, svr linear)", must(p.Score(Xte, yte)), 0.3)

	rfe := feature_selection.NewRFE(linear.NewLinearRegression())
	rfe.NFeaturesToSelect = 5
	ok(rfe.Fit(Xtr, ytr))
	check("RFE(linear, 5 features) R^2", must(rfe.Estimator.(*linear.LinearRegression).Score(
		must(rfe.Transform(Xte)), yte)), 0.3)
}

func clustering() {
	fmt.Println("unsupervised on iris:")
	X, y, err := datasets.LoadIris()
	ok(err)

	pca := decomposition.NewPCA()
	pca.NComponents = 2
	Z := must(pca.FitTransform(X))
	ratio := pca.ExplainedVarianceRatio()
	check("PCA(2) explained variance", ratio[0]+ratio[1], 0.9)

	km := cluster.NewKMeans()
	km.NClusters = 3
	km.Seed = 1
	labels := must(km.FitPredict(Z))
	check("KMeans(3) adjusted Rand index", must(metrics.AdjustedRandIndex(y, labels)), 0.5)
	check("KMeans(3) silhouette", must(metrics.SilhouetteScore(Z, labels)), 0.4)
}

func persistence() {
	fmt.Println("save / load round trip:")
	dir, err := os.MkdirTemp("", "scikitgo-quickstart-*")
	ok(err)
	defer os.RemoveAll(dir)

	X, y, err := datasets.LoadIris()
	ok(err)
	p := must(pipeline.MakePipeline(preprocessing.NewStandardScaler(), ensemble.NewRandomForestClassifier()))
	ok(p.SetParams(map[string]any{"randomforestclassifier__NTrees": 20}))
	ok(p.Fit(X, y))

	path := filepath.Join(dir, "model.gob")
	ok(p.Save(path))
	loaded := must(pipeline.LoadPipeline(path))
	want, got := must(p.Predict(X)), must(loaded.Predict(X))
	same := 0.0
	for i := range want {
		if want[i] == got[i] {
			same++
		}
	}
	check("loaded pipeline agrees with original", same/float64(len(want)), 1.0)
}
