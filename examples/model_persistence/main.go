// Command model_persistence trains a model once, saves it to disk, and loads it
// back to serve predictions, which is how a trained model moves from a training
// job to a service. It also shows that a damaged model file is rejected with an
// error instead of producing a model that misbehaves later.
//
//	go run ./examples/model_persistence            # uses a temp dir
//	go run ./examples/model_persistence -dir .     # keeps iris_model.gob in the current dir
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/ensemble"
	"github.com/ajitashwath/scikit-go/pipeline"
	"github.com/ajitashwath/scikit-go/preprocessing"
	"github.com/ajitashwath/scikit-go/utils"
)

var classNames = []string{"setosa", "versicolor", "virginica"}

func main() {
	dir := flag.String("dir", "", "directory to save the model in (default: a temporary directory)")
	flag.Parse()
	if *dir == "" {
		tmp, err := os.MkdirTemp("", "scikitgo-persistence-*")
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		defer os.RemoveAll(tmp)
		*dir = tmp
	}
	if _, err := run(os.Stdout, *dir); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// run returns "agreement" (fraction of test predictions the loaded model shares
// with the original, expected to be 1) and "corrupt_rejected" (1 if the damaged
// file was refused).
func run(w io.Writer, dir string) (map[string]float64, error) {
	X, y, err := datasets.LoadIris()
	if err != nil {
		return nil, err
	}
	Xtr, Xte, ytr, _, err := utils.TrainTestSplit(X, y, 0.3, 42)
	if err != nil {
		return nil, err
	}

	// --- training job ---------------------------------------------------------
	model, err := pipeline.MakePipeline(preprocessing.NewStandardScaler(), ensemble.NewRandomForestClassifier())
	if err != nil {
		return nil, err
	}
	if err := model.SetParams(map[string]any{"randomforestclassifier__NTrees": 50, "randomforestclassifier__Seed": 1}); err != nil {
		return nil, err
	}
	if err := model.Fit(Xtr, ytr); err != nil {
		return nil, err
	}
	path := filepath.Join(dir, "iris_model.gob")
	if err := model.Save(path); err != nil {
		return nil, err
	}
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(w, "trained on %d samples and saved %s (%d bytes)\n", len(Xtr), filepath.Base(path), info.Size())

	// --- serving process: no training data needed, only the file ------------
	loaded, err := pipeline.LoadPipeline(path)
	if err != nil {
		return nil, err
	}
	want, err := model.Predict(Xte)
	if err != nil {
		return nil, err
	}
	got, err := loaded.Predict(Xte)
	if err != nil {
		return nil, err
	}
	same := 0
	for i := range want {
		if want[i] == got[i] {
			same++
		}
	}
	agreement := float64(same) / float64(len(want))
	fmt.Fprintf(w, "loaded model agrees with the original on %d of %d test samples\n", same, len(want))

	sample := Xte[:1]
	proba, err := loaded.PredictProba(sample)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(w, "\nnew measurement %v\n", sample[0])
	for c, p := range proba[0] {
		fmt.Fprintf(w, "  P(%-10s) = %.3f\n", classNames[c], p)
	}

	// --- a damaged file is an error, not a silent failure --------------------
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	bad := filepath.Join(dir, "truncated.gob")
	if err := os.WriteFile(bad, data[:len(data)/2], 0o600); err != nil {
		return nil, err
	}
	rejected := 0.0
	if _, err := pipeline.LoadPipeline(bad); err != nil {
		rejected = 1
		fmt.Fprintf(w, "\nloading a truncated copy fails as it should:\n  %v\n", err)
	} else {
		fmt.Fprintln(w, "\nBUG: a truncated model file loaded without error")
	}
	return map[string]float64{"agreement": agreement, "corrupt_rejected": rejected}, nil
}
