// Command pca_dimensionality shows PCA as a way to shrink a dataset: how much
// variance each component keeps, how much is lost when reconstructing from a
// few components, and whether a classifier still works in the smaller space.
//
//	go run ./examples/pca_dimensionality
package main

import (
	"fmt"
	"io"
	"math"
	"os"

	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/decomposition"
	"github.com/ajitashwath/scikit-go/metrics"
	"github.com/ajitashwath/scikit-go/neighbors"
	"github.com/ajitashwath/scikit-go/preprocessing"
	"github.com/ajitashwath/scikit-go/utils"
)

func main() {
	if _, err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// run returns "variance_kept_2" (variance kept by two components),
// "reconstruction_rmse_2", "accuracy_4d" and "accuracy_2d".
func run(w io.Writer) (map[string]float64, error) {
	X, y, err := datasets.LoadIris()
	if err != nil {
		return nil, err
	}
	scaler := preprocessing.NewStandardScaler()
	Z, err := scaler.FitTransform(X)
	if err != nil {
		return nil, err
	}

	// Fit with every component first, just to see how the variance is spread.
	full := decomposition.NewPCA()
	full.NComponents = len(X[0])
	if _, err := full.FitTransform(Z); err != nil {
		return nil, err
	}
	fmt.Fprintln(w, "variance explained by each principal component of the scaled iris data:")
	cum := 0.0
	for i, r := range full.ExplainedVarianceRatio() {
		cum += r
		fmt.Fprintf(w, "  PC%d  %6.2f%%   cumulative %6.2f%%\n", i+1, 100*r, 100*cum)
	}

	// Keep two components, then map back to the original space to see what was lost.
	pca := decomposition.NewPCA()
	pca.NComponents = 2
	proj, err := pca.FitTransform(Z)
	if err != nil {
		return nil, err
	}
	kept := pca.ExplainedVarianceRatio()[0] + pca.ExplainedVarianceRatio()[1]
	back, err := pca.InverseTransform(proj)
	if err != nil {
		return nil, err
	}
	sq, n := 0.0, 0
	for i := range Z {
		for j := range Z[i] {
			d := Z[i][j] - back[i][j]
			sq += d * d
			n++
		}
	}
	rmse := math.Sqrt(sq / float64(n))
	fmt.Fprintf(w, "\n2 components keep %.2f%% of the variance; reconstruction RMSE %.4f (in standardized units)\n", 100*kept, rmse)

	// Does a classifier lose much by working in 2 dimensions instead of 4?
	acc4, err := knnAccuracy(Z, y)
	if err != nil {
		return nil, err
	}
	acc2, err := knnAccuracy(proj, y)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(w, "\nk-nearest-neighbors accuracy on a held-out 30%%:\n  all 4 features   %.4f\n  2 PCA components %.4f\n", acc4, acc2)

	return map[string]float64{
		"variance_kept_2":       kept,
		"reconstruction_rmse_2": rmse,
		"accuracy_4d":           acc4,
		"accuracy_2d":           acc2,
	}, nil
}

func knnAccuracy(X [][]float64, y []float64) (float64, error) {
	Xtr, Xte, ytr, yte, err := utils.TrainTestSplit(X, y, 0.3, 42)
	if err != nil {
		return 0, err
	}
	knn := neighbors.NewKNeighborsClassifier()
	if err := knn.Fit(Xtr, ytr); err != nil {
		return 0, err
	}
	pred, err := knn.Predict(Xte)
	if err != nil {
		return 0, err
	}
	return metrics.AccuracyScore(yte, pred)
}
