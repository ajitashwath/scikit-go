// Command moons_nonlinear shows why nonlinear models exist. Two interleaving
// half-moons cannot be separated by a straight line, so a linear-kernel SVM
// stalls while an RBF-kernel SVM, k-nearest-neighbors and a random forest cope.
// It ends with a text plot of the RBF SVM's decision regions.
//
//	go run ./examples/moons_nonlinear
package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/ensemble"
	"github.com/ajitashwath/scikit-go/metrics"
	"github.com/ajitashwath/scikit-go/neighbors"
	"github.com/ajitashwath/scikit-go/svm"
	"github.com/ajitashwath/scikit-go/utils"
)

type classifier interface {
	core.Estimator
	core.Predictor
}

func main() {
	if _, err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// run returns each model's held-out accuracy, keyed by model name.
func run(w io.Writer) (map[string]float64, error) {
	X, y, err := datasets.MakeMoons(500, 0.25, 3)
	if err != nil {
		return nil, err
	}
	Xtr, Xte, ytr, yte, err := utils.TrainTestSplit(X, y, 0.3, 42)
	if err != nil {
		return nil, err
	}

	linearSVC := svm.NewSVC()
	linearSVC.Kernel = svm.KernelLinear
	rbfSVC := svm.NewSVC() // the default kernel is RBF
	forest := ensemble.NewRandomForestClassifier()
	forest.NTrees = 100
	forest.Seed = 1

	models := []struct {
		name  string
		model classifier
	}{
		{"SVC (linear kernel)", linearSVC},
		{"SVC (rbf kernel)", rbfSVC},
		{"KNeighborsClassifier", neighbors.NewKNeighborsClassifier()},
		{"RandomForestClassifier", forest},
	}

	fmt.Fprintf(w, "two moons, noise 0.25: %d training / %d test samples\n\n", len(Xtr), len(Xte))
	scores := make(map[string]float64, len(models))
	for _, m := range models {
		if err := m.model.Fit(Xtr, ytr); err != nil {
			return nil, fmt.Errorf("%s: %w", m.name, err)
		}
		pred, err := m.model.Predict(Xte)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", m.name, err)
		}
		acc, err := metrics.AccuracyScore(yte, pred)
		if err != nil {
			return nil, err
		}
		scores[m.name] = acc
		fmt.Fprintf(w, "  %-24s accuracy %.4f\n", m.name, acc)
	}

	fmt.Fprintln(w, "\nRBF SVM decision regions ('.' = class 0, '+' = class 1; letters mark test points: o = class 0, x = class 1)")
	if err := plotRegions(w, rbfSVC, Xte, yte); err != nil {
		return nil, err
	}
	return scores, nil
}

// plotRegions predicts a grid of points with clf and prints it as text, with the
// test points drawn on top.
func plotRegions(w io.Writer, clf core.Predictor, Xte [][]float64, yte []float64) error {
	const (
		cols, rows     = 64, 22
		xMin, xMax     = -1.6, 2.6
		yMin, yMax     = -1.1, 1.6
		region0, reg1  = '.', '+'
		point0, point1 = 'o', 'x'
	)
	grid := make([][]float64, 0, cols*rows)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			grid = append(grid, []float64{
				xMin + (xMax-xMin)*(float64(c)+0.5)/cols,
				yMax - (yMax-yMin)*(float64(r)+0.5)/rows,
			})
		}
	}
	pred, err := clf.Predict(grid)
	if err != nil {
		return err
	}
	canvas := make([][]rune, rows)
	for r := range canvas {
		canvas[r] = make([]rune, cols)
		for c := range canvas[r] {
			if pred[r*cols+c] == 0 {
				canvas[r][c] = region0
			} else {
				canvas[r][c] = reg1
			}
		}
	}
	for i, p := range Xte {
		c := int((p[0] - xMin) / (xMax - xMin) * cols)
		r := int((yMax - p[1]) / (yMax - yMin) * rows)
		if c < 0 || c >= cols || r < 0 || r >= rows {
			continue
		}
		if yte[i] == 0 {
			canvas[r][c] = point0
		} else {
			canvas[r][c] = point1
		}
	}
	for _, line := range canvas {
		fmt.Fprintln(w, "  "+strings.TrimRight(string(line), " "))
	}
	return nil
}
