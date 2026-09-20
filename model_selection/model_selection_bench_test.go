package model_selection

import (
	"testing"

	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/neighbors"
)

func benchData(b *testing.B, n int) ([][]float64, []float64) {
	b.Helper()
	X, y, err := datasets.MakeClassification(n, 8, 3, 1)
	if err != nil {
		b.Fatal(err)
	}
	return X, y
}

func BenchmarkKFoldSplit_10000(b *testing.B) {
	X, y := benchData(b, 10000)
	kf := &KFold{NSplits: 5, Shuffle: true, Seed: 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := kf.Split(X, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStratifiedKFoldSplit_10000(b *testing.B) {
	X, y := benchData(b, 10000)
	skf := &StratifiedKFold{NSplits: 5, Shuffle: true, Seed: 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := skf.Split(X, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCrossValScore_KNN_600x8(b *testing.B) {
	X, y := benchData(b, 600)
	factory := func() (Model, error) { return neighbors.NewKNeighborsClassifier(), nil }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := CrossValScore(factory, X, y, NewStratifiedKFold(5), nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGridSearchCV_KNN_300x8(b *testing.B) {
	X, y := benchData(b, 300)
	grid := ParamGrid{"n_neighbors": {1, 3, 5, 7}, "weights": {"uniform", "distance"}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g := NewGridSearchCV(knnClassifierBuilder, grid)
		if err := g.Fit(X, y); err != nil {
			b.Fatal(err)
		}
	}
}
