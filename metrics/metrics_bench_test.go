package metrics

import (
	"math/rand"
	"testing"
)

func BenchmarkRegressionMetrics_1e4(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	n := 10000
	yTrue := make([]float64, n)
	yPred := make([]float64, n)
	for i := range yTrue {
		yTrue[i] = r.NormFloat64()
		yPred[i] = r.NormFloat64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := R2Score(yTrue, yPred); err != nil {
			b.Fatalf("R2Score: %v", err)
		}
		if _, err := MeanSquaredError(yTrue, yPred); err != nil {
			b.Fatalf("MeanSquaredError: %v", err)
		}
	}
}

func BenchmarkConfusionMatrix_1e4(b *testing.B) {
	n := 10000
	yTrue := make([]float64, n)
	yPred := make([]float64, n)
	for i := range yTrue {
		yTrue[i] = float64(i % 3)
		yPred[i] = float64((i + 1) % 3)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ConfusionMatrix(yTrue, yPred); err != nil {
			b.Fatalf("ConfusionMatrix: %v", err)
		}
	}
}

func BenchmarkAdjustedRandIndex_1e4(b *testing.B) {
	n := 10000
	labelsTrue := make([]float64, n)
	labelsPred := make([]float64, n)
	for i := range labelsTrue {
		labelsTrue[i] = float64(i % 10)
		labelsPred[i] = float64(i % 11)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := AdjustedRandIndex(labelsTrue, labelsPred); err != nil {
			b.Fatalf("AdjustedRandIndex: %v", err)
		}
	}
}

func BenchmarkSilhouetteScore_1000x5(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	n, p := 1000, 5
	X := make([][]float64, n)
	labels := make([]float64, n)
	for i := range X {
		row := make([]float64, p)
		for j := range row {
			row[j] = r.NormFloat64()
		}
		X[i] = row
		labels[i] = float64(i % 5)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SilhouetteScore(X, labels); err != nil {
			b.Fatalf("SilhouetteScore: %v", err)
		}
	}
}
