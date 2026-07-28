package linear

import (
	"math/rand"
	"testing"
)

// randomDataset generates a reproducible random regression dataset of the given shape
// Benchmark Fit/Predict at various scales.
func randomDataset(nSamples, nFeatures int, seed int64) ([][]float64, []float64) {
	r := rand.New(rand.NewSource(seed))
	X := make([][]float64, nFeatures)
	trueCoef := make([]float64, nFeatures)
	for j := range trueCoef {
		trueCoef[j] = rand.NormFloat64()
	}
	y := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		row := make([]float64, nFeatures)
		var target float64
		for j := 0; j < nFeatures; j++ {
			row[j] = r.NormFloat64()
			target += row[j] * trueCoef[j]
		}
		X[i] = row
		y[i] = target + r.NormFloat64()*0.1
	}
	return X, y
}

func benchmarkFit(b *testing.B, nSamples, nFeatures int) {
	X, y := randomDataset(nSamples, nFeatures, 42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model := NewLinearRegression()
		if err := model.Fit(X, y); err != nil {
			b.Fatalf("Fit failed: %w", err)
		}
	}
}

func BenchmarkFit_Small_100x10(b *testing.B)     { benchmarkFit(b, 100, 10) }
func BenchmarkFit_Medium_10000x50(b *testing.B)  { benchmarkFit(b, 10000, 50) }
func BenchmarkFit_Large_100000x100(b *testing.B) { benchmarkFit(b, 100000, 100) }

func benchmarkPredict(b *testing.B, nSamples, nFeatures int) {
	X, y := randomDataset(nSamples, nFeatures, 42)
	model := NewLinearRegression()
	if err := model.Fit(X, y); err != nil {
		b.Fatalf("Fit failed: %w", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Predict(X); err != nil {
			b.Fatalf("Predict failed: %w", err)
		}
	}
}

func BenchmarkPredict_Small_100x10(b *testing.B)     { benchmarkPredict(b, 100, 10) }
func BenchmarkPredict_Medium_10000x50(b *testing.B)  { benchmarkPredict(b, 10000, 50) }
func BenchmarkPredict_Large_100000x100(b *testing.B) { benchmarkPredict(b, 100000, 100) }
