package preprocessing

import (
	"math/rand"
	"testing"
)

// randomMatrix generates a reproducible random matrix of the given shape.
func randomMatrix(nSamples, nFeatures int, seed int64) [][]float64 {
	r := rand.New(rand.NewSource(seed))
	X := make([][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		row := make([]float64, nFeatures)
		for j := range row {
			row[j] = r.NormFloat64()
		}
		X[i] = row
	}
	return X
}

func benchmarkFit(b *testing.B, nSamples, nFeatures int) {
	X := randomMatrix(nSamples, nFeatures, 42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scaler := NewStandardScaler()
		if err := scaler.Fit(X); err != nil {
			b.Fatalf("Fit failed: %v", err)
		}
	}
}

func BenchmarkFit_Small_100x10(b *testing.B)     { benchmarkFit(b, 100, 10) }
func BenchmarkFit_Medium_10000x50(b *testing.B)  { benchmarkFit(b, 10000, 50) }
func BenchmarkFit_Large_100000x100(b *testing.B) { benchmarkFit(b, 100000, 100) }

func benchmarkTransform(b *testing.B, nSamples, nFeatures int) {
	X := randomMatrix(nSamples, nFeatures, 42)
	scaler := NewStandardScaler()
	if err := scaler.Fit(X); err != nil {
		b.Fatalf("Fit failed: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := scaler.Transform(X); err != nil {
			b.Fatalf("Transform failed: %v", err)
		}
	}
}

func BenchmarkTransform_Small_100x10(b *testing.B)     { benchmarkTransform(b, 100, 10) }
func BenchmarkTransform_Medium_10000x50(b *testing.B)  { benchmarkTransform(b, 10000, 50) }
func BenchmarkTransform_Large_100000x100(b *testing.B) { benchmarkTransform(b, 100000, 100) }
