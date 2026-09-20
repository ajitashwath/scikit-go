package tree

import (
	"testing"

	"github.com/ajitashwath/scikit-go/datasets"
)

// benchmarkFit benchmarks Fit on a deterministic synthetic regression dataset.
func benchmarkFit(b *testing.B, nSamples, nFeatures int) {
	X, y, err := datasets.MakeRegression(nSamples, nFeatures, 0.1, 42)
	if err != nil {
		b.Fatalf("MakeRegression: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model := NewDecisionTreeRegressor()
		if err := model.Fit(X, y); err != nil {
			b.Fatalf("Fit failed: %v", err)
		}
	}
}

func BenchmarkFit_Small_100x10(b *testing.B)     { benchmarkFit(b, 100, 10) }
func BenchmarkFit_Medium_10000x50(b *testing.B)  { benchmarkFit(b, 10000, 50) }
func BenchmarkFit_Large_100000x100(b *testing.B) { benchmarkFit(b, 100000, 100) }

func benchmarkPredict(b *testing.B, nSamples, nFeatures int) {
	X, y, err := datasets.MakeRegression(nSamples, nFeatures, 0.1, 42)
	if err != nil {
		b.Fatalf("MakeRegression: %v", err)
	}
	model := NewDecisionTreeRegressor()
	if err := model.Fit(X, y); err != nil {
		b.Fatalf("Fit failed: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Predict(X); err != nil {
			b.Fatalf("Predict failed: %v", err)
		}
	}
}

func BenchmarkPredict_Small_100x10(b *testing.B)     { benchmarkPredict(b, 100, 10) }
func BenchmarkPredict_Medium_10000x50(b *testing.B)  { benchmarkPredict(b, 10000, 50) }
func BenchmarkPredict_Large_100000x100(b *testing.B) { benchmarkPredict(b, 100000, 100) }
