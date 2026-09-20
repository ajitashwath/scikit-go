package feature_selection

import (
	"testing"

	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/linear"
)

func benchmarkTransform(b *testing.B, nSamples, nFeatures int) {
	X, y, err := datasets.MakeClassification(nSamples, nFeatures, 3, 42)
	if err != nil {
		b.Fatalf("MakeClassification: %v", err)
	}
	sel := NewSelectKBest()
	sel.K = nFeatures / 2
	if err := sel.Fit(X, y); err != nil {
		b.Fatalf("Fit failed: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sel.Transform(X); err != nil {
			b.Fatalf("Transform failed: %v", err)
		}
	}
}

func BenchmarkTransform_Small_100x10(b *testing.B)     { benchmarkTransform(b, 100, 10) }
func BenchmarkTransform_Medium_10000x50(b *testing.B)  { benchmarkTransform(b, 10000, 50) }
func BenchmarkTransform_Large_100000x100(b *testing.B) { benchmarkTransform(b, 100000, 100) }

func benchmarkSelectKBestFit(b *testing.B, nSamples, nFeatures int) {
	X, y, err := datasets.MakeClassification(nSamples, nFeatures, 3, 42)
	if err != nil {
		b.Fatalf("MakeClassification: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sel := NewSelectKBest()
		sel.K = nFeatures / 2
		if err := sel.Fit(X, y); err != nil {
			b.Fatalf("Fit failed: %v", err)
		}
	}
}

func BenchmarkSelectKBestFit_Small_100x10(b *testing.B)    { benchmarkSelectKBestFit(b, 100, 10) }
func BenchmarkSelectKBestFit_Medium_10000x50(b *testing.B) { benchmarkSelectKBestFit(b, 10000, 50) }

func BenchmarkVarianceThresholdFit_Medium_10000x50(b *testing.B) {
	X, _, err := datasets.MakeRegression(10000, 50, 0.1, 42)
	if err != nil {
		b.Fatalf("MakeRegression: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := NewVarianceThreshold().Fit(X, nil); err != nil {
			b.Fatalf("Fit failed: %v", err)
		}
	}
}

func BenchmarkRFEFit_Small_200x10(b *testing.B) {
	X, y, err := datasets.MakeRegression(200, 10, 0.1, 42)
	if err != nil {
		b.Fatalf("MakeRegression: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rfe := NewRFE(linear.NewLinearRegression())
		rfe.NFeaturesToSelect = 3
		if err := rfe.Fit(X, y); err != nil {
			b.Fatalf("Fit failed: %v", err)
		}
	}
}
