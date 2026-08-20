package decomposition

import (
	"testing"

	"scikit-go/datasets"
)

func benchPCAData(b *testing.B, nSamples, nFeatures int) [][]float64 {
	b.Helper()
	X, _, err := datasets.MakeRegression(nSamples, nFeatures, 0.05, 1)
	if err != nil {
		b.Fatal(err)
	}
	return X
}

func benchmarkPCAFit(b *testing.B, nSamples, nFeatures int) {
	b.Helper()
	X := benchPCAData(b, nSamples, nFeatures)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := NewPCA()
		if err := m.Fit(X, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkPCATransform(b *testing.B, nSamples, nFeatures int) {
	b.Helper()
	X := benchPCAData(b, nSamples, nFeatures)
	m := NewPCA()
	if err := m.Fit(X, nil); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Transform(X); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPCA_Fit_Small(b *testing.B)     { benchmarkPCAFit(b, 100, 10) }
func BenchmarkPCA_Fit_Medium(b *testing.B)    { benchmarkPCAFit(b, 1000, 50) }
func BenchmarkPCA_Fit_Large(b *testing.B)     { benchmarkPCAFit(b, 10000, 100) }
func BenchmarkPCA_Transform_Small(b *testing.B)  { benchmarkPCATransform(b, 100, 10) }
func BenchmarkPCA_Transform_Medium(b *testing.B) { benchmarkPCATransform(b, 1000, 50) }
func BenchmarkPCA_Transform_Large(b *testing.B)  { benchmarkPCATransform(b, 10000, 100) }