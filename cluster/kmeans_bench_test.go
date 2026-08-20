package cluster

import (
	"testing"

	"scikit-go/datasets"
)

func benchKMeans(b *testing.B, nSamples, nFeatures int) {
	b.Helper()
	X, _, err := datasets.MakeBlobs(nSamples, nFeatures, 8, 1)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := NewKMeans()
		m.NClusters = 8
		if err := m.Fit(X, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkKMeans_Fit_Small(b *testing.B)  { benchKMeans(b, 100, 10) }
func BenchmarkKMeans_Fit_Medium(b *testing.B) { benchKMeans(b, 1000, 50) }
func BenchmarkKMeans_Fit_Large(b *testing.B)  { benchKMeans(b, 10000, 100) }