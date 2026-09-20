package neighbors

import (
	"testing"

	"github.com/ajitashwath/scikit-go/datasets"
)

func benchData(b *testing.B, nSamples, nFeatures int) ([][]float64, []float64, [][]float64) {
	b.Helper()
	X, y, err := datasets.MakeRegression(nSamples, nFeatures, 0.05, 1)
	if err != nil {
		b.Fatal(err)
	}
	XTest, _, err := datasets.MakeRegression(nSamples/2, nFeatures, 0.05, 2)
	if err != nil {
		b.Fatal(err)
	}
	return X, y, XTest
}

func benchmarkFit(b *testing.B, nSamples, nFeatures int) {
	b.Helper()
	X, y, _ := benchData(b, nSamples, nFeatures)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := NewKNeighborsRegressor()
		if err := m.Fit(X, y); err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkPredict(b *testing.B, nSamples, nFeatures int) {
	b.Helper()
	X, y, XTest := benchData(b, nSamples, nFeatures)
	m := NewKNeighborsRegressor()
	if err := m.Fit(X, y); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Predict(XTest); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkKNNRegressor_Fit_Small(b *testing.B)  { benchmarkFit(b, 100, 10) }
func BenchmarkKNNRegressor_Fit_Medium(b *testing.B) { benchmarkFit(b, 1000, 50) }
func BenchmarkKNNRegressor_Fit_Large(b *testing.B)  { benchmarkFit(b, 10000, 100) }

func BenchmarkKNNRegressor_Predict_Small(b *testing.B)  { benchmarkPredict(b, 100, 10) }
func BenchmarkKNNRegressor_Predict_Medium(b *testing.B) { benchmarkPredict(b, 1000, 50) }
func BenchmarkKNNRegressor_Predict_Large(b *testing.B)  { benchmarkPredict(b, 10000, 100) }
