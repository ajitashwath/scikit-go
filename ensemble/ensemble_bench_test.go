package ensemble

import (
	"testing"

	"github.com/ajitashwath/scikit-go/datasets"
)

func benchmarkRegressorFit(b *testing.B, nSamples, nFeatures, nTrees int) {
	X, y, err := datasets.MakeRegression(nSamples, nFeatures, 0.1, 42)
	if err != nil {
		b.Fatalf("MakeRegression: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rf := NewRandomForestRegressor()
		rf.NTrees = nTrees
		if err := rf.Fit(X, y); err != nil {
			b.Fatalf("Fit failed: %v", err)
		}
	}
}

func BenchmarkRegressorFit_Small_100x10(b *testing.B)   { benchmarkRegressorFit(b, 100, 10, 20) }
func BenchmarkRegressorFit_Medium_2000x20(b *testing.B) { benchmarkRegressorFit(b, 2000, 20, 20) }
func BenchmarkRegressorFit_Large_10000x20(b *testing.B) { benchmarkRegressorFit(b, 10000, 20, 10) }

func benchmarkClassifierFit(b *testing.B, nSamples, nFeatures, nTrees int) {
	X, y, err := datasets.MakeClassification(nSamples, nFeatures, 3, 42)
	if err != nil {
		b.Fatalf("MakeClassification: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rf := NewRandomForestClassifier()
		rf.NTrees = nTrees
		if err := rf.Fit(X, y); err != nil {
			b.Fatalf("Fit failed: %v", err)
		}
	}
}

func BenchmarkClassifierFit_Small_100x10(b *testing.B)   { benchmarkClassifierFit(b, 100, 10, 20) }
func BenchmarkClassifierFit_Medium_2000x20(b *testing.B) { benchmarkClassifierFit(b, 2000, 20, 20) }
func BenchmarkClassifierFit_Large_10000x20(b *testing.B) { benchmarkClassifierFit(b, 10000, 20, 10) }

func benchmarkPredict(b *testing.B, nSamples, nFeatures int) {
	X, y, err := datasets.MakeRegression(nSamples, nFeatures, 0.1, 42)
	if err != nil {
		b.Fatalf("MakeRegression: %v", err)
	}
	rf := NewRandomForestRegressor()
	rf.NTrees = 20
	if err := rf.Fit(X, y); err != nil {
		b.Fatalf("Fit failed: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := rf.Predict(X); err != nil {
			b.Fatalf("Predict failed: %v", err)
		}
	}
}

func BenchmarkPredict_Small_100x10(b *testing.B)   { benchmarkPredict(b, 100, 10) }
func BenchmarkPredict_Medium_2000x20(b *testing.B) { benchmarkPredict(b, 2000, 20) }
func BenchmarkPredict_Large_10000x20(b *testing.B) { benchmarkPredict(b, 10000, 20) }
