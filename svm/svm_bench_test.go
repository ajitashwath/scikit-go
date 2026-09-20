package svm

import (
	"testing"

	"github.com/ajitashwath/scikit-go/datasets"
)

// SMO is quadratic-to-cubic in the sample count, so the benchmarks stop at a
// medium size rather than the 100000-row case used by the linear models.

func benchmarkSVCFit(b *testing.B, nSamples, nFeatures int, kernel string) {
	X, y, err := datasets.MakeClassification(nSamples, nFeatures, 3, 42)
	if err != nil {
		b.Fatalf("MakeClassification: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := NewSVC()
		s.Kernel = kernel
		if err := s.Fit(X, y); err != nil {
			b.Fatalf("Fit failed: %v", err)
		}
	}
}

func BenchmarkSVCFit_Small_100x10(b *testing.B)         { benchmarkSVCFit(b, 100, 10, KernelRBF) }
func BenchmarkSVCFit_Medium_1000x20(b *testing.B)       { benchmarkSVCFit(b, 1000, 20, KernelRBF) }
func BenchmarkSVCFitLinear_Medium_1000x20(b *testing.B) { benchmarkSVCFit(b, 1000, 20, KernelLinear) }

func benchmarkSVCPredict(b *testing.B, nSamples, nFeatures int) {
	X, y, err := datasets.MakeClassification(nSamples, nFeatures, 3, 42)
	if err != nil {
		b.Fatalf("MakeClassification: %v", err)
	}
	s := NewSVC()
	if err := s.Fit(X, y); err != nil {
		b.Fatalf("Fit failed: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Predict(X); err != nil {
			b.Fatalf("Predict failed: %v", err)
		}
	}
}

func BenchmarkSVCPredict_Small_100x10(b *testing.B)   { benchmarkSVCPredict(b, 100, 10) }
func BenchmarkSVCPredict_Medium_1000x20(b *testing.B) { benchmarkSVCPredict(b, 1000, 20) }

func benchmarkSVRFit(b *testing.B, nSamples, nFeatures int) {
	X, y, err := datasets.MakeRegression(nSamples, nFeatures, 0.1, 42)
	if err != nil {
		b.Fatalf("MakeRegression: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := NewSVR()
		if err := r.Fit(X, y); err != nil {
			b.Fatalf("Fit failed: %v", err)
		}
	}
}

func BenchmarkSVRFit_Small_100x10(b *testing.B)   { benchmarkSVRFit(b, 100, 10) }
func BenchmarkSVRFit_Medium_1000x20(b *testing.B) { benchmarkSVRFit(b, 1000, 20) }

func benchmarkSVRPredict(b *testing.B, nSamples, nFeatures int) {
	X, y, err := datasets.MakeRegression(nSamples, nFeatures, 0.1, 42)
	if err != nil {
		b.Fatalf("MakeRegression: %v", err)
	}
	r := NewSVR()
	if err := r.Fit(X, y); err != nil {
		b.Fatalf("Fit failed: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := r.Predict(X); err != nil {
			b.Fatalf("Predict failed: %v", err)
		}
	}
}

func BenchmarkSVRPredict_Small_100x10(b *testing.B)   { benchmarkSVRPredict(b, 100, 10) }
func BenchmarkSVRPredict_Medium_1000x20(b *testing.B) { benchmarkSVRPredict(b, 1000, 20) }

func BenchmarkSVCFitProbability_Small_100x10(b *testing.B) {
	X, y, err := datasets.MakeClassification(100, 10, 3, 42)
	if err != nil {
		b.Fatalf("MakeClassification: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := NewSVC()
		s.Probability = true
		if err := s.Fit(X, y); err != nil {
			b.Fatalf("Fit failed: %v", err)
		}
	}
}
