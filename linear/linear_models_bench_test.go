package linear

import (
	"testing"

	"github.com/ajitashwath/scikit-go/datasets"
)

func BenchmarkRidge_Fit_2000x30(b *testing.B) {
	X, y, _ := datasets.MakeRegression(2000, 30, 5, 1)
	for i := 0; i < b.N; i++ {
		if err := NewRidge().Fit(X, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLasso_Fit_2000x30(b *testing.B) {
	X, y, _ := datasets.MakeRegression(2000, 30, 5, 1)
	for i := 0; i < b.N; i++ {
		l := NewLasso()
		l.Alpha = 0.1
		if err := l.Fit(X, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkElasticNet_Fit_2000x30(b *testing.B) {
	X, y, _ := datasets.MakeRegression(2000, 30, 5, 1)
	for i := 0; i < b.N; i++ {
		e := NewElasticNet()
		e.Alpha = 0.1
		if err := e.Fit(X, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLogisticRegression_Fit_binary_2000x20(b *testing.B) {
	X, y, _ := datasets.MakeClassification(2000, 20, 2, 1)
	for i := 0; i < b.N; i++ {
		if err := NewLogisticRegression().Fit(X, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLogisticRegression_Fit_multiclass_2000x20(b *testing.B) {
	X, y, _ := datasets.MakeClassification(2000, 20, 4, 1)
	for i := 0; i < b.N; i++ {
		if err := NewLogisticRegression().Fit(X, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLogisticRegression_PredictProba_2000x20(b *testing.B) {
	X, y, _ := datasets.MakeClassification(2000, 20, 4, 1)
	m := NewLogisticRegression()
	if err := m.Fit(X, y); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.PredictProba(X); err != nil {
			b.Fatal(err)
		}
	}
}
