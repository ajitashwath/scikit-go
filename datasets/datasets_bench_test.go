package datasets

import "testing"

func BenchmarkMakeRegression_Medium_10000x50(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, _, err := MakeRegression(10000, 50, 0.1, 42); err != nil {
			b.Fatalf("MakeRegression: %v", err)
		}
	}
}

func BenchmarkMakeClassification_Medium_10000x50(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, _, err := MakeClassification(10000, 50, 5, 42); err != nil {
			b.Fatalf("MakeClassification: %v", err)
		}
	}
}

func BenchmarkMakeBlobs_Medium_10000x50(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, _, err := MakeBlobs(10000, 50, 10, 42); err != nil {
			b.Fatalf("MakeBlobs: %v", err)
		}
	}
}

func BenchmarkMakeMoons_10000(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, _, err := MakeMoons(10000, 0.05, 42); err != nil {
			b.Fatalf("MakeMoons: %v", err)
		}
	}
}

func BenchmarkLoadDiabetes(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, _, err := LoadDiabetes(); err != nil {
			b.Fatalf("LoadDiabetes: %v", err)
		}
	}
}
