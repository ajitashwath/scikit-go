package utils

import "testing"

func BenchmarkShuffle_10000x50(b *testing.B) {
	X, y := makeData(10000, 50)
	for i := 0; i < b.N; i++ {
		if _, _, err := Shuffle(X, y, 42); err != nil {
			b.Fatalf("Shuffle: %v", err)
		}
	}
}

func BenchmarkTrainTestSplit_10000x50(b *testing.B) {
	X, y := makeData(10000, 50)
	for i := 0; i < b.N; i++ {
		if _, _, _, _, err := TrainTestSplit(X, y, 0.2, 42); err != nil {
			b.Fatalf("TrainTestSplit: %v", err)
		}
	}
}
