package nprandom

import "testing"

// Expected values were produced by numpy's legacy RandomState, which is what
// scikit-learn seeds its estimators with.

func TestUint32_MatchesNumpyStream(t *testing.T) {
	// np.random.RandomState(0).randint(0, 2**32, 3, dtype=np.uint64)
	want := []uint32{2357136044, 2546248239, 3071714933}
	m := NewMT19937(0)
	for i, w := range want {
		if got := m.Uint32(); got != w {
			t.Errorf("output %d: got %d, want %d", i, got, w)
		}
	}
}

func TestIntn_MatchesNumpyRandint(t *testing.T) {
	cases := []struct {
		name string
		seed uint32
		high uint64
		want []uint32
	}{
		// rs.randint(2147483647) five times: sklearn's per-estimator seed draw.
		{"estimator seeds", 0, 2147483647, []uint32{209652396, 398764591, 924231285, 1478610112, 441365315}},
		// rs.randint(0, 10, 8): a non-power-of-two bound exercises rejection.
		{"below ten", 12345, 10, []uint32{2, 5, 1, 4, 9, 5, 2, 1}},
		// rs.randint(0, 1000, 6): bootstrap indices for 1000 samples.
		{"bootstrap indices", 3, 1000, []uint32{874, 664, 249, 643, 952, 968}},
		// rs.randint(0, 2**32, dtype=np.uint64): the full 32-bit range takes the word directly.
		{"full range", 99, 1 << 32, []uint32{2887414401, 3103284259, 2096280761}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := NewMT19937(tc.seed)
			for i, w := range tc.want {
				if got := m.Intn(tc.high); got != w {
					t.Errorf("draw %d: got %d, want %d", i, got, w)
				}
			}
		})
	}
}

// numpy returns 0 for a bound of one without consuming a random word, so a draw
// after four of them must equal the first draw of a fresh generator seeded alike:
// np.random.RandomState(7) gives randint(0, 1, 4) == [0 0 0 0] then randint(0, 100) == 47.
func TestIntn_BoundOneConsumesNothing(t *testing.T) {
	m := NewMT19937(7)
	for i := 0; i < 4; i++ {
		if got := m.Intn(1); got != 0 {
			t.Fatalf("Intn(1) = %d, want 0", got)
		}
	}
	if got := m.Intn(100); got != 47 {
		t.Errorf("draw after four Intn(1): got %d, want 47", got)
	}
	if got := NewMT19937(7).Intn(0); got != 0 {
		t.Errorf("Intn(0) = %d, want 0", got)
	}
}

func TestIntn_StaysBelowBound(t *testing.T) {
	m := NewMT19937(2024)
	for _, high := range []uint64{2, 3, 7, 100, 1 << 20, 1<<31 - 1} {
		for i := 0; i < 2000; i++ {
			if got := m.Intn(high); uint64(got) >= high {
				t.Fatalf("Intn(%d) = %d, out of range", high, got)
			}
		}
	}
}

func TestMT19937_DeterministicAndIndependent(t *testing.T) {
	a, b := NewMT19937(5), NewMT19937(5)
	for i := 0; i < 1500; i++ { // crosses the 624-word regeneration boundary twice
		if x, y := a.Uint32(), b.Uint32(); x != y {
			t.Fatalf("same seed diverged at %d: %d vs %d", i, x, y)
		}
	}
	if NewMT19937(5).Uint32() == NewMT19937(6).Uint32() {
		t.Error("different seeds produced the same first output")
	}
}
