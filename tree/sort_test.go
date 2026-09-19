package tree

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

type sortCase struct {
	Name          string    `json:"name"`
	Values        []float32 `json:"values"`
	SortedValues  []float32 `json:"sorted_values"`
	SortedIndices []int     `json:"sorted_indices"`
}

// The sort must reproduce sklearn's element for element, including the order in
// which equal values come out; that order feeds the rounding of node sums.
func TestSimultaneousSort_AgainstSklearn(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "sort_fixtures.json"))
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var cases []sortCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("parse fixtures: %v", err)
	}
	for _, c := range cases {
		c := c
		t.Run(c.Name, func(t *testing.T) {
			values := append([]float32(nil), c.Values...)
			indices := make([]int, len(values))
			for i := range indices {
				indices[i] = i
			}
			simultaneousSort(values, indices)
			for i := range values {
				if values[i] != c.SortedValues[i] || indices[i] != c.SortedIndices[i] {
					t.Fatalf("position %d: got (%v, idx %d), want (%v, idx %d)",
						i, values[i], indices[i], c.SortedValues[i], c.SortedIndices[i])
				}
			}
		})
	}
}

// heapSort is only reached once introsort exceeds its depth budget, which the
// fixtures above do not force, so check it directly.
func TestHeapSort(t *testing.T) {
	for _, n := range []int{2, 3, 16, 50, 333} {
		values := make([]float32, n)
		indices := make([]int, n)
		for i := range values {
			values[i] = float32((i * 7919) % 101)
			indices[i] = i
		}
		orig := append([]float32(nil), values...)
		heapSort(values, indices)
		if !sort.SliceIsSorted(values, func(a, b int) bool { return values[a] < values[b] }) {
			t.Fatalf("n=%d: not sorted", n)
		}
		for i, idx := range indices {
			if orig[idx] != values[i] {
				t.Fatalf("n=%d: indices do not follow values at %d", n, i)
			}
		}
	}
}
