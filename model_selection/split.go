// Package model_selection provides cross-validation splitters, CrossValScore and
// GridSearchCV, mirroring sklearn.model_selection.
//
// The splitters follow sklearn's algorithms, so without shuffling KFold and
// StratifiedKFold produce exactly the folds sklearn does. With Shuffle set the
// folds have the same sizes and guarantees but are not index-for-index identical
// to sklearn's, because the permutation comes from Go's seeded RNG instead of numpy's
// (the same deviation utils.TrainTestSplit documents).
package model_selection

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/ajitashwath/scikit-go/internal/matutil"
)

// ErrInvalidParams is returned for an unusable splitter, grid or search configuration.
var ErrInvalidParams = errors.New("model_selection: invalid parameters")

// Split is one cross-validation partition of the sample indices. Train and Test
// are disjoint, together cover every sample, and are each in ascending order.
type Split struct {
	Train []int
	Test  []int
}

// Splitter produces cross-validation splits for a dataset. y may be nil for
// splitters that ignore the labels (KFold).
type Splitter interface {
	Split(X [][]float64, y []float64) ([]Split, error)
}

// KFold partitions the samples into NSplits consecutive folds; each fold is used
// once as the test set. The first n%NSplits folds have one extra sample, as in sklearn.
type KFold struct {
	NSplits int
	Shuffle bool  // permute the samples before cutting them into folds
	Seed    int64 // seeds the permutation when Shuffle is set
}

// NewKFold returns a KFold with nSplits folds and no shuffling.
func NewKFold(nSplits int) *KFold { return &KFold{NSplits: nSplits} }

// Split returns the NSplits train/test partitions of len(X) samples. y is not used
// but, if given, must have len(X) entries.
func (k *KFold) Split(X [][]float64, y []float64) ([]Split, error) {
	n, err := checkSplitInputs("KFold.Split", k.NSplits, X, y)
	if err != nil {
		return nil, err
	}
	order := identity(n)
	if k.Shuffle {
		order = rand.New(rand.NewSource(k.Seed)).Perm(n)
	}
	testFold := make([]int, n)
	base, extra := n/k.NSplits, n%k.NSplits
	pos := 0
	for f := 0; f < k.NSplits; f++ {
		size := base
		if f < extra {
			size++
		}
		for _, idx := range order[pos : pos+size] {
			testFold[idx] = f
		}
		pos += size
	}
	return splitsFromFolds(testFold, k.NSplits), nil
}

// StratifiedKFold is KFold that keeps each class's share of every fold as close to
// its share of the whole dataset as possible. y must hold class labels.
//
// A class with fewer than NSplits members still splits (it just does not appear in
// every test fold); sklearn emits a warning for that case and this package does not.
// An error is returned only when every class is smaller than NSplits.
type StratifiedKFold struct {
	NSplits int
	Shuffle bool  // shuffle each class's samples before assigning them to folds
	Seed    int64 // seeds the shuffle when Shuffle is set
}

// NewStratifiedKFold returns a StratifiedKFold with nSplits folds and no shuffling.
func NewStratifiedKFold(nSplits int) *StratifiedKFold { return &StratifiedKFold{NSplits: nSplits} }

// Split returns the NSplits stratified train/test partitions. y is required.
func (s *StratifiedKFold) Split(X [][]float64, y []float64) ([]Split, error) {
	if y == nil {
		return nil, fmt.Errorf("StratifiedKFold.Split: %w: y is required to stratify", ErrInvalidParams)
	}
	n, err := checkSplitInputs("StratifiedKFold.Split", s.NSplits, X, y)
	if err != nil {
		return nil, err
	}

	// Encode labels by order of first appearance, as sklearn does; this decides which
	// class gets which folds when the class sizes do not divide evenly.
	code := make(map[float64]int)
	encoded := make([]int, n)
	for i, label := range y {
		c, ok := code[label]
		if !ok {
			c = len(code)
			code[label] = c
		}
		encoded[i] = c
	}
	nClasses := len(code)
	counts := make([]int, nClasses)
	for _, c := range encoded {
		counts[c]++
	}
	allSmall := true
	for _, c := range counts {
		if c >= s.NSplits {
			allSmall = false
		}
	}
	if allSmall {
		return nil, fmt.Errorf("StratifiedKFold.Split: %w: NSplits=%d is greater than the number of members in every class",
			ErrInvalidParams, s.NSplits)
	}

	// Deal the samples, sorted by class, round-robin into the folds. alloc[f][c] is how
	// many members of class c fold f gets to hold in its test set.
	sorted := make([]int, 0, n)
	for c := 0; c < nClasses; c++ {
		for j := 0; j < counts[c]; j++ {
			sorted = append(sorted, c)
		}
	}
	alloc := make([][]int, s.NSplits)
	for f := range alloc {
		alloc[f] = make([]int, nClasses)
		for i := f; i < n; i += s.NSplits {
			alloc[f][sorted[i]]++
		}
	}

	var rng *rand.Rand
	if s.Shuffle {
		rng = rand.New(rand.NewSource(s.Seed))
	}
	members := make([][]int, nClasses) // sample indices per class, in original order
	for i, c := range encoded {
		members[c] = append(members[c], i)
	}
	testFold := make([]int, n)
	for c := 0; c < nClasses; c++ {
		folds := make([]int, 0, counts[c])
		for f := 0; f < s.NSplits; f++ {
			for j := 0; j < alloc[f][c]; j++ {
				folds = append(folds, f)
			}
		}
		if rng != nil {
			rng.Shuffle(len(folds), func(a, b int) { folds[a], folds[b] = folds[b], folds[a] })
		}
		for j, idx := range members[c] {
			testFold[idx] = folds[j]
		}
	}
	return splitsFromFolds(testFold, s.NSplits), nil
}

// checkSplitInputs validates the splitter arguments and returns the sample count.
func checkSplitInputs(op string, nSplits int, X [][]float64, y []float64) (int, error) {
	if nSplits < 2 {
		return 0, fmt.Errorf("%s: %w: NSplits must be at least 2, got %d", op, ErrInvalidParams, nSplits)
	}
	var err error
	if y == nil {
		err = matutil.ValidateXMatrix(X)
	} else {
		err = matutil.ValidateXy(X, y)
	}
	if err != nil {
		return 0, fmt.Errorf("%s: %w", op, err)
	}
	if nSplits > len(X) {
		return 0, fmt.Errorf("%s: %w: NSplits=%d is greater than the number of samples (%d)", op, ErrInvalidParams, nSplits, len(X))
	}
	return len(X), nil
}

// splitsFromFolds turns a per-sample test-fold assignment into ascending index lists.
func splitsFromFolds(testFold []int, nSplits int) []Split {
	splits := make([]Split, nSplits)
	for i, f := range testFold {
		for g := range splits {
			if g == f {
				splits[g].Test = append(splits[g].Test, i)
			} else {
				splits[g].Train = append(splits[g].Train, i)
			}
		}
	}
	return splits
}

func identity(n int) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	return idx
}
