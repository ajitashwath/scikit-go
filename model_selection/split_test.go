package model_selection

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"github.com/ajitashwath/scikit-go/internal/matutil"
)

type fixtureSplit struct {
	Train []int `json:"train"`
	Test  []int `json:"test"`
}

type fixtureCV struct {
	Kind string `json:"kind"`
	K    int    `json:"k"`
}

type fixtureFile struct {
	KFold []struct {
		N      int            `json:"n"`
		K      int            `json:"k"`
		Splits []fixtureSplit `json:"splits"`
	} `json:"kfold"`
	Stratified []struct {
		Name   string         `json:"name"`
		Y      []float64      `json:"y"`
		K      int            `json:"k"`
		Splits []fixtureSplit `json:"splits"`
	} `json:"stratified"`
	CrossVal map[string]struct {
		X      [][]float64 `json:"X"`
		Y      []float64   `json:"y"`
		CV     fixtureCV   `json:"cv"`
		Scores []float64   `json:"scores"`
	} `json:"cross_val"`
	Grid map[string]struct {
		X             [][]float64      `json:"X"`
		Y             []float64        `json:"y"`
		CV            fixtureCV        `json:"cv"`
		Grid          map[string][]any `json:"grid"`
		Params        []map[string]any `json:"params"`
		FoldScores    [][]float64      `json:"fold_scores"`
		MeanTestScore []float64        `json:"mean_test_score"`
		StdTestScore  []float64        `json:"std_test_score"`
		RankTestScore []int            `json:"rank_test_score"`
		BestIndex     int              `json:"best_index"`
		BestScore     float64          `json:"best_score"`
		Predict       []float64        `json:"predict"`
		PredictX      [][]float64      `json:"predict_X"`
	} `json:"grid"`
}

func loadFixtures(t testing.TB) fixtureFile {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "model_selection_fixtures.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f fixtureFile
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func (c fixtureCV) splitter() Splitter {
	if c.Kind == "stratified" {
		return NewStratifiedKFold(c.K)
	}
	return NewKFold(c.K)
}

// dummyX returns n one-feature rows; splitters only look at its length.
func dummyX(n int) [][]float64 {
	X := make([][]float64, n)
	for i := range X {
		X[i] = []float64{float64(i)}
	}
	return X
}

func TestKFold_AgainstSklearn(t *testing.T) {
	for _, tc := range loadFixtures(t).KFold {
		got, err := NewKFold(tc.K).Split(dummyX(tc.N), nil)
		if err != nil {
			t.Fatalf("n=%d k=%d: %v", tc.N, tc.K, err)
		}
		compareSplits(t, "kfold", got, tc.Splits)
	}
}

func TestStratifiedKFold_AgainstSklearn(t *testing.T) {
	for _, tc := range loadFixtures(t).Stratified {
		got, err := NewStratifiedKFold(tc.K).Split(dummyX(len(tc.Y)), tc.Y)
		if err != nil {
			t.Fatalf("%s: %v", tc.Name, err)
		}
		compareSplits(t, tc.Name, got, tc.Splits)
	}
}

func compareSplits(t *testing.T, name string, got []Split, want []fixtureSplit) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d splits, want %d", name, len(got), len(want))
	}
	for i := range want {
		if !reflect.DeepEqual(got[i].Train, want[i].Train) || !reflect.DeepEqual(got[i].Test, want[i].Test) {
			t.Errorf("%s: split %d\n got train %v test %v\nwant train %v test %v",
				name, i, got[i].Train, got[i].Test, want[i].Train, want[i].Test)
		}
	}
}

// checkPartition asserts the invariants every splitter must keep.
func checkPartition(t *testing.T, name string, splits []Split, n, nSplits int) {
	t.Helper()
	if len(splits) != nSplits {
		t.Fatalf("%s: got %d splits, want %d", name, len(splits), nSplits)
	}
	seenTest := make([]int, n)
	for i, sp := range splits {
		if len(sp.Train)+len(sp.Test) != n {
			t.Errorf("%s: split %d has %d+%d indices, want %d in total", name, i, len(sp.Train), len(sp.Test), n)
		}
		if len(sp.Test) == 0 || len(sp.Train) == 0 {
			t.Errorf("%s: split %d has an empty side", name, i)
		}
		if !sort.IntsAreSorted(sp.Train) || !sort.IntsAreSorted(sp.Test) {
			t.Errorf("%s: split %d is not in ascending order", name, i)
		}
		inTest := make(map[int]bool)
		for _, idx := range sp.Test {
			seenTest[idx]++
			inTest[idx] = true
		}
		for _, idx := range sp.Train {
			if inTest[idx] {
				t.Errorf("%s: split %d has sample %d in both train and test", name, i, idx)
			}
		}
	}
	for idx, c := range seenTest {
		if c != 1 {
			t.Errorf("%s: sample %d is in %d test folds, want exactly 1", name, idx, c)
		}
	}
}

func TestSplitters_PartitionProperties(t *testing.T) {
	y := make([]float64, 47)
	for i := range y {
		y[i] = float64(i % 3)
	}
	for _, shuffle := range []bool{false, true} {
		for _, k := range []int{2, 3, 5, 10} {
			kf := &KFold{NSplits: k, Shuffle: shuffle, Seed: 9}
			got, err := kf.Split(dummyX(len(y)), y)
			if err != nil {
				t.Fatal(err)
			}
			checkPartition(t, "KFold", got, len(y), k)

			skf := &StratifiedKFold{NSplits: k, Shuffle: shuffle, Seed: 9}
			got, err = skf.Split(dummyX(len(y)), y)
			if err != nil {
				t.Fatal(err)
			}
			checkPartition(t, "StratifiedKFold", got, len(y), k)
		}
	}
}

func TestKFold_FoldSizes(t *testing.T) {
	// 10 samples in 3 folds: the first fold takes the extra sample.
	got, err := NewKFold(3).Split(dummyX(10), nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []int{4, 3, 3} {
		if len(got[i].Test) != want {
			t.Errorf("fold %d has %d test samples, want %d", i, len(got[i].Test), want)
		}
	}
}

func TestShuffle_DeterministicAndSeeded(t *testing.T) {
	X := dummyX(40)
	y := make([]float64, 40)
	for i := range y {
		y[i] = float64(i % 2)
	}
	splitters := map[string]func(seed int64, shuffle bool) Splitter{
		"KFold":           func(s int64, sh bool) Splitter { return &KFold{NSplits: 4, Shuffle: sh, Seed: s} },
		"StratifiedKFold": func(s int64, sh bool) Splitter { return &StratifiedKFold{NSplits: 4, Shuffle: sh, Seed: s} },
	}
	for name, mk := range splitters {
		a, _ := mk(1, true).Split(X, y)
		b, _ := mk(1, true).Split(X, y)
		c, _ := mk(2, true).Split(X, y)
		plain, _ := mk(1, false).Split(X, y)
		if !reflect.DeepEqual(a, b) {
			t.Errorf("%s: the same seed gave different splits", name)
		}
		if reflect.DeepEqual(a, c) {
			t.Errorf("%s: different seeds gave identical splits", name)
		}
		if reflect.DeepEqual(a, plain) {
			t.Errorf("%s: shuffling changed nothing", name)
		}
	}
}

// Shuffling may move which samples land in a fold but must not change how many
// samples of each class a fold holds.
func TestStratifiedKFold_ShuffleKeepsClassBalance(t *testing.T) {
	y := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2}
	X := dummyX(len(y))
	plain, err := NewStratifiedKFold(4).Split(X, y)
	if err != nil {
		t.Fatal(err)
	}
	shuf, err := (&StratifiedKFold{NSplits: 4, Shuffle: true, Seed: 5}).Split(X, y)
	if err != nil {
		t.Fatal(err)
	}
	counts := func(sp Split) map[float64]int {
		m := map[float64]int{}
		for _, i := range sp.Test {
			m[y[i]]++
		}
		return m
	}
	for f := range plain {
		if !reflect.DeepEqual(counts(plain[f]), counts(shuf[f])) {
			t.Errorf("fold %d: class counts %v (plain) vs %v (shuffled)", f, counts(plain[f]), counts(shuf[f]))
		}
	}
}

func TestSplitters_Errors(t *testing.T) {
	X10, y10 := dummyX(10), make([]float64, 10)
	for i := range y10 {
		y10[i] = float64(i % 2)
	}
	cases := []struct {
		name string
		run  func() error
		want error
	}{
		{"KFold NSplits 1", func() error { _, err := NewKFold(1).Split(X10, nil); return err }, ErrInvalidParams},
		{"KFold NSplits 0", func() error { _, err := NewKFold(0).Split(X10, nil); return err }, ErrInvalidParams},
		{"KFold more folds than samples", func() error { _, err := NewKFold(11).Split(X10, nil); return err }, ErrInvalidParams},
		{"KFold empty X", func() error { _, err := NewKFold(2).Split(nil, nil); return err }, matutil.ErrEmptyInput},
		{"KFold y length mismatch", func() error { _, err := NewKFold(2).Split(X10, y10[:5]); return err }, matutil.ErrDimMismatch},
		{"KFold ragged X", func() error { _, err := NewKFold(2).Split([][]float64{{1}, {1, 2}}, nil); return err }, matutil.ErrRaggedInput},
		{"Stratified nil y", func() error { _, err := NewStratifiedKFold(2).Split(X10, nil); return err }, ErrInvalidParams},
		{"Stratified NSplits 1", func() error { _, err := NewStratifiedKFold(1).Split(X10, y10); return err }, ErrInvalidParams},
		{"Stratified more folds than samples", func() error { _, err := NewStratifiedKFold(11).Split(X10, y10); return err }, ErrInvalidParams},
		{"Stratified every class too small", func() error { _, err := NewStratifiedKFold(6).Split(X10, y10); return err }, ErrInvalidParams},
	}
	for _, tc := range cases {
		if err := tc.run(); !errors.Is(err, tc.want) {
			t.Errorf("%s: got %v, want %v", tc.name, err, tc.want)
		}
	}
}

// One class smaller than NSplits is allowed (sklearn only warns).
func TestStratifiedKFold_SmallClassAllowed(t *testing.T) {
	y := []float64{0, 0, 0, 0, 0, 0, 1, 1}
	got, err := NewStratifiedKFold(4).Split(dummyX(len(y)), y)
	if err != nil {
		t.Fatal(err)
	}
	checkPartition(t, "small class", got, len(y), 4)
}
