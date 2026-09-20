// Package ensemble implements bootstrap-aggregated decision trees mirroring
// sklearn.ensemble: RandomForestRegressor and RandomForestClassifier. Trees are
// grown in parallel; every tree owns a seed derived from the forest's master
// seed, so results are identical across runs and across worker counts.
package ensemble

import (
	"errors"
	"fmt"
	"runtime"
	"sort"
	"sync"

	"github.com/ajitashwath/scikit-go/internal/nprandom"
)

// ErrInvalidForest is returned when forest hyperparameters fail validation.
var ErrInvalidForest = errors.New("invalid forest hyperparameters")

// forestParams carries the hyperparameters shared by both forests.
type forestParams struct {
	nTrees          int
	minSamplesSplit int
	minSamplesLeaf  int
	maxFeatures     int
}

func validateForest(p forestParams) error {
	if p.nTrees < 1 {
		return fmt.Errorf("%w: n_trees must be >= 1, got %d", ErrInvalidForest, p.nTrees)
	}
	if p.minSamplesSplit < 2 {
		return fmt.Errorf("%w: min_samples_split must be >= 2, got %d", ErrInvalidForest, p.minSamplesSplit)
	}
	if p.minSamplesLeaf < 1 {
		return fmt.Errorf("%w: min_samples_leaf must be >= 1, got %d", ErrInvalidForest, p.minSamplesLeaf)
	}
	if p.maxFeatures < 0 {
		return fmt.Errorf("%w: max_features must be >= 0, got %d", ErrInvalidForest, p.maxFeatures)
	}
	return nil
}

// maxInt32 is np.iinfo(np.int32).max, the exclusive bound sklearn uses when it
// draws a random_state for each ensemble member.
const maxInt32 = 1<<31 - 1

// treeSeeds draws one seed per tree from the master seed the way sklearn's
// _set_random_states does: successive RandomState(seed).randint(2**31 - 1)
// calls. Drawing them up front, serially, also makes the forest independent of
// goroutine scheduling. Seed is interpreted as an unsigned 32-bit value, the
// range numpy accepts.
func treeSeeds(master int64, nTrees int) []int64 {
	rs := nprandom.NewMT19937(uint32(master))
	seeds := make([]int64, nTrees)
	for i := range seeds {
		seeds[i] = int64(rs.Intn(maxInt32))
	}
	return seeds
}

// bootstrapSample returns the rows and targets of a with-replacement sample of
// size len(X), drawn like sklearn's _generate_sample_indices:
// RandomState(tree_seed).randint(0, n_samples, n_samples). Rows are shared, not copied.
func bootstrapSample(X [][]float64, y []float64, seed int64) ([][]float64, []float64) {
	rs := nprandom.NewMT19937(uint32(seed))
	n := len(X)
	Xb := make([][]float64, n)
	yb := make([]float64, n)
	for i := 0; i < n; i++ {
		j := rs.Intn(uint64(n))
		Xb[i] = X[j]
		yb[i] = y[j]
	}
	return Xb, yb
}

// runParallel calls fn(i) for every i in [0, n) on a bounded worker pool and
// returns the error of the lowest-numbered failing call, if any.
func runParallel(n, nJobs int, fn func(i int) error) error {
	workers := nJobs
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers > n {
		workers = n
	}
	errs := make([]error, n)
	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				errs[i] = fn(i)
			}
		}()
	}
	for i := 0; i < n; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			return fmt.Errorf("tree %d: %w", i, err)
		}
	}
	return nil
}

// meanImportances averages per-tree importances and renormalizes them to sum
// to one, as sklearn's forest feature_importances_ does.
func meanImportances(perTree [][]float64, nFeatures int) []float64 {
	out := make([]float64, nFeatures)
	for _, imp := range perTree {
		for j, v := range imp {
			out[j] += v
		}
	}
	var total float64
	for _, v := range out {
		total += v
	}
	if total > 0 {
		for j := range out {
			out[j] /= total
		}
	}
	return out
}

// uniqueSorted returns the sorted distinct values of y.
func uniqueSorted(y []float64) []float64 {
	seen := make(map[float64]struct{}, 8)
	for _, v := range y {
		seen[v] = struct{}{}
	}
	out := make([]float64, 0, len(seen))
	for v := range seen {
		out = append(out, v)
	}
	sort.Float64s(out)
	return out
}
