package tree

import "github.com/ajitashwath/scikit-go/internal/nprandom"

// numpy legacy MT19937 (RandomState) seeds sklearn's tree splitter, and the
// XorShift32 our_rand_r drives the in-tree feature draws.

const randRMaxUint32 = 2147483648 // RAND_R_MAX + 1 = 2^31

// initRandState mirrors sklearn's splitter init:
// self.rand_r_state = self.random_state.randint(0, RAND_R_MAX).
// With RAND_R_MAX = 2^31 - 1 the randomkit interval keeps the low 31 bits of
// the first MT19937 output.
func initRandState(seed uint32) uint32 {
	return nprandom.NewMT19937(seed).Uint32() & 0x7fffffff
}

// ourRandR mirrors sklearn.utils._random.our_rand_r (32-bit XorShift).
func ourRandR(seed *uint32) uint32 {
	if *seed == 0 {
		*seed = 1
	}
	*seed ^= *seed << 13
	*seed ^= *seed >> 17
	*seed ^= *seed << 5
	return *seed % randRMaxUint32
}

// randInt mirrors sklearn.tree._utils.rand_int.
func randInt(low, high int, seed *uint32) int {
	return low + int(ourRandR(seed)%uint32(high-low))
}
