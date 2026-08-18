package tree

// numpy legacy MT19937 (RandomState) used to seed sklearn's tree splitter, and
// the XorShift32 our_rand_r used for in-tree feature draws.

const (
	mtN            = 624
	mtM            = 397
	mtMatrixA      = 0x9908b0df
	mtUpperMask    = 0x80000000
	mtLowerMask    = 0x7fffffff
	randRMaxUint32 = 2147483648 // RAND_R_MAX + 1 = 2^31
)

// mt19937 is numpy's legacy MT19937 generator.
type mt19937 struct {
	mt  [mtN]uint32
	mti int
}

// newMT19937 seeds the generator like numpy's init_genrand.
func newMT19937(seed uint32) *mt19937 {
	m := &mt19937{}
	m.mt[0] = seed & 0xffffffff
	for i := 1; i < mtN; i++ {
		m.mt[i] = (1812433253*(m.mt[i-1]^(m.mt[i-1]>>30)) + uint32(i)) & 0xffffffff
	}
	m.mti = mtN
	return m
}

// next returns the next 32-bit tempered output.
func (m *mt19937) next() uint32 {
	if m.mti >= mtN {
		for i := 0; i < mtN; i++ {
			y := (m.mt[i] & mtUpperMask) | (m.mt[(i+1)%mtN] & mtLowerMask)
			m.mt[i] = m.mt[(i+mtM)%mtN] ^ (y >> 1)
			if y&1 != 0 {
				m.mt[i] ^= mtMatrixA
			}
		}
		m.mti = 0
	}
	y := m.mt[m.mti]
	m.mti++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

// initRandState mirrors sklearn's splitter init:
// self.rand_r_state = self.random_state.randint(0, RAND_R_MAX).
// With RAND_R_MAX = 2^31 - 1 the randomkit interval keeps the low 31 bits of
// the first MT19937 output.
func initRandState(seed uint32) uint32 {
	return newMT19937(seed).next() & 0x7fffffff
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