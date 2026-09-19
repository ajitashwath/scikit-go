// Package nprandom reproduces the parts of numpy's legacy RandomState that
// scikit-learn's estimators use to derive seeds and bootstrap samples, so that
// seeded runs here can follow the same random stream as sklearn's.
package nprandom

const (
	mtN         = 624
	mtM         = 397
	mtMatrixA   = 0x9908b0df
	mtUpperMask = 0x80000000
	mtLowerMask = 0x7fffffff
)

// MT19937 is numpy's legacy Mersenne Twister generator.
type MT19937 struct {
	mt  [mtN]uint32
	mti int
}

// NewMT19937 seeds the generator like numpy's RandomState(seed) does for an
// integer seed (init_genrand).
func NewMT19937(seed uint32) *MT19937 {
	m := &MT19937{}
	m.mt[0] = seed
	for i := 1; i < mtN; i++ {
		m.mt[i] = 1812433253*(m.mt[i-1]^(m.mt[i-1]>>30)) + uint32(i)
	}
	m.mti = mtN
	return m
}

// Uint32 returns the next 32-bit tempered output.
func (m *MT19937) Uint32() uint32 {
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

// Intn mirrors RandomState.randint(0, high) for 1 <= high <= 2^32: it draws
// masked 32-bit words and rejects values above high-1. When high is 1 the
// result is always 0 and no random word is consumed, as in numpy.
func (m *MT19937) Intn(high uint64) uint32 {
	if high <= 1 {
		return 0
	}
	rng := uint32(high - 1)
	if rng == 0xFFFFFFFF {
		return m.Uint32()
	}
	mask := rng
	mask |= mask >> 1
	mask |= mask >> 2
	mask |= mask >> 4
	mask |= mask >> 8
	mask |= mask >> 16
	for {
		if v := m.Uint32() & mask; v <= rng {
			return v
		}
	}
}
