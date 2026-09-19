package svm

import (
	"container/list"
	"fmt"
	"math"
)

// Kernel names accepted by SVC.Kernel and SVR.Kernel.
const (
	KernelLinear  = "linear"
	KernelPoly    = "poly"
	KernelRBF     = "rbf"
	KernelSigmoid = "sigmoid"
)

// kernelFunc evaluates one of libsvm's kernels with resolved parameters.
type kernelFunc struct {
	kind   string
	gamma  float64
	coef0  float64
	degree int
}

func newKernelFunc(kind string, gamma, coef0 float64, degree int) (kernelFunc, error) {
	switch kind {
	case KernelLinear, KernelPoly, KernelRBF, KernelSigmoid:
	default:
		return kernelFunc{}, fmt.Errorf("%w: unsupported kernel %q", ErrInvalidSVM, kind)
	}
	return kernelFunc{kind: kind, gamma: gamma, coef0: coef0, degree: degree}, nil
}

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// powi raises base to a non-negative integer power by repeated squaring, as
// libsvm does, so the polynomial kernel is exact for integer degrees.
func powi(base float64, times int) float64 {
	tmp, ret := base, 1.0
	for t := times; t > 0; t /= 2 {
		if t%2 == 1 {
			ret *= tmp
		}
		tmp *= tmp
	}
	return ret
}

func (k kernelFunc) eval(a, b []float64) float64 {
	switch k.kind {
	case KernelLinear:
		return dot(a, b)
	case KernelPoly:
		return powi(k.gamma*dot(a, b)+k.coef0, k.degree)
	case KernelRBF:
		var sq float64
		for i := range a {
			d := a[i] - b[i]
			sq += d * d
		}
		return math.Exp(-k.gamma * sq)
	default: // sigmoid
		return math.Tanh(k.gamma*dot(a, b) + k.coef0)
	}
}

// rowCache is an LRU cache of kernel rows. Rows evicted from the cache stay
// valid for any caller still holding them (they are ordinary slices), so the
// solver may keep two rows alive at once regardless of capacity.
type rowCache struct {
	n       int
	maxRows int
	rows    map[int]*list.Element
	lru     *list.List // front = most recently used
}

type cachedRow struct {
	index int
	data  []float64
}

// newRowCache sizes the cache from a budget in megabytes, holding at least two
// rows of n entries.
func newRowCache(n int, megabytes float64) *rowCache {
	maxRows := int(megabytes * 1024 * 1024 / 8 / float64(n))
	if maxRows < 2 {
		maxRows = 2
	}
	if maxRows > n {
		maxRows = n
	}
	return &rowCache{n: n, maxRows: maxRows, rows: make(map[int]*list.Element), lru: list.New()}
}

// get returns row i, computing it with fill on a miss.
func (c *rowCache) get(i int, fill func(dst []float64)) []float64 {
	if el, ok := c.rows[i]; ok {
		c.lru.MoveToFront(el)
		return el.Value.(*cachedRow).data
	}
	var data []float64
	if c.lru.Len() >= c.maxRows {
		oldest := c.lru.Back()
		old := oldest.Value.(*cachedRow)
		delete(c.rows, old.index)
		c.lru.Remove(oldest)
		data = make([]float64, c.n) // never reuse: a caller may still hold the old row
	} else {
		data = make([]float64, c.n)
	}
	fill(data)
	c.rows[i] = c.lru.PushFront(&cachedRow{index: i, data: data})
	return data
}
