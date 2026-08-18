package datasets

import (
	"fmt"
	"math"
	"math/rand"
)

// MakeRegression generates a deterministic linear regression dataset
// y = X @ coef + noise, mirroring sklearn.datasets.make_regression. All features
// are informative; coef is drawn from a standard normal. X ~ N(0, 1) and the
// additive noise is N(0, noise).
func MakeRegression(nSamples, nFeatures int, noise float64, seed int64) ([][]float64, []float64, error) {
	if nSamples <= 0 || nFeatures <= 0 {
		return nil, nil, fmt.Errorf("MakeRegression: %w: nSamples=%d, nFeatures=%d", ErrInvalidParams, nSamples, nFeatures)
	}
	if noise < 0 {
		return nil, nil, fmt.Errorf("MakeRegression: %w: noise must be >= 0, got %v", ErrInvalidParams, noise)
	}
	r := rand.New(rand.NewSource(seed))

	coef := make([]float64, nFeatures)
	for j := range coef {
		coef[j] = r.NormFloat64()
	}

	X := make([][]float64, nSamples)
	y := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		row := make([]float64, nFeatures)
		var target float64
		for j := 0; j < nFeatures; j++ {
			row[j] = r.NormFloat64()
			target += row[j] * coef[j]
		}
		X[i] = row
		y[i] = target + r.NormFloat64()*noise
	}
	return X, y, nil
}

// MakeClassification generates a deterministic, balanced multi-class dataset of
// separable Gaussian blobs, mirroring the spirit of sklearn.datasets.make_classification.
// Each class c draws nSamples/nClasses samples from N(center_c, I), where center_c
// is jittered to be well separated.
func MakeClassification(nSamples, nFeatures, nClasses int, seed int64) ([][]float64, []float64, error) {
	if nSamples <= 0 || nFeatures <= 0 || nClasses <= 0 {
		return nil, nil, fmt.Errorf("MakeClassification: %w: nSamples=%d, nFeatures=%d, nClasses=%d",
			ErrInvalidParams, nSamples, nFeatures, nClasses)
	}
	if nClasses > nSamples {
		return nil, nil, fmt.Errorf("MakeClassification: %w: nClasses=%d exceeds nSamples=%d",
			ErrInvalidParams, nClasses, nSamples)
	}
	r := rand.New(rand.NewSource(seed))

	centers := make([][]float64, nClasses)
	for c := range centers {
		center := make([]float64, nFeatures)
		for j := range center {
			center[j] = r.NormFloat64() * 4
		}
		centers[c] = center
	}

	X := make([][]float64, nSamples)
	y := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		c := i % nClasses
		row := make([]float64, nFeatures)
		for j := range row {
			row[j] = centers[c][j] + r.NormFloat64()
		}
		X[i] = row
		y[i] = float64(c)
	}
	return X, y, nil
}

// MakeBlobs generates deterministic isotropic Gaussian blobs, mirroring
// sklearn.datasets.make_blobs. Cluster centers are drawn uniformly from the
// center box (default [-10, 10] per feature), and each point is sampled from
// N(center, clusterStd) around the center of its (balanced) cluster.
func MakeBlobs(nSamples, nFeatures, nCenters int, seed int64) ([][]float64, []float64, error) {
	return makeBlobs(nSamples, nFeatures, nCenters, 1.0, [2]float64{-10, 10}, seed)
}

func makeBlobs(nSamples, nFeatures, nCenters int, clusterStd float64, centerBox [2]float64, seed int64) ([][]float64, []float64, error) {
	if nSamples <= 0 || nFeatures <= 0 || nCenters <= 0 {
		return nil, nil, fmt.Errorf("MakeBlobs: %w: nSamples=%d, nFeatures=%d, nCenters=%d",
			ErrInvalidParams, nSamples, nFeatures, nCenters)
	}
	if nCenters > nSamples {
		return nil, nil, fmt.Errorf("MakeBlobs: %w: nCenters=%d exceeds nSamples=%d", ErrInvalidParams, nCenters, nSamples)
	}
	if clusterStd < 0 || centerBox[0] > centerBox[1] {
		return nil, nil, fmt.Errorf("MakeBlobs: %w: clusterStd=%v, centerBox=%v", ErrInvalidParams, clusterStd, centerBox)
	}
	r := rand.New(rand.NewSource(seed))

	centers := make([][]float64, nCenters)
	for c := range centers {
		center := make([]float64, nFeatures)
		for j := range center {
			center[j] = centerBox[0] + r.Float64()*(centerBox[1]-centerBox[0])
		}
		centers[c] = center
	}

	X := make([][]float64, nSamples)
	y := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		c := i % nCenters
		row := make([]float64, nFeatures)
		for j := range row {
			row[j] = centers[c][j] + r.NormFloat64()*clusterStd
		}
		X[i] = row
		y[i] = float64(c)
	}
	return X, y, nil
}

// MakeMoons generates two interleaving half-circles, mirroring
// sklearn.datasets.make_moons (including its default shuffle). With noise=0 the
// point geometry exactly matches sklearn; noise adds N(0, noise) per coordinate
// using Go's seeded RNG. The final sample order is a seeded permutation, so it
// differs from sklearn's RNG but is fully deterministic for a given seed.
func MakeMoons(nSamples int, noise float64, seed int64) ([][]float64, []float64, error) {
	if nSamples <= 0 {
		return nil, nil, fmt.Errorf("MakeMoons: %w: nSamples=%d", ErrInvalidParams, nSamples)
	}
	if noise < 0 {
		return nil, nil, fmt.Errorf("MakeMoons: %w: noise must be >= 0, got %v", ErrInvalidParams, noise)
	}
	r := rand.New(rand.NewSource(seed))

	nOut := nSamples / 2
	nIn := nSamples - nOut

	X := make([][]float64, nSamples)
	y := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		var x, yCoord float64
		if i < nOut {
			t := float64(i) / float64(max(nOut-1, 1)) * math.Pi
			x = math.Cos(t)
			yCoord = math.Sin(t)
			y[i] = 0
		} else {
			k := i - nOut
			t := float64(k) / float64(max(nIn-1, 1)) * math.Pi
			x = 1 - math.Cos(t)
			yCoord = 1 - math.Sin(t) - 0.5
			y[i] = 1
		}
		X[i] = []float64{x + r.NormFloat64()*noise, yCoord + r.NormFloat64()*noise}
	}

	// sklearn shuffles X and y together by default.
	perm := r.Perm(nSamples)
	shuffledX := make([][]float64, nSamples)
	shuffledY := make([]float64, nSamples)
	for i, p := range perm {
		shuffledX[i] = X[p]
		shuffledY[i] = y[p]
	}
	return shuffledX, shuffledY, nil
}
