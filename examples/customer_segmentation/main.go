// Command customer_segmentation groups unlabeled customers with k-means. It
// scales the features, picks the number of clusters by silhouette score, and
// reports each segment's size and typical profile in the original units.
//
// The customers here are synthetic Gaussian blobs, so the true segment of every
// customer is known and the example can also check how well k-means recovered it.
//
//	go run ./examples/customer_segmentation
package main

import (
	"fmt"
	"io"
	"os"

	"github.com/ajitashwath/scikit-go/cluster"
	"github.com/ajitashwath/scikit-go/datasets"
	"github.com/ajitashwath/scikit-go/metrics"
	"github.com/ajitashwath/scikit-go/preprocessing"
)

var featureNames = []string{"monthly_spend", "visits", "basket_size", "tenure_months"}

// The raw blobs are centered around zero on the same scale. Give each feature
// its own offset and scale so the columns look like real customer data and are
// measured in very different units, which is exactly why k-means needs scaling.
var (
	offsets = []float64{60, 12, 40, 24}
	scales  = []float64{5, 1, 3, 2}
)

const trueSegments = 4

func main() {
	if _, err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// run returns "best_k", "silhouette" and "adjusted_rand_index".
func run(w io.Writer) (map[string]float64, error) {
	X, truth, err := datasets.MakeBlobs(800, len(featureNames), trueSegments, 7)
	if err != nil {
		return nil, err
	}
	for _, row := range X {
		for j := range row {
			row[j] = offsets[j] + scales[j]*row[j]
		}
	}

	// k-means is distance based, so put every feature on the same scale first.
	scaler := preprocessing.NewStandardScaler()
	Z, err := scaler.FitTransform(X)
	if err != nil {
		return nil, err
	}

	fmt.Fprintf(w, "%d customers, %d features. Trying k = 2..7:\n\n", len(Z), len(featureNames))
	fmt.Fprintf(w, "  %3s %12s %12s\n", "k", "inertia", "silhouette")
	var best *cluster.KMeans
	var bestLabels []float64
	bestSil := -2.0
	for k := 2; k <= 7; k++ {
		km := cluster.NewKMeans()
		km.NClusters = k
		km.Seed = 1
		labels, err := km.FitPredict(Z)
		if err != nil {
			return nil, err
		}
		sil, err := metrics.SilhouetteScore(Z, labels)
		if err != nil {
			return nil, err
		}
		fmt.Fprintf(w, "  %3d %12.1f %12.4f\n", k, km.Inertia(), sil)
		if sil > bestSil {
			best, bestLabels, bestSil = km, labels, sil
		}
	}

	ari, err := metrics.AdjustedRandIndex(truth, bestLabels)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(w, "\nbest k by silhouette: %d (silhouette %.4f)\n", best.NClusters, bestSil)
	fmt.Fprintf(w, "agreement with the true segments (adjusted Rand index): %.4f\n", ari)

	// Cluster centers live in scaled space; map them back to real units.
	centers, err := scaler.InverseTransform(best.ClusterCenters())
	if err != nil {
		return nil, err
	}
	sizes := make([]int, best.NClusters)
	for _, l := range bestLabels {
		sizes[int(l)]++
	}
	fmt.Fprintf(w, "\nsegment profiles (cluster centers in original units):\n  %-8s %6s", "segment", "size")
	for _, n := range featureNames {
		fmt.Fprintf(w, " %14s", n)
	}
	fmt.Fprintln(w)
	for c, ctr := range centers {
		fmt.Fprintf(w, "  %-8d %6d", c, sizes[c])
		for _, v := range ctr {
			fmt.Fprintf(w, " %14.2f", v)
		}
		fmt.Fprintln(w)
	}

	// Assign a brand-new customer to a segment.
	seg, err := best.Predict(mustScale(scaler, [][]float64{{center(X, 0), center(X, 1), center(X, 2), center(X, 3)}}))
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(w, "\nan average customer falls in segment %d\n", int(seg[0]))

	return map[string]float64{
		"best_k":              float64(best.NClusters),
		"silhouette":          bestSil,
		"adjusted_rand_index": ari,
	}, nil
}

// mustScale transforms rows that are known to be well formed.
func mustScale(s *preprocessing.StandardScaler, X [][]float64) [][]float64 {
	Z, err := s.Transform(X)
	if err != nil {
		panic(err)
	}
	return Z
}

// center returns the mean of column j.
func center(X [][]float64, j int) float64 {
	sum := 0.0
	for _, row := range X {
		sum += row[j]
	}
	return sum / float64(len(X))
}
