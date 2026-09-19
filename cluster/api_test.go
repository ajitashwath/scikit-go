package cluster

import (
	"errors"
	"math"
	"testing"

	"scikit-go/internal/matutil"
)

var twoBlobs = [][]float64{
	{0, 0}, {0, 1}, {1, 0}, {1, 1},
	{10, 10}, {10, 11}, {11, 10}, {11, 11},
}

func TestKMeans_FitPredictAndAccessors(t *testing.T) {
	k := NewKMeans()
	k.NClusters = 2
	k.Seed = 3
	labels, err := k.FitPredict(twoBlobs)
	if err != nil {
		t.Fatalf("FitPredict: %v", err)
	}
	if len(labels) != len(twoBlobs) {
		t.Fatalf("got %d labels for %d samples", len(labels), len(twoBlobs))
	}
	// The two blobs must be split apart, with each blob in a single cluster.
	for i := 1; i < 4; i++ {
		if labels[i] != labels[0] || labels[4+i] != labels[4] {
			t.Fatalf("blob members were split across clusters: %v", labels)
		}
	}
	if labels[0] == labels[4] {
		t.Fatalf("both blobs landed in one cluster: %v", labels)
	}

	// Labels must return a copy so callers cannot corrupt the model.
	got := k.Labels()
	got[0] = 99
	if k.Labels()[0] == 99 {
		t.Error("Labels returned the internal slice")
	}
	centers := k.ClusterCenters()
	centers[0][0] = 1e9
	if k.ClusterCenters()[0][0] == 1e9 {
		t.Error("ClusterCenters returned the internal slice")
	}

	// Score is the negative inertia; on the training data the two agree.
	score, err := k.Score(twoBlobs, nil)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if math.Abs(score+k.Inertia()) > 1e-9 {
		t.Errorf("Score %v should equal -Inertia %v on the training data", score, -k.Inertia())
	}
	if score >= 0 {
		t.Errorf("Score = %v, want a negative value", score)
	}

	// New points are assigned to the nearest fitted center.
	pred, err := k.Predict([][]float64{{0.4, 0.6}, {10.6, 10.4}})
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if pred[0] != labels[0] || pred[1] != labels[4] {
		t.Errorf("Predict = %v, want blob labels %v and %v", pred, labels[0], labels[4])
	}
}

func TestKMeans_BeforeFit(t *testing.T) {
	k := NewKMeans()
	if k.Labels() != nil || k.ClusterCenters() != nil || k.Inertia() != 0 {
		t.Error("accessors should return zero values before Fit")
	}
	if _, err := k.Predict(twoBlobs); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Predict: got %v, want ErrNotFitted", err)
	}
	if _, err := k.Score(twoBlobs, nil); !errors.Is(err, matutil.ErrNotFitted) {
		t.Errorf("Score: got %v, want ErrNotFitted", err)
	}
	if _, err := k.FitPredict(nil); !errors.Is(err, matutil.ErrEmptyInput) {
		t.Errorf("FitPredict(nil): got %v, want ErrEmptyInput", err)
	}
}
