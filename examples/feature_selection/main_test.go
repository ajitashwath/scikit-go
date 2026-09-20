package main

import "testing"

func TestRun(t *testing.T) {
	var out discard
	s, err := run(&out)
	if err != nil {
		t.Fatal(err)
	}
	if s["selectkbest_hits"] != nInformative {
		t.Errorf("SelectKBest kept %v of the %d informative columns, want all of them", s["selectkbest_hits"], nInformative)
	}
	if s["rfe_hits"] != 3 {
		t.Errorf("RFE kept %v of the 3 true features, want all 3", s["rfe_hits"])
	}
	// The point of the example: dropping noise columns must help the distance-based model.
	if s["accuracy_selected"] <= s["accuracy_all_features"] {
		t.Errorf("accuracy with selection %.4f is not above accuracy on all columns %.4f",
			s["accuracy_selected"], s["accuracy_all_features"])
	}
	if s["accuracy_selected"] < 0.95 || s["pipeline_accuracy"] < 0.95 {
		t.Errorf("selected accuracy %.4f / pipeline accuracy %.4f, want >= 0.95", s["accuracy_selected"], s["pipeline_accuracy"])
	}
}

type discard struct{}

func (discard) Write(p []byte) (int, error) { return len(p), nil }
