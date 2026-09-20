package main

import (
	"strings"
	"testing"
)

func TestRun(t *testing.T) {
	var out strings.Builder
	scores, err := run(&out)
	if err != nil {
		t.Fatal(err)
	}
	floors := map[string]float64{
		"KNeighborsClassifier":   0.85,
		"DecisionTreeClassifier": 0.80,
		"RandomForestClassifier": 0.88,
		"StandardScaler + SVC":   0.88,
	}
	for name, floor := range floors {
		if got, ok := scores[name]; !ok || got < floor {
			t.Errorf("%s accuracy = %.4f (present=%v), want >= %.2f", name, got, ok, floor)
		}
	}
	for _, want := range []string{"confusion matrix", "versicolor", "macro avg"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output is missing %q", want)
		}
	}
}

func TestRunIsDeterministic(t *testing.T) {
	var a, b strings.Builder
	if _, err := run(&a); err != nil {
		t.Fatal(err)
	}
	if _, err := run(&b); err != nil {
		t.Fatal(err)
	}
	if a.String() != b.String() {
		t.Error("two runs printed different output")
	}
}
