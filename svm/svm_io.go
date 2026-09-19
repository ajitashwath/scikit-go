package svm

import (
	"encoding/gob"
	"fmt"
	"os"

	"scikit-go/internal/matutil"
)

// svmGob is the versioned on-disk payload shared by SVC and SVR. Only the
// fields relevant to Kind are populated.
type svmGob struct {
	Version     int
	Kind        string // "svc" or "svr"
	C           float64
	Kernel      string
	Degree      int
	Gamma       float64
	Coef0       float64
	Epsilon     float64
	Tol         float64
	MaxIter     int
	CacheSize   float64
	Probability bool
	Seed        int64

	NFeatures     int
	ResolvedGamma float64
	SV            [][]float64
	SVIndex       []int
	Classes       []float64
	NSV           []int
	Coef          [][]float64 // SVC: (n_classes-1) rows; SVR: a single row
	Rho           []float64   // SVC: one per class pair; SVR: a single value
	ProbA         []float64
	ProbB         []float64
}

const svmFormatVersion = 1

func writeSVM(path, name string, payload svmGob) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("%s.Save: %w", name, err)
	}
	defer f.Close()
	payload.Version = svmFormatVersion
	if err := gob.NewEncoder(f).Encode(payload); err != nil {
		return fmt.Errorf("%s.Save: encode failed: %w", name, err)
	}
	return nil
}

func readSVM(path, name, kind string) (*svmGob, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("Load%s: %w", name, err)
	}
	defer f.Close()

	var payload svmGob
	if err := gob.NewDecoder(f).Decode(&payload); err != nil {
		return nil, fmt.Errorf("Load%s: decode failed: %w", name, err)
	}
	if payload.Version != svmFormatVersion {
		return nil, fmt.Errorf("Load%s: unsupported format version %d (expected %d)", name, payload.Version, svmFormatVersion)
	}
	if payload.Kind != kind {
		return nil, fmt.Errorf("Load%s: file contains a %s, not a %s", name, payload.Kind, kind)
	}
	return &payload, nil
}

// checkSupportVectors verifies that the support vectors match the model's feature
// count and that each of the nCoefRows coefficient rows has one entry per vector.
func checkSupportVectors(p *svmGob, nCoefRows int) error {
	if p.NFeatures < 1 {
		return fmt.Errorf("model has %d features", p.NFeatures)
	}
	for i, v := range p.SV {
		if len(v) != p.NFeatures {
			return fmt.Errorf("support vector %d has %d features, want %d", i, len(v), p.NFeatures)
		}
	}
	if len(p.Coef) != nCoefRows {
		return fmt.Errorf("%d coefficient rows, want %d", len(p.Coef), nCoefRows)
	}
	for i, row := range p.Coef {
		if len(row) != len(p.SV) {
			return fmt.Errorf("coefficient row %d has %d entries for %d support vectors", i, len(row), len(p.SV))
		}
	}
	return nil
}

// Save writes the fitted classifier to path in the versioned gob format.
func (s *SVC) Save(path string) error {
	if !s.fitted {
		return matutil.ErrNotFitted
	}
	return writeSVM(path, "SVC", svmGob{
		Kind: "svc", C: s.C, Kernel: s.Kernel, Degree: s.Degree, Gamma: s.Gamma, Coef0: s.Coef0,
		Tol: s.Tol, MaxIter: s.MaxIter, CacheSize: s.CacheSize, Probability: s.Probability, Seed: s.Seed,
		NFeatures: s.m.nFeatures, ResolvedGamma: s.m.gamma, SV: s.m.sv, SVIndex: s.m.svIdx,
		Classes: s.m.classes, NSV: s.m.nSV, Coef: s.m.coef, Rho: s.m.rho,
		ProbA: s.m.probA, ProbB: s.m.probB,
	})
}

// LoadSVC reads a fitted classifier previously written by Save.
func LoadSVC(path string) (*SVC, error) {
	p, err := readSVM(path, "SVC", "svc")
	if err != nil {
		return nil, err
	}
	k := len(p.Classes)
	if k < 2 || len(p.NSV) != k || len(p.Coef) != k-1 || len(p.Rho) != k*(k-1)/2 {
		return nil, fmt.Errorf("LoadSVC: inconsistent model shape for %d classes", k)
	}
	if err := checkSupportVectors(p, k-1); err != nil {
		return nil, fmt.Errorf("LoadSVC: %w", err)
	}
	total := 0
	for _, n := range p.NSV {
		if n < 0 {
			return nil, fmt.Errorf("LoadSVC: negative support-vector count")
		}
		total += n
	}
	if total != len(p.SV) {
		return nil, fmt.Errorf("LoadSVC: per-class support-vector counts sum to %d for %d vectors", total, len(p.SV))
	}
	if len(p.ProbA) != len(p.ProbB) || (len(p.ProbA) != 0 && len(p.ProbA) != len(p.Rho)) {
		return nil, fmt.Errorf("LoadSVC: %d probA and %d probB values for %d class pairs", len(p.ProbA), len(p.ProbB), len(p.Rho))
	}
	return &SVC{
		C: p.C, Kernel: p.Kernel, Degree: p.Degree, Gamma: p.Gamma, Coef0: p.Coef0, Tol: p.Tol,
		MaxIter: p.MaxIter, CacheSize: p.CacheSize, Probability: p.Probability, Seed: p.Seed,
		m: svcModel{classes: p.Classes, nFeatures: p.NFeatures, gamma: p.ResolvedGamma, sv: p.SV,
			svIdx: p.SVIndex, nSV: p.NSV, coef: p.Coef, rho: p.Rho, probA: p.ProbA, probB: p.ProbB},
		fitted: true,
	}, nil
}

// Save writes the fitted regressor to path in the versioned gob format.
func (r *SVR) Save(path string) error {
	if !r.fitted {
		return matutil.ErrNotFitted
	}
	return writeSVM(path, "SVR", svmGob{
		Kind: "svr", C: r.C, Kernel: r.Kernel, Degree: r.Degree, Gamma: r.Gamma, Coef0: r.Coef0,
		Epsilon: r.Epsilon, Tol: r.Tol, MaxIter: r.MaxIter, CacheSize: r.CacheSize,
		NFeatures: r.m.nFeatures, ResolvedGamma: r.m.gamma, SV: r.m.sv, SVIndex: r.m.svIdx,
		Coef: [][]float64{r.m.coef}, Rho: []float64{r.m.rho},
	})
}

// LoadSVR reads a fitted regressor previously written by Save.
func LoadSVR(path string) (*SVR, error) {
	p, err := readSVM(path, "SVR", "svr")
	if err != nil {
		return nil, err
	}
	if len(p.Coef) != 1 || len(p.Rho) != 1 {
		return nil, fmt.Errorf("LoadSVR: inconsistent model shape")
	}
	if err := checkSupportVectors(p, 1); err != nil {
		return nil, fmt.Errorf("LoadSVR: %w", err)
	}
	return &SVR{
		C: p.C, Kernel: p.Kernel, Degree: p.Degree, Gamma: p.Gamma, Coef0: p.Coef0, Epsilon: p.Epsilon,
		Tol: p.Tol, MaxIter: p.MaxIter, CacheSize: p.CacheSize,
		m: svrModel{nFeatures: p.NFeatures, gamma: p.ResolvedGamma, sv: p.SV, svIdx: p.SVIndex,
			coef: p.Coef[0], rho: p.Rho[0]},
		fitted: true,
	}, nil
}
