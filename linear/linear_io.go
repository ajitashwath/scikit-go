package linear

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"
)

// saveGob writes payload to path as a gob file. op names the caller for errors.
func saveGob(op, path string, payload any) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	defer f.Close()
	if err := gob.NewEncoder(f).Encode(payload); err != nil {
		return fmt.Errorf("%s: encode failed: %w", op, err)
	}
	return nil
}

// loadGob decodes the gob file at path into payload (a pointer).
func loadGob(op, path string, payload any) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	defer f.Close()
	if err := gob.NewDecoder(f).Decode(payload); err != nil {
		return fmt.Errorf("%s: decode failed: %w", op, err)
	}
	return nil
}

// allFinite reports whether every value is neither NaN nor infinite.
func allFinite(v ...[]float64) bool {
	for _, s := range v {
		for _, x := range s {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				return false
			}
		}
	}
	return true
}

// checkLoaded is the validation shared by the Load functions of the linear-model
// family: the format version and kind must match, and the coefficients must fit.
func checkLoaded(op string, version, wantVersion int, kind, wantKind string, nFeatures int, coef []float64, intercept float64) error {
	if version != wantVersion {
		return fmt.Errorf("%s: unsupported format version %d (expected %d)", op, version, wantVersion)
	}
	if kind != wantKind {
		return fmt.Errorf("%s: file holds a %q model, not a %q", op, kind, wantKind)
	}
	if nFeatures < 1 || len(coef) != nFeatures {
		return fmt.Errorf("%s: corrupt payload: %d coefficients for %d features", op, len(coef), nFeatures)
	}
	if !allFinite(coef, []float64{intercept}) {
		return fmt.Errorf("%s: corrupt payload: non-finite coefficients", op)
	}
	return nil
}
