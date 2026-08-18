package datasets

import (
	"embed"
	"encoding/csv"
	"fmt"
	"io"
	"strconv"
)

//go:embed testdata/diabetes_X.csv testdata/diabetes_y.csv testdata/iris_X.csv testdata/iris_y.csv
var embeddedData embed.FS

// LoadDiabetes returns the diabetes dataset (442 samples, 10 features) bundled by
// scikit-learn, with y being the numeric disease progression target.
func LoadDiabetes() ([][]float64, []float64, error) {
	X, err := readMatrix(embeddedData, "testdata/diabetes_X.csv")
	if err != nil {
		return nil, nil, fmt.Errorf("LoadDiabetes: %w", err)
	}
	y, err := readVector(embeddedData, "testdata/diabetes_y.csv", len(X))
	if err != nil {
		return nil, nil, fmt.Errorf("LoadDiabetes: %w", err)
	}
	return X, y, nil
}

// LoadIris returns the iris dataset (150 samples, 4 features) bundled by
// scikit-learn, with y holding the class index (0, 1, or 2).
func LoadIris() ([][]float64, []float64, error) {
	X, err := readMatrix(embeddedData, "testdata/iris_X.csv")
	if err != nil {
		return nil, nil, fmt.Errorf("LoadIris: %w", err)
	}
	y, err := readVector(embeddedData, "testdata/iris_y.csv", len(X))
	if err != nil {
		return nil, nil, fmt.Errorf("LoadIris: %w", err)
	}
	return X, y, nil
}

// readMatrix parses an embedded headerless CSV matrix.
func readMatrix(fsys embed.FS, path string) ([][]float64, error) {
	f, err := fsys.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1 // rows may vary; raggedness is rejected below
	var out [][]float64
	nFeatures := -1
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("parsing %s: %w", path, err)
		}
		if nFeatures == -1 {
			nFeatures = len(record)
		}
		if len(record) != nFeatures {
			return nil, fmt.Errorf("parsing %s: %w: row %d has %d features, expected %d",
				path, errRaggedCSV, len(out), len(record), nFeatures)
		}
		row := make([]float64, len(record))
		for j, cell := range record {
			row[j], err = strconv.ParseFloat(cell, 64)
			if err != nil {
				return nil, fmt.Errorf("parsing %s row %d col %d: %w", path, len(out), j, err)
			}
		}
		out = append(out, row)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: %w", path, errEmptyCSV)
	}
	return out, nil
}

// readVector parses an embedded headerless single-column CSV into a vector of
// length nSamples.
func readVector(fsys embed.FS, path string, nSamples int) ([]float64, error) {
	f, err := fsys.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 1
	out := make([]float64, 0, nSamples)
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("parsing %s: %w", path, err)
		}
		v, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			return nil, fmt.Errorf("parsing %s: %w", path, err)
		}
		out = append(out, v)
	}
	if len(out) != nSamples {
		return nil, fmt.Errorf("parsing %s: %w: got %d values, expected %d",
			path, errDimMismatchCSV, len(out), nSamples)
	}
	return out, nil
}
