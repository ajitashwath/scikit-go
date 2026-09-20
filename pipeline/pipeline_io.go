package pipeline

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/ajitashwath/scikit-go/cluster"
	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/decomposition"
	"github.com/ajitashwath/scikit-go/ensemble"
	"github.com/ajitashwath/scikit-go/feature_selection"
	"github.com/ajitashwath/scikit-go/internal/matutil"
	"github.com/ajitashwath/scikit-go/linear"
	"github.com/ajitashwath/scikit-go/neighbors"
	"github.com/ajitashwath/scikit-go/preprocessing"
	"github.com/ajitashwath/scikit-go/svm"
	"github.com/ajitashwath/scikit-go/tree"
)

// pipelineGob is the versioned on-disk payload. Every step is stored as the
// bytes of its own estimator's Save file, tagged with the kind that selects its
// loader, so each estimator keeps evolving its own format independently.
type pipelineGob struct {
	Version int
	Names   []string
	Kinds   []string
	Blobs   [][]byte
}

const pipelineFormatVersion = 1

// kindOf maps an estimator's type to the kind tag stored on disk, and loaders
// maps each tag back to its package's Load function. Estimators outside this
// registry can be used in a Pipeline but not persisted with it.
func kindOf(est any) (string, bool) {
	switch est.(type) {
	case *preprocessing.StandardScaler:
		return "standard_scaler", true
	case *linear.LinearRegression:
		return "linear_regression", true
	case *linear.Ridge:
		return "ridge", true
	case *linear.Lasso:
		return "lasso", true
	case *linear.ElasticNet:
		return "elastic_net", true
	case *linear.LogisticRegression:
		return "logistic_regression", true
	case *decomposition.PCA:
		return "pca", true
	case *cluster.KMeans:
		return "kmeans", true
	case *neighbors.KNeighborsClassifier:
		return "knn_classifier", true
	case *neighbors.KNeighborsRegressor:
		return "knn_regressor", true
	case *tree.DecisionTreeClassifier:
		return "tree_classifier", true
	case *tree.DecisionTreeRegressor:
		return "tree_regressor", true
	case *ensemble.RandomForestClassifier:
		return "forest_classifier", true
	case *ensemble.RandomForestRegressor:
		return "forest_regressor", true
	case *feature_selection.VarianceThreshold:
		return "variance_threshold", true
	case *feature_selection.SelectKBest:
		return "select_k_best", true
	case *feature_selection.RFE:
		return "rfe", true
	case *svm.SVC:
		return "svc", true
	case *svm.SVR:
		return "svr", true
	case *Pipeline:
		return "pipeline", true
	}
	return "", false
}

var loaders = map[string]func(path string) (any, error){
	"standard_scaler":     func(p string) (any, error) { return preprocessing.LoadStandardScaler(p) },
	"linear_regression":   func(p string) (any, error) { return linear.LoadLinearRegression(p) },
	"ridge":               func(p string) (any, error) { return linear.LoadRidge(p) },
	"lasso":               func(p string) (any, error) { return linear.LoadLasso(p) },
	"elastic_net":         func(p string) (any, error) { return linear.LoadElasticNet(p) },
	"logistic_regression": func(p string) (any, error) { return linear.LoadLogisticRegression(p) },
	"pca":                 func(p string) (any, error) { return decomposition.LoadPCA(p) },
	"kmeans":              func(p string) (any, error) { return cluster.LoadKMeans(p) },
	"knn_classifier":      func(p string) (any, error) { return neighbors.LoadKNeighborsClassifier(p) },
	"knn_regressor":       func(p string) (any, error) { return neighbors.LoadKNeighborsRegressor(p) },
	"tree_classifier":     func(p string) (any, error) { return tree.LoadDecisionTreeClassifier(p) },
	"tree_regressor":      func(p string) (any, error) { return tree.LoadDecisionTreeRegressor(p) },
	"forest_classifier":   func(p string) (any, error) { return ensemble.LoadRandomForestClassifier(p) },
	"forest_regressor":    func(p string) (any, error) { return ensemble.LoadRandomForestRegressor(p) },
	"variance_threshold":  func(p string) (any, error) { return feature_selection.LoadVarianceThreshold(p) },
	"select_k_best":       func(p string) (any, error) { return feature_selection.LoadSelectKBest(p) },
	"rfe":                 func(p string) (any, error) { return feature_selection.LoadRFE(p) },
	"svc":                 func(p string) (any, error) { return svm.LoadSVC(p) },
	"svr":                 func(p string) (any, error) { return svm.LoadSVR(p) },
}

// A nested Pipeline is loadable too; it is registered here because LoadPipeline
// refers back to this table, which a package-level initializer cannot do.
func init() {
	loaders["pipeline"] = func(p string) (any, error) { return LoadPipeline(p) }
}

// saveBlob runs an estimator's Save into a temporary file and returns its bytes.
func saveBlob(s core.Saver) ([]byte, error) {
	f, err := os.CreateTemp("", "scikitgo-step-*.gob")
	if err != nil {
		return nil, err
	}
	path := f.Name()
	f.Close()
	defer os.Remove(path)
	if err := s.Save(path); err != nil {
		return nil, err
	}
	return os.ReadFile(path)
}

// loadBlob writes blob to a temporary file and runs the kind's loader on it.
func loadBlob(kind string, blob []byte) (any, error) {
	load, ok := loaders[kind]
	if !ok {
		return nil, fmt.Errorf("unknown step kind %q", kind)
	}
	f, err := os.CreateTemp("", "scikitgo-step-*.gob")
	if err != nil {
		return nil, err
	}
	path := f.Name()
	defer os.Remove(path)
	if _, err := f.Write(blob); err != nil {
		f.Close()
		return nil, err
	}
	if err := f.Close(); err != nil {
		return nil, err
	}
	return load(path)
}

// Save writes the fitted pipeline, with every step, to path in the versioned
// gob format. Every step must be one of this module's persistable estimators.
func (p *Pipeline) Save(path string) error {
	if !p.fitted {
		return matutil.ErrNotFitted
	}
	payload := pipelineGob{Version: pipelineFormatVersion}
	for _, s := range p.steps {
		kind, ok := kindOf(s.Estimator)
		saver, canSave := s.Estimator.(core.Saver)
		if !ok || !canSave {
			return fmt.Errorf("Pipeline.Save: step %q (%T) cannot be persisted", s.Name, s.Estimator)
		}
		blob, err := saveBlob(saver)
		if err != nil {
			return fmt.Errorf("Pipeline.Save: step %q: %w", s.Name, err)
		}
		payload.Names = append(payload.Names, s.Name)
		payload.Kinds = append(payload.Kinds, kind)
		payload.Blobs = append(payload.Blobs, blob)
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("Pipeline.Save: %w", err)
	}
	defer f.Close()
	if err := gob.NewEncoder(f).Encode(payload); err != nil {
		return fmt.Errorf("Pipeline.Save: encode failed: %w", err)
	}
	return nil
}

// LoadPipeline reads a fitted pipeline previously written by Save.
func LoadPipeline(path string) (*Pipeline, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("LoadPipeline: %w", err)
	}
	defer f.Close()

	var payload pipelineGob
	if err := gob.NewDecoder(f).Decode(&payload); err != nil {
		return nil, fmt.Errorf("LoadPipeline: decode failed: %w", err)
	}
	if payload.Version != pipelineFormatVersion {
		return nil, fmt.Errorf("LoadPipeline: unsupported format version %d (expected %d)", payload.Version, pipelineFormatVersion)
	}
	if len(payload.Names) != len(payload.Kinds) || len(payload.Names) != len(payload.Blobs) {
		return nil, fmt.Errorf("LoadPipeline: corrupt payload: %d names, %d kinds, %d blobs",
			len(payload.Names), len(payload.Kinds), len(payload.Blobs))
	}
	steps := make([]Step, len(payload.Names))
	for i := range steps {
		est, err := loadBlob(payload.Kinds[i], payload.Blobs[i])
		if err != nil {
			return nil, fmt.Errorf("LoadPipeline: step %q: %w", payload.Names[i], err)
		}
		steps[i] = Step{Name: payload.Names[i], Estimator: est}
	}
	p, err := NewPipeline(steps...)
	if err != nil {
		return nil, fmt.Errorf("LoadPipeline: %w", err)
	}
	p.fitted = true
	return p, nil
}
