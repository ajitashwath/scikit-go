// Package pipeline chains transformers and a final estimator into a single
// estimator, mirroring sklearn.pipeline.Pipeline: Fit fits every step on the
// output of the previous one, and Predict, PredictProba, Transform and Score
// push new data through the fitted chain.
//
// Steps are ordinary estimators from this module. All but the last must be
// transformers; the last may be a transformer or a predictor.
package pipeline

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"unicode"

	"github.com/ajitashwath/scikit-go/core"
	"github.com/ajitashwath/scikit-go/internal/matutil"
)

// ErrInvalidPipeline is returned when a pipeline is built from unusable steps.
var ErrInvalidPipeline = errors.New("invalid pipeline")

// ErrUnknownParam is returned by SetParams for a parameter that does not exist
// or cannot hold the given value.
var ErrUnknownParam = errors.New("unknown or invalid pipeline parameter")

// Step is a named pipeline stage.
type Step struct {
	Name      string
	Estimator any
}

// Pipeline is a chain of transformers ending in an estimator or transformer.
type Pipeline struct {
	steps  []Step
	fitted bool
}

// Compile-time checks that Pipeline satisfies the core interfaces.
var (
	_ core.Estimator   = (*Pipeline)(nil)
	_ core.Predictor   = (*Pipeline)(nil)
	_ core.Classifier  = (*Pipeline)(nil)
	_ core.Transformer = (*Pipeline)(nil)
	_ core.Saver       = (*Pipeline)(nil)
)

// The shapes of Fit that estimators in this module use, in the order a step is
// fitted through them. Supervised transformers (PCA, SelectKBest, RFE,
// VarianceThreshold) have Fit(X, y); StandardScaler has Fit(X); anything else
// with core.Transformer is fitted through FitTransform.
type (
	transformer interface {
		Transform(X [][]float64) ([][]float64, error)
	}
	supervisedFitter interface {
		Fit(X [][]float64, y []float64) error
	}
	unsupervisedFitter interface {
		Fit(X [][]float64) error
	}
)

// canFit reports whether a step can be fitted by one of the supported routes.
func canFit(est any) bool {
	switch est.(type) {
	case supervisedFitter, unsupervisedFitter, core.Transformer:
		return true
	}
	return false
}

func isTransformer(est any) bool {
	_, ok := est.(transformer)
	return ok && canFit(est)
}

// NewPipeline validates steps and returns an unfitted Pipeline. Names must be
// non-empty, unique, and free of "__" (which separates parameter paths); every
// step but the last must be a transformer, and the last must be a transformer
// or an estimator.
func NewPipeline(steps ...Step) (*Pipeline, error) {
	if len(steps) == 0 {
		return nil, fmt.Errorf("%w: at least one step is required", ErrInvalidPipeline)
	}
	seen := make(map[string]bool, len(steps))
	for i, s := range steps {
		if s.Name == "" || strings.Contains(s.Name, "__") {
			return nil, fmt.Errorf("%w: step %d has invalid name %q", ErrInvalidPipeline, i, s.Name)
		}
		if seen[s.Name] {
			return nil, fmt.Errorf("%w: duplicate step name %q", ErrInvalidPipeline, s.Name)
		}
		seen[s.Name] = true
		if s.Estimator == nil || isNilPointer(s.Estimator) {
			return nil, fmt.Errorf("%w: step %q has a nil estimator", ErrInvalidPipeline, s.Name)
		}
		last := i == len(steps)-1
		switch {
		case !last && !isTransformer(s.Estimator):
			return nil, fmt.Errorf("%w: intermediate step %q (%T) must be a transformer with Transform and a Fit or FitTransform method",
				ErrInvalidPipeline, s.Name, s.Estimator)
		case last && !canFit(s.Estimator):
			return nil, fmt.Errorf("%w: final step %q (%T) cannot be fitted", ErrInvalidPipeline, s.Name, s.Estimator)
		}
	}
	return &Pipeline{steps: append([]Step(nil), steps...)}, nil
}

// MakePipeline builds a Pipeline from bare estimators, naming each step after
// its lower-cased type like sklearn's make_pipeline; repeated types get -1, -2,
// ... suffixes.
func MakePipeline(estimators ...any) (*Pipeline, error) {
	counts := make(map[string]int, len(estimators))
	base := make([]string, len(estimators))
	for i, est := range estimators {
		t := reflect.TypeOf(est)
		for t != nil && t.Kind() == reflect.Ptr {
			t = t.Elem()
		}
		if t == nil {
			return nil, fmt.Errorf("%w: estimator %d is nil", ErrInvalidPipeline, i)
		}
		base[i] = strings.ToLower(t.Name())
		counts[base[i]]++
	}
	steps := make([]Step, len(estimators))
	used := make(map[string]int, len(estimators))
	for i, est := range estimators {
		name := base[i]
		if counts[name] > 1 {
			used[name]++
			name = fmt.Sprintf("%s-%d", name, used[name])
		}
		steps[i] = Step{Name: name, Estimator: est}
	}
	return NewPipeline(steps...)
}

func isNilPointer(v any) bool {
	rv := reflect.ValueOf(v)
	return rv.Kind() == reflect.Ptr && rv.IsNil()
}

// Steps returns a copy of the pipeline's steps.
func (p *Pipeline) Steps() []Step { return append([]Step(nil), p.steps...) }

// Named returns the estimator of the step with the given name.
func (p *Pipeline) Named(name string) (any, bool) {
	for _, s := range p.steps {
		if s.Name == name {
			return s.Estimator, true
		}
	}
	return nil, false
}

// fitTransformStep fits a transformer on (X, y) and returns its output on X.
func fitTransformStep(name string, est any, X [][]float64, y []float64) ([][]float64, error) {
	var err error
	switch e := est.(type) {
	case supervisedFitter:
		err = e.Fit(X, y)
	case unsupervisedFitter:
		err = e.Fit(X)
	case core.Transformer:
		out, ferr := e.FitTransform(X)
		if ferr != nil {
			return nil, fmt.Errorf("pipeline step %q: %w", name, ferr)
		}
		return out, nil
	}
	if err != nil {
		return nil, fmt.Errorf("pipeline step %q: %w", name, err)
	}
	out, err := est.(transformer).Transform(X)
	if err != nil {
		return nil, fmt.Errorf("pipeline step %q: %w", name, err)
	}
	return out, nil
}

// fitOnly fits the final step without needing its output.
func fitOnly(name string, est any, X [][]float64, y []float64) error {
	var err error
	switch e := est.(type) {
	case supervisedFitter:
		err = e.Fit(X, y)
	case unsupervisedFitter:
		err = e.Fit(X)
	case core.Transformer:
		_, err = e.FitTransform(X)
	}
	if err != nil {
		return fmt.Errorf("pipeline step %q: %w", name, err)
	}
	return nil
}

// Fit fits each step in turn on the output of the previous one. y is passed to
// steps that use a target and may be nil when no step needs one.
func (p *Pipeline) Fit(X [][]float64, y []float64) error {
	p.fitted = false
	cur := X
	for _, s := range p.steps[:len(p.steps)-1] {
		out, err := fitTransformStep(s.Name, s.Estimator, cur, y)
		if err != nil {
			return fmt.Errorf("Pipeline.Fit: %w", err)
		}
		cur = out
	}
	last := p.steps[len(p.steps)-1]
	if err := fitOnly(last.Name, last.Estimator, cur, y); err != nil {
		return fmt.Errorf("Pipeline.Fit: %w", err)
	}
	p.fitted = true
	return nil
}

// transformThrough applies the transform of the first n steps.
func (p *Pipeline) transformThrough(X [][]float64, n int) ([][]float64, error) {
	cur := X
	for _, s := range p.steps[:n] {
		out, err := s.Estimator.(transformer).Transform(cur)
		if err != nil {
			return nil, fmt.Errorf("pipeline step %q: %w", s.Name, err)
		}
		cur = out
	}
	return cur, nil
}

func (p *Pipeline) lastStep() Step { return p.steps[len(p.steps)-1] }

// Predict transforms X through every step but the last and returns the last
// step's predictions.
func (p *Pipeline) Predict(X [][]float64) ([]float64, error) {
	if !p.fitted {
		return nil, matutil.ErrNotFitted
	}
	pred, ok := p.lastStep().Estimator.(core.Predictor)
	if !ok {
		return nil, fmt.Errorf("Pipeline.Predict: %w: final step %q (%T) cannot predict",
			ErrInvalidPipeline, p.lastStep().Name, p.lastStep().Estimator)
	}
	cur, err := p.transformThrough(X, len(p.steps)-1)
	if err != nil {
		return nil, fmt.Errorf("Pipeline.Predict: %w", err)
	}
	return pred.Predict(cur)
}

// PredictProba transforms X through every step but the last and returns the
// last step's class probabilities.
func (p *Pipeline) PredictProba(X [][]float64) ([][]float64, error) {
	if !p.fitted {
		return nil, matutil.ErrNotFitted
	}
	clf, ok := p.lastStep().Estimator.(core.Classifier)
	if !ok {
		return nil, fmt.Errorf("Pipeline.PredictProba: %w: final step %q (%T) has no PredictProba",
			ErrInvalidPipeline, p.lastStep().Name, p.lastStep().Estimator)
	}
	cur, err := p.transformThrough(X, len(p.steps)-1)
	if err != nil {
		return nil, fmt.Errorf("Pipeline.PredictProba: %w", err)
	}
	return clf.PredictProba(cur)
}

// Transform pushes X through every step. It requires the final step to be a
// transformer too.
func (p *Pipeline) Transform(X [][]float64) ([][]float64, error) {
	if !p.fitted {
		return nil, matutil.ErrNotFitted
	}
	if !isTransformer(p.lastStep().Estimator) {
		return nil, fmt.Errorf("Pipeline.Transform: %w: final step %q (%T) is not a transformer",
			ErrInvalidPipeline, p.lastStep().Name, p.lastStep().Estimator)
	}
	out, err := p.transformThrough(X, len(p.steps))
	if err != nil {
		return nil, fmt.Errorf("Pipeline.Transform: %w", err)
	}
	return out, nil
}

// FitTransform fits the pipeline to X and returns the transformed data. It has
// no target, so it fails if a step needs one (SelectKBest, RFE).
func (p *Pipeline) FitTransform(X [][]float64) ([][]float64, error) {
	if !isTransformer(p.lastStep().Estimator) {
		return nil, fmt.Errorf("Pipeline.FitTransform: %w: final step %q (%T) is not a transformer",
			ErrInvalidPipeline, p.lastStep().Name, p.lastStep().Estimator)
	}
	p.fitted = false
	cur := X
	for _, s := range p.steps {
		out, err := fitTransformStep(s.Name, s.Estimator, cur, nil)
		if err != nil {
			return nil, fmt.Errorf("Pipeline.FitTransform: %w", err)
		}
		cur = out
	}
	p.fitted = true
	return cur, nil
}

// Score transforms X through every step but the last and returns the last
// step's Score(X, y) (accuracy for classifiers, R^2 for regressors).
func (p *Pipeline) Score(X [][]float64, y []float64) (float64, error) {
	if !p.fitted {
		return 0, matutil.ErrNotFitted
	}
	scorer, ok := p.lastStep().Estimator.(interface {
		Score(X [][]float64, y []float64) (float64, error)
	})
	if !ok {
		return 0, fmt.Errorf("Pipeline.Score: %w: final step %q (%T) has no Score",
			ErrInvalidPipeline, p.lastStep().Name, p.lastStep().Estimator)
	}
	cur, err := p.transformThrough(X, len(p.steps)-1)
	if err != nil {
		return 0, fmt.Errorf("Pipeline.Score: %w", err)
	}
	return scorer.Score(cur, y)
}

// GetParams returns the pipeline's parameters keyed like sklearn's get_params:
// each step under its name, and every exported scalar field (numbers, strings,
// booleans) of a step's estimator as "step__Field", descending into exported
// fields that hold another estimator as "step__Field__Inner".
func (p *Pipeline) GetParams() map[string]any {
	params := make(map[string]any)
	for _, s := range p.steps {
		params[s.Name] = s.Estimator
		collectParams(params, s.Name, reflect.ValueOf(s.Estimator), 0)
	}
	return params
}

const maxParamDepth = 3

func collectParams(out map[string]any, prefix string, v reflect.Value, depth int) {
	for v.Kind() == reflect.Ptr || v.Kind() == reflect.Interface {
		if v.IsNil() {
			return
		}
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct || depth >= maxParamDepth {
		return
	}
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if !f.IsExported() {
			continue
		}
		fv := v.Field(i)
		key := prefix + "__" + f.Name
		switch fv.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
			reflect.Float32, reflect.Float64, reflect.String, reflect.Bool:
			out[key] = fv.Interface()
		case reflect.Ptr, reflect.Interface:
			if !fv.IsNil() {
				out[key] = fv.Interface()
				collectParams(out, key, fv, depth+1)
			}
		}
	}
}

// SetParams updates parameters by the keys GetParams returns. Setting a step's
// name to another estimator replaces that step (and revalidates the pipeline);
// setting "step__Field" assigns a scalar hyperparameter, converting between
// numeric types when the value is representable. The pipeline must be refitted
// afterwards; SetParams marks it unfitted.
func (p *Pipeline) SetParams(params map[string]any) error {
	// Apply on a copy so a failure leaves the pipeline untouched.
	next := append([]Step(nil), p.steps...)
	for key, val := range params {
		path := strings.Split(key, "__")
		idx := -1
		for i, s := range next {
			if s.Name == path[0] {
				idx = i
			}
		}
		if idx < 0 {
			return fmt.Errorf("%w: no step named %q (key %q)", ErrUnknownParam, path[0], key)
		}
		if len(path) == 1 {
			next[idx].Estimator = val
			continue
		}
		if err := setField(reflect.ValueOf(next[idx].Estimator), path[1:], val); err != nil {
			return fmt.Errorf("%w: %q: %v", ErrUnknownParam, key, err)
		}
	}
	validated, err := NewPipeline(next...)
	if err != nil {
		return err
	}
	p.steps = validated.steps
	p.fitted = false
	return nil
}

func setField(v reflect.Value, path []string, val any) error {
	for v.Kind() == reflect.Ptr || v.Kind() == reflect.Interface {
		if v.IsNil() {
			return errors.New("nil estimator on the path")
		}
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		return fmt.Errorf("%s is not a struct", v.Type())
	}
	name := path[0]
	if name == "" || !unicode.IsUpper(rune(name[0])) {
		return fmt.Errorf("%q is not an exported field of %s", name, v.Type())
	}
	f := v.FieldByName(name)
	if !f.IsValid() {
		return fmt.Errorf("%s has no field %q", v.Type(), name)
	}
	if len(path) > 1 {
		return setField(f, path[1:], val)
	}
	return assign(f, val)
}

func kindClass(k reflect.Kind) string {
	switch k {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return "integer"
	case reflect.Float32, reflect.Float64:
		return "float"
	case reflect.String:
		return "string"
	case reflect.Bool:
		return "bool"
	}
	return ""
}

func assign(f reflect.Value, val any) error {
	if !f.CanSet() {
		return fmt.Errorf("field is not settable")
	}
	rv := reflect.ValueOf(val)
	if !rv.IsValid() {
		return fmt.Errorf("nil value")
	}
	fc, vc := kindClass(f.Kind()), kindClass(rv.Kind())
	switch {
	case fc == "" && rv.Type().AssignableTo(f.Type()): // an estimator-valued field
		f.Set(rv)
	case fc == "" || vc == "":
		return fmt.Errorf("cannot assign %s to field of type %s", rv.Type(), f.Type())
	case fc == vc || (fc == "float" && vc == "integer") || (fc == "integer" && vc == "float"):
		conv := rv.Convert(f.Type())
		if fc == "integer" && vc == "float" && conv.Convert(rv.Type()).Interface() != rv.Interface() {
			return fmt.Errorf("value %v is not integral", val)
		}
		f.Set(conv)
	default:
		return fmt.Errorf("cannot assign %s to field of type %s", rv.Type(), f.Type())
	}
	return nil
}
