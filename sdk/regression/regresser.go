package regression

import (
	"fmt"
	"math"

	"github.com/lnashier/go-ml/sdk/base"
	"github.com/lnashier/go-ml/sdk/data"
	"gonum.org/v1/gonum/mat"
)

// Regresser provides API to build regression model
type Regresser struct {
	opts      base.Opts
	header    data.Header
	intercept float64
	coeffs    []float64
	fitted    bool
}

// SetOpts sets required opts
func (r *Regresser) SetOpts(opts base.Opts) {
	r.opts = opts
}

// SetHeader sets data header
func (r *Regresser) SetHeader(h data.Header) {
	r.header = h
}

// Load loads pretrained model
func (r *Regresser) Load(m Model) {
	r.intercept = m.Intercept
	if len(m.Coeffs) > 0 {
		r.coeffs = make([]float64, len(r.header.Vars))
		for i, f := range r.header.Vars {
			r.coeffs[i] = m.Coeffs[f]
		}
	}
}

// Get returns the trained model
func (r *Regresser) Get() Model {
	coeffs := make(map[string]float64)
	for i, f := range r.header.Vars {
		coeffs[f] = r.coeffs[i]
	}
	return Model{
		Intercept: r.intercept,
		Coeffs:    coeffs,
	}
}

// fit trains the model if there are enough data points present to run the regression.
// If training has already completed on provided data points then function returns an error.
func (r *Regresser) fit(pts []data.Point, h func(w, x *mat.VecDense) float64) error {
	fmt.Println("Regresser#Fitting ...")
	defer fmt.Println("Regresser#Fitted")
	if r.fitted {
		return ErrFitted
	}
	r.fitted = true

	coeffs, err := base.GD(&r.opts, pts, h)
	if err != nil {
		return err
	}
	r.intercept = coeffs[0]
	r.coeffs = coeffs[1:]
	return nil
}

// Predict calculates the target value using trained model
func (r *Regresser) Predict(vars []float64) float64 {
	if !r.fitted {
		return 0
	}
	p := r.intercept
	for j := 0; j < len(r.coeffs); j++ {
		p += r.coeffs[j] * vars[j]
	}
	return p
}

// Measure model performance
// https://en.wikipedia.org/wiki/Coefficient_of_determination
func (r *Regresser) Measure(pts []data.Point) Report {
	m := len(pts)
	var ototal, ptotal float64
	var predictions []float64

	for i := 0; i < m; i++ {
		predictions = append(predictions, r.Predict(pts[i].Vars))
		ototal += pts[i].Observed
		ptotal += predictions[i]
	}

	oavg := ototal / float64(m)
	pavg := ptotal / float64(m)

	// TSS = ESS + RSS
	var tss, ess, rss float64

	for i := 0; i < m; i++ {
		tss += math.Pow(pts[i].Observed-oavg, 2)
		ess += math.Pow(predictions[i]-pavg, 2)
		rss += math.Pow(pts[i].Observed-predictions[i], 2)
	}

	return Report{
		N:   m,
		MSE: rss / float64(m),
		R2:  1 - (rss / tss),
	}
}
