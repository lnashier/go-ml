package regression

import (
	"fmt"

	"github.com/lnashier/go-ml/sdk/base"
	"github.com/lnashier/go-ml/sdk/data"
	"gonum.org/v1/gonum/mat"
)

// Linear provides API to build linear regression model.
// https://en.wikipedia.org/wiki/Linear_regression
type Linear struct {
	Regresser
}

// Fit trains the model if there are enough data points
// present to run the regression. If training has already completed on provided
// data points then function returns an error.
func (r *Linear) Fit(pts []data.Point) error {
	fmt.Println("Linear#Fit ...")
	defer fmt.Println("Linear#Fitted")
	return r.fit(pts, func(w, x *mat.VecDense) float64 {
		return base.Line(w, x)
	})
}
