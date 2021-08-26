package regression

import (
	"fmt"

	"github.com/lnashier/go-ml/sdk/base"
	"github.com/lnashier/go-ml/sdk/data"
	"gonum.org/v1/gonum/mat"
)

// Logistic provides API to build logistic regression model.
// https://en.wikipedia.org/wiki/Logistic_regression
// https://en.wikipedia.org/wiki/Multinomial_logistic_regression
type Logistic struct {
	Regresser
}

// Fit trains the model if there are enough data points present to run the regression.
// If training has already completed on provided data points then function returns an error.
func (r *Logistic) Fit(pts []data.Point) error {
	fmt.Println("Logistic#Fit ...")
	defer fmt.Println("Logistic#Fitted")
	return r.fit(pts, func(w, x *mat.VecDense) float64 {
		return base.Sigmoid(w, x)
	})
}
