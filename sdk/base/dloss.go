package base

import (
	"math"

	"github.com/lnashier/go-ml/sdk/data"
	"gonum.org/v1/gonum/mat"
)

// DLoss ...
func DLoss(l string) DLossFn {
	switch l {
	case "mse":
		return DMse
	case "mae":
		return DMae
	case "rss":
		return DRss
	case "rmse":
		return DRmse
	case "msle":
		return DMsle
	case "ce":
		return DCe
	}
	return nil
}

// DLossFn defines derivative loss function w.r.t theta
type DLossFn func(m int, h float64, x *mat.VecDense, point *data.Point) *mat.VecDense

// DMse ...
func DMse(m int, h float64, x *mat.VecDense, point *data.Point) *mat.VecDense {
	x.ScaleVec((-2/float64(m))*(point.Observed-h), x)
	return x
}

// DMae ...
func DMae(m int, h float64, x *mat.VecDense, point *data.Point) *mat.VecDense {
	x.ScaleVec((1/float64(m))*math.Abs(point.Observed-h), x)
	return x
}

// DRss ...
func DRss(_ int, h float64, x *mat.VecDense, point *data.Point) *mat.VecDense {
	x.ScaleVec(-2*(point.Observed-h), x)
	return x
}

// DRmse ...
func DRmse(m int, h float64, x *mat.VecDense, point *data.Point) *mat.VecDense {
	x.ScaleVec((-1/(2*float64(m)))*(1/(point.Observed-h)), x)
	return x
}

// DMsle ...
func DMsle(m int, h float64, x *mat.VecDense, point *data.Point) *mat.VecDense {
	x.ScaleVec((2/(float64(m)*math.Pow(math.Log(10), 2)))*(math.Log(h+1)-math.Log(point.Observed)), x)
	return x
}

// DCe ...
func DCe(m int, h float64, x *mat.VecDense, point *data.Point) *mat.VecDense {
	x.ScaleVec((1/float64(m))*(h-point.Observed), x)
	return x
}
