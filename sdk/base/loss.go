package base

import (
	"math"

	"github.com/lnashier/go-ml/sdk/data"
)

// Loss ...
func Loss(l string) LossFn {
	switch l {
	case "mse":
		return Mse
	case "mae":
		return Mae
	case "rss":
		return Rss
	}
	return nil
}

// LossFn defines loss function
type LossFn func(m int, h float64, point *data.Point) float64

// Mse ...
func Mse(m int, h float64, point *data.Point) float64 {
	return math.Pow(point.Observed-h, 2) / float64(m)
}

// Mae ...
func Mae(m int, h float64, point *data.Point) float64 {
	return math.Abs(point.Observed-h) / float64(m)
}

// Rss ...
func Rss(m int, h float64, point *data.Point) float64 {
	return math.Pow(point.Observed-h, 2) / float64(m)
}
