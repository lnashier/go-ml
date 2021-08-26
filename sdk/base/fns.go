package base

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Line ...
func Line(w, x *mat.VecDense) float64 {
	return mat.Dot(w, x)
}

// Sigmoid ...
func Sigmoid(w, x *mat.VecDense) float64 {
	return 1 / (1 + math.Exp(-mat.Dot(w, x)))
}
