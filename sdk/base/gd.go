package base

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/lnashier/go-ml/sdk/data"
	"gonum.org/v1/gonum/mat"
)

// GD finds model parameters values with Gradient descent.
func GD(opts *Opts, pts []data.Point, h func(w, x *mat.VecDense) float64) ([]float64, error) {
	fmt.Println("GD#Solving ...")
	defer fmt.Println("GD#Solved")

	if len(pts) < 3 {
		return nil, data.ErrNotEnough
	}

	m := len(pts)
	n := len(pts[0].Vars)

	if m < (n + 1) {
		return nil, data.ErrTooManyVars
	}

	dloss := DLoss(opts.Loss)
	if dloss == nil {
		return nil, ErrNoLossFn
	}

	b := int(math.Max(1, math.Min(opts.Batch*float64(m), float64(m))))
	if b == m {
		b--
	}

	fmt.Printf("Batch Size: %d\n", b)

	w := mat.NewVecDense(n+1, nil)

	for e := 0; e < opts.Epochs; e++ {
		d := mat.NewVecDense(n+1, nil)
		bs := rand.Intn(m - b)
		for _, point := range pts[bs : bs+b] {
			x := mat.NewVecDense(n+1, append([]float64{1}, point.Vars...))
			d.AddVec(d, dloss(m, h(w, x), x, &point))
		}
		d.ScaleVec(opts.Lr, d)
		w.SubVec(w, d)
	}

	coeffs := make([]float64, n+1)
	coeffs[0] = w.AtVec(0)
	for c := 1; c <= n; c++ {
		coeffs[c] = w.AtVec(c)
	}

	return coeffs, nil
}
