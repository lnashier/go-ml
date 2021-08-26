package regression

import (
	"errors"
)

var (
	// ErrFitted signals that the training has already been called on the trained dataset.
	ErrFitted = errors.New("regression already fitted")
)

// Model defines the regression
type Model struct {
	Coeffs    map[string]float64 `json:"coeffs"`
	Intercept float64            `json:"intercept"`
}

// Report ...
type Report struct {
	N   int     `json:"n,omitempty"`
	R2  float64 `json:"r2,omitempty"`
	MSE float64 `json:"mse,omitempty"`
}
