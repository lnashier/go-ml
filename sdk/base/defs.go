package base

import "errors"

var (
	// ErrNoLossFn signals that loss function is missing
	ErrNoLossFn = errors.New("loss func is missing")
)

// Opts ...
type Opts struct {
	Epochs int     `json:"epochs"`
	Batch  float64 `json:"batch"`
	Loss   string  `json:"loss"`
	Lr     float64 `json:"lr"`
}
