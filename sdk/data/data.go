package data

import (
	"errors"
	"log"
	"math"
	"strconv"
)

var (
	// ErrNotEnough signals that there weren't enough datapoint to train the model.
	ErrNotEnough = errors.New("not enough data points")
	// ErrTooManyVars signals that there are too many variables for the number of data points being made.
	ErrTooManyVars = errors.New("not enough data points to support this many variables")
)

// Schema defines data
type Schema struct {
	Target   string   `json:"target"`
	Features []string `json:"features"`
}

// Header ...
type Header struct {
	Observed string   `json:"observed"`
	Vars     []string `json:"vars"`
}

// Point defines an observation that can used to train the regression
type Point struct {
	Observed float64
	Vars     []float64
}

// NewPoint creates a new data point with given observed and variables
func NewPoint(obs float64, vars []float64) Point {
	return Point{
		Observed: obs,
		Vars:     vars,
	}
}

// Parse ...
func Parse(s Schema, h map[string]int, rs [][]string) []Point {
	var points []Point

Loop:
	for _, r := range rs {
		// Parse the target values
		yVal, err := strconv.ParseFloat(r[h[s.Target]], 64)
		if err != nil {
			log.Fatal(err)
		}
		if math.IsNaN(yVal) || math.IsInf(yVal, 0) {
			continue Loop
		}

		// Parse the features' values
		var fVals []float64
		for _, f := range s.Features {
			fVal, err := strconv.ParseFloat(r[h[f]], 64)
			if err != nil {
				log.Fatal(err)
			}
			if math.IsNaN(fVal) || math.IsInf(fVal, 0) {
				continue Loop
			}
			fVals = append(fVals, fVal)
		}
		points = append(points, NewPoint(yVal, fVals))
	}

	return points
}
