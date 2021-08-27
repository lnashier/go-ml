package main

import "github.com/lnashier/go-ml/sdk/base"

// OpType defines type of operation
type OpType string

const (
	// OpTypePlot ...
	OpTypePlot OpType = "plot"
	// OpTypeSplit ...
	OpTypeSplit OpType = "split"
	// OpTypeFit ...
	OpTypeFit OpType = "fit"
	// OpTypeTest ...
	OpTypeTest OpType = "test"
)

// Op defines an operation
type Op struct {
	Type    OpType      `json:"type"`
	Enabled bool        `json:"enabled"`
	Args    interface{} `json:"args"`
}

// PlotOpArgs defines plot op arguments
type PlotOpArgs struct {
	Input struct {
		Data   string   `json:"data"`
		Charts []string `json:"charts"`
	} `json:"input"`
	Output struct {
		Plot string `json:"plot"`
	} `json:"output"`
}

// SplitOpArgs defines split op arguments
type SplitOpArgs struct {
	Input struct {
		Data    string  `json:"data"`
		Ratio   float64 `json:"ratio"`
		Shuffle bool    `json:"shuffle"`
	} `json:"input"`
	Output struct {
		Training string `json:"training"`
		Test     string `json:"test"`
	} `json:"output"`
}

// FitOpArgs defines fit op arguments
type FitOpArgs struct {
	Input struct {
		Data string    `json:"data"`
		Opts base.Opts `json:"opts"`
	} `json:"input"`
	Output struct {
		Report string `json:"report"`
		Model  string `json:"model"`
	} `json:"output"`
}

// TestOpArgs defines test op arguments
type TestOpArgs struct {
	Input struct {
		Data  string `json:"data"`
		Model string `json:"model"`
	} `json:"input"`
	Output struct {
		Report string `json:"report"`
	} `json:"output"`
}
