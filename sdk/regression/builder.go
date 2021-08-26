package regression

import (
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"github.com/lnashier/go-ml/sdk/data"
	"github.com/lnashier/go-ml/sdk/file"
	"github.com/lnashier/go-ml/sdk/file/csv"
	"github.com/lnashier/go-ml/sdk/zson"
	"github.com/spf13/viper"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"image/color"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path"
)

// Run builds regression model by running through all the ops defined in config
func Run(cfg *viper.Viper) error {
	fmt.Println("Regression starting ...")
	defer fmt.Println("Regression done !")

	var schema data.Schema

	if err := cfg.UnmarshalKey("schema", &schema); err != nil {
		return err
	}

	var ops []Op

	if err := cfg.UnmarshalKey("ops", &ops); err != nil {
		return err
	}

	for _, op := range ops {
		switch {
		case op.Type == OpTypePlot && op.Enabled:
			var args PlotOpArgs
			zson.Unmarshal(zson.Marshal(op.Args), &args)
			Plot(schema, args)
		case op.Type == OpTypeSplit && op.Enabled:
			var args SplitOpArgs
			zson.Unmarshal(zson.Marshal(op.Args), &args)
			Split(args)
		case op.Type == OpTypeFit && op.Enabled:
			var args FitOpArgs
			zson.Unmarshal(zson.Marshal(op.Args), &args)
			Fit(schema, args)
		case op.Type == OpTypeTest && op.Enabled:
			var args TestOpArgs
			zson.Unmarshal(zson.Marshal(op.Args), &args)
			Test(schema, args)
		}
	}

	return nil
}

// Plot draws scatter plots for each feature against target
func Plot(schema data.Schema, args PlotOpArgs) {
	fmt.Println("Plotting ...")
	defer fmt.Println("Plotted !")

	// Open the dataset file
	f, err := os.Open(args.Input.Data)
	if err != nil {
		log.Fatal(err)
	}
	defer func(f *os.File) {
		_ = f.Close()
	}(f)

	// Create a dataframe from the CSV file
	// The types of the columns will be inferred
	adDF := dataframe.ReadCSV(f)

	// Extract the target column.
	yVals := adDF.Col(schema.Target).Float()

	// Make sure dir exists
	if err := os.MkdirAll(args.Output.Plot, os.ModePerm); err != nil {
		log.Fatal(err)
	}

	for _, chart := range args.Input.Charts {
		switch chart {
		case "histogram":
			fmt.Println("Plotting histogram ...")

			for _, feature := range schema.Features {
				var vals plotter.Values
				for _, floatVal := range adDF.Col(feature).Float() {
					if !math.IsNaN(floatVal) {
						vals = append(vals, floatVal)
					}
				}
				ph := plot.New()
				if err != nil {
					log.Fatal(err)
				}
				ph.Title.Text = fmt.Sprintf("Histogram of %s", feature)
				h, err := plotter.NewHist(vals, 16)
				if err != nil {
					log.Fatal(err)
				}
				h.Normalize(1)
				ph.Add(h)
				if err := ph.Save(10*vg.Inch, 10*vg.Inch, path.Join(args.Output.Plot, "hist-"+feature+".png")); err != nil {
					log.Fatal(err)
				}
			}

			fmt.Println("Plotted histogram !")
		case "scatter":
			fmt.Println("Plotting scatter ...")

			for _, feature := range schema.Features {
				var pts plotter.XYs
				for i, floatVal := range adDF.Col(feature).Float() {
					if !math.IsNaN(floatVal) && !math.IsNaN(yVals[i]) {
						pts = append(pts, plotter.XY{
							X: floatVal,
							Y: yVals[i],
						})
					}
				}
				ps := plot.New()
				ps.Title.Text = fmt.Sprintf("Scatter %s - %s", schema.Target, feature)
				ps.X.Label.Text = feature
				ps.Y.Label.Text = schema.Target
				ps.Add(plotter.NewGrid())
				s, err := plotter.NewScatter(pts)
				if err != nil {
					log.Fatal(err)
				}
				s.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 255}
				s.GlyphStyle.Radius = vg.Points(3)
				ps.Add(s)
				if err := ps.Save(10*vg.Inch, 10*vg.Inch, path.Join(args.Output.Plot, "scatter-"+schema.Target+"-"+feature+".png")); err != nil {
					log.Fatal(err)
				}
			}

			fmt.Println("Plotted scatter !")
		}
	}
}

// Split divides dataset into 2 parts Training and Testing
func Split(args SplitOpArgs) {
	fmt.Println("Splitting ...")
	defer fmt.Println("Split !")

	// Open the dataset file
	f, err := os.Open(args.Input.Data)
	if err != nil {
		log.Fatal(err)
	}
	defer func(f *os.File) {
		_ = f.Close()
	}(f)

	// Create a dataframe from the CSV file.
	// The types of the columns will be inferred.
	adDF := dataframe.ReadCSV(f)

	// Calculate the number of elements in each set default: 80/20
	// https://en.wikipedia.org/wiki/Overfitting
	ratio := args.Input.Ratio
	if ratio <= 0 || ratio >= 1 {
		ratio = 0.8
	}
	trainingNum := int(ratio * float64(adDF.Nrow()))
	testNum := adDF.Nrow() - trainingNum
	if trainingNum+testNum < adDF.Nrow() {
		trainingNum++
	}

	// Create the subset indices
	trainingIdx := make([]int, trainingNum)
	testIdx := make([]int, testNum)

	// Enumerate the training indices.
	for i := 0; i < trainingNum; i++ {
		trainingIdx[i] = i
	}

	// Enumerate the test indices.
	for i := 0; i < testNum; i++ {
		testIdx[i] = trainingNum + i
	}

	if err := csv.Save(args.Output.Training, adDF.Subset(trainingIdx)); err != nil {
		log.Fatal(err)
	}
	if err := csv.Save(args.Output.Test, adDF.Subset(testIdx)); err != nil {
		log.Fatal(err)
	}
}

// Fit trains the regression model with training dataset
func Fit(schema data.Schema, args FitOpArgs) {
	fmt.Println("Fitting ...")
	defer fmt.Println("Fitted !")

	// Read training dataset
	pts := csv.Read(schema, args.Input.Data)

	// Linear regression
	var rr Linear
	rr.SetOpts(args.Input.Opts)
	rr.SetHeader(data.Header{
		Observed: schema.Target,
		Vars:     schema.Features,
	})
	// Train the regression
	if err := rr.Fit(pts); err != nil {
		log.Fatal(err)
	}
	// Measure regression performance
	rpt := rr.Measure(pts)

	if err := file.Save(args.Output.Report, zson.Marshal(&rpt)); err != nil {
		log.Fatal(err)
	}

	if err := file.Save(args.Output.Model, zson.Marshal(rr.Get())); err != nil {
		log.Fatal(err)
	}
}

// Test tests the regression model with test dataset
func Test(schema data.Schema, args TestOpArgs) {
	fmt.Println("Testing ...")
	defer fmt.Println("Tested !")

	// Read trained model
	md, err := ioutil.ReadFile(args.Input.Model)
	if err != nil {
		log.Fatal(err)
	}

	var m Model
	zson.Unmarshal(md, &m)

	// Linear regression model
	var rr Linear
	rr.Load(m)
	rr.SetHeader(data.Header{
		Observed: schema.Target,
		Vars:     schema.Features,
	})
	rpt := rr.Measure(csv.Read(schema, args.Input.Data))

	if err := file.Save(args.Output.Report, zson.Marshal(rpt)); err != nil {
		log.Fatal(err)
	}
}
