# Regression Examples (Config)

These are just some toy examples. They are provided so that you can tweak parameters in `cfg.json` of each example to
build some more understanding of both SDK and working of regression algorithm.

Linear Regression model expects the data to be normally distributed. Meaning, histogram of the data takes a bell shaped
curve.

## Linear

### Build

```
$ go run main.go -prb problem1/
```

You can send a different problem path:

```
$ go run main.go -prb problem2/
```

Alternatively, you can build the program once:

```
# go build main.go
``` 

and execute again and again.

```
$ ./main -prb problem1/
$ ./main -prb problem2/
```

### Output

Program creates <i>scratch</i> folder under same path as problem. You should see following files after successfully
running the program:

#### data

`data` folder contains test and training data created according to `cfg.json`. For instance linear/problem1 has
following:

* test.csv
* training.csv

#### plot

`plot` folder contains various plots that were requested in `cfg.json`. For instance linear/problem1 has the following
plots:

* hist-tv
* scatter-sales-tv

#### report

`report` folder stores two files `test.json` and `train.json`.

```json
{
  "n": 40,
  "r2": -5.624073665997599,
  "mse": 0.27417851128875
}
```

where

* `n`: Number of samples
* `r2`: Coefficient of determination
* `mse`: Mean squared error

#### model.json

```json
{
  "coeffs": {
    "tv": 0.5695823728693639
  },
  "intercept": 0.2158978690050924
}
```

where

* `coeffs`: Regression coefficients
* `intercept`: Constant