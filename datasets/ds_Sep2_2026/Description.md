# Parameter Estimation Benchmark Datasets

This repository contains synthetic datasets for testing and comparing parameter-estimation methods using a linear model:

\[
Ax = b
\]
![Linear matrix A visualization](A_coefficients.png)

Each dataset includes 300 samples, 20 input parameters, and 30 model outputs. Parameter values range from −1 to 1.
The matrix A is visualized below. Parameters 0-2 control Outputs 0-7; Parameters 2-4 control Outputs 8-13. The rest variables are dominated by the rest parameters.


## Purpose

The datasets provide known ground-truth parameters and model outputs, making it possible to evaluate:

- Parameter recovery accuracy
- Parameter estimation efficiency 
- How parameter estimation responds to observational and structural uncertainty

## Dataset Contents

Each dataset contains:

- `X.csv` — sampled parameter values
- `x_true.csv` — ground-truth parameter vector <---- The answer
- `Y.csv` — simulated model outputs
- `y_true.csv` — ground-truth observation vector
- `.nc` file — complete dataset, including the above 4, the coefficient matrix A and other helper information. 
Within the nc file, there is x_b and y_b, which are used to generate structural error.

## Dataset Dimensions

| Variable | Dimensions | Description |
|---|---|---|
| `X` | `(sample, input_feature)` | Sampled input parameters |
| `Y` | `(sample, output_feature)` | Corresponding model outputs |
| `x_true` | `(input_feature)` | Ground-truth parameter values |
| `y_true` | `(output_feature)` | Ground-truth model output |
| `coefficients` | `(input_feature, output_feature)` | Linear-model coefficients (A) |
| `x_b` | `(input_feature)` | Boundary-test parameter values |
| `y_b` | `(output_feature)` | Boundary-test model outputs |

## Dataset Configuration

- **Model:** Linear model (`Ax = b`)
- **Samples:** 300
- **Input parameters:** 20
- **Model outputs:** 30
- **Parameter range:** −1 to 1

## How structural error is generated
We randomly generate x_b, and apply it to get y_b: y_b = Ax_b. The certain elements in y_true are replaced by y_b.
See below on which elements are replaced

## Naming entails info on obs and structural error
For the dataset name linear_obserr_scale02_strerr_y20_y25_y29
linear:             The model is linear
obsess_scale02:     Gaussian white noise of std of 0.2 is added to Ax_true to get the final y_true. This is to simulate the observational uncertainty
strerr_y20_y25_y29:  the 20th, 25th, and 29th elements of y_true are replaced by those of y_b. 


These datasets are intended for benchmarking optimization algorithms, inverse-problem solvers, uncertainty analysis, and other parameter-estimation techniques.
