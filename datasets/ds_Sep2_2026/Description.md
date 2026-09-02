# Parameter Estimation Benchmark Datasets

This repository contains synthetic datasets for testing and comparing parameter-estimation methods using a linear model:

\[
Ax = b
\]

Each dataset includes 300 samples, 20 input parameters, and 30 model outputs. Parameter values range from −1 to 1.

## Purpose

The datasets provide known ground-truth parameters and model outputs, making it possible to evaluate:

- Parameter recovery accuracy
- Sensitivity to observation errors
- Effects of parameter scaling
- Robustness to selected output perturbations
- Differences between estimated and true parameter values

## Dataset Variants

The repository includes datasets with different scaling conditions and controlled modifications to selected outputs, including:

- `y1` and `y3`
- `y20`, `y25`, and `y29`

These variants can be used to test how parameter-estimation methods respond to uncertain or modified observations.

## Dataset Contents

Each dataset contains:

- `X.csv` — sampled parameter values
- `x_true.csv` — ground-truth parameter vector
- `Y.csv` — simulated model outputs
- `y_true.csv` — ground-truth observation vector
- `.nc` file — complete dataset with coefficients, feature groups, ground-truth values, and boundary-test values

## Dataset Dimensions

| Variable | Dimensions | Description |
|---|---|---|
| `X` | `(sample, input_feature)` | Sampled input parameters |
| `Y` | `(sample, output_feature)` | Corresponding model outputs |
| `x_true` | `(input_feature)` | Ground-truth parameter values |
| `y_true` | `(output_feature)` | Ground-truth model output |
| `coefficients` | `(input_feature, output_feature)` | Linear-model coefficients |
| `param_group_id` | `(input_feature)` | Parameter-group identifiers |
| `output_group_id` | `(output_feature)` | Output-group identifiers |
| `x_b` | `(input_feature)` | Boundary-test parameter values |
| `y_b` | `(output_feature)` | Boundary-test model outputs |

## Dataset Configuration

- **Model:** Linear model (`Ax = b`)
- **Samples:** 300
- **Input parameters:** 20
- **Model outputs:** 30
- **Parameter range:** −1 to 1

## Applications

These datasets are intended for benchmarking optimization algorithms, inverse-problem solvers, uncertainty analysis, and other parameter-estimation techniques.
