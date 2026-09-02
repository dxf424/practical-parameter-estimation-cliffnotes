This repo aims at testing, examining, and highlighting advantages and limitations of different parameter estimation and emulator methods.
We begin this project by generating different kinds of datasets and test them with the methods.

## Forward models

Three forward models, all behind the same interface:

| model | function | input range |
|---|---|---|
| `linear` | `y = x @ B`, with `full` / `band` / `block_triangular` coefficient structures | [-1, 1] |
| `morris` | Morris (1991) screening function | [0, 1] |
| `sobol_g` | Sobol' G-function, `prod_i (|4x_i - 2| + a_i)/(1 + a_i)` | [0, 1] |

```python
from data_generator_funs.dataset import generate_dataset

ds = generate_dataset("morris", n_samples=400, n_params=20, n_outputs=6,
                      noise_std=0.1,      # observation error
                      n_structural=2,     # structural error on 2 outputs
                      seed=0,
                      nc_path="datasets/my_run/my_run.nc")
```

Every array the model needs is kept in `ds["model_params"]` and written into the
`.nc`, so a saved dataset can be re-run later, not just read.

## Walkthrough notebooks

- `notebooks/linear_dataset_walkthrough.ipynb` -- the three linear coefficient
  structures, observation noise, and structural error.
- `notebooks/forward_models_walkthrough.ipynb` -- what the Morris and Sobol' G
  functions are and why they're used, with visualisations, plus the shared
  error-injection / saving / contest machinery on all three models.

## Running a model on proposed parameters

`run_model.py` evaluates a saved model at parameter values someone proposes:

```bash
python run_model.py linear_demo Yang.csv           # -> Yang_output_Iteration1.csv
python run_model.py linear_demo Yang.csv           # -> Yang_output_Iteration2.csv
python run_model.py morris_demo Yang.csv --score   # also reports RMSE vs y_true
```

- `MODEL` is a name under `datasets/` (e.g. `linear_demo`) or a path to a `.nc`.
- `PARAMS` is a `.csv` with columns `x0..x{k-1}`, one row per proposed parameter
  set, or a `.nc`.
- The output is written next to `PARAMS` and the iteration number steps past
  whatever already exists, so successive submissions never overwrite each other.
  Use `-o` to name the file yourself or `--iteration N` to pin the number.

Ready-to-use example models are in `datasets/linear_demo/`, `datasets/morris_demo/`
and `datasets/sobol_g_demo/`.
