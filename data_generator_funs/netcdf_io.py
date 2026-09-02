import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

"""Convert a dataset dict (as returned by generate_linear_dataset in
dataset.py) into a self-documenting NetCDF file: every variable carries a
`description` attribute explaining what it is, and the run's `metadata` dict
is carried over as global attributes (plus a full JSON copy) -- so opening
the file (`xr.open_dataset(path)`, or `ncdump -h`) tells you what you're
looking at without cross-referencing this codebase.

Every dataset gets at least:
    X(sample, input_feature)   -- the training ensemble's inputs (noise-free)
    Y(sample, output_feature)  -- the training ensemble's outputs (noise-free)
    x_true(input_feature)      -- one extra sampled row, held out of the
                                   ensemble: the input a calibration exercise
                                   is trying to recover
    y_true(output_feature)     -- the generating process applied to x_true,
                                   with observation noise/structural bias --
                                   the single "real" observation a
                                   calibration exercise tries to match
which is already a plain (sample x feature) table -- `ds["Y"].to_pandas()` /
`ds.to_dataframe()` gives you rows-are-samples, columns-are-outputs directly.

Everything else in the dict (coefficients, group ids, the structural-error
x_b/y_b, ...) is carried over as additional variables named after their dict
key; nested dicts are flattened with a "__" separator.
"""

_KNOWN_DESCRIPTIONS = {
    "X": "Training-ensemble inputs: sampled parameters, iid Uniform(-1, 1). Noise-free.",
    "Y": "Training-ensemble outputs: the generating process applied exactly to X, with no "
         "observation error added. Noise-free -- this is the simulator/model output, not "
         "an observation.",
    "x_true": "One extra sampled row, held out of the training ensemble: the input a "
              "calibration exercise is trying to recover.",
    "y_true": "x_true's output, with observation noise (noise_std) and/or structural error "
              "(structural_idx) applied -- the single real-world-like observation a "
              "calibration exercise tries to match. Not part of the training ensemble.",
    "x_b": "The 'wrong' input used to generate structural error (see add_structural_error) "
           "-- only present when structural_idx was non-empty.",
    "y_b": "x_b's output (X_b @ coefficients); y_true's structural_idx entries were spliced "
           "in from here, so no single x can reproduce y_true exactly.",
    "param_group_id": "Which block-diagonal sensitivity group each input belongs to (structure="
                       "'block_triangular'); that group's params are the strong, dedicated "
                       "drivers of the matching output_group_id outputs.",
    "output_group_id": "Which block-diagonal sensitivity group each output belongs to (structure="
                        "'block_triangular'); driven mainly by the matching param_group_id params.",
    "coefficients": "MODEL PARAMETER (linear): coefficient matrix B (input x output), Y = X @ B.",
    "beta0": "MODEL PARAMETER (morris): constant offset per output.",
    "beta1": "MODEL PARAMETER (morris): first-order coefficients (input x output).",
    "beta2": "MODEL PARAMETER (morris): second-order interaction coefficients "
             "(input x input x output); only the i<j upper triangle is used.",
    "beta3": "MODEL PARAMETER (morris): third-order interaction coefficients over the leading "
             "n_third_order inputs, shape (n3, n3, n3, output); only i<j<l entries are used.",
    "beta4": "MODEL PARAMETER (morris): fourth-order interaction coefficients over the leading "
             "n_fourth_order inputs, shape (n4, n4, n4, n4, output); only i<j<l<s entries are used.",
    "w_special_idx": "MODEL PARAMETER (morris): input indices using the curved transform "
                      "w = 2(1.1x/(x+0.1) - 1/2) instead of the plain w = 2(x - 1/2).",
    "param_permutation": "MODEL PARAMETER (morris): per-output relabelling of the input axis "
                          "applied before the canonical coefficients (identity unless "
                          "permute_params=True).",
    "a": "MODEL PARAMETER (sobol_g): the a_i constants (input x output) in "
         "prod_i (|4x_i - 2| + a_i)/(1 + a_i). Small a_i = influential input, large a_i = inert.",
    "a_base": "The unpermuted a vector the per-output columns of `a` were shuffled from.",
    "dominant_params": "Indices of the inputs carrying Morris's large fixed coefficients.",
}


def _describe(key):
    base_key = key.split("__")[-1]
    return _KNOWN_DESCRIPTIONS.get(base_key, f"Model array '{key}' (see metadata for the generator that produced it).")


def _json_safe(value):
    """netCDF global attrs can only hold strings/numbers/non-empty 1-D arrays of
    those (no bools, no None, no empty arrays) -- coerce or JSON-encode anything
    else (nested dicts/lists of mixed type)."""
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, np.integer, np.floating)):
        return value
    if (isinstance(value, (list, tuple)) and len(value) > 0
            and all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in value)):
        return list(value)
    return json.dumps(value)


def to_xarray(dataset):
    """Builds the xarray.Dataset described in the module docstring from a
    dataset dict. Does not write anything to disk -- see save_dataset_netcdf."""
    X, Y = dataset["X"], dataset["Y"]
    n_samples, n_inputs = X.shape
    n_outputs = Y.shape[1]
    meta = dataset.get("metadata", {})

    data_vars = {
        "X": (("sample", "input_feature"), X),
        "Y": (("sample", "output_feature"), Y),
    }

    for key, value in dataset.items():
        if key in ("X", "Y", "metadata"):
            continue
        if key == "model_params":
            # The forward model's own parameters are written as top-level
            # variables under their own names (coefficients, beta1, a, ...),
            # so run_model.py can reload them by name and re-run the model.
            for param_name, param_value in value.items():
                _add_variable(data_vars, param_name, param_value, n_samples, n_inputs, n_outputs)
            continue
        _add_variable(data_vars, key, value, n_samples, n_inputs, n_outputs)

    ds = xr.Dataset(data_vars)

    for name in ds.data_vars:
        ds[name].attrs["description"] = _describe(name)

    ds.attrs["summary"] = (
        f"Synthetic parameter-estimation dataset generated by data_generator_funs "
        f"(model={meta.get('model', 'unknown')}). X/Y = noise-free training ensemble, "
        f"x_true/y_true = the single held-out true input and its noisy/biased observation. "
        f"See each variable's 'description' attribute."
    )
    for key, value in meta.items():
        ds.attrs[key] = _json_safe(value)
    ds.attrs["metadata_json"] = json.dumps(meta)

    return ds


def _add_variable(data_vars, key, value, n_samples, n_inputs, n_outputs, prefix=""):
    name = f"{prefix}{key}"
    if isinstance(value, dict):
        if len(value) == 1 and key in value:
            # collapses the {"coefficients": B} wrapper so it
            # doesn't produce a redundant "coefficients__coefficients" variable name
            _add_variable(data_vars, key, value[key], n_samples, n_inputs, n_outputs, prefix=prefix)
            return
        for sub_key, sub_value in value.items():
            _add_variable(data_vars, sub_key, sub_value, n_samples, n_inputs, n_outputs, prefix=f"{name}__")
        return

    arr = np.asarray(value)
    if arr.ndim == 0:
        data_vars[name] = ((), arr)
    elif arr.ndim == 1:
        if arr.shape[0] == n_samples:
            dim = "sample"
        elif arr.shape[0] == n_outputs:
            dim = "output_feature"
        elif arr.shape[0] == n_inputs:
            dim = "input_feature"
        else:
            dim = f"{name}_index"
        data_vars[name] = ((dim,), arr)
    elif arr.ndim == 2:
        row_dim = "input_feature" if arr.shape[0] == n_inputs else f"{name}_row"
        col_dim = "output_feature" if arr.shape[-1] == n_outputs else f"{name}_col"
        data_vars[name] = ((row_dim, col_dim), arr)
    else:
        dims = tuple(f"{name}_dim{i}" for i in range(arr.ndim))
        data_vars[name] = (dims, arr)


def save_dataset_netcdf(dataset, path):
    """Writes `to_xarray(dataset)` to a NetCDF file at `path`, creating parent
    directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = to_xarray(dataset)
    ds.to_netcdf(path)
    return ds


def save_dataset_folder(dataset, out_dir, filename_base=None):
    """Writes one dataset into its own folder: out_dir/<filename_base>/, containing
    <filename_base>.nc (everything `save_dataset_netcdf` writes) plus X.csv, Y.csv,
    x_true.csv and y_true.csv. X/Y are the (sample x feature) training ensemble,
    columns named x0..x{n_params-1} / y0..y{n_outputs-1}. x_true/y_true are the single
    held-out row, written one value per line with those same names as the index
    (matching the ",0"-header single-column CSV format used elsewhere for obs tables).

    filename_base defaults to `default_nc_filename(dataset)` with the .nc suffix
    stripped, so the folder name matches what the .nc file would otherwise be
    called.
    """
    if filename_base is None:
        filename_base = Path(default_nc_filename(dataset)).stem

    folder = Path(out_dir) / filename_base
    folder.mkdir(parents=True, exist_ok=True)

    ds = save_dataset_netcdf(dataset, folder / f"{filename_base}.nc")

    X, Y = dataset["X"], dataset["Y"]
    x_cols = [f"x{i}" for i in range(X.shape[1])]
    y_cols = [f"y{i}" for i in range(Y.shape[1])]
    pd.DataFrame(X, columns=x_cols).to_csv(folder / "X.csv", index=True, index_label="sample")
    pd.DataFrame(Y, columns=y_cols).to_csv(folder / "Y.csv", index=True, index_label="sample")

    x_true, y_true = dataset.get("x_true"), dataset.get("y_true")
    if isinstance(x_true, np.ndarray):
        pd.Series(x_true, index=x_cols).to_csv(folder / "x_true.csv")
    if isinstance(y_true, np.ndarray):
        pd.Series(y_true, index=y_cols).to_csv(folder / "y_true.csv")

    return ds


def load_linear_dataset(path):
    """Loads a dataset written by generate_linear_dataset(..., nc_path=...):
    `path` can be either the folder save_dataset_folder wrote (the first .nc
    file found inside it is used) or a .nc file directly. Returns a dict with
    X, Y, x_true, y_true, coefficients as plain numpy arrays.
    """
    path = Path(path)
    if path.is_dir():
        nc_files = sorted(path.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"no .nc file found in {path}")
        path = nc_files[0]

    with xr.open_dataset(path) as ds:
        return {
            "X": ds["X"].values,
            "Y": ds["Y"].values,
            "x_true": ds["x_true"].values,
            "y_true": ds["y_true"].values,
            "coefficients": ds["coefficients"].values,
        }


def load_model(path):
    """Reads back everything needed to RE-RUN a saved model: its name, its
    full parameter set, and the metadata describing its shape and input range.

    This is the counterpart to the `model_params` block written by to_xarray,
    and is what run_model.py uses to evaluate a saved model at newly proposed
    parameter values. `path` may be a .nc file or a folder containing one.

    Returns (model_name, params, metadata).
    """
    path = Path(path)
    if path.is_dir():
        nc_files = sorted(path.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"no .nc file found in {path}")
        path = nc_files[0]

    with xr.open_dataset(path) as ds:
        meta = json.loads(ds.attrs["metadata_json"]) if "metadata_json" in ds.attrs else dict(ds.attrs)
        model_name = meta.get("model")
        if model_name is None:
            raise ValueError(f"{path} has no 'model' in its metadata -- cannot tell which model to run")

        param_names = meta.get("param_names")
        if not param_names:
            raise ValueError(
                f"{path} does not record 'param_names', so its model parameters cannot be "
                f"identified. It was probably written by an older version of this code -- "
                f"regenerate it with data_generator_funs.dataset.generate_dataset.")

        missing = [name for name in param_names if name not in ds]
        if missing:
            raise ValueError(f"{path} is missing model parameter(s) {missing}; cannot re-run the model")

        params = {name: ds[name].values for name in param_names}

    return model_name, params, meta


def default_nc_filename(dataset):
    """A descriptive, sortable default filename encoding the run's structure
    and key size/seed settings, e.g. "linear_full_n300-p20-o10_seed1.nc" --
    enough to tell datasets apart when browsing a directory without opening
    each file.
    """
    meta = dataset.get("metadata", {})
    parts = [str(meta.get("model", "dataset"))]

    structure = meta.get("structure")
    if structure:
        parts.append(str(structure))

    if "n_params" in meta:
        parts.append(f"n{meta['n_samples']}-p{meta['n_params']}-o{meta['n_outputs']}")
    elif "n_samples" in meta:
        parts.append(f"n{meta['n_samples']}")

    if meta.get("seed") is not None:
        parts.append(f"seed{meta['seed']}")

    return "_".join(parts) + ".nc"
