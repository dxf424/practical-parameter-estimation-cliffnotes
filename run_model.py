#!/usr/bin/env python3
"""Run a saved forward model at proposed parameter values -- the contest entry point.

    python run_model.py MODEL PARAMS [-o OUT] [--iteration N] [--score]

MODEL   which model to run. Either a path to a .nc file (or a folder holding
        one) written by generate_dataset(..., nc_path=...), or a bare name
        like "linear" / "morris" / "sobol_g", which is looked up under
        ./datasets (unambiguous partial names work too).
PARAMS  proposed parameter values, .csv or .nc. One row per ensemble member;
        columns x0..x{k-1}. A single-column "name,value" CSV (the shape of the
        x_true.csv files this repo writes) is read as one parameter set.

The output is written next to PARAMS, named after it:

    python run_model.py linear Yang.csv     ->  Yang_output_Iteration1.csv
    (run it again)                          ->  Yang_output_Iteration2.csv

The iteration number auto-increments past whatever is already on disk, so
successive submissions don't overwrite each other; pass --iteration to pin it
or -o to name the file yourself.

--score additionally reports RMSE against the model's stored y_true, when the
file has one.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_generator_funs.dataset import get_model  # noqa: E402
from data_generator_funs.netcdf_io import load_model  # noqa: E402

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


def resolve_model_path(model_arg):
    """Accepts a path to a .nc/folder, or a bare model name to look up under
    ./datasets. Returns a concrete path to a .nc file."""
    path = Path(model_arg)
    if path.exists():
        if path.is_dir():
            nc_files = sorted(path.glob("*.nc"))
            if not nc_files:
                raise FileNotFoundError(f"no .nc file found in {path}")
            return nc_files[0]
        return path

    if not DATASETS_DIR.is_dir():
        raise FileNotFoundError(f"{model_arg!r} is not a path and {DATASETS_DIR} does not exist")

    all_nc = sorted(DATASETS_DIR.rglob("*.nc"))
    # An exact folder/file-stem match wins outright, so "linear_demo" is
    # unambiguous even while the looser "linear" still matches several.
    exact = [p for p in all_nc if model_arg in (p.parent.name, p.stem)]
    candidates = exact or [p for p in all_nc if model_arg in p.name or model_arg in p.parent.name]

    if not candidates:
        available = sorted({p.parent.name for p in all_nc})
        raise FileNotFoundError(
            f"no model matching {model_arg!r} under {DATASETS_DIR}. Available: {available or 'none'}")
    if len(set(c.parent for c in candidates)) > 1:
        options = sorted({c.parent.name for c in candidates})
        raise ValueError(
            f"{model_arg!r} matches more than one model: {options}. "
            f"Use one of those names, or pass a full path.")
    return candidates[0]


def load_proposed_params(path, n_params):
    """Reads proposed parameter values into an (n_proposals, n_params) array.

    Accepts a .nc (variable X, or x_true), a wide CSV with one row per
    proposal, or a single-column "name,value" CSV holding one proposal.
    Column order is taken from x0..x{k-1} names when present, so a reordered
    or index-carrying CSV still lands in the right columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"parameter file not found: {path}")

    expected = [f"x{i}" for i in range(n_params)]

    if path.suffix == ".nc":
        with xr.open_dataset(path) as ds:
            if "X" in ds:
                values = np.atleast_2d(ds["X"].values)
            elif "x_true" in ds:
                values = np.atleast_2d(ds["x_true"].values)
            else:
                raise ValueError(f"{path} has neither an 'X' nor an 'x_true' variable")
    else:
        frame = pd.read_csv(path)
        # Single-column "name,value" layout -> one proposal, read down the column.
        if frame.shape[1] == 2 and set(expected).issubset(set(frame.iloc[:, 0].astype(str))):
            frame = frame.set_index(frame.columns[0])
            values = frame.loc[expected].to_numpy().reshape(1, -1)
        else:
            if set(expected).issubset(frame.columns):
                frame = frame[expected]
            else:
                # No x0..xk-1 header: drop a leading index column if the width
                # only matches after doing so, then take columns in order.
                if frame.shape[1] == n_params + 1:
                    frame = frame.iloc[:, 1:]
                frame = frame.select_dtypes(include=[np.number])
            values = frame.to_numpy()

    values = np.atleast_2d(np.asarray(values, dtype=float))
    if values.shape[1] != n_params:
        raise ValueError(
            f"{path} gives {values.shape[1]} parameters per row, but the model expects {n_params}. "
            f"Expected columns {expected[0]}..{expected[-1]}.")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite (NaN/inf) parameter values")
    return values


def next_output_path(params_path, iteration=None, explicit_out=None):
    """Yang.csv -> Yang_output_Iteration1.csv, then Iteration2, ... The number
    steps past whatever already exists so submissions don't clobber each other."""
    if explicit_out:
        return Path(explicit_out)

    params_path = Path(params_path)
    stem = params_path.stem
    directory = params_path.parent

    if iteration is None:
        iteration = 1
        while (directory / f"{stem}_output_Iteration{iteration}.csv").exists():
            iteration += 1
    return directory / f"{stem}_output_Iteration{iteration}.csv"


def check_in_range(values, low, high, params_path):
    """Warns (does not fail) when proposals fall outside the range the model
    was built on -- the models still evaluate, but extrapolated results are
    not what the training ensemble covers."""
    below, above = values < low, values > high
    if below.any() or above.any():
        n_rows = int(np.any(below | above, axis=1).sum())
        print(
            f"warning: {n_rows} of {values.shape[0]} proposed row(s) in {params_path} fall outside "
            f"this model's input range [{low}, {high}]; evaluating anyway.",
            file=sys.stderr,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a saved forward model at proposed parameter values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:  python run_model.py linear Yang.csv  ->  Yang_output_Iteration1.csv",
    )
    parser.add_argument("model", help="path to a model .nc/folder, or a model name under ./datasets")
    parser.add_argument("params", help="proposed parameter values (.csv or .nc)")
    parser.add_argument("-o", "--out", default=None, help="output path (default: auto-named next to PARAMS)")
    parser.add_argument("--iteration", type=int, default=None, help="pin the iteration number in the filename")
    parser.add_argument("--score", action="store_true", help="also report RMSE against the stored y_true")
    args = parser.parse_args(argv)

    model_path = resolve_model_path(args.model)
    model_name, params, meta = load_model(model_path)
    n_params, n_outputs = meta["n_params"], meta["n_outputs"]

    values = load_proposed_params(args.params, n_params)
    check_in_range(values, meta.get("param_low", -np.inf), meta.get("param_high", np.inf), args.params)

    Y = get_model(model_name).compute_Y(values, params)

    out_path = next_output_path(args.params, args.iteration, args.out)
    frame = pd.DataFrame(Y, columns=[f"y{i}" for i in range(n_outputs)])
    frame.index.name = "proposal"
    frame.to_csv(out_path)

    print(f"model      : {model_name}  ({model_path})")
    print(f"proposals  : {values.shape[0]} row(s) x {n_params} parameters")
    print(f"outputs    : {Y.shape[0]} row(s) x {n_outputs} outputs")
    print(f"written    : {out_path}")

    if args.score:
        with xr.open_dataset(model_path) as ds:
            y_true = ds["y_true"].values if "y_true" in ds else None
        if y_true is None:
            print("score      : unavailable (this model file has no y_true)")
        else:
            rmse = np.sqrt(np.mean((Y - y_true[None, :]) ** 2, axis=1))
            best = int(np.argmin(rmse))
            for i, value in enumerate(rmse):
                print(f"score      : proposal {i} RMSE vs y_true = {value:.6g}")
            print(f"best       : proposal {best} (RMSE {rmse[best]:.6g})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
