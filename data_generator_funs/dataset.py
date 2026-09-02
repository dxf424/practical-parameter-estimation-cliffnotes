import numpy as np

from . import linear, morris, sobol_g
from .netcdf_io import save_dataset_netcdf  # noqa: F401 (re-exported)

"""Top-level dataset assembly, shared by every forward model.

Sample an (n_samples + 1)-row ensemble, generate the model's parameters,
compute Y, hold out the last row as the single "true" input/output pair, and
package everything into a dict.

Every dataset has this shape:
    X, Y          -- the training ensemble: n_samples rows, noise-free (Y is
                     exactly the forward model applied to X).
    x_true        -- one extra sampled row, held out of the ensemble: the
                     input a calibration exercise is trying to recover.
    y_true        -- x_true's output, with observation noise and/or
                     structural error optionally applied -- the single "real"
                     observation a calibration exercise tries to match.
    model_params  -- EVERY array the forward model needs to be re-evaluated
                     from scratch (see each model module's PARAM_NAMES). This
                     is what makes a saved dataset runnable later: run_model.py
                     reloads exactly these and can reproduce Y from any X.

A model module plugs in here by exposing PARAM_RANGE, PARAM_NAMES,
generate_parameters(n_params, n_outputs, ..., seed) -> (params, extra), and
compute_Y(X, params) -> (n_samples, n_outputs).
"""

MODELS = {
    "linear": linear,
    "morris": morris,
    "sobol_g": sobol_g,
}


def get_model(name):
    """Looks up a forward-model module by name."""
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"unknown model: {name!r} (expected one of {sorted(MODELS)})") from None


def sample_X(n_samples, n_params, model="linear", seed=None):
    """Samples inputs uniformly over the model's own parameter range -- [-1, 1]
    for the linear model, [0, 1] for Morris and Sobol' G."""
    low, high = get_model(model).PARAM_RANGE
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=(n_samples, n_params))


def _substreams(seed, n):
    """n independent RNG seeds derived from one master seed, so the ensemble,
    the model parameters, x_b and the observation noise don't share a draw
    sequence (passing `seed` to each of them directly would make, e.g., x_b
    come out identical to the ensemble's first row)."""
    return np.random.SeedSequence(seed).spawn(n)


def add_observation_noise(y, noise_std, seed=None):
    """Adds iid N(0, noise_std) measurement noise to y (any shape). A
    correctly-specified model can average this out given enough data."""
    if noise_std <= 0:
        return y
    rng = np.random.default_rng(seed)
    return y + rng.normal(scale=noise_std, size=np.shape(y))


def add_structural_error(x_true, model, params, structural_idx, seed=None):
    """Injects structural error the way it actually shows up: y_true becomes
    unreachable by ANY single set of parameters, not even x_true itself.

    Draws a second, "wrong" input x_b from the model's parameter range,
    computes what it would produce (y_b = model(x_b)), and splices y_b's
    values into y_true on `structural_idx` outputs only. Since x_b != x_true,
    no single x can reproduce both the untouched outputs (which need x_true)
    and the spliced ones (which need x_b) -- that mismatch is what makes it
    "structural" rather than just noisy: no amount of averaging or better
    sampling removes it.

    Returns (y_true_with_structural_error, x_b, y_b).
    """
    module = get_model(model)
    rng = np.random.default_rng(seed)
    low, high = module.PARAM_RANGE
    n_params = np.asarray(x_true).shape[0]

    x_b = rng.uniform(low, high, size=n_params)
    y_b = module.compute_Y(x_b[None, :], params)[0]
    y_true = module.compute_Y(np.asarray(x_true)[None, :], params)[0].copy()
    y_true[list(structural_idx)] = y_b[list(structural_idx)]
    return y_true, x_b, y_b


def _resolve_structural_idx(n_outputs, structural_idx, n_structural, structural_position, seed):
    """Turns either of the two ways of asking for structural error -- an
    explicit `structural_idx` list, or a count `n_structural` -- into the
    list of output indices to corrupt. Returns [] if neither is given.

    With a count, `structural_position` picks which outputs: "last" (the
    default, matching "replace the last few elements of y") or "random".
    """
    if structural_idx is not None and n_structural is not None:
        raise ValueError("pass either structural_idx or n_structural, not both")

    if structural_idx is not None:
        idx = sorted(set(int(i) for i in structural_idx))
        if idx and (idx[0] < 0 or idx[-1] >= n_outputs):
            raise ValueError(f"structural_idx entries must be in [0, {n_outputs}), got {idx}")
    elif n_structural is not None:
        if not 0 <= n_structural <= n_outputs:
            raise ValueError(f"n_structural must be in [0, n_outputs={n_outputs}], got {n_structural}")
        # Corrupting every output would make y_true exactly y_b -- which x_b
        # then reproduces perfectly, so there'd be no conflicting evidence
        # left and it would stop being structural error at all.
        if n_structural == n_outputs and n_outputs > 0:
            raise ValueError(
                f"n_structural={n_structural} would replace every output, leaving y_true == y_b, "
                f"which x_b reproduces exactly -- that is not structural error. "
                f"Use at most n_outputs - 1 = {n_outputs - 1}.")
        if structural_position == "last":
            idx = list(range(n_outputs - n_structural, n_outputs))
        elif structural_position == "random":
            rng = np.random.default_rng(seed)
            idx = sorted(rng.choice(n_outputs, size=n_structural, replace=False).tolist())
        else:
            raise ValueError(f"structural_position must be 'last' or 'random', got {structural_position!r}")
    else:
        idx = []

    return idx


def generate_dataset(model, n_samples, n_params, n_outputs, model_kwargs=None,
                     noise_std=0.0, structural_idx=None, n_structural=None,
                     structural_position="last", seed=None, nc_path=None):
    """Generates one dataset from any registered forward model.

    model: "linear", "morris" or "sobol_g".
    model_kwargs: passed through to that model's generate_parameters, e.g.
        linear   -> structure="band", structure_kwargs=dict(n_sensitive_para=3)
        morris   -> n_third_order=5, n_fourth_order=4, permute_params=False
        sobol_g  -> a=[...], permute_per_output=True

    Structural error -- ask for it either way (not both):
        n_structural:   HOW MANY outputs to corrupt. `structural_position`
                        chooses which ones: "last" (default) or "random".
        structural_idx: exactly WHICH output indices to corrupt.
    Giving neither skips structural error. See add_structural_error.
    """
    module = get_model(model)
    model_kwargs = model_kwargs or {}
    seed_X, seed_params, seed_struct, seed_noise, seed_pick = _substreams(seed, 5)

    structural_idx = _resolve_structural_idx(
        n_outputs, structural_idx, n_structural, structural_position, seed_pick)

    X_all = sample_X(n_samples + 1, n_params, model=model, seed=seed_X)
    X, x_true = X_all[:-1], X_all[-1]

    params, extra = module.generate_parameters(n_params, n_outputs, seed=seed_params, **model_kwargs)
    Y = module.compute_Y(X, params)

    if structural_idx:
        y_true, x_b, y_b = add_structural_error(x_true, model, params, structural_idx, seed=seed_struct)
    else:
        y_true, x_b, y_b = module.compute_Y(x_true[None, :], params)[0], None, None
    y_true = add_observation_noise(y_true, noise_std, seed=seed_noise)

    low, high = module.PARAM_RANGE
    dataset = {
        "X": X,
        "Y": Y,
        "x_true": x_true,
        "y_true": y_true,
        "model_params": params,
        "metadata": {
            "model": model,
            "model_kwargs": model_kwargs,
            "param_names": list(module.PARAM_NAMES),
            "param_low": low,
            "param_high": high,
            "n_samples": n_samples,
            "n_params": n_params,
            "n_outputs": n_outputs,
            "noise_std": noise_std,
            "structural_idx": structural_idx,
            "n_structural": len(structural_idx),
            "seed": seed,
        },
    }
    dataset.update(extra)
    if x_b is not None:
        dataset["x_b"] = x_b
        dataset["y_b"] = y_b

    if nc_path:
        save_dataset_netcdf(dataset, nc_path)
    return dataset


def generate_linear_dataset(n_samples, n_params, n_outputs, structure="full", structure_kwargs=None,
                             noise_std=0.0, structural_idx=None, n_structural=None,
                             structural_position="last", seed=None, nc_path=None):
    """Convenience wrapper around generate_dataset(model="linear", ...) that
    keeps `structure`/`structure_kwargs` as top-level arguments."""
    return generate_dataset(
        "linear", n_samples, n_params, n_outputs,
        model_kwargs=dict(structure=structure, structure_kwargs=structure_kwargs or {}),
        noise_std=noise_std, structural_idx=structural_idx, n_structural=n_structural,
        structural_position=structural_position, seed=seed, nc_path=nc_path)
