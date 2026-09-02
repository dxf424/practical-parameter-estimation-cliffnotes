import numpy as np

"""Linear forward model: y = x @ B, with three choices of structure for the
coefficient matrix B (see the generate_coefficients_* functions)."""

PARAM_RANGE = (-1.0, 1.0)
PARAM_NAMES = ("coefficients",)


def sample_X(n_samples, n_params, low=-1.0, high=1.0, seed=None):
    """Draws the input matrix X ~ Uniform(low, high), shape (n_samples, n_params)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=(n_samples, n_params))


def generate_coefficients_full(n_params, n_outputs, seed=None):
    """Structure 1 -- full random matrix: every output depends on every input,
    coefficients ~ N(0, 1). No sparsity, no structure."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_params, n_outputs))


def generate_coefficients_band(n_params, n_outputs, n_sensitive_para, seed=None):
    """Structure 2 -- sliding diagonal band: output j is driven only by params
    [j, j + n_sensitive_para). E.g. n_sensitive_para=3 means x0,x1,x2 drive y0,
    x1,x2,x3 drive y1, x2,x3,x4 drive y2, etc. -- everything off that band is
    exactly zero.
    """
    rng = np.random.default_rng(seed)
    coefficients = np.zeros((n_params, n_outputs))
    for j in range(n_outputs):
        start, end = j, min(j + n_sensitive_para, n_params)
        if start < end:
            coefficients[start:end, j] = rng.normal(size=end - start)
    return coefficients


def generate_coefficients_block_triangular(n_params, n_outputs, param_group_sizes, output_group_sizes,
                                            sensitivity_scale=3.0, background_scale=0.2, seed=None):
    """Structure 3 -- block-diagonal sensitivity with a small upper-triangular
    background: partitions params/outputs into matching contiguous groups.
    Group g's params are the strong, dedicated drivers of group g's outputs
    (the block diagonal, scaled by `sensitivity_scale`). On top of that,
    EARLIER param groups also get small non-zero coefficients on LATER output
    groups (upper-right, scaled by `background_scale`) -- an early parameter
    has a faint downstream effect -- while LATER param groups get exactly zero
    on EARLIER output groups (lower-left): no backward influence.

    param_group_sizes / output_group_sizes: lists of group sizes, same number
    of groups, summing to n_params / n_outputs respectively.

    Returns (coefficients, param_group_id, output_group_id): the last two are
    (n_params,) / (n_outputs,) arrays giving each param's/output's group index.
    """
    if sum(param_group_sizes) != n_params:
        raise ValueError(f"param_group_sizes must sum to n_params ({n_params}), got {sum(param_group_sizes)}")
    if sum(output_group_sizes) != n_outputs:
        raise ValueError(f"output_group_sizes must sum to n_outputs ({n_outputs}), got {sum(output_group_sizes)}")
    if len(param_group_sizes) != len(output_group_sizes):
        raise ValueError("param_group_sizes and output_group_sizes must have the same number of groups")

    rng = np.random.default_rng(seed)
    n_groups = len(param_group_sizes)
    coefficients = np.zeros((n_params, n_outputs))

    param_group_id = np.repeat(np.arange(n_groups), param_group_sizes)
    output_group_id = np.repeat(np.arange(n_groups), output_group_sizes)
    param_starts = np.concatenate([[0], np.cumsum(param_group_sizes)])
    output_starts = np.concatenate([[0], np.cumsum(output_group_sizes)])

    for g in range(n_groups):
        p0, p1 = param_starts[g], param_starts[g + 1]
        for h in range(n_groups):
            o0, o1 = output_starts[h], output_starts[h + 1]
            if h == g:
                coefficients[p0:p1, o0:o1] = sensitivity_scale * rng.normal(size=(p1 - p0, o1 - o0))
            elif h > g:
                coefficients[p0:p1, o0:o1] = background_scale * rng.normal(size=(p1 - p0, o1 - o0))
            # h < g (lower-left of the block grid): stays zero.

    return coefficients, param_group_id, output_group_id


def generate_parameters(n_params, n_outputs, structure="full", structure_kwargs=None, seed=None):
    """Model-interface entry point: builds the coefficient matrix under the
    requested structure. Returns (params, extra) -- extra carries the group
    ids for structure="block_triangular", and is empty otherwise."""
    structure_kwargs = structure_kwargs or {}
    if structure == "full":
        coefficients = generate_coefficients_full(n_params, n_outputs, seed=seed, **structure_kwargs)
        extra = {}
    elif structure == "band":
        coefficients = generate_coefficients_band(n_params, n_outputs, seed=seed, **structure_kwargs)
        extra = {}
    elif structure == "block_triangular":
        coefficients, param_group_id, output_group_id = generate_coefficients_block_triangular(
            n_params, n_outputs, seed=seed, **structure_kwargs)
        extra = {"param_group_id": param_group_id, "output_group_id": output_group_id}
    else:
        raise ValueError(
            f"unknown structure: {structure!r} (expected 'full', 'band', or 'block_triangular')")
    return {"coefficients": coefficients}, extra


def compute_Y(X, params):
    """y = X @ B, shape (n_samples, n_outputs). `params` is the model-parameter
    dict ({"coefficients": B}); a bare array is also accepted."""
    coefficients = params["coefficients"] if isinstance(params, dict) else params
    return np.atleast_2d(np.asarray(X, dtype=float)) @ coefficients
