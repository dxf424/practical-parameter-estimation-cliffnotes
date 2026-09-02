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


def sliding_param_groups(n_groups, group_size, stride):
    """Convenience builder for OVERLAPPING param groups, to hand to
    generate_coefficients_block_triangular's `param_groups`. Group g is
    [g*stride, g*stride + group_size). With stride < group_size consecutive
    groups share parameters: sliding_param_groups(2, 3, 2) -> [[0,1,2], [2,3,4]],
    i.e. x2 is a strong driver of BOTH output groups."""
    return [list(range(g * stride, g * stride + group_size)) for g in range(n_groups)]


def generate_coefficients_block_triangular(n_params, n_outputs, param_group_sizes=None,
                                            output_group_sizes=None, param_groups=None,
                                            sensitivity_scale=3.0, background_scale=0.2, seed=None):
    """Structure 3 -- block-diagonal sensitivity with a small upper-triangular
    background: partitions the outputs into contiguous groups and assigns each
    output group a group of params that are its strong, dedicated drivers (the
    block diagonal, scaled by `sensitivity_scale`). On top of that, params from
    EARLIER groups also get small non-zero coefficients on LATER output groups
    (upper-right, scaled by `background_scale`) -- an early parameter has a
    faint downstream effect -- while params belonging only to LATER groups get
    exactly zero on EARLIER output groups (lower-left): no backward influence.

    Outputs are always a contiguous partition (`output_group_sizes`). The param
    groups come from exactly one of:
      param_group_sizes -- list of sizes summing to n_params: contiguous,
                           DISJOINT groups, so each param drives one output
                           group strongly.
      param_groups      -- explicit list of index lists, which MAY OVERLAP:
                           [[0,1,2], [2,3,4]] makes x2 a strong driver of both
                           output group 0 and output group 1. See
                           sliding_param_groups() for the sliding case. A param
                           in no group at all is inert everywhere (allowed).

    With overlap the three-way rule is applied per parameter, not per group: for
    output group h, a param IN group h is strong (that wins even if the param
    also belongs to a later group), a param in some EARLIER group only is small,
    and a param in only LATER groups is exactly zero.

    Returns (coefficients, param_group_id, output_group_id, param_group_membership):
    param_group_id is (n_params,) giving each param's FIRST group (-1 if it
    belongs to none) -- with overlap that is only a label, so the full picture
    lives in param_group_membership, an (n_params, n_groups) 0/1 matrix.
    """
    if output_group_sizes is None:
        raise ValueError("output_group_sizes is required")
    if (param_group_sizes is None) == (param_groups is None):
        raise ValueError("pass exactly one of param_group_sizes (contiguous, disjoint groups) "
                         "or param_groups (explicit index lists, may overlap)")

    if param_group_sizes is not None:
        if sum(param_group_sizes) != n_params:
            raise ValueError(f"param_group_sizes must sum to n_params ({n_params}), got {sum(param_group_sizes)}")
        starts = np.concatenate([[0], np.cumsum(param_group_sizes)])
        param_groups = [list(range(starts[g], starts[g + 1])) for g in range(len(param_group_sizes))]

    param_groups = [np.asarray(group, dtype=int) for group in param_groups]
    if sum(output_group_sizes) != n_outputs:
        raise ValueError(f"output_group_sizes must sum to n_outputs ({n_outputs}), got {sum(output_group_sizes)}")
    if len(param_groups) != len(output_group_sizes):
        raise ValueError("there must be one param group per output group "
                         f"(got {len(param_groups)} param groups, {len(output_group_sizes)} output groups)")
    for g, group in enumerate(param_groups):
        if group.size == 0:
            raise ValueError(f"param group {g} is empty")
        if group.min() < 0 or group.max() >= n_params:
            raise ValueError(f"param group {g} has indices outside [0, {n_params})")

    rng = np.random.default_rng(seed)
    n_groups = len(param_groups)
    coefficients = np.zeros((n_params, n_outputs))

    # (n_params, n_groups) 0/1: which groups each param is a strong driver of.
    membership = np.zeros((n_params, n_groups), dtype=int)
    for g, group in enumerate(param_groups):
        membership[group, g] = 1

    output_starts = np.concatenate([[0], np.cumsum(output_group_sizes)])
    output_group_id = np.repeat(np.arange(n_groups), output_group_sizes)
    # First group each param belongs to; -1 for a param in no group at all.
    param_group_id = np.where(membership.any(axis=1), membership.argmax(axis=1), -1)

    for h in range(n_groups):
        o0, o1 = output_starts[h], output_starts[h + 1]
        strong = membership[:, h].astype(bool)
        # "Earlier group" is evaluated per param: being in group h wins over
        # also being in an earlier one, so overlap upgrades rather than dilutes.
        earlier = membership[:, :h].any(axis=1) & ~strong
        if strong.any():
            coefficients[strong, o0:o1] = sensitivity_scale * rng.normal(size=(strong.sum(), o1 - o0))
        if earlier.any():
            coefficients[earlier, o0:o1] = background_scale * rng.normal(size=(earlier.sum(), o1 - o0))
        # params in only later groups (lower-left of the block grid): stay zero.

    return coefficients, param_group_id, output_group_id, membership


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
        coefficients, param_group_id, output_group_id, membership = generate_coefficients_block_triangular(
            n_params, n_outputs, seed=seed, **structure_kwargs)
        extra = {"param_group_id": param_group_id, "output_group_id": output_group_id,
                 "param_group_membership": membership}
    else:
        raise ValueError(
            f"unknown structure: {structure!r} (expected 'full', 'band', or 'block_triangular')")
    return {"coefficients": coefficients}, extra


def compute_Y(X, params):
    """y = X @ B, shape (n_samples, n_outputs). `params` is the model-parameter
    dict ({"coefficients": B}); a bare array is also accepted."""
    coefficients = params["coefficients"] if isinstance(params, dict) else params
    return np.atleast_2d(np.asarray(X, dtype=float)) @ coefficients
