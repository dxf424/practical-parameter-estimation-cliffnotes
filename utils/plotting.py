import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BAR_COLOR = "#2a78d6"
_HIGHLIGHT = "#4a3aa7"


def plot_correlation(X, Y, sensitive_param_idx=None, sensitive_output_idx=None):
    """Heatmap of the correlation between each column of Y and each column of X.
    No axis tick labels and no per-cell numeric labels -- color only.
    sensitive_param_idx / sensitive_output_idx (if given) outline the rows/
    columns the generator marked as sensitive, so you can see whether the
    amplified coefficients actually show up as stronger correlation.
    """
    n_params = X.shape[1]
    combined = np.hstack([X, Y])
    corr = np.corrcoef(combined, rowvar=False)
    corr_yx = corr[n_params:, :n_params]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    im = ax.imshow(corr_yx, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x", color=_INK_SECONDARY)
    ax.set_ylabel("y", color=_INK_SECONDARY)
    ax.set_title("Correlation between y's and x's", color=_INK_PRIMARY)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")

    if sensitive_param_idx is not None and len(sensitive_param_idx) > 0:
        for j in sensitive_param_idx:
            ax.add_patch(Rectangle((j - 0.5, -0.5), 1, corr_yx.shape[0],
                                    fill=False, edgecolor=_HIGHLIGHT, linewidth=1.5))
    if sensitive_output_idx is not None and len(sensitive_output_idx) > 0:
        for i in sensitive_output_idx:
            ax.add_patch(Rectangle((-0.5, i - 0.5), corr_yx.shape[1], 1,
                                    fill=False, edgecolor=_HIGHLIGHT, linewidth=1.5))

    fig.tight_layout()
    plt.show()
    return corr_yx


def sparsity_per_output(coefficients, tol=1e-8):
    """Fraction of near-zero coefficient entries per output column.

    `coefficients` is the dict stored on a dataset (`{"coefficients": B}`).
    """
    arr = coefficients["coefficients"]
    return (np.abs(arr) < tol).sum(axis=0) / arr.shape[0]


def nonlinearity_per_output(X, Y_true):
    """1 - R^2 of the best linear (OLS) fit of Y_true from X, per output column.

    For this repo's generators this should come out ~0 everywhere -- it's the
    check that the data really is linear, not a measurement of anything.
    """
    n_samples = X.shape[0]
    X_design = np.hstack([np.ones((n_samples, 1)), X])
    beta, *_ = np.linalg.lstsq(X_design, Y_true, rcond=None)
    Y_hat = X_design @ beta

    ss_res = ((Y_true - Y_hat) ** 2).sum(axis=0)
    ss_tot = ((Y_true - Y_true.mean(axis=0)) ** 2).sum(axis=0)
    ss_tot = np.where(ss_tot == 0, 1.0, ss_tot)

    return np.clip(ss_res / ss_tot, 0, 1)


def plot_data_overview(dataset):
    """Bar charts summarising a dataset per output: coefficient sparsity (for
    the linear model, which is the only one with a coefficient matrix) and
    nonlinearity -- 1 - R² of the best linear fit, which is ~0 for the linear
    model and large for Morris / Sobol' G."""
    X = dataset["X"]
    Y = dataset["Y"]
    model_params = dataset.get("model_params", {})

    nonlinearity = nonlinearity_per_output(X, Y)
    panels = []
    if "coefficients" in model_params:
        panels.append((sparsity_per_output(model_params),
                       "Coefficient sparsity per output", "fraction zero"))
    panels.append((nonlinearity,
                   "Nonlinearity per output (1 - R² of best linear fit; ~0 = linear)", "1 - R²"))

    n_outputs = Y.shape[1]
    idx = np.arange(n_outputs)

    fig, axes = plt.subplots(len(panels), 1, figsize=(max(6, n_outputs * 0.35), 3 * len(panels)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.patch.set_facecolor(_SURFACE)
    for ax, (values, title, ylabel) in zip(axes, panels):
        ax.set_facecolor(_SURFACE)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=_GRIDLINE, linewidth=0.8)
        ax.bar(idx, values, color=_BAR_COLOR, width=0.7, zorder=3)
        ax.set_ylim(0, 1)
        ax.set_title(title, color=_INK_PRIMARY, loc="left", fontsize=11)
        ax.set_ylabel(ylabel, color=_INK_MUTED, fontsize=9)
        ax.tick_params(colors=_INK_MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xlabel("output index", color=_INK_MUTED, fontsize=9)
    axes[-1].set_xticks(idx)
    fig.tight_layout()
    plt.show()
    return nonlinearity


def plot_scatter_examples(dataset, n_examples=3, seed=None):
    """Small-multiples scatter of Y vs. its most correlated X column, for a
    few outputs -- a quick visual check that the relationship looks straight
    (as a linear generator should)."""
    X = dataset["X"]
    Y = dataset["Y"]
    n_outputs = Y.shape[1]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_outputs, size=min(n_examples, n_outputs), replace=False).tolist()

    fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 3.5))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(_SURFACE)

    for ax, k in zip(axes, chosen):
        corrs = [np.corrcoef(X[:, i], Y[:, k])[0, 1] for i in range(X.shape[1])]
        best_param = int(np.argmax(np.abs(corrs)))
        ax.set_facecolor(_SURFACE)
        ax.scatter(X[:, best_param], Y[:, k], s=10, color=_BAR_COLOR, alpha=0.6, edgecolors="none")
        ax.set_title(f"output {k} vs param {best_param}", color=_INK_PRIMARY, fontsize=10, loc="left")
        ax.set_xlabel(f"x[{best_param}]", color=_INK_MUTED, fontsize=9)
        ax.set_ylabel(f"y[{k}]", color=_INK_MUTED, fontsize=9)
        ax.tick_params(colors=_INK_MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    plt.show()


def plot_block_structure(dataset):
    """Heatmap of |coefficients| for a structure="block_triangular" dataset,
    with lines marking the block boundaries from param_group_id/output_group_id
    -- the direct visual check that each group's diagonal block is strong,
    the upper-right background blocks are small-but-nonzero, and the
    lower-left blocks are exactly zero.
    """
    param_group_id = dataset.get("param_group_id")
    output_group_id = dataset.get("output_group_id")
    if param_group_id is None or output_group_id is None:
        raise ValueError(
            "dataset has no param_group_id/output_group_id -- "
            "was it generated with structure='block_triangular'?")

    coefficients = dataset["model_params"]["coefficients"]

    fig, (ax, ax_mask) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(_SURFACE)

    ax.set_facecolor(_SURFACE)
    im = ax.imshow(np.abs(coefficients), cmap="Blues", aspect="auto")
    ax.set_xlabel("output", color=_INK_SECONDARY)
    ax.set_ylabel("param", color=_INK_SECONDARY)
    ax.set_title("|coefficients| (magnitude)", color=_INK_PRIMARY, loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|coefficient|")

    # The magnitude panel's scale is set by the strong diagonal blocks, which
    # renders the small upper-right background and the exactly-zero lower-left
    # identically white -- this panel separates those three cases explicitly.
    mask = np.zeros_like(coefficients)
    mask[np.abs(coefficients) > 0] = 1
    strong = np.abs(coefficients) > 0.5 * np.abs(coefficients).max()
    mask[strong] = 2
    ax_mask.set_facecolor(_SURFACE)
    im2 = ax_mask.imshow(mask, cmap="viridis", aspect="auto", vmin=0, vmax=2)
    ax_mask.set_xlabel("output", color=_INK_SECONDARY)
    ax_mask.set_title("zero / small / strong", color=_INK_PRIMARY, loc="left", fontsize=11)
    cbar = fig.colorbar(im2, ax=ax_mask, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["exactly 0", "small", "strong"])

    for axis in (ax, ax_mask):
        for boundary in np.flatnonzero(np.diff(output_group_id)) + 0.5:
            axis.axvline(boundary, color=_HIGHLIGHT, linewidth=1.2)
        for boundary in np.flatnonzero(np.diff(param_group_id)) + 0.5:
            axis.axhline(boundary, color=_HIGHLIGHT, linewidth=1.2)

    fig.tight_layout()
    plt.show()
    return coefficients


def _random_baselines(meta, n_baselines, seed):
    """Random points in the model's input range, used as the "hold the other
    parameters here" states for one-at-a-time sweeps.

    Deliberately NOT the range midpoint: for the Sobol' G-function every
    factor hits its minimum at the midpoint, and any parameter with a_i = 0
    drives its factor to exactly 0 there -- which zeroes the whole product and
    makes every other parameter look perfectly inert. Random baselines avoid
    that degeneracy and, by varying where the other inputs sit, also expose
    interaction effects instead of hiding them.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(meta["param_low"], meta["param_high"], size=(n_baselines, meta["n_params"]))


def oat_sweep(dataset, param_idx, output_idx=0, n_points=101, baseline=None, seed=0):
    """One-at-a-time sweep: vary parameter `param_idx` across the model's full
    input range while holding every other parameter at `baseline`, and
    evaluate the forward model.

    This is the most direct way to see the *shape* a forward model gives each
    input -- a straight line for the linear model, a V for Sobol' G, a ramp or
    a sharp curve for Morris depending on the parameter.

    baseline defaults to a random point in the input range (see
    _random_baselines for why not the midpoint).

    Returns (grid, values): the swept parameter values and the output.
    """
    from data_generator_funs.dataset import get_model

    meta = dataset["metadata"]
    module = get_model(meta["model"])
    low, high = meta["param_low"], meta["param_high"]

    if baseline is None:
        baseline = _random_baselines(meta, 1, seed)[0]
    grid = np.linspace(low, high, n_points)

    X = np.tile(np.asarray(baseline, dtype=float), (n_points, 1))
    X[:, param_idx] = grid
    return grid, module.compute_Y(X, dataset["model_params"])[:, output_idx]


def plot_oat_curves(dataset, param_indices=None, output_idx=0, n_points=101,
                    n_baselines=8, seed=0, ncols=5):
    """Small-multiples of `oat_sweep` for several parameters at once -- the
    "what does this forward model actually look like" panel.

    Each subplot draws the sweep from `n_baselines` different random states of
    the *other* parameters. That serves two purposes: it avoids any single
    degenerate baseline, and the spread between curves is itself the signal --
    parallel curves mean the parameter acts independently, curves that cross or
    change shape mean it interacts with the others. Subplots share a y-axis, so
    a visibly flat panel is a parameter that barely moves this output.
    """
    meta = dataset["metadata"]
    if param_indices is None:
        param_indices = range(min(meta["n_params"], 10))
    param_indices = list(param_indices)

    baselines = _random_baselines(meta, n_baselines, seed)
    curves = {
        idx: [oat_sweep(dataset, idx, output_idx, n_points, baseline) for baseline in baselines]
        for idx in param_indices
    }
    all_values = np.concatenate([v for sweeps in curves.values() for _g, v in sweeps])
    lo, hi = all_values.min(), all_values.max()
    pad = 0.05 * (hi - lo) if hi > lo else 1.0

    nrows = int(np.ceil(len(param_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.3 * nrows),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(_SURFACE)

    for ax, idx in zip(axes.ravel(), param_indices):
        ax.set_facecolor(_SURFACE)
        for grid, values in curves[idx]:
            ax.plot(grid, values, color=_BAR_COLOR, linewidth=1.2, alpha=0.55)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"x[{idx}]", color=_INK_PRIMARY, fontsize=10, loc="left")
        ax.tick_params(colors=_INK_MUTED, labelsize=8)
        ax.grid(color=_GRIDLINE, linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.ravel()[len(param_indices):]:
        ax.set_visible(False)

    fig.suptitle(
        f"One-at-a-time response of output {output_idx} -- {meta['model']} "
        f"({n_baselines} random states of the other parameters)",
        color=_INK_PRIMARY, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    return curves


def oat_sensitivity(dataset, n_points=51, n_baselines=32, seed=0):
    """Model-agnostic sensitivity estimate: for every (parameter, output) pair,
    how far the output travels as that parameter sweeps its full range --
    averaged over `n_baselines` random states of the other parameters.

    Averaging over baselines rather than using one fixed point matters. A
    single baseline gives a purely local answer, and for the Sobol' G-function
    the obvious choice (the range midpoint) is outright degenerate: every
    factor bottoms out there, so one a_i = 0 parameter zeroes the whole product
    and every other input looks inert. Averaging also lets some of the
    interaction effect show up, since the sweep is repeated at different
    positions of the other inputs.

    Returns (n_params, n_outputs).
    """
    from data_generator_funs.dataset import get_model

    meta = dataset["metadata"]
    module = get_model(meta["model"])
    low, high = meta["param_low"], meta["param_high"]
    n_params, n_outputs = meta["n_params"], meta["n_outputs"]

    baselines = _random_baselines(meta, n_baselines, seed)
    grid = np.linspace(low, high, n_points)

    sensitivity = np.zeros((n_params, n_outputs))
    for i in range(n_params):
        ranges = np.zeros((n_baselines, n_outputs))
        for b, baseline in enumerate(baselines):
            X = np.tile(baseline, (n_points, 1))
            X[:, i] = grid
            Y = module.compute_Y(X, dataset["model_params"])
            ranges[b, :] = Y.max(axis=0) - Y.min(axis=0)
        sensitivity[i, :] = ranges.mean(axis=0)
    return sensitivity


def plot_sensitivity_heatmap(dataset, normalize=True):
    """Heatmap of `oat_sensitivity` -- which inputs actually drive which
    outputs. Normalised per output by default so columns are comparable."""
    sensitivity = oat_sensitivity(dataset)
    shown = sensitivity.copy()
    if normalize:
        col_max = shown.max(axis=0, keepdims=True)
        shown = shown / np.where(col_max == 0, 1.0, col_max)

    meta = dataset["metadata"]
    fig, ax = plt.subplots(figsize=(max(5, meta["n_outputs"] * 0.45), max(4, meta["n_params"] * 0.28)))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    im = ax.imshow(shown, cmap="Blues", aspect="auto")
    ax.set_xlabel("output", color=_INK_SECONDARY)
    ax.set_ylabel("parameter", color=_INK_SECONDARY)
    ax.set_title(f"One-at-a-time sensitivity -- {meta['model']}"
                 + (" (normalised per output)" if normalize else ""),
                 color=_INK_PRIMARY, loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="output range" + (" (relative)" if normalize else ""))
    fig.tight_layout()
    plt.show()
    return sensitivity


def plot_sobol_indices(dataset, output_idx=0):
    """For a Sobol' G dataset: the analytic first-order and total sensitivity
    indices next to the `a` constants that produce them. Small a_i means an
    influential parameter -- this panel is the direct check of that claim."""
    from data_generator_funs import sobol_g

    if dataset["metadata"]["model"] != "sobol_g":
        raise ValueError("plot_sobol_indices expects a model='sobol_g' dataset")

    S1, ST = sobol_g.analytic_indices(dataset["model_params"])
    a = dataset["model_params"]["a"][:, output_idx]
    idx = np.arange(len(a))
    width = 0.4

    fig, (ax_a, ax_s) = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor(_SURFACE)

    for ax in (ax_a, ax_s):
        ax.set_facecolor(_SURFACE)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=_GRIDLINE, linewidth=0.8)
        ax.tick_params(colors=_INK_MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_a.bar(idx, a, color=_INK_MUTED, width=0.7, zorder=3)
    ax_a.set_title(f"a values (output {output_idx}) -- small a = influential",
                   color=_INK_PRIMARY, loc="left", fontsize=11)
    ax_a.set_xlabel("parameter", color=_INK_MUTED, fontsize=9)

    ax_s.bar(idx - width / 2, S1[:, output_idx], width, color=_BAR_COLOR, label="S1 (first order)", zorder=3)
    ax_s.bar(idx + width / 2, ST[:, output_idx], width, color=_HIGHLIGHT, label="ST (total)", zorder=3)
    ax_s.set_title("Analytic Sobol' indices", color=_INK_PRIMARY, loc="left", fontsize=11)
    ax_s.set_xlabel("parameter", color=_INK_MUTED, fontsize=9)
    ax_s.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    plt.show()
    return S1[:, output_idx], ST[:, output_idx]
