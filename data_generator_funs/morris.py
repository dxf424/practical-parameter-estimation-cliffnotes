import itertools

import numpy as np

"""Morris (1991) function -- the standard screening/sensitivity test function
from "Factorial Sampling Plans for Preliminary Computational Experiments",
Technometrics 33(2), 161-174.

    y = b0 + sum_i b1_i w_i
           + sum_{i<j} b2_ij w_i w_j
           + sum_{i<j<l} b3_ijl w_i w_j w_l
           + sum_{i<j<l<s} b4_ijls w_i w_j w_l w_s

with inputs x in [0, 1]^k and

    w_i = 2 (x_i - 1/2)                       for most i
    w_i = 2 (1.1 x_i / (x_i + 0.1) - 1/2)     for i in W_SPECIAL_IDX

Morris's published coefficient assignment makes the first handful of
parameters dominant and deliberately mixes strong main effects with strong
interactions, so a screening method has something real to find:

    b1_i    = +20  for the first N_LARGE_FIRST params, else ~ N(0, 1)
    b2_ij   = -15  for the first N_LARGE_SECOND params, else ~ N(0, 1)
    b3_ijl  = -10  for the first n_third_order params, else 0
    b4_ijls =  +5  for the first n_fourth_order params, else 0

The third/fourth-order coefficients are only ever nonzero among the first
few parameters, so they are stored over just that index range rather than
as full (k, k, k) / (k, k, k, k) arrays -- which would blow up as k grows.

This module generates `n_outputs` independent instances of the function
(the published function is scalar-valued); the fixed large coefficients
follow the assignment above in every output while the remaining N(0, 1)
coefficients are drawn independently per output. With permute_params=True
each output also gets its own shuffle of which parameters are the dominant
ones, so different outputs are sensitive to different inputs.
"""

PARAM_RANGE = (0.0, 1.0)
PARAM_NAMES = ("beta0", "beta1", "beta2", "beta3", "beta4",
               "w_special_idx", "param_permutation")

# 0-based; Morris's paper names these parameters 3, 5 and 7.
W_SPECIAL_IDX = (2, 4, 6)
N_LARGE_FIRST = 10
N_LARGE_SECOND = 6
LARGE_FIRST, LARGE_SECOND, LARGE_THIRD, LARGE_FOURTH = 20.0, -15.0, -10.0, 5.0


def generate_parameters(n_params, n_outputs, n_third_order=5, n_fourth_order=4,
                        permute_params=False, seed=None):
    """Draws one Morris-function parameter set per output column.

    n_third_order / n_fourth_order: how many of the leading parameters carry
    third-/fourth-order interaction terms (Morris uses 5 and 4).
    permute_params: if True, each output shuffles which parameters are the
    dominant ones -- useful when you want outputs that constrain different
    inputs. If False (default, faithful to the paper) every output is
    dominated by the same leading parameters.

    Returns (params, extra) where params holds every array needed to
    re-evaluate the function and extra carries diagnostics.
    """
    if n_third_order > n_params or n_fourth_order > n_params:
        raise ValueError("n_third_order/n_fourth_order cannot exceed n_params")
    rng = np.random.default_rng(seed)

    beta0 = np.zeros(n_outputs)
    beta1 = rng.normal(size=(n_params, n_outputs))
    beta2 = np.zeros((n_params, n_params, n_outputs))
    beta3 = np.zeros((n_third_order,) * 3 + (n_outputs,))
    beta4 = np.zeros((n_fourth_order,) * 4 + (n_outputs,))

    iu = np.triu_indices(n_params, k=1)
    beta2[iu[0], iu[1], :] = rng.normal(size=(len(iu[0]), n_outputs))

    n_first = min(N_LARGE_FIRST, n_params)
    n_second = min(N_LARGE_SECOND, n_params)
    for out in range(n_outputs):
        beta1[:n_first, out] = LARGE_FIRST
        for i, j in itertools.combinations(range(n_second), 2):
            beta2[i, j, out] = LARGE_SECOND
        for combo in itertools.combinations(range(n_third_order), 3):
            beta3[combo + (out,)] = LARGE_THIRD
        for combo in itertools.combinations(range(n_fourth_order), 4):
            beta4[combo + (out,)] = LARGE_FOURTH

    # A permutation is applied as a relabelling of the input axis at
    # evaluation time, so the coefficient arrays above stay in the canonical
    # Morris layout and remain directly comparable across outputs.
    if permute_params:
        param_permutation = np.stack([rng.permutation(n_params) for _ in range(n_outputs)], axis=1)
    else:
        param_permutation = np.tile(np.arange(n_params)[:, None], (1, n_outputs))

    params = {
        "beta0": beta0,
        "beta1": beta1,
        "beta2": beta2,
        "beta3": beta3,
        "beta4": beta4,
        "w_special_idx": np.array([i for i in W_SPECIAL_IDX if i < n_params], dtype=int),
        "param_permutation": param_permutation,
    }
    extra = {
        "dominant_params": np.array(sorted(range(n_first)), dtype=int),
    }
    return params, extra


def transform_w(X, w_special_idx):
    """x -> w. The plain map is a rescale of [0, 1] onto [-1, 1]; the special
    indices instead use 2(1.1 x / (x + 0.1) - 1/2), which is strongly curved
    near x = 0 and nearly flat above it -- that asymmetry is what makes the
    Morris function a non-trivial screening target."""
    W = 2.0 * (np.asarray(X, dtype=float) - 0.5)
    if len(w_special_idx) > 0:
        idx = np.asarray(w_special_idx, dtype=int)
        Xs = np.asarray(X, dtype=float)[:, idx]
        W[:, idx] = 2.0 * (1.1 * Xs / (Xs + 0.1) - 0.5)
    return W


def compute_Y(X, params):
    """Evaluates the Morris function at every row of X (shape (n_samples,
    n_params)), returning (n_samples, n_outputs)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    beta1 = params["beta1"]
    n_params, n_outputs = beta1.shape
    if X.shape[1] != n_params:
        raise ValueError(f"X has {X.shape[1]} parameters, model expects {n_params}")

    beta2, beta3, beta4 = params["beta2"], params["beta3"], params["beta4"]
    permutation = params["param_permutation"]
    W_canonical = transform_w(X, params["w_special_idx"])

    n_third = beta3.shape[0]
    n_fourth = beta4.shape[0]
    iu = np.triu_indices(n_params, k=1)
    combos3 = list(itertools.combinations(range(n_third), 3))
    combos4 = list(itertools.combinations(range(n_fourth), 4))

    Y = np.zeros((X.shape[0], n_outputs))
    for out in range(n_outputs):
        # Relabel inputs for this output, then apply the canonical coefficients.
        W = W_canonical[:, permutation[:, out]]
        y = params["beta0"][out] + W @ beta1[:, out]
        y = y + (W[:, iu[0]] * W[:, iu[1]]) @ beta2[iu[0], iu[1], out]
        for combo in combos3:
            y = y + beta3[combo + (out,)] * np.prod(W[:, combo], axis=1)
        for combo in combos4:
            y = y + beta4[combo + (out,)] * np.prod(W[:, combo], axis=1)
        Y[:, out] = y
    return Y
