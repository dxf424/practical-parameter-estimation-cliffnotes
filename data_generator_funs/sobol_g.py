import numpy as np

"""Sobol' G-function -- the standard variance-based sensitivity benchmark.

    g(x) = prod_{i=1}^{k} ( |4 x_i - 2| + a_i ) / ( 1 + a_i ),    x_i in [0, 1]

Every factor is a V shape in x_i with its minimum at x_i = 0.5. The constant
a_i >= 0 controls how sharp that V is, and therefore how much parameter i
matters:

    a_i = 0     -> factor swings over [0, 2]; parameter is highly influential
    a_i small   -> still influential
    a_i large   -> factor is pinned near 1; parameter barely matters at all

That is the whole appeal of this function: the sensitivity of every input is
known analytically before you run anything. Each factor has mean 1 (so the
product has mean 1 too) and variance

    V_i = (1/3) / (1 + a_i)^2

with the total variance being prod_i (1 + V_i) - 1, so exact first-order and
total Sobol' indices are available in closed form (see analytic_indices).

The published function is scalar-valued; this module generates `n_outputs`
independent instances. Since the `a` vector alone determines the function,
every output would otherwise be identical -- so by default each output gets
its own permutation of `a`, making different outputs sensitive to different
inputs (set permute_per_output=False only if you deliberately want identical
output columns).
"""

PARAM_RANGE = (0.0, 1.0)
PARAM_NAMES = ("a",)


def default_a(n_params):
    """a_i = i / 2 for i = 0, 1, 2, ... -- the conventional ramp, so parameter
    0 is the most influential and importance falls off monotonically."""
    return np.arange(n_params, dtype=float) / 2.0


def generate_parameters(n_params, n_outputs, a=None, permute_per_output=True, seed=None):
    """Builds the (n_params, n_outputs) matrix of `a` constants.

    a: the base vector of a_i values (default `default_a(n_params)`); a scalar
       broadcasts to every parameter. Must be >= 0.
    permute_per_output: give each output its own shuffle of the base vector,
       so outputs are sensitive to different parameters (default True).
    """
    rng = np.random.default_rng(seed)
    a_base = default_a(n_params) if a is None else np.broadcast_to(
        np.asarray(a, dtype=float), (n_params,)).copy()
    if np.any(a_base < 0):
        raise ValueError("Sobol' G-function requires a_i >= 0")

    if permute_per_output:
        a_matrix = np.stack([rng.permutation(a_base) for _ in range(n_outputs)], axis=1)
    else:
        a_matrix = np.tile(a_base[:, None], (1, n_outputs))

    params = {"a": a_matrix}
    extra = {"a_base": a_base}
    return params, extra


def compute_Y(X, params):
    """Evaluates the G-function at every row of X (shape (n_samples,
    n_params)), returning (n_samples, n_outputs)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    a = params["a"]
    n_params, _n_outputs = a.shape
    if X.shape[1] != n_params:
        raise ValueError(f"X has {X.shape[1]} parameters, model expects {n_params}")

    # (n_samples, n_params, 1) against (1, n_params, n_outputs)
    factors = (np.abs(4.0 * X - 2.0)[:, :, None] + a[None, :, :]) / (1.0 + a[None, :, :])
    return np.prod(factors, axis=1)


def analytic_indices(params):
    """Exact first-order and total Sobol' sensitivity indices, per output.

    V_i = (1/3)/(1 + a_i)^2 is factor i's variance; the factors are
    independent with mean 1, so total variance is prod(1 + V_i) - 1. This is
    ground truth to check any estimated sensitivity against.

    Returns (S1, ST), both (n_params, n_outputs).
    """
    a = params["a"]
    V = (1.0 / 3.0) / (1.0 + a) ** 2
    total_variance = np.prod(1.0 + V, axis=0) - 1.0

    S1 = V / total_variance[None, :]
    # Total effect: everything that survives when only factor i is left free.
    prod_all = np.prod(1.0 + V, axis=0)[None, :]
    ST = V * (prod_all / (1.0 + V)) / total_variance[None, :]
    return S1, ST
