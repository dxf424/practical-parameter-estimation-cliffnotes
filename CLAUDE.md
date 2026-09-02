This is a repo that 
    1. stores different parameter estimation methods, 
    2. Generate datasets of varying qualities (how non-linear, size of the dataset)
    3. Apply these methods to test their performance, and also illustrate their pros and cons



2026 July 17th
Currently, we are working on Point 2. Generate the data. We are skipping Points 1 and 3 for now
One important concern is that we need to generate the data systematically across different kind of data (linear, non-linear, PDE and etc)
We want to highlight the full (if possible) spectrum of challenges in practical parameter estimation.
A few challenges are listed here:
    a) Parameter estimation efficiency (e.g., the use of more efficient sampling strategy);
    b) Poor emulator performance;
    c) Structural error;
    d) The strategy if we are given too few (hard to constrian the parameters effectively) or too many targets (structural error-prone); 
    e) When we do waves/iterations of parameter estimation, what is the best strategy (whether take all previous runs in training or a mixture, etc);
    f) Using scores or raw model outputs as emulator targets;
    g) If sensitivity test could help inform more efficient sampling strategy? 
    h) The presence of multi-modes in the solutions.
    i) The relative importance of parameters (some parameters are important after others are better constrained)

Another challenge is to make all the generated data formated systematically. For example, a nc file that has everything in it or 
csv files and etc. 

Similarly, once we have an ensemble of estimated parameters, we also need to be able to apply them to the moedel and check the results. This 
will help us deploy the methodology of simulation (which is what is this for)-emulation-sampling iteration.

It is also good to be able to inject different kind of errors or uncertainties to the data (e.g., obs error, structural error). Here the structural error can be something like a set of x_1 values that map to y_1 given a model m_1, in addition, we are also given some y_2 that is generated from a different x_2 given model y_2. x_1 and x_2 share the same parameter space. This set up is to simulate the scenario where there is no parameter values that could satisify y_1 and y_2 given m_1 and m_2


2026 August 15th
Narrowed the repo to LINEAR generators only -- the exponential/polynomial/ODE/Lorenz-96
generators were removed to keep the linear case clean and readable. Bring them back from
git history (commit 273e5b7^) if/when they're needed again.

data_generator_funs/linear.py now has exactly three coefficient structures:
    1. full              -- dense random matrix, every x drives every y.
    2. band              -- sliding diagonal window: y_j is driven only by
                            x_j .. x_{j+n_sensitive_para-1}; everything else exactly 0.
    3. block_triangular  -- param/output groups; each group's diagonal block is strongly
                            sensitive, earlier param groups have a SMALL nonzero influence
                            on later output groups (upper-right), and later param groups
                            have EXACTLY ZERO influence on earlier ones (lower-left).

Error injection lives in dataset.py:
    - add_observation_noise: iid N(0, noise_std) on y_true. Averages out with more data.
    - add_structural_error: takes a second, different input x_b, computes y_b = x_b @ A,
      and splices y_b's values into y_true on the chosen outputs only. Because
      x_b != x_true, NO single parameter vector reproduces the whole observation -- the
      untouched outputs need x_true, the spliced ones need x_b. That irreducibility is
      the point; it's what separates structural error from noise.
      Ask for it by COUNT (n_structural=k, placed at the last k outputs by default or
      scattered with structural_position="random") or by exact indices (structural_idx),
      not both. n_structural == n_outputs is rejected on purpose: replacing every output
      leaves y_true == y_b, which x_b reproduces exactly, so no conflict would remain.

notebooks/linear_dataset_walkthrough.ipynb tests and visualizes all of the above.


2026 August 15th (later)
Added two more FORWARD MODELS beside the linear one, behind a shared interface, so
everything above (obs error, structural error, saving, plotting) works on all three:
    - morris.py   -- Morris (1991) screening function. Strong main effects on the first
                     10 params, interactions up to 4th order, and a curved transform on
                     params 3/5/7. Inputs live on [0, 1].
    - sobol_g.py  -- Sobol' G-function, prod_i (|4x_i - 2| + a_i)/(1 + a_i). a_i sets each
                     input's influence (small a = influential), and the exact Sobol'
                     indices are available in closed form via analytic_indices().
                     Inputs live on [0, 1].

A model module plugs in by exposing PARAM_RANGE, PARAM_NAMES, generate_parameters()
and compute_Y(X, params); dataset.MODELS is the registry. Use dataset.generate_dataset(
model=...) for any of them (generate_linear_dataset is a thin wrapper kept for
convenience). NOTE the input range differs per model -- [-1, 1] for linear, [0, 1] for
the other two -- so always sample via dataset.sample_X(..., model=...).

ALL MODEL PARAMETERS ARE SAVED. Every array a model needs is stored under the dataset's
"model_params" key and written into the .nc as top-level variables. netcdf_io.load_model()
reads them back, which is what makes a saved dataset re-runnable rather than just readable.

CONTEST RUNNER: run_model.py evaluates a saved model at proposed parameter values.
    python run_model.py linear_demo Yang.csv          -> Yang_output_Iteration1.csv
    python run_model.py linear_demo Yang.csv          -> Yang_output_Iteration2.csv
    python run_model.py morris_demo Yang.csv --score  -> also RMSE vs the stored y_true
The iteration number auto-increments past whatever is on disk so submissions never
overwrite each other. Example models live in datasets/{linear,morris,sobol_g}_demo/.

GOTCHA worth remembering: do NOT use the range midpoint as the "hold the others here"
baseline for one-at-a-time sensitivity. For the Sobol' G-function every factor bottoms
out at x=0.5, so a single a_i=0 parameter zeroes the entire product and makes every other
input look inert. utils/plotting.py averages over random baselines instead.

notebooks/forward_models_walkthrough.ipynb explains both new models with visuals and
demonstrates the shared machinery + the contest workflow end to end.

