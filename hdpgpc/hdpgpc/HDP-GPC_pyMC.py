
# ============================================================================
# STANDARD VECTOR LINEAR DYNAMICAL SYSTEM (LDS) IMPLEMENTATION IN PyMC
# ============================================================================

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import pytensor
from sklearn.preprocessing import StandardScaler
# Load your data
y_data = np.load('C:/Users/Adrian/Projects/Thesis/HDP-GPC/hdpgpc/data/mitbih/100.npy')[:4,:,0]
y_data = StandardScaler().fit_transform(y_data.T).T
time_vector = np.array([np.arange(y_data.shape[1])]*y_data.shape[0])

# Model configuration
n_obs_dim = y_data.shape[1]      # Number of observed dimensions (ECG channels)
n_states = y_data.shape[1]       # Number of latent state dimensions
T = 4             # Number of time steps

# ============================================================================
# MODEL: Linear Dynamical System
# ============================================================================
# State Equations:
#   f_0 ~ N(mu_0, Sigma_0)                    # Initial state
#   f_t = A @ f_{t-1} + omega_t               # State evolution
#   omega_t ~ N(0, Q)                         # Process noise
#
# Observation Equations:
#   x_t = C @ f_t + epsilon_t                 # Observation model
#   epsilon_t ~ N(0, R)                       # Observation noise
# ============================================================================

with pm.Model() as lds_model:

    # ------------------------------------------------------------------------
    # 1. INITIAL STATE DISTRIBUTION GP: f_0 ~ GP(mu_0, K_0)
    # ------------------------------------------------------------------------
    X_init = time_vector[[0]]

    # GP hyperparameters for initial state
    # These control the prior belief about f_0
    ls_init = pm.Gamma('ls_init', alpha=2, beta=1)
    eta_init = pm.HalfNormal('eta_init', sigma=2.0)

    # Define GP covariance function
    # ExpQuad (RBF) kernel is common choice
    cov_init = eta_init ** 2 * pm.gp.cov.ExpQuad(
        input_dim=X_init.shape[1],
        ls=ls_init
    )

    # Define GP mean function
    # Zero mean (common default)
    mean_func_init = pm.gp.mean.Zero()

    # Create GP object
    gp_init = pm.gp.Latent(mean_func=mean_func_init, cov_func=cov_init)

    f_0_gp = gp_init.prior('f_0_gp', X=X_init * np.ones((n_states, 1)))

    # Initial covariance (uncertainty around GP mean)
    chol_0, _, _ = pm.LKJCholeskyCov(
        "chol_0",
        n=n_states,
        eta=2.0,
        sd_dist=pm.Exponential.dist(0.5, shape=n_states)
    )

    # Sample initial state with GP-informed mean
    f_0 = pm.MvNormal("f_0", mu=f_0_gp, chol=chol_0, shape=n_states)

    # ------------------------------------------------------------------------
    # 2. STATE TRANSITION MATRIX: A (controls dynamics)
    # ------------------------------------------------------------------------
    A_raw = pm.Normal("A_raw", mu=0, sigma=0.5, shape=(n_states, n_states))

    # Constrain A for stability: eigenvalues should have magnitude < 1
    # Using tanh to map to [-0.9, 0.9]
    A = pm.Deterministic("A", 0.9 * pt.tanh(A_raw))

    # ------------------------------------------------------------------------
    # 3. PROCESS NOISE COVARIANCE: Q
    # ------------------------------------------------------------------------
    chol_Q, corr_Q, sd_Q = pm.LKJCholeskyCov(
        "chol_Q",
        n=n_states,
        eta=2.0,
        sd_dist=pm.Exponential.dist(1.0, shape=n_states),
        compute_corr=True  # Also return correlation and standard deviations
    )

    # Optionally store full covariance matrix
    Q = pm.Deterministic("Q", pt.dot(chol_Q, chol_Q.T))

    # ------------------------------------------------------------------------
    # 4. OBSERVATION/EMISSION MATRIX: C (maps states to observations)
    # ------------------------------------------------------------------------
    C = pm.Normal("C", mu=0, sigma=1.0, shape=(n_obs_dim, n_states))

    # ------------------------------------------------------------------------
    # 5. OBSERVATION NOISE COVARIANCE: R
    # ------------------------------------------------------------------------
    chol_R, corr_R, sd_R = pm.LKJCholeskyCov(
        "chol_R",
        n=n_obs_dim,
        eta=2.0,
        sd_dist=pm.HalfNormal.dist(sigma=0.1, shape=n_obs_dim),
        compute_corr=True
    )

    R = pm.Deterministic("R", pt.dot(chol_R, chol_R.T))

    # ------------------------------------------------------------------------
    # 6. STATE EVOLUTION: Generate latent trajectory
    # ------------------------------------------------------------------------

    f = [f_0]
    # Generate trajectory from t=1 to t=T-1
    for t in range(1, T):
        f_t = pm.MvNormal(f'f_{t}',mu=A @ f[t - 1], chol=chol_Q)
        f.append(f_t)

    # ------------------------------------------------------------------------
    # 6. OBSERVATION MODEL: Generate observations
    # ------------------------------------------------------------------------

    x = []
    #Generate observations from t=0 to t=T
    for t in range(T):
        x_t = pm.MvNormal(f'x_{t}', mu=C @ f[t], chol=chol_R, observed=y_data[t])
        x.append(x_t)

# ============================================================================
# INFERENCE: Sample from the posterior
# ============================================================================
with lds_model:

    # Prior predictive check (optional)
    prior_pred = pm.sample_prior_predictive(samples=10)

    # MCMC sampling
    trace = pm.sample(
        draws=5000,           # Number of samples per chain
        tune=2000,            # Number of tuning steps
        target_accept=0.90,   # Target acceptance rate for NUTS
        random_seed=42,
        nuts_sampler='numpyro'
    )

    # Posterior predictive check
    post_pred = pm.sample_posterior_predictive(trace)
    latent_states_mean = trace.posterior['f_3'].mean(dim=['chain', 'draw']).values
    A_posterior = trace.posterior['A'].mean(dim=['chain', 'draw']).values

    plt.plot(y_data[0],'.')
    plt.plot(post_pred.observed_data.x_0.values)
    plt.plot(latent_states_mean)
    plt.show()
# ============================================================================
# EXTRACT RESULTS
# ============================================================================
import arviz as az

# Convergence diagnostics
print(az.summary(trace, var_names=['A', 'C', 'chol_Q', 'chol_R']))

# Extract latent states
latent_states_mean = trace.posterior['f_trajectory'].mean(dim=['chain', 'draw']).values
latent_states_std = trace.posterior['f_trajectory'].std(dim=['chain', 'draw']).values

# Extract predicted observations
x_pred_mean = trace.posterior['x_mean'].mean(dim=['chain', 'draw']).values
x_pred_std = trace.posterior['x_mean'].std(dim=['chain', 'draw']).values

# Transition matrix


print(f"Latent states shape: {latent_states_mean.shape}")
print(f"Predicted observations shape: {x_pred_mean.shape}")
print(f"Transition matrix A shape: {A_posterior.shape}")

