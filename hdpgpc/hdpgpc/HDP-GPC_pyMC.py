import pymc as pm
import numpy as np
import pytensor.tensor as pt
import pymc_extras as pmx

from scipy.special import digamma
import arviz as az


class HDPGPC:
    """
    HDP-GPC: Hierarchical Dirichlet Process Gaussian Process Clustering

    This model implements the generative process described in the paper:
    - HDP prior for switching between linear dynamical systems
    - Each LDS evolves a Gaussian Process representing a cluster morphology
    - Warping GPs model temporal misalignment
    """

    def __init__(self, X, Y, K_max=10, alpha=10.0, kappa=20.0,
                 inducing_points=None):
        """
        Parameters
        ----------
        X : list of arrays
            List of time indices for each sequence [t_1, t_2, ..., t_N]
        Y : list of arrays
            List of observations for each sequence [y_1, y_2, ..., y_N]
        K_max : int
            Truncation level for stick-breaking (maximum number of clusters)
        alpha : float
            Concentration parameter for DP (controls number of clusters)
        kappa : float
            Top-level concentration parameter for HDP
        inducing_points : array-like, optional
            Inducing points for sparse GP approximation
        """
        self.X = X
        self.Y = Y
        self.N = len(Y)  # Number of sequences
        self.K_max = K_max
        self.alpha = alpha
        self.kappa = kappa

        # Store dimensions
        self.q = len(Y[0]) if len(Y) > 0 else 0  # Sequence length
        self.d = 1 if Y[0].ndim == 1 else Y[0].shape[1]  # Observation dimension

        # Setup inducing points for sparse GP
        if inducing_points is None:
            # Use a subset of observed time points
            all_times = np.concatenate(X)
            self.inducing_points = np.linspace(all_times.min(),
                                               all_times.max(),
                                               min(20, len(all_times)))
        else:
            self.inducing_points = inducing_points

        self.p = len(self.inducing_points)  # Number of inducing points

    def build_model(self):
        """
        Build the PyMC model for HDP-GPC following the generative process
        described in equations (15)-(25) of the paper.
        """
        with pm.Model() as model:
            # ==========================================
            # HDP Prior (Equations 15-18)
            # ==========================================

            # Top-level stick-breaking weights β ~ GEM(γ)
            v = pm.Beta('v', alpha=1, beta=self.alpha, shape=self.K_max)

            # Compute stick-breaking weights using equation (8)
            def stick_breaking(v):
                """Equation (8): β_k = v_k * ∏_{j<k}(1 - v_j)"""
                portions_remaining = pt.concatenate([
                    pt.ones((1,)),
                    pt.extra_ops.cumprod(1 - v)[:-1]
                ])
                return v * portions_remaining

            beta = pm.Deterministic('beta', stick_breaking(v))

            # Transition-specific stick-breaking (Equation 16)
            # π_j ~ DP(κ, β) for each state j
            # For simplicity, we use β as the transition probabilities
            # A full implementation would have separate π_j for each state

            # Initial state distribution π_0
            pi_0 = pm.Deterministic('pi_0', beta / pt.sum(beta))

            # ==========================================
            # GP Hyperparameters for each cluster k
            # (Equation 19)
            # ==========================================

            # Squared exponential kernel hyperparameters
            # Following the paper: σ_f controls amplitude, ℓ controls smoothness
            lengthscale_f = pm.Gamma('lengthscale_f', alpha=2, beta=0.5,
                                     shape=self.K_max)
            variance_f = pm.HalfNormal('variance_f', sigma=300.0,
                                       shape=self.K_max)
            noise_f = pm.HalfNormal('noise_f', sigma=5.0,
                                    shape=self.K_max)

            # Warping GP hyperparameters (Equation 22)
            lengthscale_g = pm.Gamma('lengthscale_g', alpha=2, beta=0.25,
                                     shape=self.K_max)
            variance_g = pm.HalfNormal('variance_g', sigma=1.0,
                                       shape=self.K_max)
            noise_g = pm.HalfNormal('noise_g', sigma=1.0,
                                    shape=self.K_max)

            # ==========================================
            # LDS Parameters (Equations 20-21)
            # ==========================================

            # Evolution matrix A_k and observation matrix C_k
            # Using matrix-normal inverse-Wishart prior (Equation 40)

            # Simplified: use independent priors for each element
            A = pm.Normal('A', mu=1.0, sigma=0.5,
                          shape=(self.K_max, self.q, self.q))
            C = pm.Normal('C', mu=1.0, sigma=0.5,
                          shape=(self.K_max, self.q, self.q))

            # Covariance matrices for dynamics and observations
            # Σ_η and Σ_ε from equations (20)-(21)
            sigma_eta = pm.HalfNormal('sigma_eta', sigma=10.0,
                                      shape=self.K_max)
            sigma_epsilon = pm.HalfNormal('sigma_epsilon', sigma=5.0,
                                          shape=self.K_max)

            # ==========================================
            # Switching states (Equation 18)
            # ==========================================

            # For each sequence n, sample switching state s_n
            s = pm.Categorical('s', p=pi_0, shape=self.N)

            # ==========================================
            # Latent GP functions and observations
            # (Equations 19-25)
            # ==========================================

            # Store latent functions for each cluster and sequence
            f_list = []
            x_list = []
            g_list = []
            y_pred_list = []

            for n in range(self.N):
                # Get the cluster assignment for this sequence
                s_n = s[n]

                # For each possible cluster k (we'll index by s_n later)
                f_k_list = []
                x_k_list = []
                g_k_list = []
                y_k_list = []

                for k in range(self.K_max):
                    # ====================================
                    # Initial GP f^k_1 ~ GP(0, K^k)
                    # (Equation 19)
                    # ====================================

                    # Sparse GP using inducing points (Equation 4)
                    # Define kernel at inducing points
                    cov_func_f = variance_f[k] ** 2 * pm.gp.cov.ExpQuad(
                        1, ls=lengthscale_f[k]
                    )

                    # GP prior at inducing points
                    gp_f = pm.gp.Latent(cov_func=cov_func_f)
                    f_inducing = gp_f.prior(f'f_inducing_k{k}_n{n}',
                                            X=self.inducing_points[:, None])

                    # Project to observation locations (Equation 4)
                    f_k = gp_f.conditional(f'f_k{k}_n{n}',
                                           Xnew=self.X[n][:, None])

                    # ====================================
                    # LDS Evolution: f^k_n = A_k f^k_{n-1} + η
                    # (Equation 20)
                    # ====================================

                    # For n > 1, apply linear dynamics
                    # This is simplified - full model needs message passing
                    if n > 0:
                        # f_k would be updated via Kalman filtering
                        # For now, add dynamics noise
                        f_k_evolved = pm.Normal(f'f_evolved_k{k}_n{n}',
                                                mu=f_k,
                                                sigma=sigma_eta[k],
                                                shape=self.q)
                    else:
                        f_k_evolved = f_k

                    # ====================================
                    # Pseudo-observation: x^k_n = C_k f^k_n + ε
                    # (Equation 21)
                    # ====================================

                    x_k = pm.Normal(f'x_k{k}_n{n}',
                                    mu=f_k_evolved,
                                    sigma=sigma_epsilon[k],
                                    shape=self.q)

                    # ====================================
                    # Warping GP: g^k ~ GP(0, K^k_g)
                    # (Equation 22)
                    # ====================================

                    cov_func_g = variance_g[k] ** 2 * pm.gp.cov.ExpQuad(
                        1, ls=lengthscale_g[k]
                    )

                    gp_g = pm.gp.Latent(cov_func=cov_func_g)

                    # Warped time: t^w_n = g^k(t_n)
                    # In practice, approximate as additive warp: t^w = t + g(t)
                    g_k = gp_g.prior(f'g_k{k}_n{n}', X=self.X[n][:, None])
                    t_warped = self.X[n][:, None] + g_k[:, None]

                    # ====================================
                    # Final observation model
                    # (Equations 23-25)
                    # ====================================

                    # Project x_k to warped time locations
                    # This is a simplification of the full joint distribution
                    y_k = pm.Normal(f'y_k{k}_n{n}',
                                    mu=x_k,
                                    sigma=noise_f[k],
                                    shape=self.q)

                    f_k_list.append(f_k_evolved)
                    x_k_list.append(x_k)
                    g_k_list.append(g_k)
                    y_k_list.append(y_k)

                # Stack all cluster predictions
                y_all_k = pt.stack(y_k_list, axis=0)  # (K_max, q)

                # Select prediction based on cluster assignment s_n
                y_pred = y_all_k[s_n, :]

                # Observed data
                y_obs = pm.Normal(f'y_obs_n{n}',
                                  mu=y_pred,
                                  sigma=1.0,
                                  observed=self.Y[n])

                f_list.append(f_k_list)
                x_list.append(x_k_list)
                g_list.append(g_k_list)
                y_pred_list.append(y_pred)

        return model


def create_simplified_hdp_gpc(X, Y, K_max=10, alpha=10.0):
    """
    Simplified HDP-GPC model suitable for MCMC inference

    This version uses a more tractable formulation while maintaining
    the key components from the paper.
    """
    N = len(Y)
    q = len(Y[0])

    with pm.Model() as model:
        # Stick-breaking for cluster weights (Equation 8)
        v = pm.Beta('v', alpha=1, beta=alpha, shape=K_max)

        def stick_breaking(v):
            portions = pt.concatenate([pt.ones((1,)),
                                       pt.extra_ops.cumprod(1 - v)[:-1]])
            return v * portions

        w = pm.Deterministic('w', stick_breaking(v))
        weights = pm.Deterministic('weights', w / pt.sum(w))

        # GP hyperparameters per cluster
        ls = pm.Gamma('ls', alpha=2, beta=0.5, shape=K_max)
        eta = pm.HalfNormal('eta', sigma=300, shape=K_max)
        sigma = pm.HalfNormal('sigma', sigma=5, shape=K_max)

        # LDS parameters
        A = pm.Normal('A', mu=1.0, sigma=0.3, shape=K_max)
        sigma_dyn = pm.HalfNormal('sigma_dyn', sigma=10, shape=K_max)

        # Cluster assignments
        z = pm.Categorical('z', p=weights, shape=N)

        # GP latent functions for each cluster
        for n in range(N):
            cluster_means = []

            for k in range(K_max):
                # GP for each cluster
                cov_k = eta[k] ** 2 * pm.gp.cov.ExpQuad(1, ls=ls[k])
                gp_k = pm.gp.Latent(cov_func=cov_k)

                # Latent function
                f_k = gp_k.prior(f'f_{k}_{n}', X=X[n][:, None])

                # Apply LDS dynamics (simplified)
                if n > 0:
                    # Could reference previous sequences here
                    f_k_evolved = pm.Deterministic(f'f_evo_{k}_{n}',
                                                   A[k] * f_k)
                else:
                    f_k_evolved = f_k

                cluster_means.append(f_k_evolved)

            # Stack and select based on assignment
            all_means = pt.stack(cluster_means, axis=0)

            # Observation likelihood
            mu_n = all_means[z[n], :]
            y_obs = pm.Normal(f'y_{n}', mu=mu_n, sigma=sigma[z[n]],
                              observed=Y[n])

    return model


def create_hdp_gpc_marginalizable(X, Y, K_max=5, alpha=10.0):
    """
    HDP-GPC model that can be marginalized using pymc-experimental
    """
    N = len(Y)
    q = len(Y[0])

    with pm.Model() as model:
        # Stick-breaking for cluster weights
        v = pm.Beta('v', alpha=1, beta=alpha, shape=K_max)

        def stick_breaking(v):
            portions = pt.concatenate([pt.ones((1,)),
                                       pt.extra_ops.cumprod(1 - v)[:-1]])
            return v * portions

        w = stick_breaking(v)
        weights = pm.Deterministic('weights', w / pt.sum(w))

        # GP hyperparameters per cluster
        ls = pm.Gamma('ls', alpha=2, beta=0.5, shape=K_max)
        eta = pm.HalfNormal('eta', sigma=300, shape=K_max)
        sigma = pm.HalfNormal('sigma', sigma=5, shape=K_max)

        # Discrete cluster assignments (will be marginalized)
        z = pm.Categorical('z', p=weights, shape=N)

        # Observations
        for n in range(N):
            # Create GP means for each cluster
            means_k = []
            for k in range(K_max):
                cov_k = eta[k] ** 2 * pm.gp.cov.ExpQuad(1, ls=ls[k])
                gp_k = pm.gp.Latent(cov_func=cov_k)
                f_k = gp_k.prior(f'f_{k}_{n}', X=X[n][:, None])
                means_k.append(f_k)

            # Stack means: shape (K_max, q)
            means_all = pt.stack(means_k, axis=0)

            # Select mean based on cluster assignment
            mu_n = means_all[z[n], :]

            # Likelihood
            y_obs = pm.Normal(f'y_{n}', mu=mu_n, sigma=sigma[z[n]],
                              observed=Y[n])

    return model


def create_hdp_gpc_mixture(X, Y, K_max=5, alpha=10.0):
    """
    HDP-GPC using Mixture distributions - works with NUTS!
    This approach marginalizes automatically without pymc-experimental
    """
    N = len(Y)
    q = len(Y[0])

    with pm.Model() as model:
        # Stick-breaking for cluster weights (HDP prior)
        v = pm.Beta('v', alpha=1, beta=alpha, shape=K_max)

        def stick_breaking(v):
            portions = pt.concatenate([
                pt.ones((1,)),
                pt.extra_ops.cumprod(1 - v)[:-1]
            ])
            return v * portions

        w = stick_breaking(v)
        weights = w / pt.sum(w)

        # GP hyperparameters per cluster
        ls = pm.Gamma('ls', alpha=2, beta=0.5, shape=K_max)
        eta = pm.HalfNormal('eta', sigma=300, shape=K_max)
        sigma = pm.HalfNormal('sigma', sigma=5, shape=K_max)

        # For each sequence, create a mixture of GP components
        for n in range(N):
            # Build component distributions
            comp_dists = []

            for k in range(K_max):
                # GP for this cluster
                cov_k = eta[k] ** 2 * pm.gp.cov.ExpQuad(1, ls=ls[k])
                gp_k = pm.gp.Latent(cov_func=cov_k)

                # Latent GP function for this sequence and cluster
                f_k = gp_k.prior(f'f_{k}_{n}', X=X[n][:, None])

                # Create component distribution (not observed yet)
                comp_dist = pm.Normal.dist(mu=f_k, sigma=sigma[k])
                comp_dists.append(comp_dist)

            # Mixture observation - automatically marginalizes over clusters
            y_obs = pm.Mixture(
                f'y_{n}',
                w=weights,
                comp_dists=comp_dists,
                observed=Y[n]
            )

    return model


# Example usage
if __name__ == "__main__":
    # Generate synthetic time series data
    np.random.seed(42)
    N = 5
    q = 20
    K_max = 3

    X = [np.linspace(0, 1, q) for _ in range(N)]
    Y = [np.sin(2 * np.pi * X[i]) + np.random.randn(q) * 0.1 for i in range(N)]

    # Create model
    model = create_hdp_gpc_mixture(X, Y, K_max=K_max)

    # Now sample with NUTS (no discrete variables!)
    with model:
        trace = pm.sample(
            draws=500,
            tune=500,
            chains=2,
            nuts_sampler='numpyro',
            return_inferencedata=True
        )

    # Print cluster assignments
    print("\nCluster assignments (mode):")
    print(trace.posterior['chain'].mode(dim=['chain', 'draw']).values)

    # Print weights
    print("\nCluster weights:")
    print(trace.posterior['draw'].mean(dim=['chain', 'draw']).values)
