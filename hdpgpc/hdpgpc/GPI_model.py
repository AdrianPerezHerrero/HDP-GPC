#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:55:40 2021
@author: adrian.perez
"""
import hdpgpc.GPI as GPI
import numpy as np
import torch
from tqdm import trange
import math
from bisect import bisect_right
from hdpgpc.lds import (
    expected_gaussian_log_likelihood,
    expected_residual_scatter,
    gaussian_conditional_entropy,
    gaussian_kl,
    inverse_wishart_expected_logdet,
    inverse_wishart_kl,
    matrix_normal_inverse_wishart_kl,
    stable_cholesky,
)
dtype = torch.float64
torch.set_default_dtype(dtype)

class GPI_model():
    """Model that sum up all the information to characterise a GPR_iterative_model.
        Parameters
        ----------
        kernel : kernel from sklearn class for covariance computation.

        x_basis : array-like of shape (s_samples) domain points where is wanted
        to focus the learning.

        ini_Sigma : positive double or array-like of shape (n_samples) noise associated with subyacent process.
        Returns
        -------
        self : returns an instance of self.
        """

    def __init__(self, kernel, x_basis, annealing=True, bayesian=False, cuda=False, inducing_points=False,
                 estimation_limit=None, free_deg_MNIV=5, verbose=True,
                 learn_observation_matrix=True, couple_gp_lds_noise=True):
        self.gp = GPI.IterativeGaussianProcess(kernel, x_basis, cuda=cuda, verbose=verbose)
        self.x_basis = self.cond_to_torch(x_basis)
        self.x_train = []
        self.y_train = []
        self.f_star = []
        self.f_star_sm = []
        self.cov_f = []
        self.cov_f_sm = []
        self.cross_cov_f_sm = []
        self.y_var = []
        self.var = []
        self.A = []
        self.Gamma = []
        self.C = []
        self.Sigma = []
        self.likelihood = []
        self.N = 0
        self.indexes = []
        self.sample_weights = []
        self.K = self.cond_to_torch(kernel(self.cond_to_cpu(x_basis), self.cond_to_cpu(x_basis)))
        self.annealing = annealing
        self.bayesian = bayesian
        self.cuda = cuda
        self.inducing_points = inducing_points
        self.learn_observation_matrix = bool(learn_observation_matrix)
        self.couple_gp_lds_noise = bool(couple_gp_lds_noise)
        if bayesian:
            self.internal_params = None
            self.observation_params = None
        if estimation_limit is None:
            estimation_limit = np.inf
        self.estimation_limit = estimation_limit
        self.fitted = False
        self.A_def, self.Gamma_def, self.C_def, self.Sigma_def = None, None, None, None
        self.ini_cov_def = None
        self.ini_kernel_theta = self.gp.kernel.theta

        if self.cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.sq_lat_last = torch.zeros(1, device=self.device)
        self.free_deg_MNIV = free_deg_MNIV
        self.verbose = verbose
        self.disable = not verbose

    def _eye_cached(self, n, device, dtype):
        if not hasattr(self, "_eye_cache"):
            self._eye_cache = {}
        key = (n, str(device), dtype)
        if key not in self._eye_cache:
            self._eye_cache[key] = torch.eye(n, device=device, dtype=dtype)
        return self._eye_cache[key]

    def _chol_spd(self, M, jitter_scale=1e-8):
        M = 0.5 * (M + M.T)
        eye = self._eye_cached(M.shape[0], M.device, M.dtype)
        diag_mean = torch.mean(torch.diag(M).abs()).clamp_min(torch.finfo(M.dtype).eps)
        last_error = None
        for multiplier in (1.0, 10.0, 100.0, 1_000.0, 10_000.0):
            try:
                return torch.linalg.cholesky(
                    M + multiplier * jitter_scale * diag_mean * eye
                )
            except torch.linalg.LinAlgError as error:
                last_error = error
        raise torch.linalg.LinAlgError(
            "matrix is not positive definite after adaptive jitter"
        ) from last_error

    def _log2pi(self, ref):
        return torch.tensor(math.log(2.0 * math.pi), device=ref.device, dtype=ref.dtype)

    def _gaussian_score_shared_cov(self, Y, mean, cov, mean_covariance=None):
        """
        Y    : (B,T,1) or (B,T)
        mean : (T,1) or (T,)
        cov  : (T,T)
        returns: (B,)
        """
        if Y.ndim == 3:
            Y2 = Y[..., 0].T  # (T,B)
        else:
            Y2 = Y.T  # (T,B)

        if mean.ndim == 2:
            m = mean[:, 0:1]  # (T,1)
        else:
            m = mean.unsqueeze(1)  # (T,1)

        diff = Y2 - m  # (T,B)
        L = self._chol_spd(cov)
        alpha = torch.cholesky_solve(diff, L)
        q = diff.shape[0]
        logdet = 2.0 * torch.log(torch.diag(L)).sum()
        scores = (
            -0.5 * torch.sum(diff * alpha, dim=0)
            - 0.5 * logdet
            - 0.5 * q * self._log2pi(diff)
        )
        if mean_covariance is not None:
            latent_trace = torch.trace(torch.cholesky_solve(mean_covariance, L))
            scores = scores - 0.5 * latent_trace
        return scores

    def _latent_parameters_at(self, t=None, params=None):
        """Return latent and observation moments used by an ELBO emission term."""
        if params is not None:
            mean, covariance, C, Sigma = params
            return mean, covariance, C, Sigma
        if len(self.indexes) == 0:
            return self.f_star_sm[0], self.cov_f_sm[0], self.C[0], self.Sigma[0]
        if t is None or len(self.indexes) <= t:
            return self.f_star_sm[-1], self.cov_f_sm[-1], self.C[-1], self.Sigma[-1]
        parameter_index = -1 if self.estimation_limit <= t else t
        _, _, C, Sigma = self.get_params(parameter_index)
        return self.f_star_sm[t], self.cov_f_sm[t], C, Sigma

    def _emission_moments(self, x_train, t=None, params=None):
        """Observation mean, latent-induced covariance, and observation noise."""
        mean, covariance, C, Sigma = self._latent_parameters_at(t, params)
        mean = self.cond_to_torch(mean)
        covariance = self.cond_to_torch(covariance)
        C = self.cond_to_torch(C)
        Sigma = self.cond_to_torch(Sigma)
        basis_mean = C @ mean
        basis_covariance = C @ covariance @ C.T
        if torch.equal(x_train, self.x_basis):
            return basis_mean, basis_covariance, Sigma
        projected_mean, noise_covariance = self.gp.pred_dist(
            x_train, self.x_basis, basis_mean, Sigma
        )
        _, projected_latent_covariance = self.gp.pred_latent_dist(
            x_train, self.x_basis, basis_mean, basis_covariance
        )
        return projected_mean, projected_latent_covariance, noise_covariance

    def initial_conditions(self, ini_mean=None, ini_cov=None,
                           ini_A=None, ini_Gamma=None, ini_C=None, ini_Sigma=None):
        """
        Incorporate initial conditions to the model, if none is specified default
        model is a dynamic one.
        Parameters
        ----------
        ini_mean : array like
            DESCRIPTION. Initial mean of the model. The default is 0.
        ini_cov : matrix like
            DESCRIPTION. Initial covariance of the model, should be computed
            using hyperparameters kernel stimation. The default is None.
        ini_A : matrix like
            DESCRIPTION. Projection step matrix. The default is Id.
        ini_Gamma : matrix like
            DESCRIPTION. Noise associated with the subsequent process. The default is 0.01.
        ini_C : matrix like
            DESCRIPTION. Projection observations matrix. The default is Id.
        ini_Sigma : matrix like
            DESCRIPTION. Noise associated with observations. The default is 0.5**2.

        Returns
        -------
        None.

        """
        # Incorporate initial parameters
        if ini_mean is None:
            self.f_star.append(self.compute_mean())
            self.f_star_sm.append(self.compute_mean())
        else:
            self.f_star.append(self.cond_to_torch(ini_mean))
            self.f_star_sm.append(self.cond_to_torch(ini_mean))
        if ini_cov is None:
            self.cov_f.append(self.cond_to_torch(self.K))
            self.cov_f_sm.append(self.cond_to_torch(self.K))
            self.ini_cov_def = self.K.clone()
        else:
            self.cov_f.append(self.cond_to_torch(ini_cov))
            self.cov_f_sm.append(self.cond_to_torch(ini_cov))
            self.ini_cov_def = ini_cov
        lds_parameters = (ini_A, ini_Gamma, ini_C, ini_Sigma)
        if all(value is None for value in lds_parameters):
            ini_A, ini_Gamma, ini_C, ini_Sigma = self.GPR_dynamic()
        elif any(value is None for value in lds_parameters):
            raise ValueError(
                "ini_A, ini_Gamma, ini_C and ini_Sigma must be provided together"
            )

        self.A.append(ini_A)
        self.Gamma.append(ini_Gamma)
        self.C.append(ini_C)
        self.Sigma.append(ini_Sigma)
        self.A_def = self.cond_to_cuda(self.cond_to_torch(ini_A))
        self.Gamma_def = self.cond_to_cuda(self.cond_to_torch(ini_Gamma))
        self.C_def = self.cond_to_cuda(self.cond_to_torch(ini_C))
        self.Sigma_def = self.cond_to_cuda(self.cond_to_torch(ini_Sigma))
        self.var.append(np.atleast_2d(np.diag(ini_Gamma)).T)
        self.y_var.append(np.atleast_2d(np.diag(ini_Sigma)).T)
        # DEGREES OF FREEDOM AT LEAST 3
        if self.bayesian:
            self.internal_params = matrix_normal_inv_wishart(ini_A, np.eye(ini_A.shape[0]),  self.free_deg_MNIV, ini_Gamma)
            if (
                torch.all(ini_Gamma == torch.zeros(ini_Gamma.shape))
                or not self.learn_observation_matrix
            ):
                self.observation_params = inv_wishart(self.free_deg_MNIV, ini_Sigma, ini_C)
            else:
                self.observation_params = matrix_normal_inv_wishart(ini_C, np.eye(ini_C.shape[0]), self.free_deg_MNIV, ini_Sigma)


    def GPR_static(self, ini_Sigma=None):
        """ Define static conditions using initial Sigma as a double.
        """
        shape = len(self.x_basis)
        ini_A = torch.eye(shape)
        ini_Gamma = torch.zeros((shape, shape))
        ini_C = torch.eye(shape)
        if ini_Sigma is None:
            ini_Sigma = torch.mul(0.5 ** 2, torch.eye(shape))
        else:
            ini_Sigma = torch.mul(ini_Sigma, torch.eye(shape))
        return ini_A, ini_Gamma, ini_C, ini_Sigma

    def GPR_dynamic(self, gamma=None, sigma=None):
        """ Define dynamic conditions using initial Sigma as a double.
        """
        shape = len(self.x_basis)
        ini_A = torch.eye(shape)
        if gamma is None:
            ini_Gamma = torch.mul(0.01, torch.eye(shape))
        else:
            ini_Gamma = torch.mul(gamma, torch.eye(shape))
        ini_C = torch.eye(shape)
        if sigma is None:
            ini_Sigma = torch.mul(0.5 ** 2, torch.eye(shape))
        else:
            ini_Sigma = torch.mul(sigma, torch.eye(shape))
        return ini_A, ini_Gamma, ini_C, ini_Sigma

    def fit_kernel_params(self, x_train, y, alpha_ini, gamma_ini, valid=True):
        """ Optimize RBF kernel hyperparameters
        """
        alph_ = self.cond_to_torch(alpha_ini[0][0])
        gam_ = self.cond_to_torch(gamma_ini[0][0])
        if valid:
            fitted = self.gp.fit_torch(self.cond_to_torch(x_train), self.cond_to_torch(y), alph_, gam_,
                                       reduced_points=self.inducing_points, verbose=self.verbose)
        noise = self.cond_to_torch(self.gp.kernel.get_params()["k2__noise_level"])
        self.x_basis = self.cond_to_cuda(self.cond_to_torch(self.gp.x_basis))
        if self.couple_gp_lds_noise:
            self.Sigma[-1] = self.cond_to_cuda(
                self.cond_to_torch(
                    torch.mul(
                        noise,
                        torch.eye(len(self.x_basis), device=noise.device),
                    )
                )
            )
            self.Sigma_def = torch.clone(self.Sigma[-1])
        self.y_var[-1] = self.cond_to_cuda(self.cond_to_torch(np.atleast_2d(np.diag(self.cond_to_cpu(self.Sigma[-1]))).T))
        self.C[-1] = self.cond_to_cuda(torch.eye(len(self.x_basis)))
        self.A[-1] = self.cond_to_cuda(torch.eye(len(self.x_basis)))
        self.Gamma[-1] = self.cond_to_cuda(self.cond_to_torch(torch.mul(torch.mean(torch.diag(self.Gamma[-1])),
                                                                        torch.eye(len(self.x_basis), device=self.device))))
        self.f_star[-1] = self.cond_to_cuda(self.cond_to_torch(self.compute_mean()))
        self.f_star_sm[-1] = self.cond_to_cuda(self.cond_to_torch(self.compute_mean()))
        ini_cov = self.cond_to_cuda(self.cond_to_torch(
            self.gp.kernel(self.cond_to_numpy(self.cond_to_cpu(self.x_basis)),
                           self.cond_to_numpy(self.cond_to_cpu(self.x_basis)))))# + self.Sigma[-1]
        self.ini_cov_def = ini_cov
        self.cov_f[-1] = ini_cov
        self.cov_f_sm[-1] = ini_cov
        if self.bayesian:
            if self.couple_gp_lds_noise:
                self.observation_params.set_scale(self.Sigma[-1])
            if isinstance(self.observation_params, matrix_normal_inv_wishart):
                self.observation_params.m_mean = self.C[-1]
            self.internal_params.set_scale(self.Gamma[-1])
            self.internal_params.m_mean = self.A[-1]
        self.fitted = True
        print("---Kernel estimated---")
        print(self.gp.kernel)
        return self.x_basis, ini_cov

    def log_lik_sample(self, y):
        """Returns log-likelihood of a single sample using the last state of the model.
        """
        lik_post = self.gp.log_likelihood(self.N, self.N, self.f_star_sm, self.cov_f_sm, self.A[-1], self.Gamma[-1], y,
                                          self.C[-1], self.Sigma[-1])
        return lik_post

    def log_sq_error(self, x_train, y, mean=None, cov=None, C=None, Sigma=None, i=None, proj=False, first=False):
        """Compute E_q[log p(y | f)] for one sample and latent posterior."""
        y = self.cond_to_cuda(self.cond_to_torch(y)).to(torch.float64)
        mean = self.cond_to_torch(mean)
        cov = self.cond_to_torch(cov)
        C = self.cond_to_torch(C)
        Sigma = self.cond_to_torch(Sigma)

        if x_train is None:
            x_train = self.x_basis

        params = None if mean is None else [mean, cov, C, Sigma]
        if (
            torch.equal(x_train, self.x_basis)
            and self.bayesian
            and self.observation_params is not None
        ):
            latent_mean, latent_covariance, _, _ = self._latent_parameters_at(
                i, params
            )
            zero = torch.zeros_like(latent_covariance)
            return self.observation_params.expected_log_likelihood(
                y,
                latent_mean,
                torch.zeros_like(self.Sigma[-1]),
                latent_covariance,
                zero,
            )
        f_star, latent_covariance, noise_covariance = self._emission_moments(
            x_train, i, params
        )

        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if f_star.ndim == 1:
            f_star = f_star.unsqueeze(-1)
        return expected_gaussian_log_likelihood(
            y, f_star, latent_covariance, noise_covariance
        )

    def log_lat_error(self, i, h_ini):
        """Latent ELBO contribution, including the structured Gaussian entropy."""
        latent_current = self.f_star_sm[i + 1]
        covariance_current = self.cov_f_sm[i + 1]
        if i == 0:
            return -gaussian_kl(
                latent_current,
                covariance_current,
                self.f_star_sm[0],
                self.cov_f_sm[0],
            )

        latent_previous = self.f_star_sm[i]
        covariance_previous = self.cov_f_sm[i]
        cross_index = i - 1
        if (
            cross_index >= len(self.cross_cov_f_sm)
            or self.cross_cov_f_sm[cross_index] is None
        ):
            raise RuntimeError("missing lag-one smoothing covariance for latent ELBO")
        cross = self.cross_cov_f_sm[cross_index]
        if self.bayesian and self.internal_params is not None:
            expected_transition = self.internal_params.expected_log_likelihood(
                latent_current,
                latent_previous,
                covariance_current,
                covariance_previous,
                cross,
            )
        else:
            parameter_index = min(i + 1, len(self.Gamma) - 1)
            Gamma_mat = self.Gamma[parameter_index]
            A = self.A[min(parameter_index, len(self.A) - 1)]
            scatter = expected_residual_scatter(
                latent_current,
                latent_previous,
                A,
                covariance_current,
                covariance_previous,
                cross,
            )
            expected_transition = expected_gaussian_log_likelihood(
                torch.zeros_like(latent_current),
                torch.zeros_like(latent_current),
                scatter,
                Gamma_mat,
            )
        entropy = gaussian_conditional_entropy(
            covariance_current, covariance_previous, cross
        )
        return expected_transition + entropy

    def include_sample(self, index, x_train, y, x_warped=None, h=1.0, posterior=True, embedding=True, include_index=False):
        """ Method to include sample in the model, compute the posterior and add the data.
        """
        if posterior:
            self.N = self.N + 1
            self.indexes.append(index)
            self.sample_weights.append(float(h))
            self.x_train.append(x_train)
            self.y_train.append(self.cond_to_torch(y))
            f_star_, cov_f_ = self.gp.posterior(self.f_star_sm[-1], self.cov_f_sm[-1], self.y_train[-1], self.A[-1],
                                                self.Gamma[-1], self.C[-1], self.Sigma[-1] / h, x_train=x_train,
                                                x_warped=x_warped, embedding=embedding)
            self.f_star.append(f_star_)
            self.f_star_sm.append(f_star_)
            self.cov_f.append(cov_f_)
            self.cov_f_sm.append(cov_f_)
            return self.f_star_sm[-1], self.cov_f_sm[-1]
        else:
            if include_index:
                self.indexes.append(index)
                self.sample_weights.append(0.0)
                self.x_train.append(x_train)
                self.y_train.append(self.cond_to_torch(y))
                f_star_, cov_f_ = self.f_star_sm[-1], self.cov_f_sm[-1]
                self.f_star.append(f_star_)
                self.f_star_sm.append(f_star_)
                self.cov_f.append(cov_f_)
                self.cov_f_sm.append(cov_f_)
            return self.f_star_sm[-1], self.cov_f_sm[-1]

    def include_weighted_sample(self, index, x_train, x_warped, y, h, snr=None):
        """Method to include the sample depending on the responsibility h.
        """
        y = self.cond_to_cuda(self.cond_to_torch(y))
        x_train = self.cond_to_cuda(self.cond_to_torch(x_train))
        new_x_basis = self.x_basis
        if h > 0.0:
            if self.N == 0 and not self.fitted:
                if torch.allclose(torch.from_numpy(self.gp.kernel.theta), torch.from_numpy(self.ini_kernel_theta)):
                    new_x_basis, _ = self.fit_kernel_params(x_train, y, self.Sigma[-1], self.Gamma[-1], valid=True)
                else:
                    new_x_basis, _ = self.fit_kernel_params(x_train, y, self.Sigma[-1], self.Gamma[-1], valid=False)
            if snr is not None:
                if snr > 0.5:
                    self.include_sample(index, x_train, y, x_warped, h=h)
                else:
                    self.include_sample(index, x_train, y, x_warped, posterior=False, include_index=True)
            else:
                self.include_sample(index, x_train, y, x_warped, h=h)
        else:
            self.include_sample(index, x_train, y, x_warped, posterior=False)
        return new_x_basis

    def full_pass_weighted(self, x_trains, y_trains, resp, q=None, q_lat=None, snr=None,
                           min_responsibility=None):
        """Full forward/smoothing pass followed by one conjugate LDS update."""
        if min_responsibility is None:
            min_responsibility = getattr(self, "min_responsibility", 1e-4)
        if len(torch.nonzero(self.Gamma[-1])) < 1:
            model_type = 'static'
        else:
            model_type = 'dynamic'

        active = torch.nonzero(resp > min_responsibility, as_tuple=False).squeeze(1)
        if active.numel() == 0:
            return q, q_lat

        internal_prior = (
            self.internal_params.clone() if self.bayesian else None
        )
        observation_prior = (
            self.observation_params.clone() if self.bayesian else None
        )

        for index in active.tolist():
            h = float(resp[index].item())
            # SNR is applied when output scores are combined by GPI_HDP. Using
            # it here as a second gate would fit each output on a different
            # subset while still evaluating that output on the complete batch.
            self.include_weighted_sample(
                index,
                x_trains[index],
                x_trains[index],
                y_trains[index],
                h,
            )
            if model_type == 'dynamic':
                self.backwards_pair(h)

        if model_type == 'dynamic':
            self.backwards()
        if self.bayesian:
            self._bayesian_batch_update(
                model_type,
                internal_prior=internal_prior,
                observation_prior=observation_prior,
            )

        q_ = self.compute_sq_err_all(x_trains, y_trains)
        q_lat_ = self.compute_q_lat_all(x_trains)
        return q_, q_lat_

    def reinit_GP(self, save_last=False, save_index=False):
        """ Method to reinitiate GP parameters. Can save some of them.
        """
        if save_last:
            self.y_var = [self.y_var[0],self.y_var[-1]]
            self.var = [self.var[0],self.var[-1]]
            self.f_star = [self.f_star[0],self.f_star[-1]]
            self.f_star_sm = [torch.clone(self.f_star[0]),torch.clone(self.f_star[-1])]
            self.cov_f = [self.cov_f[0],self.cov_f[-1]]
            self.cov_f_sm = [self.cov_f_sm[0],self.cov_f_sm[-1]]
            self.y_train = []
            self.x_train = []
            if not save_index:
                self.indexes = [0]

        else:
            self.y_var = self.y_var[:1]
            self.var = self.var[:1]
            self.f_star = self.f_star[:1]
            self.f_star_sm = self.f_star[:1].copy()
            self.cov_f = [torch.clone(self.ini_cov_def)]
            self.cov_f_sm = [torch.clone(self.ini_cov_def)]
            self.indexes = []
            self.y_train = []
            self.x_train = []
        self.sample_weights = []
        self.likelihood = []
        self.N = 0

    def start_new_batch(self, *, carry_latent_state=False):
        """Reset sequence data without discarding learned GP/LDS distributions."""
        if self.bayesian:
            self._batch_internal_prior = self.internal_params.clone()
            self._batch_observation_prior = self.observation_params.clone()
        if carry_latent_state:
            mean = self.f_star_sm[-1].detach().clone()
            covariance = self.cov_f_sm[-1].detach().clone()
        else:
            mean = self.compute_mean().detach().clone()
            covariance = self.ini_cov_def.detach().clone()
        self.f_star = [mean]
        self.f_star_sm = [mean.clone()]
        self.cov_f = [covariance]
        self.cov_f_sm = [covariance.clone()]
        self.cross_cov_f_sm = []
        self.A = [self.A[-1].detach().clone()]
        self.Gamma = [self.Gamma[-1].detach().clone()]
        self.C = [self.C[-1].detach().clone()]
        self.Sigma = [self.Sigma[-1].detach().clone()]
        self.var = [torch.atleast_2d(torch.sqrt(torch.diag(self.Gamma[-1]))).T]
        self.y_var = [torch.atleast_2d(torch.sqrt(torch.diag(self.Sigma[-1]))).T]
        self.x_train = []
        self.y_train = []
        self.indexes = []
        self.sample_weights = []
        self.likelihood = []
        self.N = 0
        self.cross_cov_f_sm = []


    def reinit_LDS(self, save_last=False, save_last_diag=False, return_likelihood=False):
        """ Method to reinitiate LDS parameters. Can save some of them.
        """
        if (
            not save_last
            and hasattr(self, "_batch_internal_prior")
            and hasattr(self, "_batch_observation_prior")
        ):
            if return_likelihood:
                A_, Gam_, C_, Sig_ = (
                    self.A[-1], self.Gamma[-1], self.C[-1], self.Sigma[-1]
                )
            self.internal_params = self._batch_internal_prior.clone()
            self.observation_params = self._batch_observation_prior.clone()
            self.A = [self.internal_params.get_mean()]
            self.Gamma = [self.internal_params.get_scale()]
            if isinstance(self.observation_params, inv_wishart):
                observation_mean = self.observation_params.get_C()
            else:
                observation_mean = self.observation_params.get_mean()
            self.C = [observation_mean]
            self.Sigma = [self.observation_params.get_scale()]
            if return_likelihood:
                internal_score = self.internal_params.log_likelihood_MNIW(A_, Gam_)
                if isinstance(self.observation_params, inv_wishart):
                    observation_score = self.observation_params.log_likelihood_IW(Sig_)
                else:
                    observation_score = self.observation_params.log_likelihood_MNIW(C_, Sig_)
                return internal_score, observation_score
            return
        if save_last:
            ind_ = -1
            if save_last_diag:
                ini_A, ini_Gamma, ini_C, ini_Sigma = self.A_def, torch.diag(torch.diag(self.Gamma[ind_])) * 3.0, self.C_def, torch.diag(torch.diag(self.Sigma[ind_])) * 3.0
            else:
                ini_A, ini_Gamma, ini_C, ini_Sigma = self.A[ind_], self.Gamma[ind_], self.C[ind_], self.Sigma[ind_]
        else:
            ini_A, ini_Gamma, ini_C, ini_Sigma = self.A_def, self.Gamma_def, self.C_def, self.Sigma_def
            if return_likelihood:
                A_, Gam_, C_, Sig_ = self.A[-1], self.Gamma[-1], self.C[-1], self.Sigma[-1]
        self.A = [ini_A]
        self.Gamma = [ini_Gamma]
        self.C = [ini_C]
        self.Sigma = [ini_Sigma]
        self.internal_params = matrix_normal_inv_wishart(ini_A, torch.eye(ini_A.shape[0], device=self.device), self.free_deg_MNIV, ini_Gamma)
        if torch.count_nonzero(ini_Gamma) == 0 or not self.learn_observation_matrix:
            self.observation_params = inv_wishart(
                self.free_deg_MNIV, ini_Sigma, ini_C
            )
        else:
            self.observation_params = matrix_normal_inv_wishart(
                ini_C,
                torch.eye(ini_C.shape[0], device=self.device),
                self.free_deg_MNIV,
                ini_Sigma,
            )
        for name in ("_batch_internal_prior", "_batch_observation_prior"):
            if hasattr(self, name):
                delattr(self, name)
        if return_likelihood:
            internal_score = self.internal_params.log_likelihood_MNIW(A_, Gam_)
            if isinstance(self.observation_params, matrix_normal_inv_wishart):
                observation_score = self.observation_params.log_likelihood_MNIW(C_, Sig_)
            else:
                observation_score = self.observation_params.log_likelihood_IW(Sig_)
            return internal_score, observation_score

    def return_LDS_param_likelihood(self, first=False):
        """Return negative KLs for the Bayesian LDS parameter posteriors."""
        if not self.bayesian:
            return torch.zeros((), device=self.x_basis.device, dtype=self.x_basis.dtype)

        parameter_elbo = torch.zeros(
            (), device=self.x_basis.device, dtype=self.x_basis.dtype
        )
        if torch.count_nonzero(self.Gamma_def) > 0:
            internal_prior = getattr(self, "_batch_internal_prior", None)
            if internal_prior is None:
                internal_prior = matrix_normal_inv_wishart(
                    self.A_def,
                    torch.eye(
                        self.A_def.shape[1],
                        device=self.A_def.device,
                        dtype=self.A_def.dtype,
                    ),
                    self.free_deg_MNIV,
                    self.Gamma_def,
                )
            parameter_elbo = parameter_elbo - self.internal_params.kl_divergence(
                internal_prior
            )

        observation_prior = getattr(self, "_batch_observation_prior", None)
        if observation_prior is None:
            if isinstance(self.observation_params, inv_wishart):
                observation_prior = inv_wishart(
                    self.free_deg_MNIV, self.Sigma_def, self.C_def
                )
            else:
                observation_prior = matrix_normal_inv_wishart(
                    self.C_def,
                    torch.eye(
                        self.C_def.shape[1],
                        device=self.C_def.device,
                        dtype=self.C_def.dtype,
                    ),
                    self.free_deg_MNIV,
                    self.Sigma_def,
                )
        parameter_elbo = parameter_elbo - self.observation_params.kl_divergence(
            observation_prior
        )
        return parameter_elbo

    def compute_sq_err_all(self, x_trains, y_trains, no_first=False):
        """Method to compute the squared error over all provided examples y_trains."""
        n_samps = x_trains.shape[0]
        device = x_trains.device if torch.is_tensor(x_trains) else x_trains[0].device
        sq_err = torch.zeros(n_samps, device=device, dtype=torch.float64)

        if len(self.indexes) == 0:
            return sq_err

        idx_t = torch.as_tensor(self.indexes, device=device, dtype=torch.long)
        sample_ids = torch.arange(n_samps, device=device, dtype=torch.long)

        pos_of_sample = torch.full((n_samps,), -1, device=device, dtype=torch.long)
        pos_of_sample[idx_t] = torch.arange(idx_t.numel(), device=device, dtype=torch.long)

        exact_mask = pos_of_sample >= 0
        closest_pos0 = torch.searchsorted(idx_t, sample_ids, right=True) - 1
        closest_pos0 = torch.clamp(closest_pos0, min=0)

        # Keep your current behavior
        i_vals = torch.where(
            exact_mask,
            pos_of_sample + 1,
            torch.clamp(closest_pos0, min=1)
        )
        # Fast path only if all samples share the same x-grid
        shared_grid = torch.equal(x_trains, x_trains[0:1].expand_as(x_trains))
        if shared_grid:
            x0 = x_trains[0]
            for time_index in torch.unique(i_vals):
                ids = torch.nonzero(
                    i_vals == time_index, as_tuple=False
                ).squeeze(1)
                i_cur = int(time_index.item())

                if (
                    torch.equal(x0, self.x_basis)
                    and
                    self.bayesian
                    and self.observation_params is not None
                ):
                    latent_mean, latent_covariance, _, _ = self._latent_parameters_at(
                        i_cur, params=None
                    )
                    sq_err[ids] = self.observation_params.expected_log_likelihood_batch(
                        y_trains[ids], latent_mean, latent_covariance
                    )
                    continue

                f_star, latent_covariance, noise_covariance = self._emission_moments(
                    x0, i_cur, params=None
                )
                sq_err[ids] = self._gaussian_score_shared_cov(
                    y_trains[ids],
                    f_star,
                    noise_covariance,
                    mean_covariance=latent_covariance,
                )

            return sq_err

        # Fallback for irregular x-grids
        i_vals_cpu = i_vals.cpu().tolist()

        for index in trange(n_samps, desc="Compute_sq_error", disable=self.disable):
            sq_err[index] = self.log_sq_error(
                x_trains[index],
                y_trains[index],
                i=i_vals_cpu[index],
            )

        return sq_err

    def compute_q_lat_all(self, x_trains, h_ini=1.0):
        """Method to compute the latent squared error accumulated.
        """
        sq_err = torch.zeros(x_trains.shape[0], device=x_trains[0].device)
        if self.N == 0:
            return sq_err
        if len(torch.nonzero(self.Gamma[-1]))< 1:
            return sq_err
        for j, index in enumerate(self.indexes):
            sq_err[index] = self.log_lat_error(j, h_ini)
        return sq_err

    def posterior_weighted(self, x_train, y, h, t=None):
        """ Method to compute the posterior depending on the responsibility h.
        """
        y = self.cond_to_torch(y)
        x_train = self.cond_to_torch(x_train)
        if h > 0.0:
            if t is not None and len(self.indexes) > t:
                f_star_sm_, cov_f_sm_ = self.f_star[t], self.cov_f[t]
                A, Gamma, C, Sigma = self.get_params(t)
            else:
                f_star_sm_ = self.f_star[-1]
                cov_f_sm_ = self.cov_f[-1]
                A = self.A[-1]
                Gamma = self.Gamma[-1]
                C = self.C[-1]
                Sigma = self.Sigma[-1]
            f_star_, cov_f_ = self.gp.posterior(f_star_sm_, cov_f_sm_, y, A,
                                                Gamma / h, C, Sigma / h, x_train=x_train, h=h)
        else:
            f_star_ = torch.clone(self.f_star[-1])
            cov_f_ = torch.clone(self.cov_f[-1])
        return f_star_, cov_f_

    def find_closest_lower(self, t):
        """Method to compute the closest last sample added to the model.
        """
        #List is assumed sorted
        lst = self.indexes
        idx = bisect_right(lst, t)
        if idx:
            return idx-1
        else:
            return 0

    def step_forward_last(self, x_post, params=None):
        """ Compute the observation over new x_post.
        """
        if params is None:
            C = self.C[-1]
            A = self.A[-1]
            Gamma = self.Gamma[-1]
            Sigma = self.Sigma[-1]
            mean = self.f_star_sm[-1]
            cov = self.cov_f_sm[-1]
        else:
            A = self.A[-1]
            Gamma = self.Gamma[-1]
            mean = params[0]
            cov = params[1]
            C = params[2]
            Sigma = params[3]
        mean = torch.linalg.multi_dot([C, mean])
        Sigma = Sigma# + torch.linalg.multi_dot([C, Gamma, C.T])
        x_basis = self.x_basis
        return self.gp.pred_dist(x_post, x_basis, mean, Sigma)

    def observe_last(self, x_post):
        """ Compute the last observation distribution over new x_post.
        """
        C = self.C[-1]
        Sigma = self.Sigma[-1]
        mean = torch.matmul(C, self.f_star_sm[-1])
        x_basis = self.x_basis
        return self.gp.pred_dist(x_post, x_basis, mean, Sigma)

    def observe(self, x_post, t, params=None, proj=False):
        """
        Method to resample the emission GP distribution at time t or using the parameters given.
        :param x_post: time set to resample
        :param t: step n where to resample the model
        :param params: tuple (mean, cov, Gamma, Sigma)
        """
        if params is None:
            #Case when model is not initialited
            if len(self.indexes) == 0:
                C = self.C[0]
                Sigma = self.Sigma[0]
                mean = torch.matmul(C, self.f_star[0])
            #Case when computing error with last (predict)
            elif len(self.indexes) <= t:
                C = self.C[-1]
                Sigma = self.Sigma[-1]
                A = self.A[-1]
                Gamma = self.Gamma[-1]
                mean = torch.linalg.multi_dot([C, self.f_star[-1]])
            elif self.estimation_limit <= t:
                C = self.C[-1]
                Sigma = self.Sigma[-1]
                if proj:
                    Sigma = Sigma + self.Gamma[-1]
                mean = torch.matmul(C, self.f_star[t])
            else:
                A, Gamma, C, Sigma = self.get_params(t)
                if proj:
                    Sigma = Sigma + Gamma
                mean = torch.matmul(C, self.f_star[t])
        else:
            mean = params[0]
            Sigma = params[3]
            mean = torch.matmul(params[2], mean)
        x_basis = self.x_basis
        return self.gp.pred_dist(x_post, x_basis, mean, Sigma)

    def get_params(self, t):
        """Method to return params on a specific iteration of the model
        """
        rest_len = len(self.C)
        ind = t if t < rest_len else -1
        return self.A[ind], self.Gamma[ind], self.C[ind], self.Sigma[ind]

    def resample_latent_mean(self, x_post, t=None, params=None):
        """Method to resample the latent process on a specific iteration of the model
        """
        if params is None:
            if t is None or t> len(self.indexes):
                mean = self.f_star_sm[-1]
                cov = self.cov_f_sm[-1]
            else:
                mean = self.f_star_sm[t]
                cov = self.cov_f_sm[t]
        else:
            mean = params[0]
            cov = params[1]
        x_basis = self.x_basis
        return self.gp.pred_latent_dist(x_post, x_basis, mean, cov)

    def backwards(self, h=1.0):
        """Method to compute backward recursion weighted by the responsibility.
        """
        if len(torch.nonzero(self.Gamma[-1])) < 1:
            model_type = 'static'
        else:
            model_type = 'dynamic'
        if h > 0.0:
            mean = list(self.f_star[1:])
            covs = list(self.cov_f[1:])
            if model_type == 'dynamic':
                transition_matrices = self.A[1:] if len(self.A) > 1 else self.A
                transition_covariances = (
                    self.Gamma[1:] if len(self.Gamma) > 1 else self.Gamma
                )
                aux_f_star, aux_cov_f, cross = self.gp.backward(
                    transition_matrices,
                    transition_covariances,
                    mean,
                    covs,
                    return_cross=True,
                )
            else:
                aux_f_star, aux_cov_f, cross = self.gp.backward(
                    self.A[0], self.Gamma[0], mean, covs, return_cross=True
                )
            for i in range(len(mean)):
                self.f_star_sm[i + 1] = aux_f_star[i]
                self.cov_f_sm[i + 1] = aux_cov_f[i]
            self.cross_cov_f_sm = cross

    def backwards_pair(self, h, snr=None):
        """ Fast method to compute the last two backward iterations.
        """
        if len(self.indexes) > 1:
            if h > 0.0:
                if snr is None:
                    mean = list(self.f_star[-2:])
                    covs = list(self.cov_f[-2:])
                    aux_f_star, aux_cov_f, cross = self.gp.backward_notrange(
                        self.A[-1], self.Gamma[-1], mean, covs, return_cross=True
                    )
                    for i in range(len(mean)):
                        self.f_star_sm[-(i + 1)] = aux_f_star[-(i + 1)]
                        self.cov_f_sm[-(i + 1)] = aux_cov_f[-(i + 1)]
                    if cross:
                        if self.cross_cov_f_sm:
                            self.cross_cov_f_sm[-1] = cross[-1]
                        else:
                            self.cross_cov_f_sm = [cross[-1]]
                else:
                    if snr > 0.5:
                        mean = self.f_star_sm[-2:]
                        covs = self.cov_f_sm[-2:]
                        aux_f_star, aux_cov_f, cross = self.gp.backward_notrange(
                            self.A[-1], self.Gamma[-1], mean, covs, return_cross=True
                        )
                        for i in range(len(mean)):
                            self.f_star_sm[-(i + 1)] = aux_f_star[-(i + 1)]
                            self.cov_f_sm[-(i + 1)] = aux_cov_f[-(i + 1)]
                        if cross:
                            if self.cross_cov_f_sm:
                                self.cross_cov_f_sm[-1] = cross[-1]
                            else:
                                self.cross_cov_f_sm = [cross[-1]]

    def smoother_weighted(self, x_train, y, h):
        """ Method to compute the conditioned posterior and distribution if a sample is added.
        """
        f_star_aux, cov_f_aux = self.posterior_weighted(x_train, y, h)
        means = self.f_star.copy()
        means.append(f_star_aux)
        covs = self.cov_f.copy()
        covs.append(cov_f_aux)
        C = self.C.copy()
        C.append(self.C[-1])
        Sigma = self.Sigma.copy()
        Sigma.append(self.Sigma[-1])
        return means, covs, C, Sigma

    def smoother_weighted_index(self, x_train, y, h, t):
        """ Method to return the conditioned posterior and LDS parameters of a specific iteration of the model.
        """
        f_star_aux, cov_f_aux = self.posterior_weighted(x_train, y, h, t)
        A, Gamma, C, Sigma = self.get_params(t)
        return f_star_aux, cov_f_aux, C, Sigma

    def new_params(self, batch=None, reestimate=True, model_type='dynamic', verbose=True, check_var=False):
        """ Maximum Likelihood computation of the new LDS params.
        """
        if batch is None or batch >= self.N:
            batch = self.N
        if reestimate:
            converged = False
            # Declare parameters to start iterations.
            # As we include all y_train, we should select only indexed samples
            N = self.N
            means = self.f_star_sm[1:]
            covs = self.cov_f_sm[1:]
            y_samples = self.y_train
            A_prior = self.A[-1]
            Gamma_prior = self.Gamma[-1]
            C_prior = self.C[-1]
            Sigma_prior = self.Sigma[-1]
            if self.annealing:
                Gamma_prior = Gamma_prior - self.Gamma[0] / (2 * N)
                Sigma_prior = Sigma_prior - self.Sigma[0] / (2 * N)
            # Compute previous likelihood
            try:
                lik_pre = self.gp.log_likelihood(N - batch, batch - 1, means[N - batch:], covs[N - batch:], A_prior,
                                                 Gamma_prior, y_samples[N - batch:], C_prior, Sigma_prior)
            except RuntimeWarning:
                print("Starting parameters are divergent, using initials.")
                converged = True
            lik_best = lik_pre
            trials = 0
            if N < 101:
                try_max = 6
            else:
                try_max = 4
            A_best = A_prior
            Gamma_best = Gamma_prior
            C_best = C_prior
            Sigma_best = Sigma_prior
            while not converged and trials < try_max:
                try:
                    A_new, Gamma_new, C_new, Sigma_new = self.gp.new_params_LDS(A_prior, Gamma_prior, C_prior, Sigma_prior,
                                                                                y_samples[N - batch:], means[N - batch:],
                                                                                covs[N - batch:], model_type)
                    means, covs = self.gp.backward(A_new, Gamma_new, means, covs)
                    lik_post = self.gp.log_likelihood(N - batch, batch - 1, means[N - batch:], covs[N - batch:], A_new,
                                                      Gamma_new, y_samples[N - batch:], C_new, Sigma_new)
                except RuntimeWarning:
                    if verbose:
                        print("Finded parameters are divergent, using initials.")
                        break
                if not torch.isnan(lik_post) and torch.isclose(lik_best, lik_post, 0.01) and lik_best <= lik_post and not torch.isinf(lik_post):
                    converged = True
                    if verbose:
                        print('Last iteration: ', torch.abs(lik_pre - lik_post))
                    A_best = A_new
                    Gamma_best = Gamma_new
                    C_best = C_new
                    Sigma_best = Sigma_new
                else:
                    if torch.isnan(lik_post):
                        if verbose:
                            print('Singular matrix detected, using previous.')
                            trials = try_max
                    elif lik_best > lik_post:
                        if verbose:
                            print('Divergence detected, using previous.')
                            trials = try_max
                    elif torch.isinf(lik_post):
                        if verbose:
                            print('Divergence detected, using previous.')
                            trials = try_max
                    else:
                        if verbose:
                            print('Iterating: Step-', lik_post - lik_pre, ' Diference with best-', lik_post - lik_best)
                        if lik_best <= lik_post:
                            # Here we are obtaining a better model so we save it
                            lik_best = lik_post
                            A_best = A_new
                            Gamma_best = Gamma_new
                            C_best = C_new
                            Sigma_best = Sigma_new
                        # We keep iterating until we reach some critical point.
                        lik_pre = lik_post
                        A_prior = A_new
                        Gamma_prior = Gamma_new
                        C_prior = C_new
                        Sigma_prior = Sigma_new
                    trials = trials + 1
            if converged:
                if verbose:
                    print('Converged estimation of new LDS parameters.')
                if self.annealing:
                    Gamma_best = Gamma_best + self.Gamma[0] / (2 * N)
                    Sigma_best = Sigma_best + self.Sigma[0] / (2 * N)
                self.A.append(A_best)
                self.Gamma.append(Gamma_best)
                self.C.append(C_best)
                if check_var:
                    Sigma_best = self.check_bound_sigma(Sigma_best)
                self.Sigma.append(Sigma_best)
                self.var.append(torch.atleast_2d(torch.diag(Gamma_best)).T)
                self.y_var.append(torch.atleast_2d(torch.diag(Sigma_best)).T)
            else:
                self.A.append(self.A[-1])
                self.Gamma.append(self.Gamma[-1])
                self.C.append(self.C[-1])
                self.Sigma.append(self.Sigma[-1])
                self.var.append(torch.atleast_2d(torch.diag(self.Gamma[-1])).T)
                self.y_var.append(torch.atleast_2d(torch.diag(self.Sigma[-1])).T)
        else:
            self.A.append(self.A[-1])
            self.Gamma.append(self.Gamma[-1])
            self.C.append(self.C[-1])
            self.Sigma.append(self.Sigma[-1])
            self.var.append(torch.atleast_2d(torch.diag(self.Gamma[-1])).T)
            self.y_var.append(torch.atleast_2d(torch.diag(self.Sigma[-1])).T)

    def check_bound_sigma(self, S):
        """ Method to check variance bounds (not actually used)
        """
        bounds = np.exp(self.gp.kernel.bounds[0]) ** 2
        for i in range(S.shape[0]):
            if S[i][i] < bounds[0]:
                S[i][i] = bounds[0]
            elif S[i][i] > bounds[1]:
                S[i][i] = bounds[1]
        return S

    def new_params_weighted(self, h, batch=None, reestimate=True, model_type='dynamic', min_samples=1, max_samples=6,
                            div_samples=15, verbose=True, check_var=False):
        """ Method to compute iteratively LDS params conditioned on the responsibility.
        """
        if not np.isclose(h, 0, rtol=1e-1, atol=1e-1):
            num_included = self.N
            if num_included > 500:
                div_samples = 10
            if num_included > min_samples and num_included < max_samples or (
                    num_included % div_samples == 0 and not num_included == 0):
                self.backwards()
                self.new_params(batch, reestimate, model_type, verbose=verbose, check_var=check_var)
            else:
                self.new_params(0, reestimate=False, verbose=verbose)

    def revise_constraint_noise_step(self, Sigma_prior, Sigma_new):
        """ Method of natural gradient for covariance estimation (not actually used)
        """
        # Condition to ensure a minimal variance is required (occurs when two samples
        # are so similar and we assume a very certain model)
        if np.trace(Sigma_new) / np.trace(Sigma_prior) < 0.3 or np.trace(Sigma_new) / np.trace(Sigma_prior) > 2.0:
            # If reduces a third or doubles their medium diagonal variance value we assume is a strong assumption.
            Sigma_new = (Sigma_new + Sigma_prior) / 2
        return Sigma_new

    def KL_divergence(self, t, gpmodel, t_gp, smoothed=True, x_bas=None):
        """ Method to compute the Kullback-Leibler divergence over every iteration of two LDS models.
        """
        l1 = t
        l2 = t_gp
        if smoothed:
            f_m1 = self.f_star_sm[l1 + 1]
            f_m2 = gpmodel.f_star_sm[l2 + 1]
            cov_m1 = self.cov_f_sm[l1 + 1]
            cov_m2 = gpmodel.cov_f_sm[l2 + 1]
        else:
            f_m1 = self.f_star[l1 + 1]
            f_m2 = gpmodel.f_star[l2 + 1]
            cov_m1 = self.cov_f[l1 + 1]
            cov_m2 = gpmodel.cov_f[l2 + 1]
        if self.estimation_limit <= t:
            t = -1
        if gpmodel.estimation_limit <= t_gp:
            t_gp = -1
        if len(self.Gamma) == 0:
            return self.gp.KL_divergence(f_m1, cov_m1, f_m2, cov_m2)
        elif torch.all(self.Gamma[-1] == 0):
            return self.gp.KL_divergence(f_m1, cov_m1, f_m2, cov_m2)
        else:
            if x_bas is not None and not torch.equal(x_bas, self.x_basis):
                mean1, cov1 = self.observe(x_bas, t, params=[f_m1, cov_m1, self.C[t], self.Sigma[t]])
                mean2, cov2 = gpmodel.observe(x_bas, t_gp, params=[f_m2, cov_m2, gpmodel.C[t_gp], gpmodel.Sigma[t_gp]])
            else:
                mean1 = torch.matmul(self.C[t], f_m1)
                mean2 = torch.matmul(gpmodel.C[t_gp], f_m2)
                cov1 = torch.linalg.multi_dot([self.C[t], cov_m1, self.C[t].T]) + self.Sigma[t]
                cov2 = torch.linalg.multi_dot([gpmodel.C[t_gp], cov_m2, gpmodel.C[t_gp].T]) + gpmodel.Sigma[t_gp]
            return self.gp.KL_divergence(mean1, cov1, mean2, cov2)

    def compute_mean(self):
        """ Method to resample mean (usually zero mean)
        """
        return self.gp.compute_mean(self.x_basis)

    def plot_last(self, num_model):
        """ Predefined method to plot last iteration of the model.
        """
        if len(self.indexes) == 0:
            y = np.repeat(0, len(self.x_basis))
            x_train = np.repeat(0, len(self.x_basis))
        else:
            y = self.y_train[-1]
            x_train = self.x_train[-1]

        self.gp.plotGP(len(self.indexes) + num_model, self.x_basis,
                       np.dot(self.C[-1], self.f_star[-1]), np.sqrt(self.var[-1]),
                       x_train, y, np.sqrt(self.y_var[-1]),
                       title=True, label_model=num_model, labels=True)

    def sample_last(self, num_samples=1, random_state=0):
        """ Method to resample last GP as a distribution.
        """
        samples = self.gp.sample_y(self.f_star_sm[-1], self.cov_f_sm[-1], self.C[-1], self.Sigma[-1], num_samples,
                                   random_state).T
        rav_samples = []
        for i in range(num_samples):
            rav_samples.append(samples[i][0])
        return rav_samples

    def reduce_noise_matrix(self, x_basis=None, x_train=None):
        return self.gp.projection_matrix(x_basis, x_train)

    @staticmethod
    def _as_column(value):
        value = torch.as_tensor(value)
        return value[:, None] if value.ndim == 1 else value

    def _project_observation_to_basis(self, position):
        """Represent one observation on the LDS basis without moving the latent state."""
        observation = self.cond_to_cuda(self.cond_to_torch(self.y_train[position]))
        observation = self._as_column(observation)
        x_train = self.cond_to_cuda(self.cond_to_torch(self.x_train[position]))
        if torch.equal(x_train, self.x_basis):
            return observation
        projection = self.reduce_noise_matrix(self.x_basis, x_train)
        projection = projection.to(
            device=observation.device, dtype=observation.dtype
        )
        return projection @ observation

    def _parameter_observation_count(self):
        count = len(self.y_train)
        if np.isfinite(self.estimation_limit):
            count = min(count, max(int(self.estimation_limit), 0))
        return count

    def _weight_at(self, position):
        if position < len(self.sample_weights):
            return float(self.sample_weights[position])
        return 1.0

    def _append_bayesian_parameter_snapshot(self, model_type):
        if model_type == 'dynamic':
            transition_mean = self.internal_params.get_mean()
            transition_covariance = self.internal_params.get_scale()
        else:
            transition_mean = self.A[-1]
            transition_covariance = self.Gamma[-1]

        if isinstance(self.observation_params, inv_wishart):
            observation_mean = self.observation_params.get_C()
        else:
            observation_mean = self.observation_params.get_mean()
        observation_covariance = self.observation_params.get_scale()

        self.A.append(transition_mean)
        self.Gamma.append(transition_covariance)
        self.C.append(observation_mean)
        self.Sigma.append(observation_covariance)
        self.var.append(
            torch.atleast_2d(torch.sqrt(torch.diag(transition_covariance))).T
        )
        self.y_var.append(
            torch.atleast_2d(torch.sqrt(torch.diag(observation_covariance))).T
        )

    def _bayesian_observation_update(self, prior, positions):
        """Update (C, Sigma) from weighted E[f f^T], y f^T and y y^T."""
        if isinstance(prior, matrix_normal_inv_wishart):
            s_xx = torch.zeros_like(prior.m_r_cov)
            s_yx = torch.zeros_like(prior.m_mean)
            s_yy = torch.zeros_like(prior.scale)
        else:
            scatter = torch.zeros_like(prior.scale)
        total_weight = 0.0

        for position in positions:
            weight = self._weight_at(position)
            if weight <= 0.0:
                continue
            observation = self._project_observation_to_basis(position)
            latent_mean = self._as_column(self.f_star_sm[position + 1])
            latent_covariance = self.cov_f_sm[position + 1]
            if isinstance(prior, matrix_normal_inv_wishart):
                s_xx = s_xx + weight * (
                    latent_covariance + latent_mean @ latent_mean.T
                )
                s_yx = s_yx + weight * (observation @ latent_mean.T)
                s_yy = s_yy + weight * (observation @ observation.T)
            else:
                zero_observation_covariance = torch.zeros_like(prior.scale)
                zero_cross_covariance = torch.zeros_like(latent_covariance)
                scatter = scatter + weight * expected_residual_scatter(
                    observation,
                    latent_mean,
                    prior.C_fixed,
                    zero_observation_covariance,
                    latent_covariance,
                    zero_cross_covariance,
                )
            total_weight += weight

        if isinstance(prior, matrix_normal_inv_wishart):
            return prior.posterior_from_sufficient_statistics(
                total_weight, s_xx, s_yx, s_yy
            )
        return prior.posterior_from_sufficient_statistics(total_weight, scatter)

    def _bayesian_transition_update(self, prior, positions):
        """Update (A, Gamma) using lag-one RTS cross-covariances."""
        s_xx = torch.zeros_like(prior.m_r_cov)
        s_yx = torch.zeros_like(prior.m_mean)
        s_yy = torch.zeros_like(prior.scale)
        total_weight = 0.0

        for position in positions:
            weight = self._weight_at(position)
            if weight <= 0.0:
                continue
            cross_index = position - 1
            if (
                cross_index >= len(self.cross_cov_f_sm)
                or self.cross_cov_f_sm[cross_index] is None
            ):
                raise RuntimeError(
                    "missing lag-one smoothing covariance for LDS posterior"
                )
            current_mean = self._as_column(self.f_star_sm[position + 1])
            previous_mean = self._as_column(self.f_star_sm[position])
            current_covariance = self.cov_f_sm[position + 1]
            previous_covariance = self.cov_f_sm[position]
            cross_covariance = self.cross_cov_f_sm[cross_index]
            s_xx = s_xx + weight * (
                previous_covariance + previous_mean @ previous_mean.T
            )
            s_yx = s_yx + weight * (
                cross_covariance + current_mean @ previous_mean.T
            )
            s_yy = s_yy + weight * (
                current_covariance + current_mean @ current_mean.T
            )
            total_weight += weight

        return prior.posterior_from_sufficient_statistics(
            total_weight, s_xx, s_yx, s_yy
        )

    def _bayesian_batch_update(
        self, model_type, *, internal_prior, observation_prior
    ):
        """One exact conjugate M-step from final smoothed batch moments."""
        count = self._parameter_observation_count()
        positions = range(count)
        self.observation_params = self._bayesian_observation_update(
            observation_prior, positions
        )
        if model_type == 'dynamic' and count > 1:
            # There are N observation factors but only N-1 transition factors.
            self.internal_params = self._bayesian_transition_update(
                internal_prior, range(1, count)
            )
        else:
            self.internal_params = internal_prior.clone()
        self._append_bayesian_parameter_snapshot(model_type)

    def _online_transition_cross_covariance(self):
        if self.cross_cov_f_sm and self.cross_cov_f_sm[-1] is not None:
            return self.cross_cov_f_sm[-1]
        transition = self.A[-1]
        transition_covariance = self.Gamma[-1]
        previous_covariance = self.cov_f_sm[-2]
        prediction_covariance = (
            transition @ previous_covariance @ transition.T
            + transition_covariance
        )
        smoother_gain = torch.cholesky_solve(
            transition @ previous_covariance.T,
            self._chol_spd(prediction_covariance),
        ).T
        return self.cov_f_sm[-1] @ smoother_gain.T

    def bayesian_new_params(self, h, model_type='dynamic', full_data=False, q=None, force=False, snr=1.0):
        """Update the LDS posterior online, or recompute it from smoothed data.

        The inverse-Wishart prior is the regularizer. ``annealing`` is therefore
        intentionally not added again to the posterior covariance estimate.
        """
        if not self.bayesian or h <= 0.0 or snr <= 0.5:
            return
        if model_type not in {'dynamic', 'static'}:
            raise ValueError("model_type must be 'dynamic' or 'static'")
        if torch.count_nonzero(self.Gamma[-1]) == 0:
            model_type = 'static'

        if full_data:
            internal_prior = getattr(
                self, "_batch_internal_prior", self.internal_params
            ).clone()
            observation_prior = getattr(
                self, "_batch_observation_prior", self.observation_params
            ).clone()
            self._bayesian_batch_update(
                model_type,
                internal_prior=internal_prior,
                observation_prior=observation_prior,
            )
            return

        if not self.y_train or self.N > self.estimation_limit:
            return
        position = len(self.y_train) - 1
        self.observation_params = self._bayesian_observation_update(
            self.observation_params, [position]
        )
        if model_type == 'dynamic' and len(self.y_train) > 1:
            cross = self._online_transition_cross_covariance()
            if self.cross_cov_f_sm:
                self.cross_cov_f_sm[-1] = cross
            else:
                self.cross_cov_f_sm = [cross]
            self.internal_params = self._bayesian_transition_update(
                self.internal_params, [position]
            )
        self._append_bayesian_parameter_snapshot(model_type)


    #Methods to perform the conversion from NumPy to torch and from Cuda to cpu. Still have to solve this.
    def cond_to_numpy(self, x):
        if x is not None:
            if type(x) is torch.Tensor:
                x = x.detach().numpy()
        return x

    def cond_to_torch(self, x):
        if x is not None:
            if type(x) is not torch.Tensor:
                x = torch.from_numpy(np.array(x))
                x.requires_grad = False
            else:
                x.requires_grad = False
        return x

    def cond_to_cuda(self, x):
        if self.cuda:
            if x is not None:
                if type(x) is torch.Tensor and torch.cuda.is_available():
                    x = x.cuda()
                    x.requires_grad = False
        return x

    def cond_to_cpu(self, x):
        if x is not None:
            if type(x) is torch.Tensor:
                if x.is_cuda:
                    x = x.cpu()
        return x

    def model_to_numpy(self):
        def recursive_numpy(x):
            if type(x) is torch.Tensor:
                x = self.cond_to_numpy(x)
            elif type(x) is list:
                if len(x) > 0:
                    if type(x[0]) is torch.Tensor:
                        for j, i in enumerate(x):
                            x[j] = self.cond_to_numpy(i)
            else:
                x = x
            return x

        self.x_basis = recursive_numpy(self.x_basis)
        self.x_train = recursive_numpy(self.x_train)
        self.y_train = recursive_numpy(self.y_train)
        self.f_star = recursive_numpy(self.f_star)
        self.f_star_sm = recursive_numpy(self.f_star_sm)
        self.cov_f = recursive_numpy(self.cov_f)
        self.cov_f_sm = recursive_numpy(self.cov_f_sm)
        self.y_var = recursive_numpy(self.y_var)
        self.var = recursive_numpy(self.var)
        self.A = recursive_numpy(self.A)
        self.Gamma = recursive_numpy(self.Gamma)
        self.C = recursive_numpy(self.C)
        self.Sigma = recursive_numpy(self.Sigma)
        self.likelihood = recursive_numpy(self.likelihood)
        self.K = recursive_numpy(self.K)
        self.internal_params.to_numpy()
        self.observation_params.to_numpy()

    def model_to_torch(self):
        def recursive_torch(x):
            if type(x) is np.ndarray:
                x = self.cond_to_torch(x)
            elif type(x) is list:
                if len(x) > 0:
                    if type(x[0]) is np.ndarray or type(x[0]) is np.float64:
                        for j, i in enumerate(x):
                            x[j] = self.cond_to_torch(i)
            else:
                x = x
            return x

        self.x_basis = recursive_torch(self.x_basis)
        self.x_train = recursive_torch(self.x_train)
        self.y_train = recursive_torch(self.y_train)
        self.f_star = recursive_torch(self.f_star)
        self.f_star_sm = recursive_torch(self.f_star_sm)
        self.cov_f = recursive_torch(self.cov_f)
        self.cov_f_sm = recursive_torch(self.cov_f_sm)
        self.y_var = recursive_torch(self.y_var)
        self.var = recursive_torch(self.var)
        self.A = recursive_torch(self.A)
        self.Gamma = recursive_torch(self.Gamma)
        self.C = recursive_torch(self.C)
        self.Sigma = recursive_torch(self.Sigma)
        self.likelihood = recursive_torch(self.likelihood)
        self.K = recursive_torch(self.K)
        self.internal_params.to_torch()
        self.observation_params.to_torch()

    def model_to_cuda(self):
        if torch.cuda.is_available() and self.cuda:
            def recursive_cuda(x):
                if type(x) is torch.Tensor:
                    x = x.cuda()
                    x.requires_grad = False
                elif type(x) is list:
                    if len(x) > 0:
                        if type(x[0]) is torch.Tensor or type(x[0]) is torch.float64:
                            for j, i in enumerate(x):
                                x[j] = i.cuda()
                                x[j].requires_grad = False
                else:
                    x = x
                return x

            self.x_basis = recursive_cuda(self.x_basis)
            self.x_train = recursive_cuda(self.x_train)
            self.y_train = recursive_cuda(self.y_train)
            self.f_star = recursive_cuda(self.f_star)
            self.f_star_sm = recursive_cuda(self.f_star_sm)
            self.cov_f = recursive_cuda(self.cov_f)
            self.cov_f_sm = recursive_cuda(self.cov_f_sm)
            self.y_var = recursive_cuda(self.y_var)
            self.var = recursive_cuda(self.var)
            self.A = recursive_cuda(self.A)
            self.Gamma = recursive_cuda(self.Gamma)
            self.C = recursive_cuda(self.C)
            self.Sigma = recursive_cuda(self.Sigma)
            self.likelihood = recursive_cuda(self.likelihood)
            self.K = recursive_cuda(self.K)
            self.internal_params.to_cuda()
            self.observation_params.to_cuda()
            self.cuda = True
            self.gp.cuda = True

    def model_to_cpu(self):
        def recursive_cpu(x):
            if type(x) is torch.Tensor:
                x = x.cpu()
            elif type(x) is list:
                if len(x) > 0:
                    if type(x[0]) is torch.Tensor or type(x[0]) is torch.float64:
                        for j, i in enumerate(x):
                            x[j] = i.cpu()
            else:
                x = x
            return x

        self.x_basis = recursive_cpu(self.x_basis)
        self.x_train = recursive_cpu(self.x_train)
        self.y_train = recursive_cpu(self.y_train)
        self.f_star = recursive_cpu(self.f_star)
        self.f_star_sm = recursive_cpu(self.f_star_sm)
        self.cov_f = recursive_cpu(self.cov_f)
        self.cov_f_sm = recursive_cpu(self.cov_f_sm)
        self.y_var = recursive_cpu(self.y_var)
        self.var = recursive_cpu(self.var)
        self.A = recursive_cpu(self.A)
        self.Gamma = recursive_cpu(self.Gamma)
        self.C = recursive_cpu(self.C)
        self.Sigma = recursive_cpu(self.Sigma)
        self.likelihood = recursive_cpu(self.likelihood)
        self.K = recursive_cpu(self.K)
        self.internal_params.to_cpu()
        self.observation_params.to_cpu()
        self.cuda = False
        self.gp.cuda = False

class matrix_normal_inv_wishart():
    """Matrix-normal inverse-Wishart with explicit natural parameters.

    ``n0`` is a prior strength. Internally the inverse-Wishart degrees of
    freedom are ``dimension + 1 + n0`` and ``scale`` stores Psi.
    """

    def __init__(self, m_mean, m_r_cov, n0, scale, _natural=False):
        self.m_mean = torch.as_tensor(m_mean).detach().clone()
        self.m_r_cov = torch.as_tensor(
            m_r_cov, device=self.m_mean.device, dtype=self.m_mean.dtype
        ).detach().clone()
        self.n0 = float(n0)
        if self.n0 <= 0:
            raise ValueError("inverse-Wishart strength must be positive")
        value = torch.as_tensor(
            scale, device=self.m_mean.device, dtype=self.m_mean.dtype
        ).detach().clone()
        self.scale = value if _natural else value * self.n0
        self.b = 0.7

    @property
    def degrees_of_freedom(self):
        return self.scale.shape[0] + 1.0 + self.n0

    def clone(self):
        return matrix_normal_inv_wishart(
            self.m_mean,
            self.m_r_cov,
            self.n0,
            self.scale,
            _natural=True,
        )

    def expected_precision(self):
        factor = stable_cholesky(self.scale)
        return self.degrees_of_freedom * torch.cholesky_inverse(factor)

    def expected_logdet(self):
        return inverse_wishart_expected_logdet(
            self.degrees_of_freedom, self.scale
        )

    def expected_log_likelihood(
        self,
        y_mean,
        x_mean,
        y_covariance,
        x_covariance,
        cross_covariance,
    ):
        """E[log N(y | Mx, Sigma)] under MNIW and Gaussian moments."""
        y_mean = torch.as_tensor(
            y_mean, device=self.scale.device, dtype=self.scale.dtype
        )
        x_mean = torch.as_tensor(
            x_mean, device=self.scale.device, dtype=self.scale.dtype
        )
        if y_mean.ndim == 1:
            y_mean = y_mean[:, None]
        if x_mean.ndim == 1:
            x_mean = x_mean[:, None]
        y_covariance = torch.as_tensor(
            y_covariance, device=self.scale.device, dtype=self.scale.dtype
        )
        x_covariance = torch.as_tensor(
            x_covariance, device=self.scale.device, dtype=self.scale.dtype
        )
        cross_covariance = torch.as_tensor(
            cross_covariance, device=self.scale.device, dtype=self.scale.dtype
        )
        scatter = expected_residual_scatter(
            y_mean,
            x_mean,
            self.m_mean,
            y_covariance,
            x_covariance,
            cross_covariance,
        )
        x_second_moment = x_covariance + x_mean @ x_mean.T
        column_covariance = torch.cholesky_inverse(
            stable_cholesky(self.m_r_cov)
        )
        row_dimension = y_mean.shape[0]
        coefficient_uncertainty = row_dimension * torch.trace(
            column_covariance @ x_second_moment
        )
        quadratic = torch.trace(self.expected_precision() @ scatter)
        return -0.5 * (
            quadratic
            + coefficient_uncertainty
            + self.expected_logdet()
            + row_dimension * y_mean.new_tensor(math.log(2.0 * math.pi))
        )

    def expected_log_likelihood_batch(self, observations, x_mean, x_covariance):
        observations = torch.as_tensor(
            observations, device=self.scale.device, dtype=self.scale.dtype
        )
        if observations.ndim == 3:
            observations = observations[..., 0]
        x_mean = torch.as_tensor(
            x_mean, device=self.scale.device, dtype=self.scale.dtype
        )
        if x_mean.ndim == 1:
            x_mean = x_mean[:, None]
        x_covariance = torch.as_tensor(
            x_covariance, device=self.scale.device, dtype=self.scale.dtype
        )
        predicted = self.m_mean @ x_mean
        residuals = observations.T - predicted
        expected_precision = self.expected_precision()
        residual_quadratics = torch.sum(
            residuals * (expected_precision @ residuals), dim=0
        )
        latent_trace = torch.trace(
            expected_precision @ self.m_mean @ x_covariance @ self.m_mean.T
        )
        column_covariance = torch.cholesky_inverse(
            stable_cholesky(self.m_r_cov)
        )
        x_second_moment = x_covariance + x_mean @ x_mean.T
        row_dimension = observations.shape[1]
        coefficient_uncertainty = row_dimension * torch.trace(
            column_covariance @ x_second_moment
        )
        constant = (
            latent_trace
            + coefficient_uncertainty
            + self.expected_logdet()
            + row_dimension * observations.new_tensor(math.log(2.0 * math.pi))
        )
        return -0.5 * (residual_quadratics + constant)

    def kl_divergence(self, prior):
        if not isinstance(prior, matrix_normal_inv_wishart):
            raise TypeError("MNIW KL requires another matrix_normal_inv_wishart")
        return matrix_normal_inverse_wishart_kl(
            self.m_mean,
            self.m_r_cov,
            self.degrees_of_freedom,
            self.scale,
            prior.m_mean,
            prior.m_r_cov,
            prior.degrees_of_freedom,
            prior.scale,
        )

    def posterior(self, n_k, y1, y2, cov, cov_, cov_cross, sse_matrix=None, annealing=False):
        weight = float(n_k)
        if weight <= 0:
            return self
        y1 = torch.as_tensor(y1, device=self.scale.device, dtype=self.scale.dtype)
        y2 = torch.as_tensor(y2, device=self.scale.device, dtype=self.scale.dtype)
        cov = torch.as_tensor(cov, device=self.scale.device, dtype=self.scale.dtype)
        cov_ = torch.as_tensor(cov_, device=self.scale.device, dtype=self.scale.dtype)
        cov_cross = torch.as_tensor(cov_cross, device=self.scale.device, dtype=self.scale.dtype)
        if y1.ndim == 1:
            y1 = y1[:, None]
        if y2.ndim == 1:
            y2 = y2[:, None]
        if sse_matrix is not None:
            projection = torch.as_tensor(
                sse_matrix, device=self.scale.device, dtype=self.scale.dtype
            )
            y1 = projection @ y1
            y2 = projection @ y2
            cov = projection @ cov @ projection.T
            cov_ = projection @ cov_ @ projection.T
            cov_cross = projection @ cov_cross @ projection.T

        statistic_weight = weight if y1.shape[1] == 1 else 1.0
        s_xx = statistic_weight * (y2 @ y2.T + cov_)
        s_yx = statistic_weight * (y1 @ y2.T + cov_cross)
        s_yy = statistic_weight * (y1 @ y1.T + cov)
        return self.posterior_from_sufficient_statistics(
            weight, s_xx, s_yx, s_yy
        )

    def posterior_from_sufficient_statistics(self, weight, s_xx, s_yx, s_yy):
        """Conjugate MNIW update from weighted regression statistics."""
        weight = float(weight)
        if weight <= 0:
            return self.clone()
        s_xx = torch.as_tensor(
            s_xx, device=self.scale.device, dtype=self.scale.dtype
        )
        s_yx = torch.as_tensor(
            s_yx, device=self.scale.device, dtype=self.scale.dtype
        )
        s_yy = torch.as_tensor(
            s_yy, device=self.scale.device, dtype=self.scale.dtype
        )
        prior_precision = self.m_r_cov
        posterior_precision = prior_precision + s_xx
        rhs = self.m_mean @ prior_precision + s_yx
        posterior_mean = torch.linalg.solve(posterior_precision.T, rhs.T).T
        posterior_psi = (
            self.scale
            + s_yy
            + self.m_mean @ prior_precision @ self.m_mean.T
            - posterior_mean @ posterior_precision @ posterior_mean.T
        )
        posterior_psi = 0.5 * (posterior_psi + posterior_psi.T)
        return matrix_normal_inv_wishart(
            posterior_mean,
            posterior_precision,
            self.n0 + weight,
            posterior_psi,
            _natural=True,
        )

    def log_likelihood_MNIW(self, M, Sigma, n0=None):
        """Normalized joint log density log p(M, Sigma) under this MNIW."""
        M = torch.as_tensor(M, device=self.scale.device, dtype=self.scale.dtype)
        Sigma = torch.as_tensor(
            Sigma, device=self.scale.device, dtype=self.scale.dtype
        )
        row_dimension, column_dimension = M.shape
        nu = Sigma.new_tensor(self.degrees_of_freedom)
        logdet_sigma = 2.0 * torch.log(torch.diag(stable_cholesky(Sigma))).sum()
        logdet_scale = 2.0 * torch.log(torch.diag(stable_cholesky(self.scale))).sum()
        logdet_precision = 2.0 * torch.log(
            torch.diag(stable_cholesky(self.m_r_cov))
        ).sum()
        sigma_factor = stable_cholesky(Sigma)
        inverse_scale_trace = torch.trace(
            torch.cholesky_solve(self.scale, sigma_factor)
        )
        difference = M - self.m_mean
        mean_quadratic = torch.trace(
            torch.cholesky_solve(
                difference @ self.m_r_cov @ difference.T, sigma_factor
            )
        )
        iw_log_density = (
            0.5 * nu * logdet_scale
            - 0.5 * nu * row_dimension * math.log(2.0)
            - torch.mvlgamma(0.5 * nu, row_dimension)
            - 0.5 * (nu + row_dimension + 1.0) * logdet_sigma
            - 0.5 * inverse_scale_trace
        )
        matrix_normal_log_density = (
            -0.5 * row_dimension * column_dimension * math.log(2.0 * math.pi)
            + 0.5 * row_dimension * logdet_precision
            - 0.5 * column_dimension * logdet_sigma
            - 0.5 * mean_quadratic
        )
        return iw_log_density + matrix_normal_log_density

    def get_mean(self):
        return self.m_mean

    def get_scale(self, final=False):
        return self.scale / self.n0


    def set_scale(self, scale):
        value = torch.as_tensor(scale, device=self.m_mean.device, dtype=self.m_mean.dtype)
        self.scale = value * self.n0

    def cond_to_numpy(self, x):
        if x is not None:
            if type(x) is torch.Tensor:
                x = x.detach().numpy()
        return x

    def cond_to_torch(self, x):
        if x is not None:
            if type(x) is not torch.Tensor:
                x = torch.from_numpy(np.array(x))
                x.requires_grad = False
        return x

    def to_numpy(self):
        self.m_mean = self.cond_to_numpy(self.m_mean)
        self.m_r_cov = self.cond_to_numpy(self.m_r_cov)
        self.scale = self.cond_to_numpy(self.scale)

    def to_torch(self):
        self.m_mean = self.cond_to_torch(self.m_mean)
        self.m_r_cov = self.cond_to_torch(self.m_r_cov)
        self.scale = self.cond_to_torch(self.scale)

    def to_cuda(self):
        self.m_mean = self.m_mean.cuda()
        self.m_r_cov = self.m_r_cov.cuda()
        self.scale = self.scale.cuda()

    def to_cpu(self):
        self.m_mean = self.m_mean.cpu()
        self.m_r_cov = self.m_r_cov.cpu()
        self.scale = self.scale.cpu()


class inv_wishart():
    """Inverse-Wishart posterior for a fixed observation transform."""

    def __init__(self, n0, scale, C_fixed, _natural=False):
        self.C_fixed = torch.as_tensor(C_fixed).detach().clone()
        self.n0 = float(n0)
        if self.n0 <= 0:
            raise ValueError("inverse-Wishart strength must be positive")
        value = torch.as_tensor(
            scale, device=self.C_fixed.device, dtype=self.C_fixed.dtype
        ).detach().clone()
        self.scale = value if _natural else value * self.n0
        self.b = 0.7

    @property
    def degrees_of_freedom(self):
        return self.scale.shape[0] + 1.0 + self.n0

    def clone(self):
        return inv_wishart(
            self.n0, self.scale, self.C_fixed, _natural=True
        )

    def expected_precision(self):
        factor = stable_cholesky(self.scale)
        return self.degrees_of_freedom * torch.cholesky_inverse(factor)

    def expected_logdet(self):
        return inverse_wishart_expected_logdet(
            self.degrees_of_freedom, self.scale
        )

    def expected_log_likelihood(
        self,
        y_mean,
        x_mean,
        y_covariance,
        x_covariance,
        cross_covariance,
    ):
        """E[log N(y | Cx, Sigma)] for fixed C and inverse-Wishart Sigma."""
        y_mean = torch.as_tensor(
            y_mean, device=self.scale.device, dtype=self.scale.dtype
        )
        x_mean = torch.as_tensor(
            x_mean, device=self.scale.device, dtype=self.scale.dtype
        )
        if y_mean.ndim == 1:
            y_mean = y_mean[:, None]
        if x_mean.ndim == 1:
            x_mean = x_mean[:, None]
        scatter = expected_residual_scatter(
            y_mean,
            x_mean,
            self.C_fixed,
            torch.as_tensor(
                y_covariance, device=self.scale.device, dtype=self.scale.dtype
            ),
            torch.as_tensor(
                x_covariance, device=self.scale.device, dtype=self.scale.dtype
            ),
            torch.as_tensor(
                cross_covariance, device=self.scale.device, dtype=self.scale.dtype
            ),
        )
        row_dimension = y_mean.shape[0]
        return -0.5 * (
            torch.trace(self.expected_precision() @ scatter)
            + self.expected_logdet()
            + row_dimension * y_mean.new_tensor(math.log(2.0 * math.pi))
        )

    def expected_log_likelihood_batch(self, observations, x_mean, x_covariance):
        observations = torch.as_tensor(
            observations, device=self.scale.device, dtype=self.scale.dtype
        )
        if observations.ndim == 3:
            observations = observations[..., 0]
        x_mean = torch.as_tensor(
            x_mean, device=self.scale.device, dtype=self.scale.dtype
        )
        if x_mean.ndim == 1:
            x_mean = x_mean[:, None]
        x_covariance = torch.as_tensor(
            x_covariance, device=self.scale.device, dtype=self.scale.dtype
        )
        predicted = self.C_fixed @ x_mean
        residuals = observations.T - predicted
        expected_precision = self.expected_precision()
        residual_quadratics = torch.sum(
            residuals * (expected_precision @ residuals), dim=0
        )
        latent_trace = torch.trace(
            expected_precision @ self.C_fixed @ x_covariance @ self.C_fixed.T
        )
        row_dimension = observations.shape[1]
        constant = (
            latent_trace
            + self.expected_logdet()
            + row_dimension * observations.new_tensor(math.log(2.0 * math.pi))
        )
        return -0.5 * (residual_quadratics + constant)

    def kl_divergence(self, prior):
        if not isinstance(prior, inv_wishart):
            raise TypeError("inverse-Wishart KL requires another inv_wishart")
        if not torch.allclose(self.C_fixed, prior.C_fixed):
            raise ValueError("fixed observation transforms differ")
        return inverse_wishart_kl(
            self.degrees_of_freedom,
            self.scale,
            prior.degrees_of_freedom,
            prior.scale,
        )

    def log_likelihood_IW(self, Sigma):
        """Normalized inverse-Wishart log density at ``Sigma``."""
        Sigma = torch.as_tensor(
            Sigma, device=self.scale.device, dtype=self.scale.dtype
        )
        dimension = Sigma.shape[0]
        nu = Sigma.new_tensor(self.degrees_of_freedom)
        logdet_sigma = 2.0 * torch.log(torch.diag(stable_cholesky(Sigma))).sum()
        logdet_scale = 2.0 * torch.log(
            torch.diag(stable_cholesky(self.scale))
        ).sum()
        inverse_scale_trace = torch.trace(
            torch.cholesky_solve(self.scale, stable_cholesky(Sigma))
        )
        return (
            0.5 * nu * logdet_scale
            - 0.5 * nu * dimension * math.log(2.0)
            - torch.mvlgamma(0.5 * nu, dimension)
            - 0.5 * (nu + dimension + 1.0) * logdet_sigma
            - 0.5 * inverse_scale_trace
        )

    def posterior(self, n_k, y1, y2, cov, cov_, cov_cross, sse_matrix=None, annealing=False):
        weight = float(n_k)
        if weight <= 0:
            return self
        y1 = torch.as_tensor(y1, device=self.scale.device, dtype=self.scale.dtype)
        y2 = torch.as_tensor(y2, device=self.scale.device, dtype=self.scale.dtype)
        cov = torch.as_tensor(cov, device=self.scale.device, dtype=self.scale.dtype)
        cov_ = torch.as_tensor(cov_, device=self.scale.device, dtype=self.scale.dtype)
        cov_cross = torch.as_tensor(cov_cross, device=self.scale.device, dtype=self.scale.dtype)
        if y1.ndim == 1:
            y1 = y1[:, None]
        if y2.ndim == 1:
            y2 = y2[:, None]
        transform = self.C_fixed
        if sse_matrix is not None:
            projection = torch.as_tensor(
                sse_matrix, device=self.scale.device, dtype=self.scale.dtype
            )
            y1 = projection @ y1
            y2 = projection @ y2
            cov = projection @ cov @ projection.T
            cov_ = projection @ cov_ @ projection.T
            cov_cross = projection @ cov_cross @ projection.T
        scatter = expected_residual_scatter(
            y1, y2, transform, cov, cov_, cov_cross
        )
        if y1.shape[1] == 1:
            scatter = weight * scatter
        return self.posterior_from_sufficient_statistics(weight, scatter)

    def posterior_from_sufficient_statistics(self, weight, scatter):
        """Conjugate inverse-Wishart update from weighted residual scatter."""
        weight = float(weight)
        if weight <= 0:
            return self.clone()
        scatter = torch.as_tensor(
            scatter, device=self.scale.device, dtype=self.scale.dtype
        )
        return inv_wishart(
            self.n0 + weight,
            self.scale + scatter,
            self.C_fixed,
            _natural=True,
        )

    def get_scale(self, final=False):
        return self.scale / self.n0

    def set_scale(self, scale):
        value = torch.as_tensor(scale, device=self.C_fixed.device, dtype=self.C_fixed.dtype)
        self.scale = value * self.n0

    def get_C(self):
        return self.C_fixed

    def get_mean(self):
        return self.C_fixed

    def cond_to_numpy(self, x):
        if x is not None:
            if type(x) is torch.Tensor:
                x = x.detach().numpy()
        return x

    def cond_to_torch(self, x):
        if x is not None:
            if type(x) is not torch.Tensor:
                x = torch.from_numpy(np.array(x))
        return x

    def to_numpy(self):
        self.scale = self.cond_to_numpy(self.scale)
        self.C_fixed = self.cond_to_numpy(self.C_fixed)

    def to_torch(self):
        self.scale = self.cond_to_torch(self.scale)
        self.C_fixed = self.cond_to_torch(self.C_fixed)

    def to_cuda(self):
        self.scale = self.scale.cuda()
        self.C_fixed = self.C_fixed.cuda()

    def to_cpu(self):
        self.scale = self.scale.cpu()
        self.C_fixed = self.C_fixed.cpu()
