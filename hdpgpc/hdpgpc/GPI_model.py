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
from bisect import bisect_right
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
                 estimation_limit=None, free_deg_MNIV=5):
        self.gp = GPI.IterativeGaussianProcess(kernel, x_basis, cuda=cuda)
        self.x_basis = self.cond_to_torch(x_basis)
        self.x_train = []
        self.y_train = []
        self.f_star = []
        self.f_star_sm = []
        self.cov_f = []
        self.cov_f_sm = []
        self.y_var = []
        self.var = []
        self.A = []
        self.Gamma = []
        self.C = []
        self.Sigma = []
        self.likelihood = []
        self.N = 0
        self.indexes = []
        self.K = self.cond_to_torch(kernel(self.cond_to_cpu(x_basis), self.cond_to_cpu(x_basis)))
        self.annealing = annealing
        self.bayesian = bayesian
        self.cuda = cuda
        self.inducing_points = inducing_points
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
        else:
            self.cov_f.append(self.cond_to_torch(ini_cov))
            self.cov_f_sm.append(self.cond_to_torch(ini_cov))
        if ini_A is None and ini_Gamma is None and ini_C is None and ini_Sigma is None:
            ini_A, ini_Gamma, ini_C, ini_Sigma = self.GPR_dynamic()
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
            if torch.all(ini_Gamma == torch.zeros(ini_Gamma.shape)):
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
                                       reduced_points=self.inducing_points, verbose=True)
        noise = self.gp.kernel.get_params()["k2__noise_level"]
        noise = alph_
        self.x_basis = self.cond_to_cuda(self.cond_to_torch(self.gp.x_basis))
        self.Sigma[-1] = self.cond_to_cuda(self.cond_to_torch(torch.mul(noise, torch.eye(len(self.x_basis), device=noise.device))))
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
            self.observation_params.set_scale(self.cond_to_cuda(self.cond_to_torch(torch.mul(noise, torch.eye(len(self.x_basis), device=noise.device)))))
            self.observation_params.m_mean = self.C[-1]
            self.internal_params.set_scale(self.Gamma[-1])
            self.internal_params.m_mean = self.A[-1]
        self.fitted = True
        return self.x_basis, ini_cov

    def log_lik_sample(self, y):
        """Returns log-likelihood of a single sample using the last state of the model.
        """
        lik_post = self.gp.log_likelihood(self.N, self.N, self.f_star_sm, self.cov_f_sm, self.A[-1], self.Gamma[-1], y,
                                          self.C[-1], self.Sigma[-1])
        return lik_post

    def log_sq_error(self, x_train, y, mean=None, cov=None, C=None, Sigma=None, i=None, proj=False, first=False):
        """Compute the log-squared-error of a sample assuming a specific iteration of the model.
        """
        y = self.cond_to_torch(y)
        mean = self.cond_to_torch(mean)
        cov = self.cond_to_torch(cov)
        C = self.cond_to_torch(C)
        Sigma = self.cond_to_torch(Sigma)
        t = len(x_train)
        if x_train is None:
            x_train = self.x_basis
        if mean is None:
            params = None
        else:
            params = [mean, cov, C, Sigma]

        if i is not None:
            f_star, cov_f = self.observe(x_train, i, params, proj=proj)
            lat_f, lat_cov = self.resample_latent_mean(x_train, i, params)
            if C is None:
                if x_train.shape[0] == self.x_basis.shape[0]:
                    _, _, C, _ = self.get_params(i)
                else:
                    C = torch.eye(x_train.shape[0], device=self.device)
        else:
            f_star, cov_f = self.step_forward_last(x_train, params)
            lat_f, lat_cov = self.resample_latent_mean(x_train)
            if C is None:
                if x_train.shape[0] == self.x_basis.shape[0]:
                    C = self.C[-1]
                else:
                    C = torch.eye(x_train.shape[0], device=self.device)

        #If first iteration we add the kernel noise.
        if first:
            #ini_noise = self.cond_to_cuda(self.cond_to_torch(self.gp.kernel.get_params()["k2__noise_level"])) * 1e-4
            ini_noise = torch.mean(torch.diag(self.Sigma[0])) * 1e-1
            cov_f = cov_f + torch.mul(ini_noise, torch.eye(len(x_train), device=ini_noise.device))
        # If we have a projection we have to add an extra noise because of the smooth conditions of the
        # interpolation. We assume this condition if we have as basis points less than a third of the training.
        #if len(self.x_basis) / len(x_train) <= 0.5:
            #cov_f = 0.5 * torch.diag(torch.diag(cov_f))
            #cov_f = 0.01 * cov_f + 0.99 * torch.diag(torch.diag(cov_f))
        exp_t_t_ = lat_cov + torch.matmul(lat_f, lat_f.T)
        #exp_t_t = lat_cov + torch.matmul(f_star, f_star.T)
        Sigma_inv = torch.linalg.solve(cov_f, self.cond_to_cuda(torch.eye(t)))
        err = -1 / 2 * torch.linalg.multi_dot([y.T, Sigma_inv, y])\
              + torch.linalg.multi_dot([y.T, Sigma_inv, f_star]) \
              - 1 / 2 * torch.trace(torch.linalg.multi_dot([C.T, Sigma_inv, C, exp_t_t_])) \
              #- 1 / 2 * torch.trace(torch.linalg.multi_dot([Sigma_inv, exp_t_t]))
              # - 1 / 2 * torch.trace(cov_f)
        #Scale with dimension:
        #err = err / y.shape[0]
        return err

    def log_lat_error(self, i, h_ini):
        """Compute the log-squared-error of the latent process.
        """
        err = 0.0
        if i == 0:
            cov_f_ = self.cov_f[i + 1]
            lat_f_ = self.f_star[i + 1]
            Gamma_inv = torch.linalg.solve(self.Gamma[-1] * h_ini, self.cond_to_cuda(torch.eye(self.Gamma[-1].shape[0])))
            A = self.A[-1]
        else:
            cov_f_ = self.cov_f[i]
            lat_f_ = self.f_star[i]
            if i+1 < len(self.Gamma):
                Gamma_inv = torch.linalg.solve(self.Gamma[i+1], self.cond_to_cuda(torch.eye(self.Gamma[i].shape[0])))
                A = self.A[i+1]
            else:
                Gamma_inv = self.gamma_inv
                A = self.A[-1]
        #cov_f = self.cov_f_sm[i + 1]
        lat_f = self.f_star[i + 1]
        #Gamma = self.Gamma[i]

        #t = Gamma.shape[0]
        exp_t_t_ = cov_f_ + torch.matmul(lat_f_, lat_f_.T)
        #xp_t_t = cov_f + torch.matmul(lat_f, lat_f.T)
        #A = self.A[i + 1]
        #Gamma_inv = torch.linalg.solve(Gamma, self.cond_to_cuda(torch.eye(Gamma.shape[0])))
        err = -1 / 2 * torch.linalg.multi_dot([lat_f.T, Gamma_inv, lat_f]) \
              + torch.linalg.multi_dot([lat_f.T, Gamma_inv, A, lat_f_]) \
              - 1 / 2 * torch.trace(torch.linalg.multi_dot([A.T, Gamma_inv, A, exp_t_t_])) \
              #- 1 / 2 * torch.trace(torch.linalg.multi_dot([Gamma_inv, exp_t_t]))
              # - 1 / 2 * torch.trace(Gamma)
        return err# / 2.0

    def include_sample(self, index, x_train, y, x_warped=None, h=1.0, posterior=True, embedding=True, include_index=False):
        """ Method to include sample in the model, compute the posterior and add the data.
        """
        if posterior:
            self.N = self.N + 1
            self.indexes.append(index)
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
        #Responsability truncated to 0.9 for stability purposes.
        if h > 0.9:
            if self.N == 0 and not self.fitted:
                if torch.allclose(torch.from_numpy(self.gp.kernel.theta), torch.from_numpy(self.ini_kernel_theta)):
                    new_x_basis, _ = self.fit_kernel_params(x_train, y, self.Sigma[-1], self.Gamma[-1], valid=True)
                else:
                    new_x_basis, _ = self.fit_kernel_params(x_train, y, self.Sigma[-1], self.Gamma[-1], valid=False)
            if snr is not None:
                if snr > 0.5:
                    self.include_sample(index, x_train, y, x_warped, h=1.0)
                else:
                    self.include_sample(index, x_train, y, x_warped, posterior=False, include_index=True)
            else:
                self.include_sample(index, x_train, y, x_warped, h=1.0)
        else:
            self.include_sample(index, x_train, y, x_warped, posterior=False)
        return new_x_basis

    def full_pass_weighted(self, x_trains, y_trains, resp, q=None, q_lat=None, snr=None):
        """ Full forward pass method. Compute the full forward message passing and reestimation of 
            dynamic parameters in a bayesian way.
            Used in the offline scheme.

            Returns:
            q : squared error of each of the observations

            q_lat: accumulated squared error of the latent process.
        """
        q_ = torch.zeros(len(x_trains), device=self.device)
        q_lat_ = torch.zeros(1, device=self.device)
        ind = torch.where(resp > self.cond_to_cuda(self.cond_to_torch(0.9)))[0]
        if len(ind) == 0:
            return q, q_lat
        n_samp = x_trains.shape[0]
        step_lik = 0
        if len(ind) > 0:
            for index in trange(n_samp, desc="Forward_pass"):
                self.include_weighted_sample(index, x_trains[index], x_trains[index],
                                             y_trains[index], resp[index])#, snr=snr_)
                self.backwards_pair(resp[index])  # , snr=snr_)
                self.bayesian_new_params(resp[index])
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
        self.likelihood = []
        self.N = 0


    def reinit_LDS(self, save_last=False, save_last_diag=False, return_likelihood=False):
        """ Method to reinitiate LDS parameters. Can save some of them.
        """
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
        self.observation_params = matrix_normal_inv_wishart(ini_C, torch.eye(ini_C.shape[0], device=self.device), self.free_deg_MNIV, ini_Sigma)
        if return_likelihood:
            return self.internal_params.log_likelihood_MNIW(A_, Gam_), self.observation_params.log_likelihood_MNIW(C_, Sig_)

    def return_LDS_param_likelihood(self, first=False):
        """ Method to compute the likelihood of LDS parameters over the prior.
        """
        ini_A, ini_Gamma, ini_C, ini_Sigma = self.A_def, self.Gamma_def, self.C_def, self.Sigma_def
        if first:
            #ini_noise = self.cond_to_cuda(self.cond_to_torch(self.gp.kernel.get_params()["k2__noise_level"]))
            ini_noise = torch.mean(torch.diag(self.Sigma[-1])) * 1e-1
            ini_noise_ = torch.mean(torch.diag(self.Gamma[-1])) * 1e-1
            A_, Gam_, C_, Sig_ = (self.A[-1],
                                  self.Gamma[-1] + torch.mul(ini_noise_, torch.eye(self.Sigma[-1].shape[0],
                                                                                  device=ini_noise.device)),
                                  self.C[-1],
                                  self.Sigma[-1] + torch.mul(ini_noise, torch.eye(self.Sigma[-1].shape[0],
                                                                                  device=ini_noise.device)))
            # A_, Gam_, C_, Sig_ = (self.A[-1], self.Gamma[-1] + self.cov_f[-1], self.C[-1],
            #                        self.Sigma[-1] + self.cov_f[-1])
        else:
            A_, Gam_, C_, Sig_ = self.A[-1], self.Gamma[-1], self.C[-1], self.Sigma[-1]
        int_params = matrix_normal_inv_wishart(ini_A, torch.eye(ini_A.shape[0], device=self.device), self.free_deg_MNIV, ini_Gamma)
        obs_params = matrix_normal_inv_wishart(ini_C, torch.eye(ini_C.shape[0], device=self.device), self.free_deg_MNIV, ini_Sigma)
        return int_params.log_likelihood_MNIW(A_, Gam_) + obs_params.log_likelihood_MNIW(C_, Sig_)

    def compute_sq_err_all(self, x_trains, y_trains, no_first=False):
        """ Method to compute the squared error over all provided examples y_trains.
        """
        n_samps = x_trains.shape[0]
        sq_err = torch.zeros(n_samps, device=x_trains[0].device)
        for index in trange(n_samps, desc="Compute_sq_error"):
            if len(self.indexes) > 0:
                if index in self.indexes:
                    ind = self.indexes.index(index) + 1
                    if ind == 1 and not no_first:
                        sq_err[index] = self.log_sq_error(x_trains[index], y_trains[index], i=ind, first=True)
                    else:
                        sq_err[index] = self.log_sq_error(x_trains[index], y_trains[index], i=ind)
                else:
                    ind = np.max([self.find_closest_lower(index), 1])
                    sq_err[index] = self.log_sq_error(x_trains[index], y_trains[index], i=ind)
        return sq_err

    def compute_q_lat_all(self, x_trains, h_ini=1.0):
        """Method to compute the latent squared error accumulated.
        """
        sq_err = torch.zeros(x_trains.shape[0], device=x_trains[0].device)
        if self.N == 0:
            return sq_err
        self.gamma_inv = torch.linalg.solve(self.Gamma[-1], self.cond_to_cuda(torch.eye(self.Gamma[-1].shape[0])))
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
                mean = self.f_star[-1]
                cov = self.cov_f[-1]
            else:
                mean = self.f_star[t]
                cov = self.cov_f[t]
        else:
            mean = params[0]
            cov = params[1]
        x_basis = self.x_basis
        return self.gp.pred_latent_dist(x_post, x_basis, mean, cov)

    def backwards(self, h=1.0):
        """Method to compute backward recursion weighted by the responsibility.
        """
        if h > 0.99:
            mean = self.f_star_sm[1:]
            covs = self.cov_f_sm[1:]
            aux_f_star, aux_cov_f = self.gp.backward(self.A, self.Gamma, mean, covs)
            for i in range(len(mean)):
                self.f_star_sm[i + 1] = aux_f_star[i]
                self.cov_f_sm[i + 1] = aux_cov_f[i]

    def backwards_pair(self, h, snr=None):
        """ Fast method to compute the last two backward iterations.
        """
        if len(self.indexes) > 1:
            if h > 0.9:
                if snr is None:
                    mean = self.f_star_sm[-2:]
                    covs = self.cov_f_sm[-2:]
                    aux_f_star, aux_cov_f = self.gp.backward_notrange(self.A[-1], self.Gamma[-1], mean, covs)
                    for i in range(len(mean)):
                        self.f_star_sm[-(i + 1)] = aux_f_star[-(i + 1)]
                        self.cov_f_sm[-(i + 1)] = aux_cov_f[-(i + 1)]
                else:
                    if snr > 0.5:
                        mean = self.f_star_sm[-2:]
                        covs = self.cov_f_sm[-2:]
                        aux_f_star, aux_cov_f = self.gp.backward_notrange(self.A[-1], self.Gamma[-1], mean, covs)
                        for i in range(len(mean)):
                            self.f_star_sm[-(i + 1)] = aux_f_star[-(i + 1)]
                            self.cov_f_sm[-(i + 1)] = aux_cov_f[-(i + 1)]

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

    def bayesian_new_params(self, h, model_type='dynamic', full_data=False, q=None, force=False, snr=1.0):
        """ Method to compute the variational Bayesian step for LDS parameters. Can deal with dynamic or static models.
            Can use full data batch n-step estimation or 1-step estimation.
        """
        if h > 0.9:
            if snr > 0.5:
                if (full_data and 1 < self.N) or 1 < self.N < self.estimation_limit or force:
                    try:
                        # if not full_data:
                        #     self.backwards()
                        if model_type == 'dynamic':
                            if not full_data:
                                A, Gamma = self.A[-1], self.Gamma[-1]
                                # samples_A = self.f_star_sm[-1]
                                # samples_A_ = self.f_star_sm[-2]
                                samples_A = self.f_star[-1]
                                samples_A_ = self.f_star[-2]
                                #samples_A_ = torch.matmul(A, self.f_star_sm[-2])
                                cov = self.cov_f_sm[-1]
                                cov_ = self.cov_f_sm[-2]
                                cov_f_ = self.cov_f[-1]
                                cov_f__ = self.cov_f[-2]
                                P = torch.matmul(A, torch.matmul(cov_, A.T)) + Gamma
                                cov_cross = torch.matmul(torch.linalg.solve(P.T, torch.matmul(A, cov_.T)).T, cov)
                                cov_cross = (cov_cross + cov_cross.T) / 2.0
                                if True:
                                    cov = torch.zeros(cov.shape, device=cov.device)
                                    cov_ = torch.zeros(cov.shape, device=cov.device)
                                    cov_cross = torch.zeros(cov.shape, device=cov.device)
                            else:
                                n_f = min(self.estimation_limit,len(self.f_star_sm)-2) if (
                                        self.estimation_limit != np.PINF) else len(self.f_star_sm)-2
                                samples_A = torch.stack(self.f_star_sm[2:n_f+2])[:, :, 0].T
                                samples_A_ = torch.stack(self.f_star_sm[1:n_f+1])[:, :, 0].T
                                cov = torch.sum(torch.stack(self.cov_f_sm[2:n_f+2]), axis=0)
                                cov_ = torch.sum(torch.stack(self.cov_f_sm[1:n_f+1]), axis=0)
                                cov_cross = torch.zeros(cov.shape, device=cov.device)
                                A, Gamma = self.A[-1], self.Gamma[-1]
                                for t in range(n_f+1):
                                    P = torch.matmul(A, torch.matmul(self.cov_f_sm[t], A.T)) + Gamma
                                    cov_cross = cov_cross + torch.matmul(
                                        torch.linalg.solve(P.T, torch.matmul(A, self.cov_f_sm[t].T)).T,
                                        self.cov_f_sm[t + 1])
                                cov_cross = (cov_cross + cov_cross.T)/2.0
                            if not full_data:
                                N_k = 1
                            elif self.estimation_limit != np.PINF:
                                N_k = self.estimation_limit
                            else:
                                N_k = samples_A.shape[1]
                            new_int_dist = self.internal_params.posterior(N_k, samples_A, samples_A_, cov, cov_, cov_cross,
                                                                          annealing=self.annealing)
                        elif model_type == 'static':
                            N_k = 1
                            new_int_dist = self.internal_params
                        else:
                            print("Only programmed static and dynamic models.")
                            new_int_dist = self.internal_params
                        if not full_data:
                            samples_C = self.y_train[-1]
                            #samples_C_, _ = self.observe_last(self.x_train[-1])
                            samples_C_ = self.f_star_sm[-1]
                            if model_type == 'static':
                                samples_C_, _ = self.resample_latent_mean(self.x_train[-1])
                            samples_C_ = self.cond_to_cuda(samples_C_)
                            C, Sigma = self.C[-1], self.Sigma[-1]
                            cov = torch.zeros(Sigma.shape, device=Sigma.device)
                            cov_ = torch.zeros(cov.shape, device=cov.device)
                            cov_cross = torch.zeros(cov.shape, device=cov.device)
                        else:
                            samples_C  = torch.stack(self.y_train[:n_f])[:, :, 0].T
                            samples_C_ = torch.stack(self.f_star_sm[1:n_f+1])[:, :, 0].T
                            cov_ = torch.sum(torch.stack(self.cov_f_sm[1:n_f+1]), axis=0)
                            cov = torch.zeros(cov_.shape, device=cov_.device)
                            cov_cross = torch.zeros(cov_.shape, device=cov_.device)
                            C, Sigma = self.C[-1], self.Sigma[-1]
                            for t in range(n_f+1):
                                P = torch.matmul(C, torch.matmul(self.cov_f_sm[t], C.T)) + Sigma
                                cov_cross = cov_cross + torch.matmul(
                                    torch.linalg.solve(P.T, torch.matmul(C, self.cov_f_sm[t].T)).T,
                                    Sigma)
                            cov = cov + Sigma
                            if False:
                                cov = torch.zeros(cov.shape, device=cov.device)
                                cov_ = torch.zeros(cov.shape, device=cov.device)
                                cov_cross = torch.zeros(cov.shape, device=cov.device)
                        if torch.equal(self.x_basis, self.x_train[-1]):
                            project_mat = None
                        else:
                            project_mat = self.reduce_noise_matrix(self.x_basis, self.x_train[-1])
                        new_obs_dist = self.observation_params.posterior(N_k, samples_C, samples_C_, cov, cov_, cov_cross,
                                                                         sse_matrix=project_mat, annealing=self.annealing)
                    except torch.linalg.LinAlgError:
                        print("Alg error matrix ill conditioned.")
                        new_int_dist = self.internal_params
                        new_obs_dist = self.observation_params
                else:
                    new_int_dist = self.internal_params
                    new_obs_dist = self.observation_params
                self.internal_params = new_int_dist
                self.observation_params = new_obs_dist
                if 1 < self.N:
                    Gamma_ = new_int_dist.get_scale(final=full_data)
                    Sigma_ = new_obs_dist.get_scale(final=full_data)
                else:
                    Gamma_ = self.Gamma[-1]
                    Sigma_ = self.Sigma[-1]
                if self.annealing:
                    if model_type == 'static':
                        factor_S = self.Sigma[0] / (self.N ** 2)
                        factor_G = self.Gamma[0]
                    else:
                        factor_G = self.Gamma[0] / (self.N ** 2)
                        factor_S = self.Sigma[0] / (self.N ** 2)
                    Gamma_ = Gamma_ + factor_G
                    Sigma_ = Sigma_ + factor_S
                if self.N < self.estimation_limit or full_data:
                    self.A.append(new_int_dist.get_mean())
                    self.Gamma.append(Gamma_)
                    if model_type == 'static':
                        self.C.append(new_obs_dist.get_C())
                    else:
                        self.C.append(new_obs_dist.get_mean())
                    self.Sigma.append(Sigma_)
                    self.var.append(torch.atleast_2d(torch.sqrt(torch.diag(Gamma_))).T)
                    self.y_var.append(torch.atleast_2d(torch.sqrt(torch.diag(Sigma_))).T)
            else:
                new_int_dist = self.internal_params
                new_obs_dist = self.observation_params
                Gamma_ = new_int_dist.get_scale(final=full_data)
                Sigma_ = new_obs_dist.get_scale(final=full_data)
                self.A.append(new_int_dist.get_mean())
                self.Gamma.append(Gamma_)
                if model_type == 'static':
                    self.C.append(new_obs_dist.get_C())
                else:
                    self.C.append(new_obs_dist.get_mean())
                self.Sigma.append(Sigma_)
                self.var.append(torch.atleast_2d(torch.sqrt(torch.diag(Gamma_))).T)
                self.y_var.append(torch.atleast_2d(torch.sqrt(torch.diag(Sigma_))).T)


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
    """ Class to define the Matrix Normal Inverse Wishart distribution (MNIW) for the LDS parameters.
    """
    def __init__(self, m_mean, m_r_cov, n0, scale):
        if type(m_mean) is torch.Tensor:
            self.m_mean = m_mean
        else:
            self.m_mean = torch.from_numpy(m_mean)
        if type(m_r_cov) is torch.Tensor:
            self.m_r_cov = m_r_cov
        else:
            self.m_r_cov = torch.from_numpy(m_r_cov)
        self.n0 = n0
        self.scale = scale
        if type(scale) is torch.Tensor:
            self.set_scale(scale)
        else:
            self.set_scale(scale)
        self.b = 0.7


    def posterior(self, n_k, y1, y2, cov, cov_, cov_cross, sse_matrix=None, annealing=False):
        y_k = y1
        y_k_ = y2
        new_n0 = self.n0 + n_k
        id = torch.eye(self.scale.shape[0])
        id_x = torch.eye(y1.shape[0])
        if torch.cuda.is_available() and self.scale.is_cuda:
            id = id.cuda()
            id_x = id_x.cuda()
        if sse_matrix is None:
            sse_matrix = id
        scale = self.m_r_cov
        jitter = 1e-2 * id
        scale = (scale + scale.T)/2 + jitter
        scale_c = torch.linalg.cholesky(scale)
        #
        if n_k == 1:
            scale_inv = torch.cholesky_solve(id, scale_c)
        else:
            scale_inv = torch.cholesky_solve(id, scale_c)
        exp_f_f_ = torch.linalg.multi_dot([sse_matrix, torch.linalg.multi_dot([y_k_, y_k_.T]) + cov_, sse_matrix.T])
        exp_ff_ = torch.linalg.multi_dot([sse_matrix, torch.linalg.multi_dot([y_k, y_k_.T]) + cov_cross, sse_matrix.T])
        exp_ff = torch.linalg.multi_dot([sse_matrix,torch.linalg.multi_dot([y_k, y_k.T]) + cov, sse_matrix.T])

        S__ = exp_f_f_ + scale_inv
        S_ = exp_ff_ + torch.linalg.multi_dot([self.m_mean, scale_inv])
        S = exp_ff + torch.matmul(self.m_mean, torch.matmul(scale_inv, self.m_mean.T))
        S__c = torch.linalg.cholesky(S__)
        S__inv = torch.cholesky_solve(id, S__c)
        part_mean = torch.linalg.multi_dot([S_, S__inv])
        if n_k == 1:
            #Step by step computation
            new_m_mean = ((self.n0 - 2) * self.m_mean + part_mean) / (new_n0 - 2)
        else:
            new_m_mean = part_mean
        if n_k == 1:
            e = (y_k - y_k_)
            e2 = torch.matmul(e, e.T)
            e2 = torch.linalg.multi_dot([sse_matrix, e2, sse_matrix.T])
            #Step by step computation
            new_scale = ((self.n0 - 2) * self.scale + e2) / (new_n0 - 2)
        else:
            e = (y_k - torch.matmul(new_m_mean, y_k_))
            e2 = torch.matmul(e, e.T)
            new_scale = ((self.n0 - 2) * self.scale + e2) / (new_n0 - 2)

        new_m_r_cov = S__
        return matrix_normal_inv_wishart(new_m_mean, new_m_r_cov, new_n0, new_scale)

    def log_likelihood_MNIW(self, M, Sigma):
        """ Method to compute the likelihood of the MNIW parameters, some parts removed
            because of the inherent dependence on the prior.
        """
        d = M.shape[0]
        id = torch.eye(self.scale.shape[0])
        if torch.cuda.is_available() and self.scale.is_cuda:
            id = id.cuda()
        sig_c = torch.linalg.cholesky(Sigma)
        sig_inv = torch.cholesky_solve(id, sig_c)
        mean_lik = - 0.5 * torch.trace(torch.linalg.multi_dot([(M-self.m_mean).T,
                                                               sig_inv,
                                                               (M-self.m_mean),
                                                               self.m_r_cov]))
                   # - d * 0.5 * torch.logdet(self.m_r_cov)\
                   # - self.n0 * 0.5 * torch.logdet(self.scale)\
                   # - self.n0 * d * 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=self.scale.device))
        scale_lik = - (self.n0 + 1) * 0.5 * torch.logdet(Sigma) \
                    - 0.5 * torch.trace(torch.matmul(sig_inv, self.scale))
                    #- 0.5 * torch.trace(Sigma)
                    # - self.n0 * 0.5 * torch.logdet(self.scale)\
                    # - self.n0 * d * 0.5 * torch.log(torch.tensor(2.0, device=self.scale.device))\
                    # - torch.special.multigammaln(torch.tensor((self.n0 + d)*0.5, device=self.scale.device), d)
        #Scale with dimension:
        #scale_lik = scale_lik / self.scale.shape[0]
        #return scale_lik# / d
        #return (mean_lik + scale_lik) / d
        return mean_lik + scale_lik

    def get_mean(self):
        return self.m_mean

    def get_scale(self, final=False):
        if final:
            return self.scale
        else:
            return self.scale * self.n0 / (self.n0 - 2)


    def set_scale(self, scale):
        if type(self.scale) is torch.Tensor:
            if self.scale.is_cuda:
                self.scale = scale.cuda()
            else:
                self.scale = scale
        else:
            self.scale = torch.from_numpy(scale)

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
    """ Class to define the Inverse Wishart distribution (IW) for the LDS static parameters.
    """
    def __init__(self, n0, scale, C_fixed):
        if type(C_fixed) is torch.Tensor:
            self.C_fixed = C_fixed
        else:
            self.m_mean = torch.from_numpy(C_fixed)
        if type(scale) is torch.Tensor:
            self.scale = scale
        else:
            self.scale = torch.from_numpy(scale)
        self.n0 = n0
        self.b = 0.7

    def posterior(self, n_k, y1, y2, cov, cov_, cov_cross, sse_matrix=None, annealing=False):
        y_k = y1
        y_k_ = y2
        new_n0 = self.n0 + n_k
        id = torch.eye(self.scale.shape[0])
        if torch.cuda.is_available() and self.scale.is_cuda:
            id = id.cuda()
        if sse_matrix is None:
            sse_matrix = id
        # e = (y_k - torch.matmul(self.C_fixed, y_k_))
        e = (y_k - y_k_)
        e2 =  torch.matmul(e, e.T)
        e2 = torch.linalg.multi_dot([sse_matrix, e2, sse_matrix.T])
        new_scale = ((self.n0 - 2) * self.scale + e2) / (new_n0 - 2)
        return inv_wishart(new_n0, new_scale, self.C_fixed)

    def get_scale(self, final=False):
        # return self.scale / (self.n0 - self.scale.shape[0] - 1)
        return self.scale * self.n0 / (self.n0 - 2)

    def set_scale(self, scale):
        if type(scale) is torch.Tensor and type(self.scale) is torch.Tensor:
            if self.scale.is_cuda:
                self.scale = scale.cuda()
            else:
                self.scale = scale
        else:
            self.scale = torch.from_numpy(scale)

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
