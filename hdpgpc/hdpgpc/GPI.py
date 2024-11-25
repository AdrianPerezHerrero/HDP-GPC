# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:56:03 2021

@author: adrianperez
"""

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.linalg import cholesky, cho_solve
import scipy.optimize
from sklearn.utils import check_random_state
import warnings
import torch
import gpytorch
from hdpgpc.GPI_models_pytorch import ExactGPModel, ProjectedGPModel, VarProjectedGPModel
from hdpgpc.util_plots import print_hyperparams
from tqdm.notebook import trange
torch.set_default_dtype(torch.float64)


# Definition class of an Iterative Gaussian Process. Takes a kernel as parameters from the
# sklearn gaussian_process class.
class IterativeGaussianProcess():
    """Gaussian Process from an iterative point of view, can compute posterior and
    update them recursively from new observations. It works by storing the posterior distribution 
    and mixing a standard GP of 0 mean and kernel covariance function with posterior
    mean and posterior covariance between inputs.

        Parameters
        ----------
        kernel : kernel from sklearn class for covariance computation.

        x_basis : array-like of shape (s_samples) domain points where is wanted
        to focus the learning.

        alpha : positive double or array-like of shape (n_samples) noise associated with subyacent process.

        Returns
        -------
        self : returns an instance of self.
        """

    def __init__(self, kernel, x_basis, cuda=False):
        self.kernel = kernel
        self.x_basis = torch.tensor(x_basis) if not type(x_basis) is torch.Tensor else x_basis
        self.alpha_ini = 0.0
        self.gamma_ini = 0.0
        self.zero = self.cond_to_torch([0.0])
        if type(x_basis) is torch.Tensor:
            if x_basis.is_cuda:
                self.K_X_X = torch.tensor(kernel(x_basis.cpu()), requires_grad=False)
            else:
                self.K_X_X = torch.tensor(kernel(x_basis), requires_grad=False)
        else:
            self.K_X_X = torch.tensor(kernel(x_basis), requires_grad=False)
        self.K_inv = 0.0  # np.linalg.inv(self.K_X_X + alpha_ini)
        self.optimizer = "fmin_l_bfgs_b"
        self.fitted = False
        self.log_marginal_fitted_value = 0.0
        self.cuda = cuda
        if self.cuda:
            self.zero = self.zero.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def posterior(self, mean_prior, cov_prior, y_train, A, Gamma, C, Sigma, warped_mean=None, x_train=None,
                  x_warped=None, embedding=True):
        """Compute posterior distribution using an observation as an input and returns
        the mean and the covariance centered into the set of basis points.
        Assuming we follow the system described by:
            f_t = A f_{t-1} + Gamma
            y_t = C f_t + Sigma

        Parameters
        ----------
        mean_prior : array-like of shape (s_samples) prior mean on basis vectors

        cov_prior : matrix-like of shape (sxs_samples) prior covariance matrix on basis vectors

        y_train : array-like of shape (n_samples) target values paired with x_basis.

        A: matrix-like of shape (sxs_samples) matrix associated with the linear transformation of the dynamical model.

        Gamma: matrix-like of shape (sxs_samples) matrix represents the noise of projection of the previous mean.

        C: matrix-like of shape (sxs_samples) matrix associated with the projection of the latent variable to the observations.

        Sigma: matrix-like of shape (sxs_samples) matrix associated with noise of the observations (alpha simpler case).

        Returns
        -------
        mean_post :  array-like of shape (s_samples) posterior mean on basis vectors

        cov_post : matrix-like of shape (sxs_samples) posterior covariance matrix on basis vectors
        """

        y = y_train
        x_basis_mean = torch.matmul(A, mean_prior)
        x_basis_cov = cov_prior
        x_basis_ = self.x_basis
        if x_train is None:
            x_train = self.cond_to_torch(np.atleast_2d(np.arange(0, y_train.shape[0], dtype=np.float64)).T)
        if x_warped is None:
            x_warped = x_train.detach()
        if self.cuda and torch.cuda.is_available():
            x_basis_cov = x_basis_cov.cuda()
            x_basis_mean = x_basis_mean.cuda()
            y = y.cuda()
            x_train = x_train.cpu()
            x_warped = x_warped.cpu()
            x_basis_ = x_basis_.cpu()

        l = x_basis_mean.shape[0]
        id = torch.eye(l)
        if torch.cuda.is_available() and self.cuda:
            id = id.cuda()
        jitter = 1e-4 * id
        K_X_X = self.cond_to_torch(self.kernel(x_basis_, x_basis_))
        # K_X_X = self.cond_to_torch(self.kernel(self.x_basis))
        K_Xs_X = self.cond_to_torch(self.kernel(x_warped, x_basis_))
        if self.cuda and torch.cuda.is_available():
            K_X_X = K_X_X.cuda()
            K_Xs_X = K_Xs_X.cuda()
        if torch.equal(x_warped, x_basis_):
            K_cov = id
        else:
            K_cov = torch.linalg.solve((K_X_X + jitter).T, K_Xs_X.T).T
        P_t = torch.linalg.multi_dot([A, x_basis_cov, A.T]) + Gamma
        # First step where initial covariance is needed as a prior, using initial Sigma computed by the GP optim.
        if torch.equal(cov_prior, K_X_X):
            P_t = x_basis_cov
            f_star = torch.zeros(x_warped.shape, device=self.device)
            cov_f = self.cond_to_torch(self.kernel(x_train)) - self.cond_to_torch(self.kernel(x_train, x_train))
        else:
            mean = torch.matmul(C, x_basis_mean)
            f_star, cov_f = self.pred_dist(x_warped, self.x_basis, mean, Sigma)
        # Forward equations of Kalman.
        K_t = torch.linalg.solve((torch.linalg.multi_dot([K_cov, C, P_t, C.T, K_cov.T]) + cov_f).T,
                                 torch.linalg.multi_dot([K_cov, C, P_t.T])).T
        mean_post = x_basis_mean + torch.matmul(K_t, (y - f_star))
        # Joshep form
        cov_post = torch.linalg.multi_dot([id - torch.linalg.multi_dot([K_t, K_cov, C]), P_t,
                                           (id - torch.linalg.multi_dot([K_t, K_cov, C])).T]) \
                                            + torch.linalg.multi_dot([K_t, cov_f, K_t.T])
        return mean_post, cov_post
    
    def projection_matrix(self, x_basis=None, x_train=None):
        """Compute projection matrix using the optimised GP:
            K_t = K_{t,t_n}K_{t_n,t_n}^{-1}

        Parameters
        ----------
        x_basis : array-like of shape (s_samples) basis vectors

        x_train : array-like of shape (s*_samples) train vector

        Returns
        -------
        K_nn_inv :  matrix-like of shape (s*xs_samples) projection matrix
        """
        if x_basis is None:
            x_basis_ = self.x_basis
        else:
            x_basis_ = x_basis
        if x_train is None:
            x_train_ = self.x_train
        else:
            x_train_ = x_train
        if self.cuda and torch.cuda.is_available():
            x_basis_ = x_basis_.cpu()
            x_train_ = x_train.cpu()
        if torch.equal(x_basis_, x_train_):
            K_nn_inv = torch.eye(x_basis.shape[0])
            if self.cuda and torch.cuda.is_available():
                K_nn_inv = K_nn_inv.cuda()
        else:
            x_basis_ = self.cond_to_numpy(x_basis_)
            x_train_ = self.cond_to_numpy(x_train_)
            jitter = 1e-4
            K_mn = self.cond_to_torch(self.kernel(x_basis_, x_train_))
            K_nn = self.cond_to_torch(self.kernel(x_train_, x_train_)) + jitter * torch.eye(x_train_.shape[0])
            if self.cuda and torch.cuda.is_available():
                K_mn = K_mn.cuda()
                K_nn = K_nn.cuda()
            K_nn_inv = torch.linalg.solve(K_nn.T, K_mn.T).T
        return K_nn_inv

    def project_y(self, x_train, y, cov, C, Sigma, x_basis = None):
        """
        Compute projection of observations into basis dimension.

        Parameters
        -----------
        x_train: domain of the observations
        y: observations
        x_basis : domain of output
        Returns:
        y_p: projected observations using GP
        y_p_cov: projected covariance of observations
        """
        if x_basis is None:
            x_basis_ = self.x_basis
        else:
            x_basis_ = x_basis
        if x_train is None:
            x_train_ = self.x_train
        else:
            x_train_ = x_train
        if self.cuda and torch.cuda.is_available():
            x_basis_ = x_basis_.cpu()
            x_train_ = x_train.cpu()
        if torch.equal(x_basis_, x_train_):
            y_p = y
            y_p_cov = Sigma
            if self.cuda and torch.cuda.is_available():
                y_p = y_p.cuda()
                y_p_cov = y_p_cov.cuda()
        else:
            x_basis_ = self.cond_to_numpy(x_basis_)
            x_train_ = self.cond_to_numpy(x_train_)
            jitter = 1e-4
            K_mn = self.cond_to_torch(self.kernel(x_basis_, x_train_))
            K_nn = self.cond_to_torch(self.kernel(x_train_, x_train_)) + jitter * torch.eye(x_train_.shape[0])
            K_mm = self.cond_to_torch(self.kernel(x_basis_, x_basis_)) + jitter * torch.eye(x_basis_.shape[0])
            if self.cuda and torch.cuda.is_available():
                K_mn = K_mn.cuda()
                K_nn = K_nn.cuda()
                K_mm = K_mm.cuda()
            K_nn_inv = torch.linalg.solve(K_nn.T, torch.matmul(C, K_mn).T).T
            y_p = torch.linalg.multi_dot([K_nn_inv, y])
            y_p_cov = Sigma
        return y_p, y_p_cov

    def backward(self, A_prior, Gamma_prior, means, covars):
        """
        Compute backward mean and cov incorporating full sequence
        information to previous latent states.

        Parameters
        ----------
        A_prior: matrix (sxs_samples)
            Matrix that determine linear transformation.
        Gamma_prior: matrix (sxs_samples)
            Matrix of error projection of linear transformation.
        means : ordered array of arrays (txs_samples)
            List of means computed till moment
        covars : ordered array of matrix (txsxs_samples)
            List of covariance computed till moment


        Returns
        -------
        Lists of new means (txs_samples) and covars (txsxs_samples).

        """
        T = len(means)
        for t in trange(T - 2, -1, -1, desc="Backward_pass"):
            P_t = torch.linalg.multi_dot([A_prior, covars[t], A_prior.T]) + Gamma_prior
            J_t = torch.matmul(torch.matmul(covars[t],A_prior.T),torch.linalg.inv(P_t))
            means[t] = means[t] + torch.matmul(J_t, (means[t + 1] - torch.matmul(A_prior, means[t])))
            covars[t] = covars[t] + torch.linalg.multi_dot([J_t, (covars[t + 1] - P_t), J_t.T])
        return means, covars

    def backward_notrange(self, A_prior, Gamma_prior, means, covars):
        """
        Compute backward mean and cov incorporating full sequence
        information to previous latent states.

        Parameters
        ----------
        A_prior: matrix (sxs_samples)
            Matrix that determine linear transformation.
        Gamma_prior: matrix (sxs_samples)
            Matrix of error projection of linear transformation.
        means : ordered array of arrays (txs_samples)
            List of means computed till moment
        covars : ordered array of matrix (txsxs_samples)
            List of covariance computed till moment


        Returns
        -------
        Lists of new means (txs_samples) and covars (txsxs_samples).

        """
        T = len(means)
        for t in range(T - 2, -1, -1):
            P_t = torch.linalg.multi_dot([A_prior, covars[t], A_prior.T]) + Gamma_prior
            J_t = torch.linalg.solve(P_t.T, torch.matmul(A_prior, covars[t].T)).T
            means[t] = means[t] + torch.matmul(J_t, (means[t + 1] - torch.matmul(A_prior, means[t])))
            covars[t] = covars[t] + torch.linalg.multi_dot([J_t, (covars[t + 1] - P_t), J_t.T])
        return means, covars

    def new_params_LDS(self, A_prior, Gamma_prior, C_prior, Sigma_prior, y_samples, means, covars, model='dynamic'):
        """Compute new matrices of linear tranformation using ML approach following
        the model described by:
            f_t = A f_{t-1} + Gamma
            y_t = C f_t + Sigma
        Parameters
        ----------
        A_priors : list of matrix (txsxs_samples)
            Previous matrices of linear transformation.
        Gamma_priors : list of matrix (txsxs_samples)
            Previous matrices of projection noise.
        C_priors : list of matrix (txsxs_samples)
            Previous matrices observations projection.
        Sigma_priors : list of matrix (txsxs_samples)
            Previous matrices observation projection noise.
        y_samples : ordered array of arrays (txs_samples)
            List of ovservations received till moment.
        means : ordered array of arrays (txs_samples)
            List of means computed till moment
        covars : ordered array of matrix (txsxs_samples)
            List of covariance computed till moment
        Returns
        -------
        Matrices (sxs_samples) new matrices of linear transformation model
        which maximize likelihood of the model (A,Gamma, C, Sigma).
        """

        # Start computing essential means
        exp_ft_ft1 = []
        exp_ft1_ft = []
        exp_ft_ft = []
        T = len(means)
        P = []
        J = []

        for t in range(T):
            P.append(torch.matmul(A_prior, torch.matmul(covars[t], A_prior.T)) + Gamma_prior)
            J.append(torch.linalg.solve(P[t].T, torch.matmul(A_prior, covars[t].T)).T)

        for t in range(T - 1):
            exp_ft_ft1.append(torch.matmul(covars[t + 1], J[t].T) + torch.matmul(means[t + 1], means[t].T))
            exp_ft1_ft.append(torch.matmul(J[t], covars[t + 1]) + torch.matmul(means[t], means[t + 1].T))

        for t in range(T):
            exp_ft_ft.append(covars[t] + torch.matmul(means[t], means[t].T))

        # Start computing of new matrices
        # A
        A_aux1 = torch.zeros(A_prior.shape)
        A_aux2 = torch.zeros(A_prior.shape)
        # Gamma
        Gamma_aux = torch.zeros(Gamma_prior.shape)
        # C
        C_aux1 = torch.zeros(C_prior.shape)
        C_aux2 = torch.zeros(C_prior.shape)
        # Sigma
        Sigma_aux = torch.zeros(Sigma_prior.shape)
        zer = torch.tensor([0.0])
        if torch.cuda.is_available() and self.cuda:
            A_aux1 = A_aux1.cuda()
            A_aux2 = A_aux2.cuda()
            Gamma_aux = Gamma_aux.cuda()
            C_aux1 = C_aux1.cuda()
            C_aux2 = C_aux2.cuda()
            Sigma_aux = Sigma_aux.cuda()
            zer = zer.cuda()

        if model == 'static':

            jitter = torch.mul(1e-8, torch.eye(A_prior.shape[0]))
            A_new = torch.eye(A_prior.shape[0])
            C_new = torch.eye(C_prior.shape[0])
            Gamma_new = torch.zeros(Gamma_prior.shape)
            if torch.cuda.is_available() and self.cuda:
                jitter = jitter.cuda()
                A_new = A_new.cuda()
                C_new = C_new.cuda()
            for t in range(T):
                # Sigma computations
                Sigma_aux = Sigma_aux + (torch.matmul(y_samples[t], y_samples[t].T)
                                         - torch.matmul(means[t], y_samples[t].T)
                                         - torch.matmul(y_samples[t], means[t].T) + exp_ft_ft[t])
            Sigma_new = Sigma_aux / T
            Sigma_new = (Sigma_new + Sigma_new.T) / 2

            if torch.isclose(torch.linalg.det(Sigma_new), zer):
                Sigma_new = Sigma_new + jitter

        elif model == 'dynamic':

            jitter = torch.mul(1e-8, torch.eye(A_prior.shape[0]))
            if torch.cuda.is_available() and self.cuda:
                jitter = jitter.cuda()
            for t in range(T - 1):
                # A computations
                A_aux1 = A_aux1 + exp_ft_ft1[t]
                A_aux2 = A_aux2 + exp_ft_ft[t]

            for t in range(T):
                # C computations
                C_aux1 = C_aux1 + torch.matmul(y_samples[t], means[t].T)
                C_aux2 = C_aux2 + exp_ft_ft[t]

            # Numerical stability
            if torch.isclose(torch.linalg.det(A_aux2), zer):
                A_aux2 = A_aux2 + jitter
            if torch.isclose(torch.linalg.det(A_aux1), zer):
                A_aux1 = A_aux1 + jitter

            if torch.isclose(torch.linalg.det(C_aux2), zer):
                C_aux2 = C_aux2 + jitter
            if torch.isclose(torch.linalg.det(C_aux1), zer):
                C_aux1 = C_aux1 + jitter

            A_new = torch.linalg.solve(A_aux2.T, A_aux1.T).T
            C_new = torch.linalg.solve(C_aux2.T, C_aux1.T).T

            for t in range(1, T):
                # Gamma computations
                Gamma_aux = Gamma_aux + (exp_ft_ft[t] - torch.matmul(A_new, exp_ft1_ft[t - 1])
                                         - torch.matmul(exp_ft_ft1[t - 1], A_new.T)
                                         + torch.linalg.multi_dot([A_new, exp_ft_ft[t - 1], A_new.T]))

            for t in range(T):
                # Sigma computations
                Sigma_aux = Sigma_aux + (torch.matmul(y_samples[t], y_samples[t].T)
                                         - torch.linalg.multi_dot([C_new, means[t], y_samples[t].T])
                                         - torch.linalg.multi_dot([y_samples[t], means[t].T, C_new.T])
                                         + torch.linalg.multi_dot([C_new, exp_ft_ft[t], C_new.T]))

            # Compute the final covariance matrix and add a small noise for stability
            if T == 1:
                Gamma_new = Gamma_aux  # + jitter_gamma
            else:
                Gamma_new = Gamma_aux / (T - 1)  # + jitter_gamma/T

            # Ensure symmetry of the covariance matrices
            Gamma_new = (Gamma_new + Gamma_new.T) / 2

            if torch.isclose(torch.linalg.det(Gamma_new), zer):
                Gamma_new = Gamma_new + jitter

            Sigma_new = Sigma_aux / T  # + jitter_alpha/T

            # Ensure symmetry of the covariance matrices
            Sigma_new = (Sigma_new + Sigma_new.T) / 2

            if torch.isclose(torch.linalg.det(Sigma_new), zer):
                Sigma_new = Sigma_new + jitter

        else:
            print("ERROR: Only dynamic and static models implemented.")

        return A_new, Gamma_new, C_new, Sigma_new

    def pred_dist(self, x_post, x_fixed, mean_prior, Sigma):
        """Compute posterior distribution using prior given and over a domain X of x_post.
            It also defines likelihood of y targets.
        Parameters
        ----------
        x_post : array-like of shape (m_samples) target inputs.

        Returns
        -------
        self : returns an instance of self.
        """
        x_post_mean = self.compute_mean(x_post)
        x_train_mean = self.compute_mean(x_fixed)
        x_f = x_fixed
        x_p = x_post
        if torch.cuda.is_available() and self.cuda:
            x_f = x_f.cpu()
            x_p = x_p.cpu()
        if torch.equal(x_f, x_p):
            f_star = mean_prior
            cov_f = Sigma
        else:
            ker = self.kernel.clone_with_theta(self.kernel.theta)
            x_f = self.cond_to_numpy(x_f)
            x_p = self.cond_to_numpy(x_p)
            K_X_X = self.cond_to_torch(ker(x_f, x_f))
            id = torch.eye(K_X_X.shape[0])
            id_Xs = torch.eye(x_p.shape[0])
            if torch.cuda.is_available() and self.cuda:
                id = id.cuda()
                id_Xs = id_Xs.cuda()
            K_X_Xs = self.cond_to_torch(ker(x_f, x_p))
            K_Xs_X = self.cond_to_torch(ker(x_p, x_f))
            K_Xs_Xs = self.cond_to_torch(self.kernel(x_p))
            if torch.cuda.is_available() and self.cuda:
                K_X_X = K_X_X.cuda()
                K_X_Xs = K_X_Xs.cuda()
                K_Xs_X = K_Xs_X.cuda()
                K_Xs_Xs = K_Xs_Xs.cuda()
            jitter = 1e-4 * torch.mean(torch.diag(Sigma)) * id
            cov = K_X_X + jitter
            cov_inv = torch.linalg.solve(cov.T,K_X_Xs).T
            f_star = x_post_mean + torch.linalg.multi_dot([cov_inv, (mean_prior - x_train_mean)])
            if torch.all(torch.isclose(torch.diag(Sigma), torch.mean(torch.diag(Sigma)))):
                cov_f = torch.mean(torch.diag(Sigma)) * id_Xs
            else:
                cov_f = K_Xs_Xs - torch.linalg.multi_dot([cov_inv, K_X_Xs]) + \
                        torch.linalg.multi_dot([cov_inv, Sigma, cov_inv.T])
                cov_f = cov_f + 1e-6 * id_Xs
                while torch.any(torch.diag(cov_f) < 0.0):
                    cov_f = cov_f + 1e-2 * torch.mean(torch.diag(Sigma)) * id_Xs
                    print("Error: negative diagonal")
        return f_star, cov_f

    def pred_latent_dist(self, x_post, x_fixed, mean_prior, cov_prior):
        """Compute posterior distribution using prior given and over a domain X of x_post.
            It also defines likelihood of y targets.
        Parameters
        ----------
        x_post : array-like of shape (m_samples) target inputs.

        Returns
        -------
        self : returns an instance of self.
        """
        x_post_mean = self.compute_mean(x_post)
        x_train_mean = self.compute_mean(x_fixed)
        x_f = x_fixed
        x_p = x_post
        if torch.equal(x_f, x_p):
            f_star = mean_prior
            cov_f = cov_prior
        else:
            if torch.cuda.is_available() and self.cuda:
                x_f = x_f.cpu()
                x_p = x_p.cpu()
            x_f = self.cond_to_numpy(x_f)
            x_p = self.cond_to_numpy(x_p)
            K_X_X = self.cond_to_torch(self.kernel(x_f, x_f))
            id = torch.eye(K_X_X.shape[0])
            if torch.cuda.is_available() and self.cuda:
                id = id.cuda()
            K_X_Xs = self.cond_to_torch(self.kernel(x_f, x_p))
            K_Xs_X = self.cond_to_torch(self.kernel(x_p, x_f))
            K_Xs_Xs = self.cond_to_torch(self.kernel(x_p, x_p))
            if torch.cuda.is_available() and self.cuda:
                K_X_X = K_X_X.cuda()
                K_X_Xs = K_X_Xs.cuda()
                K_Xs_X = K_Xs_X.cuda()
                K_Xs_Xs = K_Xs_Xs.cuda()
            jitter = 1e-4 * id
            cov = K_X_X + jitter
            cov_inv = torch.linalg.solve(cov, id)
            f_star = x_post_mean + torch.linalg.multi_dot([K_Xs_X, cov_inv, mean_prior - x_train_mean])
            cov_f = K_Xs_Xs - torch.linalg.multi_dot([K_Xs_X, cov_inv, K_X_Xs]) + \
                    torch.linalg.multi_dot([K_Xs_X, cov_inv, cov_prior, cov_inv.T, K_X_Xs])
        return f_star, cov_f


    def sample_y(self, f_mean, f_cov, C, Sigma, n_samples=1, random_state=0):
        """Draw samples from Gaussian process and evaluate at X.

        Parameters
        ----------
        f_mean : mean of the distribution (s-array) paired with x_basis

        f_cov : covariance of the distribution (sxs-array)

        C: matrix (sxs_array) which represent projection over mean to observations.

        Sigma: matrix (sxs_array) which represent noise associated with observations.

        n_samples : int, default=1
            The number of samples drawn from the Gaussian process

        random_state : int, RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.
            See :term: `Glossary <random_state>`.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, [n_output_dims], n_samples)
            Values of n_samples samples drawn from Gaussian process and
            evaluated at query points.
        """
        f_mean = self.cond_to_numpy(f_mean)
        f_cov = self.cond_to_numpy(f_cov)
        C = self.cond_to_numpy(C)
        Sigma = self.cond_to_numpy(Sigma)

        rng = check_random_state(random_state)
        mean = np.dot(C, f_mean)
        cov = np.dot(C, np.dot(f_cov, C.T)) + Sigma
        if f_mean.ndim == 1:
            y_samples = rng.multivariate_normal(mean, cov, n_samples).T
        else:
            y_samples = \
                [rng.multivariate_normal(mean[:, i], cov,
                                         n_samples).T[:, np.newaxis]
                 for i in range(f_mean.shape[1])]
            y_samples = np.hstack(y_samples)
        return y_samples

    def fit_torch(self, x, y, alpha_ini, gamma_ini, reduced_points=False, verbose=False):
        """Optimize the parameters for a RBF kernel of the GP using one sample.
         Parameters
        ----------
        x : array (s_array) x train of the example y
        
        y : array (s_array) example y

        Returns
        -------
        fitted : True of False depending on the optimization succesful or not
        """
        if not self.fitted:
            if verbose:
                print("\n Fitting_GP: \n")
            x_ = x.detach().T[0]
            y_ = y.detach().T[0]
            x_basis = self.x_basis.T[0].detach().clone()
            if self.cuda and torch.cuda.is_available():
                x_basis = x_basis.cuda()
            alpha_ini_bounds = self.kernel.k2.noise_level_bounds

            if hasattr(self.kernel.k1, 'k2'):
                lengthscale_bounds = self.kernel.k1.k2.length_scale_bounds
            else:
                lengthscale_bounds = self.kernel.k1.length_scale_bounds
            if hasattr(self.kernel.k1, 'k1'):
                gamma_ini_bounds = self.kernel.k1.k1.constant_value_bounds
            else:
                gamma_ini_bounds = (1e-2, 1e+3)
            outputscale_constraint = gpytorch.constraints.GreaterThan(gamma_ini_bounds[0])
            if reduced_points or not torch.equal(x_basis, x_):
                if not reduced_points:
                    lik = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(alpha_ini_bounds[0]))
                else:
                    lik = gpytorch.likelihoods.GaussianLikelihood(
                        noise_constraint=gpytorch.constraints.Interval(alpha_ini_bounds[0], alpha_ini_bounds[1]))
                gp = ProjectedGPModel(x_, y_, lik, x_basis)
                if not reduced_points:
                    gp.covar_module.base_kernel.base_kernel.raw_lengthscale_constraint = \
                        gpytorch.constraints.Interval(lengthscale_bounds[0], lengthscale_bounds[1])
            else:
                lik = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.Interval(alpha_ini_bounds[0], alpha_ini_bounds[1]))
                gp = ExactGPModel(x_, y_, lik)

            lik.train()
            gp.train()

            training_iter = 4000
            
            if not reduced_points and not torch.equal(x_basis, x_):
                optimizer = torch.optim.Adam([{'params': gp.covar_module.base_kernel.parameters(), 'lr': 0.05},
                                                {'params': gp.likelihood.parameters()}], lr=0.05)
                training_iter = 2000
            elif hasattr(gp.covar_module, 'inducing_points'):
                optimizer = torch.optim.Adam([{'params': gp.covar_module.inducing_points, 'lr': 0.1},
                                              {'params': gp.covar_module.base_kernel.parameters(), 'lr': 0.1},
                                              {'params': gp.likelihood.parameters(), 'lr': 0.1}])
                training_iter = 5000
            else:
                optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, gp)

            losses = []

            # CUDA OPTION
            if torch.cuda.is_available() and self.cuda:
                gp = gp.cuda()
                lik = lik.cuda()

            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = gp(x_)
                # Calc loss and backprop gradients
                loss = -mll(output, y_)
                loss.backward()
                losses.append(loss.item())
                if verbose and i % 500 == 0:
                    print('Iter %d/%d - Loss: %.3f' % (
                        i + 1, training_iter, loss.item()))
                optimizer.step()
                if len(losses) > 1000:
                    # if np.isclose(np.sum(np.subtract(losses[-10:], losses[-11:-1])), 0, atol=1e-4):
                    if np.isclose(np.sum(np.subtract(losses[-10:], losses[-11:-1])), 0, atol=1e-4):
                        break

            gp.eval()
            lik.eval()

            if verbose:
                print_hyperparams(gp, self.cuda)
                
            if type(gp) is ExactGPModel:
                if hasattr(self.kernel.k1, "k1"):
                    self.kernel.k1.k1.theta = np.log(np.array([gp.covar_module.outputscale.item()]))
                if hasattr(self.kernel.k1, "k2"):
                    self.kernel.k1.k2.theta = np.log(np.array([gp.covar_module.base_kernel.lengthscale.item()]))
                else:
                    self.kernel.k1.theta = np.log(np.array([gp.covar_module.base_kernel.lengthscale.item()]))
                self.kernel.k2.theta = np.log(np.array([lik.noise.item()]))
            elif type(gp) is ProjectedGPModel:
                self.x_basis, _ = torch.sort(torch.atleast_2d(x_basis).T, axis=0)
                #Check if some points collapsed
                if reduced_points:
                    x_basis_clean = torch.clone(self.x_basis.detach())
                    collapsed = False
                    for i in torch.arange(self.x_basis.shape[0]-1):
                        if self.x_basis[i+1] - self.x_basis[i] < torch.log(gp.covar_module.base_kernel.base_kernel.lengthscale):
                            print("Inducing point removed, ", str(self.x_basis[i]))
                            x_basis_clean = x_basis_clean[x_basis_clean != self.x_basis[i]]
                            collapsed = True
                    if collapsed:
                        self.x_basis = torch.atleast_2d(x_basis_clean).T
                if self.cuda:
                    self.x_basis = self.x_basis.cuda()
                if hasattr(self.kernel.k1, "k1"):
                    self.kernel.k1.k1.theta = np.log(np.array([gp.covar_module.base_kernel.outputscale.item()]))
                if hasattr(self.kernel.k1, "k2"):
                    self.kernel.k1.k2.theta = np.log(np.array([gp.covar_module.base_kernel.base_kernel.lengthscale.item()]))
                else:
                    self.kernel.k1.theta = np.log(np.array([gp.covar_module.base_kernel.base_kernel.lengthscale.item()]))
                self.kernel.k2.theta = np.log(np.array([lik.noise.item()]))
            elif type(gp) is VarProjectedGPModel:
                self.x_basis = torch.clone(torch.sort(gp.variational_strategy.inducing_points, axis=0)[0]).detach()
                if hasattr(self.kernel.k1, "k1"):
                    self.kernel.k1.k1.theta = np.log(np.array([gp.covar_module.outputscale.item()]))
                if hasattr(self.kernel.k1, "k2"):
                    self.kernel.k1.k2.theta = np.log(np.array([gp.covar_module.base_kernel.lengthscale.item()]))
                else:
                    self.kernel.k1.theta = np.log(np.array([gp.covar_module.base_kernel.lengthscale.item()]))
                self.kernel.k2.theta = np.log(np.array([lik.noise.item()]))
            if torch.cuda.is_available() and self.cuda:
                x_ = self.cond_to_numpy(self.x_basis.cpu())
                self.K_X_X = self.cond_to_torch(self.kernel(x_, x_)).cuda()
            else:
                x_ = self.cond_to_numpy(self.x_basis)
                self.K_X_X = self.cond_to_torch(self.kernel(x_, x_))
            self.K_inv = self.inv_r("kernelMat", self.K_X_X)
            self.fitted = True
            try:
                id = torch.eye(self.x_basis.shape[0])
                alph_ = self.cond_to_torch(self.kernel.k2.noise_level)
                gam_ = self.cond_to_torch(gamma_ini)
                if torch.cuda.is_available() and self.cuda:
                    id = id.cuda()
                    alph_ = alph_.cuda()
                    gam_ = gam_.cuda()
                alph_ = torch.mul(alph_, id)
                gam_ = torch.mul(gam_, id)
                self.assign_alpha_ini(alph_, gam_)
            except AttributeError:
                self.assign_alpha_ini(alpha_ini, gamma_ini)
        return self.fitted

    def fit(self, x, y, alpha_ini, gamma_ini, n_restarts_optimizer, random_state=None, valid=True):
        """Compute optimal hyperparameter for kernel maximizing marginal likelihood.

        Parameters
        ----------
        x_post : array-like of shape (m_samples) target inputs.

        y_mean : array-like of shape (s_samples) mean of basis vectors.

        n_restarts_optimizer : int number of points from where is started the search
        of the hyperparameters

        random_state : int for reproducibility of results

        Returns
        -------
        self : an instance of self with hyperparameter stablished on kernel

        parameters : return a double or array of doubles with optimal hyperparameters of kernel.

        log_marginal_likelihood_value : double indicating value of log marginal likelihood for
        computed hyperparameters.
        """
        x = self.cond_to_numpy(x)
        y = self.cond_to_numpy(y)
        alpha_ini = self.cond_to_numpy(alpha_ini)
        gamma_ini = self.cond_to_numpy(gamma_ini)

        if self.fitted == False:
            # Normalize y_target.
            y_train_mean = np.mean(y, axis=0)
            y_train_std = np.std(y, axis=0)

            # Remove mean and make unit variance
            if not y_train_std == 0:
                y = (y - y_train_mean) / y_train_std
                alpha_ini = alpha_ini / y_train_std ** 2
            else:
                y = y - y_train_mean
                alpha_ini = alpha_ini

            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(x, y, alpha_ini,
                                                             theta, eval_gradient=True, clone_kernel=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(x, y, alpha_ini, theta,
                                                         clone_kernel=True)

                # First optimize starting from theta specified in kernel

            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel.theta,
                                                      self.kernel.bounds))]

            rng = check_random_state(random_state)
            # Additional runs are performed from log-uniform chosen initial
            # theta
            if n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel.bounds
                for iteration in range(n_restarts_optimizer):
                    theta_initial = \
                        rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood and transforming it into theta which is in log form.
            lml_values = list(map(itemgetter(1), optima))
            self.kernel.theta = optima[np.argmin(lml_values)][0]
            # self.kernel._check_bounds_params()

            self.log_marginal_likelihood_value = -np.min(lml_values)

            # Recompute values of the GP model in terms of new kernel
            if type(self.x_basis) is torch.Tensor:
                if self.x_basis.is_cuda:
                    x_ = self.x_basis.cpu()
                    self.K_X_X = self.cond_to_torch(self.kernel(x_, x_))
                else:
                    self.K_X_X = self.cond_to_torch(self.kernel(self.x_basis, self.x_basis))
            else:
                self.K_X_X = self.cond_to_torch(self.kernel(self.x_basis, self.x_basis))
            # self.K_inv = np.linalg.inv(self.K_X_X)
            self.K_inv = self.cond_to_torch(self.inv_r("K_X_X", self.K_X_X))
            if torch.cuda.is_available() and self.cuda:
                self.K_X_X = self.K_X_X.cuda()
                self.K_X_X.requires_grad = False
                self.K_inv = self.K_inv.cuda()
                self.K_inv.requires_grad = False
            # Assign alpha_ini to the model
            try:
                self.assign_alpha_ini(self.kernel.k2.noise_level ** 2, gamma_ini)
            except AttributeError:
                self.assign_alpha_ini(alpha_ini, gamma_ini)
            if valid:
                self.fitted = True
        return self.kernel.theta, self.log_marginal_likelihood_value

    # Revisar computo verosimilitud para una muestra mas amplia.
    def log_likelihood(self, t0, t1, means, covars, A=None, Gamma=None, y=None, C=None, Sigma=None):
        """
        Return log transformed likelihood of the full joint distribution
        depending on what inputs were given.

        Parameters
        ----------
        t0: integer time of starting sequence whose we want to compute likelihood.
        t1: integer time of ending sequence whose we want to compute likelihood.
        means : list of arrays-like with shape (txs_samples)
            Means of the distributions generated by the iteration of GP
        covars : list of matrix-like with shape (txsxs_samples)
            Covariance matrices of the distributions generated by the iteration of GP
        A : matrix-like with shape (sxs_samples)
            Matrix that characterize the linear model
        Gamma: matrix-like with shape (sxs_samples)
            Matrix of noise associated with linear model
        C: matrix-like with shape (sxs_samples)
            Matrix of linear projection between latent state and observations
        Sigma: matrix-like with shape (sxs_samples)
            Matrix of noise associated with observations
        y : list of array-like with shape (txs_samples), optional
            If included the likelihood is computed also in terms of the likelihood of the observations.
            The default is None.

        Returns
        -------
        Negative scalar, log-likelihood of the model with given parameters.

        """
        #Ensure y is a tensor
        if type(y) is list:
            if type(y[0]) is torch.Tensor:
                y = y
        else:
            y = self.cond_to_torch(y)
        # If all None then simplest system
        if A is None:
            A = torch.eye(means[0].shape[0])
        if Gamma is None:
            Gamma = torch.zeros(covars[0].shape)
        if C is None:
            C = torch.eye(means[0].shape[0])
        if Sigma is None:
            Sigma = torch.zeros(covars[0].shape)
        n = means[0].shape[0]
        sum_0 = 0.0
        if t0 == 0:
            if isinstance(y, list):
                sum_0 = self.log_marginal_likelihood(self.x_basis, y[0], self.alpha_ini, self.kernel.theta,
                                                     clone_kernel=True)
            else:
                sum_0 = self.log_marginal_likelihood(self.x_basis, y, self.alpha_ini, self.kernel.theta,
                                                     clone_kernel=True)
        sum_0 = torch.tensor(sum_0)
        sum_1 = torch.tensor(0.0)
        T = t1 - t0
        zer = torch.zeros(covars[0].shape)
        if torch.cuda.is_available() and self.cuda:
            sum_0 = sum_0.cuda()
            sum_1 = sum_1.cuda()
            zer = zer.cuda()
            zer.requires_grad = False
        if not torch.allclose(Gamma, zer):
            if t1 > 1:
                C_t = Gamma
                det = self.log_det('Gamma', C_t)
                C_t_inv = self.inv_r('Gamma', C_t)
                for t in range(max(t0, 1), t1):
                    exp_t_t = covars[t] + torch.matmul(means[t], means[t].T)
                    sum_1 = sum_1 - torch.linalg.multi_dot([means[t + 1].T, C_t_inv, means[t + 1]]) \
                            + 2 * torch.linalg.multi_dot([means[t + 1].T, C_t_inv, C, means[t]])\
                            - torch.trace(torch.linalg.multi_dot([C.T, C_t_inv, C, exp_t_t])) - det
                sum_1 = sum_1 - T * n * np.log(2 * np.pi)
                sum_1 = 0.5 * sum_1
        lik = sum_0 + sum_1
        # lik = sum_1
        if y is not None:
            sum_2 = torch.tensor(0.0)
            var = Sigma
            det = self.log_det('Sigma', var)
            var_inv = self.inv_r('Sigma', var)
            for t in range(t0, t1 + 1):
                exp_t_t = covars[t] + torch.matmul(means[t], means[t].T)
                if isinstance(y, list):
                    sum_2 = sum_2 - torch.linalg.multi_dot([y[t].T, var_inv, y[t]]) \
                            + 2 * torch.linalg.multi_dot([y[t].T, var_inv, C, means[t]]) \
                            - torch.trace(torch.linalg.multi_dot([C.T, var_inv, C, exp_t_t])) - det
                else:
                    sum_2 = sum_2 - torch.linalg.multi_dot([y.T, var_inv, y]) \
                            + 2 * torch.linalg.multi_dot([y.T, var_inv, C, means[t]]) \
                            - torch.trace(torch.linalg.multi_dot([C.T, var_inv, C, exp_t_t])) - det
            sum_2 = sum_2 - (T + 1) * n * np.log(2 * torch.pi)
            sum_2 = 0.5 * sum_2
            lik = lik + sum_2
        return lik

    def log_marginal_likelihood(self, x_train, y_train, alpha_ini, theta=None, eval_gradient=False,
                                clone_kernel=True):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        x_train : array-like of shape (s_samples) inputs associated with training inputs y

        y_train : array-like of shape (s_samples) observations of the sample

        theta : array-like of shape (n_kernel_params,) default=None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default=False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        clone_kernel : bool, default=True
            If True, the kernel attribute is copied. If False, the kernel
            attribute is modified, but may result in a performance improvement.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : ndarray of shape (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if torch.cuda.is_available() and self.cuda:
            x_train = x_train.cpu()
            y_train = y_train.cpu()
        x_train = self.cond_to_numpy(x_train)
        y_train = self.cond_to_numpy(y_train)
        theta = self.cond_to_numpy(theta)
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")

        if clone_kernel:
            kernel = self.kernel.clone_with_theta(theta)
        else:
            kernel = self.kernel
            kernel.theta = theta

        if eval_gradient:
            K, K_gradient = kernel(x_train, eval_gradient=True)
        else:
            K = kernel(x_train)

        # K += alpha_ini
        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        alpha = cho_solve((L.dot(L.T), True), y_train)  # Line 3
        log_likelihood = -1 / 2 * y_train.T.dot(alpha) - np.log(np.diag(L)).sum() - K.shape[0] / 2 * np.log(2 * np.pi)

        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,jik->kl", tmp, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def KL_divergence(self, mean1, cov1, mean2, cov2):
        """
        Compute symmetric Kullback-Leibler divergence for two
        Gaussian distributions.

        Parameters
        ----------
        mean1 : array like of s_samples
            Mean of the first distribution
        cov1 : matrix like of sxs_samples
            Covariance matrix of first distribution
        mean2 : array like of s_samples
            Mean of the second distribution
        cov2 : matrix like of sxs_samples
            Covariance matrix of second distribution

        Returns
        -------
        Positive real value KL-divergence of distributions.

        """
        # Second term
        invcov1 = torch.linalg.inv(cov1)
        invcov2 = torch.linalg.inv(cov2)
        cov = torch.matmul(invcov2, cov1) + torch.matmul(invcov1, cov2)
        tr = (torch.trace(cov) - 2 * len(cov)) / 4

        # First term
        difmean = mean1 - mean2
        sumcov = invcov1 + invcov2
        prod = torch.matmul(sumcov, difmean)
        first = torch.sum(prod * difmean) / 4

        # Total
        dist = first + tr

        return dist.item()

    def compute_mean(self, x):
        """Compute prior mean of an input vector x

        Parameters
        ----------
        x : array-like of shape (m_samples) inputs.

        Returns
        -------
        mean : vector with prior mean over the inputs x
        """
        # Now assuming a 0 mean prior.
        if torch.cuda.is_available() and self.cuda:
            return torch.tensor(np.atleast_2d(np.repeat(0.0, x.shape[0])).T, requires_grad=False).cuda()
        else:
            return torch.tensor(np.atleast_2d(np.repeat(0.0, x.shape[0])).T, requires_grad=False)

    # Private function of optimization used for hyperparameter computation.
    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt = {'maxiter': 50000, 'disp': False}
            opt_res = scipy.optimize.minimize(
                obj_func, initial_theta, method="L-BFGS-B", jac=True,
                bounds=bounds, options=opt)
            # _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif self.optimizer == "CG":
            opt_res = scipy.optimize.minimize(
                obj_func, initial_theta, method="CG", jac=True)  # , options=opt)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = \
                self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min

    # Standard plot function
    def plotGP(self, numberFigure, x_post, y_pred, gamma, x_train=None, y_train=None, sigma=None, title=False,
               label_model=0, labels=False):
        x_post = self.cond_to_numpy(x_post)
        y_pred = self.cond_to_numpy(y_pred)
        gamma = self.cond_to_numpy(gamma)
        x_train = self.cond_to_numpy(x_train)
        y_train = self.cond_to_numpy(y_train)
        sigma = self.cond_to_numpy(sigma)

        plt.figure(numberFigure, figsize=(12, 6))
        if title == True:
            tit = 'Gaussian process with example:' + str(numberFigure)
            if not label_model == 0:
                tit = 'Gaussian process example:' + str(numberFigure - label_model) + ' with model:' + str(
                    int(label_model / 1000))
            plt.title(tit)
        if x_train is not None and y_train is not None:
            plt.plot(x_train, y_train, 'k.', markersize=2, label='Observations QRS complex')
        plt.plot(x_post, y_pred, 'b-', label='Prediction')
        plt.fill(np.concatenate([x_post, x_post[::-1]]),
                 np.concatenate([y_pred - 1.9600 * gamma,
                                 (y_pred + 1.9600 * gamma)[::-1]]),
                 alpha=.35, fc='b', ec='None', label='95% confidence interval')
        if sigma is not None:
            plt.fill(np.concatenate([x_post, x_post[::-1]]),
                     np.concatenate([y_pred - 1.9600 * sigma,
                                     (y_pred + 1.9600 * sigma)[::-1]]),
                     alpha=.2, fc='cyan', ec='None')
        if labels == True:
            plt.xlabel('$t$')

    # Robust computing of determinant
    def log_det(self, name, M):
        # Compute using formula log(det(C))=2*trace(log(L)) where L is cholesky lower decomposition
        M = self.cond_to_torch(M)
        id = torch.eye(M.shape[0])
        if torch.cuda.is_available() and self.cuda:
            M = M.cuda()
            id = id.cuda()
        try:
            # Compute order of magnitude
            od = torch.floor(torch.log10(torch.max(torch.diag(M))))
            # Compute the inverse to "normalize matrix"
            k = 10 ** -od
            # Apply reescale to the matrix
            M_aux = torch.mul(k, M)
            L = torch.linalg.cholesky(M_aux)
            log_diag = torch.log(torch.diag(L))
            # Apply reescale again to return to main order
            det = 2 * sum(log_diag) - len(M) * torch.log(k)
        except RuntimeError:
            od = torch.floor(torch.log10(torch.max(torch.diag(M))))
            k = 10 ** -od
            M_aux = torch.mul(k, M)
            jitter = torch.mul(1e-6, id)
            with warnings.catch_warnings(record=True) as w:
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")
                det = torch.log(torch.linalg.det(M_aux + jitter)) - len(M) * torch.log(k)
                # if w is not None:
                #     print("Stop")
        except ValueError:
            return torch.from_numpy(-np.inf)
        return det

    # Robust computing of the inverse
    def inv_r(self, name, M):
        M = self.cond_to_torch(M)
        if torch.cuda.is_available() and self.cuda:
            M = M.cuda()
        try:
            # Compute order of magnitude
            id = torch.eye(M.shape[0])
            if torch.cuda.is_available() and self.cuda:
                id = id.cuda()
                id.requires_grad = False
            od = torch.floor(torch.log10(torch.max(torch.diag(M))))
            # Compute the inverse to "normalize matrix"
            k = 10 ** -od
            # Apply reescale to the matrix
            M_aux = torch.mul(k, M)
            inv_aux = torch.linalg.solve(M_aux, id)
            # Apply reescale again to return to main order
            inv = torch.mul(k, inv_aux)
        except RuntimeError:
            inv = torch.linalg.solve(M, id)
        return inv

    def assign_alpha_ini(self, alpha, gamma):
        self.alpha_ini = alpha
        self.gamma_ini = gamma
        # self.covariance_kernel = WhiteKernel(noise_level=alpha*0.2, noise_level_bounds=(1e-10, 1e+1))

    def cond_to_numpy(self, x):
        if x is None:
            x = x
        elif type(x) is torch.Tensor:
            x = x.clone().detach().numpy()
        return x

    def cond_to_torch(self, x):
        if x is None:
            x = x
        elif type(x) is not torch.Tensor:
            x = torch.tensor(x, requires_grad=False)
        return x
