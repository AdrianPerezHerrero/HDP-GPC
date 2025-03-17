#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 8 10:33:40 2022
@author: adrian.perez
"""

import hdpgpc.GPI_model as gp
from hdpgpc.warping_system import Warping_system
from hdpgpc.OptimizerRhoOmega import find_optimum_multiple_tries, kvec

import numpy as np
np.random.seed(42)
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.special._ufuncs import psi as digamma
from scipy.special._ufuncs import gammaln
import pandas as pd
from torchmetrics.audio import SignalNoiseRatio
from tqdm import trange
import pickle as plk
import torch
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)



class GPI_HDP():
    """
    Model composed by an HMM whose states are iterative Gaussian Processes.
    The model define a structure that allow to switch between states determined by
    an LDS with Gaussian Process Prior.
    Parameters
    ----------
    M : (int) initial number of states. (usually 2)

    x_basis : array-like of shape (s_samples) domain points.

    n_outputs: int number of leads where receiving information. data should have dimensions
        [number of examples, length of examples, n_outputs]

    x_basis_warp: x_basis of the warp GP.

    kernels: predefinition of the kernels to use, default if output_var * RBF(lengthscale) + sigma_var
    
    model_type: list of strings indicating type of model for each state: 'static' or 'dynamic'. If for
            index m the model is dynamic, arrays with index m of ini_gamma and ini_sigma should be added.
            In other case default parameters would be used gamma=0.01 and sigma=0.5.

    ini_lengthscale, bound_lengthscale: initial legthscale value for kernel.
    
    ini_gamma, bound_gamma: array of matrix of shape (Mxsxs_samples) initial noise of the process.

    ini_sigma, bound_sigma: array of matrix of shape (Mxsxs_samples) initial noise used to compute kernel hyperparameters.
    
    ini_outputscale: initial value for output_var of the kernel, usually max amplitude of the signal

    noise_warp, bound_noise_warp: bound for noise of the GP warp.

    reest_conditions: conditions for reestimating LDS when using Maximum Likelihood.

    recursive_warp: use last computed warp to compute the next, not working.

    warp_updating: give a static LDS structure to GP update for the warp.

    method_compute_warp: 'standard', 'greedy', 'greedy_bound'

    mode_warp: 'balanced', 'fine', 'rough'

    verbose:

    annealing: updates of LDS reducing over iterations (for ML approach only)

    hmm_switch: if dependence over SLDS wants to be computed or LDS independent.

    max_models: bound on number of models.

    batch: number of samples to compute LDS parameters (for ML approach only)

    check_var: if check bound for variance estimation (not used).

    bayesian_params: if a bayesian update of LDS params is used.

    cuda: if cuda computations

    inducing_points: if x_basis is smaller than x_trains then should be setted True
    
    estimation_limit: bound for the estimation of LDS (both ML and Bayesian)

    reestimate_initial_params: if reestimation of Sigma and Gamma wants to be performed for offline approach
        using the classical estimation variance and 1-step variance.
    
    Returns
    -------
    self : returns an instance of self..
    """

    def __init__(self, x_basis, M=None, n_outputs=1, x_basis_warp=None, kernels=None, model_type='dynamic', ini_lengthscale=None,
                 bound_lengthscale=None, ini_gamma=None, ini_sigma=None, ini_outputscale=None, bound_sigma=(1e-10, 1e+10),
                 bound_gamma=(1e-1, 1e+2), bound_noise_warp=(1e-10, 1e+10), reest_conditions=[1, 20, 5],
                 noise_warp=0.05, recursive_warp=False, warp_updating=False, method_compute_warp='greedy', mode_warp='rough',
                 verbose=False, annealing=True, hmm_switch=True, max_models=None, batch=None,
                 check_var=False, bayesian_params=True, cuda=False, inducing_points=False, estimation_limit=None, reestimate_initial_params=False,
                 n_explore_steps=10, free_deg_MNIV=5):
        if M is None:
            M = 1
        self.M = M
        self.verbose = verbose
        self.actual_state = 0
        self.cuda = cuda
        self.device = 'cuda' if self.cuda else 'cpu'
        self.n_outputs = n_outputs

        # Default cases
        # Be careful at using []*M because this replicates the object.
        if not isinstance(x_basis, list):
            x_basis = [x_basis] * M
        if x_basis_warp is None:
            x_basis_warp = x_basis
        elif not isinstance(x_basis_warp, list):
            x_basis_warp = [x_basis_warp] * M
        if not isinstance(bound_sigma, list) and not isinstance(bound_sigma, np.ndarray):
            bound_sigma = [bound_sigma] * M
        if not isinstance(bound_gamma, list) and not isinstance(bound_gamma, np.ndarray):
            bound_gamma = [bound_gamma] * M
        if not isinstance(bound_noise_warp, list) and not isinstance(bound_noise_warp, np.ndarray):
            bound_noise_warp = [bound_noise_warp] * M
        if not isinstance(ini_lengthscale, list) and not isinstance(ini_lengthscale, np.ndarray):
            ini_lengthscale = [ini_lengthscale] * M
        if not isinstance(ini_outputscale, list) and not isinstance(ini_outputscale, np.ndarray):
            ini_outputscale = [ini_outputscale] * M
        if not isinstance(bound_lengthscale, list) and not isinstance(bound_lengthscale, np.ndarray):
            bound_lengthscale = [bound_lengthscale] * M
        if not isinstance(inducing_points, list) and not isinstance(inducing_points, np.ndarray):
            inducing_points = [inducing_points] * M
        if not isinstance(estimation_limit, list) and not isinstance(estimation_limit, np.ndarray):
            estimation_limit = [estimation_limit] * M
        if not isinstance(ini_gamma, list) and not isinstance(ini_gamma, np.ndarray):
            ini_gamma = [ini_gamma] * M
        if not isinstance(ini_sigma, list) and not isinstance(ini_sigma, np.ndarray):
            ini_sigma = [ini_sigma] * M
        if not isinstance(model_type, list) and not isinstance(model_type, np.ndarray):
            model_type = [model_type] * M
        if not isinstance(annealing, list) and not isinstance(annealing, np.ndarray):
            annealing = [annealing] * M
        if not isinstance(warp_updating, list) and not isinstance(warp_updating, np.ndarray):
            warp_updating = [warp_updating] * M
        if not isinstance(recursive_warp, list) and not isinstance(recursive_warp, np.ndarray):
            recursive_warp = [recursive_warp] * M
        if ini_outputscale[0] is None:
            ini_outputscale = ini_sigma.copy()
        if kernels is None:
            kernels = []
            # WhiteNoise always last for stability.
            for m in range(M):
                #Outputscale should be bounded if a learning with few t basis will be performed
                kernels.append(ConstantKernel(ini_outputscale[m], (ini_outputscale[m], ini_outputscale[m]*5.0)) * RBF(ini_lengthscale[m], bound_lengthscale[m])
                               + WhiteKernel(bound_sigma[m][0], bound_sigma[m]))
                # kernels.append(RBF(ini_lengthscale[m], bound_lengthscale[m])
                #                + WhiteKernel(ini_sigma[m], bound_sigma[m]))
        # Save default options to generate new model
        self.set_default_options(kernels[0], ini_sigma[0], ini_gamma[0], ini_outputscale[0], bound_sigma[0], bound_gamma[0], bound_noise_warp[0],
                                 annealing[0], method_compute_warp, model_type[0], recursive_warp[0], warp_updating[0], inducing_points[0],
                                 estimation_limit[0], free_deg_MNIV)
        # Define some characteristics of the model with an initial M decided
        self.ini_lengthscale = ini_lengthscale
        self.bound_lengthscale = bound_lengthscale
        #self.static_factor = ini_sigma[0] / (ini_sigma[0] + ini_gamma[0])
        #self.dynamic_factor = ini_gamma[0] / (ini_sigma[0] + ini_gamma[0])
        self.static_factor = ini_sigma[0] / ini_sigma[0]
        self.dynamic_factor = ini_gamma[0] / ini_gamma[0]
        self.bound_sigma = bound_sigma
        self.bound_gamma = bound_gamma
        self.bound_sigma_warp = bound_noise_warp
        self.annealing = annealing
        self.hmm_switch = hmm_switch
        self.method_compute_warp = method_compute_warp
        self.recursive_warp = recursive_warp
        self.model_type = model_type
        self.warp_updating = warp_updating
        self.max_models = max_models
        self.batch = batch
        self.check_var = check_var
        self.bayesian_params = bayesian_params
        self.x_basis_warp = x_basis_warp
        self.inducing_points = inducing_points
        self.estimation_limit = estimation_limit
        self.reestimate_initial_params = reestimate_initial_params
        self.n_explore_steps = n_explore_steps
        self.free_deg_MNIV = free_deg_MNIV
        self.train_elbo = []
        self.resp_assigned = []
        self.f_ind_old = torch.zeros(M, device= self.device).long()
        # Define reestimation conditions
        self.min_samples = reest_conditions[0]
        self.max_samples = reest_conditions[1]
        self.div_samples = reest_conditions[2]
        # Define the parameters associated with samples and warp
        self.T = 0
        self.y = []
        self.y_w = []
        self.x_w = []
        # self.x_warped = []
        self.liks = []
        self.noise_warp = noise_warp
        self.mode_warp = mode_warp
        self.wp_sys = []
        for ld in range(self.n_outputs):
            wp_sys_ = []
            for m in range(M):
                wp_sys_.append(Warping_system(x_basis_warp[m], noise_warp, bound_noise_warp[m],
                                                  recursive=recursive_warp[m], bayesian=self.bayesian_params,
                                                  cuda=self.cuda, mode=self.mode_warp))
            self.wp_sys.append(wp_sys_)

        # Define the parameters of the state model:
        self.x_basis = x_basis
        self.x_basis_ini = x_basis[0].copy()
        self.x_train = []
        self.y_train = torch.tensor([])

        # Initial vector following a uniform distribution in log form
        # self.pi = [np.log(np.repeat(1 / M, M))]

        # Uniform matrix describing the initial transitions between states.
        self.trans_A = [np.log(np.matrix([[1 / M] * M] * M))]

        # Messages associated with state-space model
        # self.alpha = []
        # self.beta = []

        # Variational local parameters
        self.h = []
        self.q = []
        self.fmsg = None
        self.margPrObs = None

        # Variational global parameters
        self.rho = []
        self.omega = []
        self.theta = []
        self.transTheta = []
        self.startTheta = []

        # Hyperparameters of HDP
        # Gamma represent the variability in the prior weights for the rows of the transition matrix
        # transAlpha represent a factor of prior probability of state-state transition
        # startAlpha represent a factor of prior probability of starting state
        # kappa represent the sticky factor of self transition
        # self.gamma = 700.0
        # self.transAlpha = 1400.0
        # self.startAlpha = 1400.0
        # self.kappa = 0.0
        # self.gamma = 100.0
        # self.transAlpha = 200.0
        # self.startAlpha = 200.0
        # self.kappa = 0.0
        self.gamma = 0.4
        self.transAlpha = 0.4
        self.startAlpha = 0.4
        self.kappa = 0.0

        # Model associated with each state
        # Default hyperparameters of GP defining the model.
        self.gpmodels = [[] for _ in range(n_outputs)]
        for ld in range(n_outputs):
            for m in range(M):
                # Create GP
                if self.bayesian_params:
                    gp_ = gp.GPI_model(kernels[m], x_basis[m], annealing=self.annealing[m],
                                       bayesian=self.bayesian_params, cuda=self.cuda, inducing_points=inducing_points[m],
                                       estimation_limit=estimation_limit[m], free_deg_MNIV=self.free_deg_MNIV, verbose=self.verbose)
                else:
                    gp_ = gp.GPI_model(kernels[m], x_basis[m], annealing=self.annealing[m], cuda=self.cuda,
                                       inducing_points=inducing_points[m], estimation_limit=estimation_limit[m],
                                       free_deg_MNIV=self.free_deg_MNIV, verbose=self.verbose)
                # Initiate GP
                if model_type[m] == 'static':
                    cond = gp_.GPR_static(ini_sigma[m])
                elif model_type[m] == 'dynamic':
                    cond = gp_.GPR_dynamic(ini_gamma[m], ini_sigma[m])
                else:
                    print("Chosen model should be dynamic or static.")
                gp_.initial_conditions(ini_mean=None, ini_cov=None,
                                       ini_A=cond[0], ini_Gamma=cond[1], ini_C=cond[2], ini_Sigma=cond[3])
                self.gpmodels[ld].append(gp_)

        self.init_global_params(len(x_basis), self.M)
        if self.cuda:
            self.full_model_to_torch()
            self.full_model_to_cuda()

    #Methods to compute HDP variational computations.
    def init_global_params(self, d_dim, M):
        self.rho = self.create_initrho(M)
        self.omega = (1.0 + self.gamma) * self.cond_cuda(torch.ones(M))
        transStateCount = self.cond_cuda(torch.ones((M, M)))  # Cause we are on initial moment
        startStateCount = self.cond_cuda(torch.ones(M))  # Cause we are on initial moment
        self.transTheta, self.startTheta = self._calcThetaFull(transStateCount, startStateCount, M + 1)

    def reinit_global_params(self, M, transStateCount_, startStateCount_):
        self.rho = self.create_initrho(M)
        self.omega = (1.0 + self.gamma) * torch.ones(M)
        self.transTheta, self.startTheta = self._calcThetaFull(transStateCount_, startStateCount_, M=M)

    def temp_reinit_global_params(self, M, transStateCount_, startStateCount_, rho=None, omega=None):
        if rho is None:
            rho = self.rho
        if omega is None:
            omega = self.omega
        rho_ = self.create_initrho(M)
        rho_[:rho.shape[0]] = rho
        omega_ = (1.0 + self.gamma) * torch.ones(M)
        omega_[:omega.shape[0]] = omega
        transTheta_, startTheta_ = self._calcThetaFull(transStateCount_, startStateCount_, M + 1, rho_)
        return rho_, omega_, transTheta_, startTheta_

    def create_initrho(self, M):
        remMass = np.minimum(0.1, 1.0 / (M * M))
        delta = (-1 + remMass) * np.arange(0, M, 1, dtype=np.float64)
        rho = (1 - remMass) / (M + delta)
        return self.cond_cuda(self.cond_to_torch(rho))

    def _calcThetaPost(self, transStateCount, startStateCount, M):
        transStateCount_ = transStateCount
        Ebeta = self.rho2beta(self.cond_cpu(self.rho), returnSize='K+1')
        alphaEbeta = self.transAlpha * Ebeta

        transTheta = self.cond_cuda(torch.zeros((M, M)))
        transTheta += alphaEbeta[np.newaxis, :]
        if not len(transStateCount_.shape) == 0 and not transStateCount_.shape == (1, 1):
            kp_ = self.cond_cuda(torch.zeros((M, M)))
            kp_[:-1, :-1] = self.kappa * self.cond_cuda(torch.eye(M-1))
            transTheta[:M, :M] += transStateCount_ + kp_
        else:
            transTheta[:M, :M] += transStateCount_ + self.kappa * self.cond_cuda(torch.eye(M))

        startTheta = self.startAlpha * Ebeta
        startTheta[:M] += startStateCount

        return transTheta, startTheta

    def _calcThetaFull(self, transStateCount, startStateCount, M=None, rho=None, kappa=None):
        transStateCount_ = transStateCount
        if M is None:
            M = self.M + 1
        if rho is None:
            rho = self.rho
        if kappa is None:
            kappa = self.kappa
        if M == rho.shape[0]:
            Ebeta = self.rho2beta(self.cond_cpu(rho), returnSize='K')
        else:
            Ebeta = self.rho2beta(self.cond_cpu(rho), returnSize='K+1')
        alphaEbeta = self.transAlpha * Ebeta

        transTheta = self.cond_cuda(torch.zeros((M, M)))
        transTheta += alphaEbeta[np.newaxis, :]
        transTheta[:M-1, :M-1] += transStateCount_[:M-1,:M-1] + kappa * self.cond_cuda(torch.eye(M-1))

        startTheta = self.startAlpha * Ebeta
        startTheta[:M-1] += startStateCount[:M-1]

        return transTheta, startTheta
        
    def compute_Pi(self):
        """ Compute transition matrix
        """
        return torch.exp(digamma(self.cond_cpu(self.transTheta)) -
                         torch.log(torch.sum(torch.exp(digamma(self.cond_cpu(self.transTheta))), axis=1))[:, np.newaxis])

    def rho2beta(self, rho, returnSize='K+1'):
        rho = np.asarray(rho, dtype=np.float64)
        if returnSize == 'K':
            beta = rho.copy()
            beta[1:] *= np.cumprod(1 - rho[:-1])
        else:
            beta = np.append(rho, 1.0)
            beta[1:] *= np.cumprod(1.0 - rho)
        return self.cond_cuda(self.cond_to_torch(beta))

    def beta2rho(self, beta, K):
        ''' Get rho vector that can deterministically produce provided beta.

        Returns
        ------
        rho : 1D array, size K
            Each entry rho[k] >= 0.
        '''
        beta = np.asarray(beta, dtype=np.float64)
        rho = beta.copy()
        beta_gteq = 1 - np.cumsum(beta[:-1])
        rho[1:] /= np.maximum(1e-100, beta_gteq)
        if beta.size == K + 1:
            return self.cond_cuda(self.cond_to_torch(rho[:-1]))
        elif beta.size == K:
            return self.cond_cuda(self.cond_to_torch(rho))
        else:
            raise ValueError('Provided beta needs to be of length K or K+1')


    def keep_last_all(self):
        """ Method to reset full model and save space
        """
        for ld in range(self.n_outputs):
            for gp in self.gpmodels[ld]:
                gp.reinit_LDS(save_last=True)
                gp.reinit_GP(save_last=True, save_index=True)


    def set_default_options(self, kernel, ini_sigma, ini_gamma, ini_outputscale, bound_sigma, bound_gamma,
                            bound_noise_warp, annealing, method_compute_warp,
                            model_type, recursive_warp, warp_updating, inducing_points, estimation_limit, free_deg_MNIV):
        """ Default options definition.
        """
        self.kernel_def = kernel.clone_with_theta(kernel.theta)
        self.ini_sigma_def = ini_sigma
        self.ini_gamma_def = ini_gamma
        self.ini_outputscale_def = ini_outputscale
        self.bound_sigma_def = bound_sigma
        self.bound_gamma_def = bound_gamma
        self.bound_sigma_warp_def = bound_noise_warp
        self.annealing_def = annealing
        self.method_compute_warp_def = method_compute_warp
        self.model_type_def = model_type
        self.recursive_warp_def = recursive_warp
        self.warp_updating_def = warp_updating
        self.inducing_points_def = inducing_points
        self.estimation_limit_def = estimation_limit
        self.free_deg_MNIV = free_deg_MNIV

    def get_default_options(self):
        return self.kernel_def, self.ini_sigma_def, self.ini_gamma_def, self.ini_outputscale_def, self.bound_sigma_def,\
            self.bound_gamma_def, self.bound_sigma_warp_def, self.annealing_def, self.method_compute_warp_def,\
            self.model_type_def, self.recursive_warp_def, self.warp_updating_def, self.inducing_points_def,\
            self.estimation_limit_def, self.free_deg_MNIV

    def create_gp_default(self, i=None):
        """ Create a GP default when a birth happens.
        """
        if i is None or len(self.wp_sys) <= i:
            kernel, ini_sigma, ini_gamma, ini_outputscale, bound_sigma, bound_gamma, bound_noise_warp, annealing, method_compute_warp,\
                model_type, recursive_warp, warp_updating, inducing_points, estimation_limit, free_deg_MNIV = self.get_default_options()
            kernel = kernel.clone_with_theta(kernel.theta)
            gp_ = gp.GPI_model(kernel, self.x_basis_ini, annealing=annealing, bayesian=self.bayesian_params, cuda=self.cuda,
                               inducing_points=inducing_points, estimation_limit=estimation_limit, free_deg_MNIV=free_deg_MNIV, verbose=self.verbose)
            if model_type == 'static':
                cond = gp_.GPR_static(ini_sigma)
            elif model_type == 'dynamic':
                cond = gp_.GPR_dynamic(ini_gamma, ini_sigma)
            else:
                print("Chosen model should be dynamic or static.")
            gp_.initial_conditions(ini_mean=None, ini_cov=None,
                                   ini_A=cond[0], ini_Gamma=cond[1], ini_C=cond[2], ini_Sigma=cond[3])
            # for ld in range(self.n_outputs):
            #     self.wp_sys[ld].append(Warping_system(self.x_basis_warp[0], self.noise_warp, bound_noise_warp,
            #                                       recursive=recursive_warp, cuda=self.cuda, mode=self.mode_warp))
            self.wp_sys.append(Warping_system(self.x_basis_warp[0], self.noise_warp, bound_noise_warp,
                                                  recursive=recursive_warp, cuda=self.cuda, mode=self.mode_warp))
            if self.cuda and torch.cuda.is_available():
                gp_.model_to_cuda()
                self.wp_sys[-1].warp_gp.model_to_cuda()
            self.bound_sigma.append(bound_sigma)
            self.bound_gamma.append(bound_gamma)
            self.bound_sigma_warp.append(bound_noise_warp)
            self.annealing.append(annealing)
            self.recursive_warp.append(recursive_warp)
            self.warp_updating.append(warp_updating)
            self.model_type.append(model_type)
            self.x_basis.append(self.x_basis_ini)
            self.bound_sigma.append(bound_sigma)
            self.bound_sigma_warp.append(bound_noise_warp)
            self.inducing_points.append(inducing_points)
            self.estimation_limit.append(estimation_limit)
            for t, y_ in enumerate(self.y[:-1]):
                gp_.include_sample(t, y_, 0, posterior=False)
            return gp_
        else:
            kernel, ini_sigma, ini_gamma, ini_outputscale, bound_sigma, bound_gamma, bound_noise_warp, annealing, method_compute_warp, \
                model_type, recursive_warp, warp_updating, inducing_points, estimation_limit, free_deg_MNIV = self.get_default_options()
            kernel = kernel.clone_with_theta(kernel.theta)
            gp_ = gp.GPI_model(kernel, self.x_basis_ini, annealing=annealing, bayesian=self.bayesian_params,
                               cuda=self.cuda, inducing_points=inducing_points, estimation_limit=estimation_limit,
                               free_deg_MNIV=free_deg_MNIV, verbose=self.verbose)
            if model_type == 'static':
                cond = gp_.GPR_static(ini_sigma)
            elif model_type == 'dynamic':
                cond = gp_.GPR_dynamic(ini_gamma, ini_sigma)
            else:
                print("Chosen model should be dynamic or static.")
            gp_.initial_conditions(ini_mean=None, ini_cov=None,
                                   ini_A=cond[0], ini_Gamma=cond[1], ini_C=cond[2], ini_Sigma=cond[3])
            self.wp_sys[i] = Warping_system(self.x_basis_warp[0], self.noise_warp, bound_noise_warp,
                                              recursive=recursive_warp, cuda=self.cuda, mode=self.mode_warp)
            if self.cuda and torch.cuda.is_available():
                gp_.model_to_cuda()
                self.wp_sys[i].warp_gp.model_to_cuda()
            self.bound_sigma[i] = bound_sigma
            self.bound_gamma[i] = bound_gamma
            self.bound_sigma_warp[i] = bound_noise_warp
            self.annealing[i] = annealing
            self.recursive_warp[i] = recursive_warp
            self.warp_updating[i] = warp_updating
            self.model_type[i] = model_type
            self.x_basis[i] = self.x_basis_ini
            self.bound_sigma[i] = bound_sigma
            self.bound_sigma_warp[i] = bound_noise_warp
            self.inducing_points[i] = inducing_points
            self.estimation_limit[i] = estimation_limit
            for t, y_ in enumerate(self.y[:-1]):
                gp_.include_sample(t, y_, 0, posterior=False)
            return gp_


    def variational_local_terms(self, q, transTheta=None, startTheta=None, liks=None, classify=False):
        """
        Compute variational terms of the SLDS for one example. Online approach.

        Parameters
        ----------
        y : vector double
            observation vector (or warped versions).
        liks : vector, optional
            likelihoods of the warpings.
        """
        # Parameters of model
        M = self.M + 1
        q = torch.clone(q)
        # Compute the initial values
        if transTheta is None:
            transTheta = self.transTheta
        if startTheta is None:
            startTheta = self.startTheta
        if liks is None:
            liks = np.zeros(M)
        digammaSumTransTheta = torch.log(torch.sum(torch.exp(digamma(self.cond_cpu(transTheta[:M, :M + 1]))), axis=1))
        transPi = digamma(self.cond_cpu(transTheta[:M, :M])) - digammaSumTransTheta[:, np.newaxis]
        self.trans_A = transPi
        startPi = digamma(self.cond_cpu(startTheta[:M])) - torch.log(torch.sum(torch.exp(digamma(self.cond_cpu(startTheta[:M + 1])))))
        liks_ = np.array(liks)[:,np.newaxis]# * 0.5
        q[-1] = q[-1] + liks_
        if classify:
            for q_aux in q:
                q_aux[-1] = -np.inf
        m_ = np.argmax(self.weight_mean(q)[-1])
        q_, _ = self.LogLik(self.weight_mean(q))
        alpha, margprob = self.forward(startPi, transPi, q_)
        beta = self.backward(transPi, q_, margprob)
        resp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
        m_resp = np.argmax(self.cond_cpu(resp[-1]))
        respPair, _ = self.LogLik(self.coupled_state_coef(alpha, beta, transPi, q_, margprob), axis=1)
        if m_ != m_resp and self.verbose:
            print("Mismatch between SSE ("+str(m_+1)+") and Resp ("+str(m_resp+1)+").")
        if torch.any(torch.isnan(resp)):
            print("Error")
        if classify:
            return q, torch.exp(resp), resp, torch.exp(respPair), respPair
        else:
            return torch.exp(resp), resp, torch.exp(respPair), respPair


    def LogLik(self, logSoftEv, axis=1):
        ''' Return element-wise normalized input log likelihood

        Numerically safe, guaranteed not to underflow

        Returns
        --------
        SoftEv : 2D array, size TxK
                  equal to logSoftEv, up to prop constant for each row
        lognormC : 1D array, size T
                  gives log of the prop constant for each row
        '''
        if type(logSoftEv) is torch.Tensor:
            # lognormC = self.cond_to_numpy(self.cond_cpu(torch.max(logSoftEv, axis)[0]))
            lognormC = torch.max(logSoftEv, dim=axis)[0]
            if torch.any(torch.isinf(lognormC)):
                return logSoftEv, lognormC
        else:
            lognormC = np.max(logSoftEv, axis)
            if np.any(np.isinf(lognormC)):
                return logSoftEv, lognormC
        if axis == 0:
            if type(lognormC) is np.float64:
                lognormC = np.array([lognormC])
            SoftEv = logSoftEv - lognormC[np.newaxis, :]
        elif axis == 1:
            SoftEv = logSoftEv - lognormC[:, np.newaxis]
        else:
            SoftEv = logSoftEv
        return SoftEv, lognormC

    def signaltonoise(self, a, axis=0, ddof=0):
        """ Compute signal to noise ratio
        """
        a = np.asanyarray(a)
        #m = a.mean(axis)** 2
        m = 100.0
        #m = a.max(axis)[0] ** 2
        sd = a.std(axis=axis, ddof=ddof) ** 2
        return np.where(sd == 0, 0, m / sd)

    def rolling_snr(self, signal, window_size: int):
        """ Compute rolled signal to noise ratio.
        """
        signal_series = pd.Series(signal)
        rolling_mean = signal_series.rolling(window=window_size).mean()[window_size:].mean()
        rolling_max = signal_series.rolling(window=window_size).max().max() ** 2
        rolling_min = signal_series.rolling(window=window_size).min().min() ** 2
        rolling_std = signal_series.rolling(window=window_size).std()[window_size:].mean()
        rolling_snr = 10 * np.log10(
            (rolling_mean ** 2) / (rolling_std ** 2))#.replace(0, np.finfo(float).eps))  # type: ignore
        return rolling_snr

    def weight_mean(self, q, snr=None):
        """ Weight the log squared error with the snr computed to join the information on each lead.
        """
        if len(q.shape) > 2:
            if snr is None:
                return torch.einsum('ijk,ik->ij', q, self.snr_norm)
            else:
                snr_ = torch.softmax(torch.max(snr, dim=1)[0], dim=1)
                return torch.einsum('ijk,ik->ij', q, snr_)
        else:
            if snr is None:
                snr_frac = torch.sum(self.snr_norm, dim=0) / torch.sum(self.snr_norm)
                return torch.einsum('ij,j->i', q, snr_frac)
            else:
                snr_ = torch.softmax(torch.max(snr, dim=1)[0], dim=1)
                snr_frac = torch.sum(snr_, dim=0) / torch.sum(snr_)
                return torch.einsum('ij,j->i', q, snr_frac)

    def compute_snr_ini(self, y_trains):
        """ Initial computation of snr
        """
        n_samples = np.array(y_trains).shape[0]
        n_outputs = np.array(y_trains).shape[2]
        snr = torch.zeros(n_samples, n_outputs)
        snr_comp = SignalNoiseRatio()
        for ld in range(n_outputs):
            snr[:, ld] = torch.tensor(
                [snr_comp(torch.from_numpy(y),
                     torch.mean(torch.from_numpy(y_trains)[:, :, ld], dim=0)) for y in
                y_trains[:, :, ld]])
        self.snr_norm = torch.softmax(snr, dim=1)

    def compute_snr(self, y_trains, gp):
        """ Iterative computation of snr
        """
        snr = torch.zeros(y_trains.shape[0])
        snr_comp = SignalNoiseRatio()
        for t in range(y_trains.shape[0]):
            j = np.min([np.max([gp.find_closest_lower(t), 1]), len(gp.f_star_sm) - 1])
            snr[t] = snr_comp(y_trains[t], gp.f_star_sm[j].T[0])
        #snr = torch.tensor([0.5] * y_trains.shape[0])
        return snr

    def normalize_snr(self, snr):
        """ Normalize snr using softmax
        """
        return torch.softmax(torch.max(torch.clone(snr), dim=1)[0], dim=1)

    def include_batch(self, x_trains, y_trains, with_warp=False, force_model=None, it_limit=None, warp=False):
        """
        Include the batch of samples received and restimate all the parameters of the
        full model, including the state-space and the GPS.
        Parameters
        ----------
        x_trains : time indexes of each sample (nxs_samples
        y_trains : array sample of shape (nxs_samples)

        Returns
        -------
        self : returns an instance of self.
        """
        # # Redefine HDP hyperparams for batch inclusion
        self.gamma = 1.0
        self.transAlpha = 1.0
        self.startAlpha = 1.0
        self.kappa = 0.0
        print("------ HDP Hyperparameters ------", flush=True)
        print("gamma: " + str(self.gamma))
        print("transAlpha: " + str(self.transAlpha))
        print("startAlpha: " + str(self.startAlpha))
        print("kappa: " + str(self.kappa))
        print("---------------------------------", flush=True)
        n_samples = np.array(y_trains).shape[0]
        n_outputs = np.array(y_trains).shape[2]
        t = self.T
        self.T = self.T + n_samples
        #Compute SNR to ponderate q on each lead.
        self.compute_snr_ini(y_trains)
        M = self.M
        y_trains = self.cond_cuda(self.cond_to_torch(y_trains))
        x_trains = self.cond_cuda(self.cond_to_torch(x_trains))
        iteration = 0
        reparam = True
        #Start loop of EM
        elbo = -np.inf
        resp = self.cond_cuda(torch.zeros((n_samples, M)))
        respPair = self.cond_cuda(torch.zeros((n_samples, M, M)))
        respPair[:,0,0] = respPair[:,0,0] + 1.0
        q = self.cond_cuda(torch.zeros((n_samples, M, n_outputs)))
        q_lat = self.cond_cuda(torch.zeros((n_samples, M, n_outputs)))
        snr = self.snr_norm
        resp[:, 0] = resp[:, 0] + 1.0
        y_trains_w = torch.clone(y_trains)
        warp_computed = False
        # Initial estimation of full group and conditioning the covariance matrices.
        if self.reestimate_initial_params:
            self.redefine_default(x_trains, y_trains, resp)
        startStateCount = None
        transStateCount = None
        reallocate = False
        while True:
            resp, respPair, q, q_lat, snr, end = self.refill(resp, respPair, startStateCount, transStateCount, q, q_lat, snr)
            M = self.M
            if resp.shape[1] == 1:
                startStateCount = self.cond_cuda(resp[0])
                transStateCount = self.cond_cuda(torch.sum(respPair, axis=0))
                self.reinit_global_params(M, transStateCount, startStateCount)
                nIters = 2
                for giter in range(nIters):
                    self.transTheta, self.startTheta = self._calcThetaFull(self.cond_cuda(transStateCount),
                                                                           self.cond_cuda(startStateCount), M + 1)
                    self.rho, self.omega = self.find_optimum_rhoOmega()
            if end:
                break
            resp, respPair, q, q_lat, snr, y_trains_w, reallocate = self.variational_local_terms_batch(M, x_trains, y_trains, y_trains_w,
                                                                    self.transTheta, self.startTheta, resp, respPair, q, q_lat, snr, reallocate, reparam)
            #resp, respPair = self.refill_resp(resp, respPair)
            if resp.shape[1] > M:
                self.M = M + 1
                M = self.M
                #n_samples = self.T
                #resp__ = self.cond_cuda(torch.zeros((n_samples, M + 1)))
                #resp__[:, :-1] = torch.clone(resp)
                #respPair__ = self.cond_cuda(torch.zeros((n_samples, M + 1, M + 1)))
                #respPair__[:, :-1, :-1] = torch.clone(respPair)
            resp__ = torch.clone(resp)
            respPair__= torch.clone(respPair)
            # Update HDP variational params.
            if self.hmm_switch:
                startStateCount = self.cond_cuda(resp__[0])
                transStateCount = self.cond_cuda(torch.sum(respPair__, axis=0))
            else:
                transStateCount = self.cond_cuda(torch.ones((M + 1, M + 1)))  # Cause we are in independent observations
                startStateCount = self.cond_cuda(torch.ones(M + 1))

            self.reinit_global_params(M, transStateCount, startStateCount)
            nIters = 2
            for giter in range(nIters):
                self.transTheta, self.startTheta = self._calcThetaFull(self.cond_cuda(transStateCount),
                                                                       self.cond_cuda(startStateCount), M + 1)
                self.rho, self.omega = self.find_optimum_rhoOmega()

            #Update transition matrix
            digammaSumTransTheta = torch.log(
                torch.sum(torch.exp(digamma(self.cond_cpu(self.transTheta[:M, :M + 1]))), axis=1))
            transPi = digamma(self.cond_cpu(self.transTheta[:M, :M])) - digammaSumTransTheta[:, np.newaxis]
            self.trans_A = transPi
            # Compute ELBO and start steps of EM
            # TODO: adapt this option to full cuda implementation.
            if self.T > 1:
                elbo_ = self.calcELBO_NonlinearTerms(resp=self.cond_to_numpy(self.cond_cpu(resp)),
                                                             respPair=self.cond_to_numpy(self.cond_cpu(respPair)))
                print('\n-------End Lower Bound Iteration ' + str(iteration) + '-------')
                q_obs, elbo_lin = self.compute_q_elbo(resp, respPair, self.weight_mean(q), self.weight_mean(q_lat), self.gpmodels, self.M, snr='saved', post=False)
                elbo_ = elbo_ + elbo_lin + q_obs
                print("ELBO + Nonlinear: "+ str(elbo_))
                iteration = iteration + 1
                print('\n-------Start lower Bound Iteration '+str(iteration)+'-------')
                if it_limit is not None and iteration >= it_limit:
                    self.train_elbo.append(elbo_)
                    self.resp_assigned.append(torch.where(resp == 1.0)[1])
                    break
                resp_group = torch.sum(resp, axis=0)
                self.train_elbo.append(elbo_)
                self.resp_assigned.append(torch.where(resp == 1.0)[1])
                if (torch.where(resp_group==0.0)[0].shape[0] > 1.0 or
                        (len(self.resp_assigned) >1 and torch.all(self.resp_assigned[-2] == self.resp_assigned[-1]))):
                    if warp_computed or not warp:
                        if not warp:
                            self.y_train = y_trains
                        break
                    else:
                        elbo = elbo_
                        q, q_lat, warp_computed = self.compute_warp_actual_state(x_trains, y_trains, q=q, q_lat=q_lat)
                else:
                    elbo = elbo_
                    self.y_train = y_trains
            else:
                break

    def compute_warp_actual_state(self, x_trains, y_trains, q=None, q_lat=None, snr=None):
        """ Compute warp for every sample with an assignation to a gpmodel in the actual state of the model.
        """
        y_trains = self.cond_cuda(self.cond_to_torch(y_trains))
        x_trains = self.cond_cuda(self.cond_to_torch(x_trains))
        y_trains_w = torch.clone(y_trains)
        self.x_w = torch.zeros(y_trains.shape)
        self.liks_w = torch.zeros((y_trains.shape[0], self.n_outputs))
        for ld in range(self.n_outputs):
            print("--Compute warp--")
            for j in trange(y_trains.shape[0]):
                y = y_trains[j, :, [ld]]
                m_ = 0
                for m, gp in enumerate(self.gpmodels[ld]):
                    if j in gp.indexes:
                        m_ = m
                y_w, x_w, lik = self.compute_warp_y(x_trains[j], y, force_model=m_,
                                                    gpmodel=self.gpmodels[ld][m_])
                y_trains_w[j, :, [ld]] = y_w[m_]
                self.x_w[j, :, [ld]] = x_w[m_]
                self.liks_w[j, ld] = lik[m_]
            if q is not None:
                for m, gp in enumerate(self.gpmodels[ld]):
                    q[:, m, ld] = gp.compute_sq_err_all(x_trains, y_trains_w[:, :, ld])
                    q_lat[:, m, ld] = gp.compute_q_lat_all(x_trains)
        warp_computed = True
        self.y_train = y_trains_w
        return q, q_lat, warp_computed

    def elbo_Linears(self, resp, respPair, post=False):
        """ Compute ELBO for linear terms. HDP.
        """
        startStateCount = resp[0]
        transStateCount = torch.sum(respPair, axis=0)
        M = resp.shape[1]

        # Augment suff stats to be sure have 0 in final column,
        # which represents inactive states.
        if startStateCount.shape[0] == M:
            startStateCount = torch.hstack([startStateCount, torch.zeros(1)])
        if transStateCount.shape[-1] == M:
            transStateCount = torch.hstack([transStateCount, torch.zeros((M, 1))])
            transStateCount = torch.vstack([transStateCount, torch.zeros((1, M + 1))])

        if self.rho.shape[0] == M:
            rho_ = torch.clone(self.rho)
            omega_ = torch.clone(self.omega)
        else:
            rho_, omega_, transTheta_, startTheta_ = self.temp_reinit_global_params(M, torch.clone(transStateCount),
                                                                                    torch.clone(startStateCount))
        # nIters = 4
        # for giter in range(nIters):
        if post:
            nIters = 1
            for giter in range(nIters):
                transTheta_, startTheta_ = self._calcThetaFull(self.cond_cuda(torch.clone(transStateCount)),
                                                        self.cond_cuda(torch.clone(startStateCount)), M + 1, rho=rho_)
                rho_, omega_ = self.find_optimum_rhoOmega(startTheta=startTheta_,
                                                          transTheta=transTheta_, rho=rho_, omega=omega_, M=M)
        else:
            transTheta_, startTheta_ = self._calcThetaFull(self.cond_cuda(torch.clone(transStateCount)),
                                                        self.cond_cuda(torch.clone(startStateCount)), M + 1, rho=rho_)
        # if not post:
        #     # if M > 1:
        #     #     M = M - 1

        #     rho_ = torch.clone(self.rho)
        #     omega_ = torch.clone(self.omega)
        #     transTheta_, startTheta_ = self._calcThetaFull(self.cond_cuda(torch.clone(transStateCount)),
        #                                                     self.cond_cuda(torch.clone(startStateCount)),
        #                                                     rho=rho_, M=M)
        #     # for giter in range(2):
        #     #     rho_, omega_ = self.find_optimum_rhoOmega(startTheta_, transTheta_, rho=rho_, omega=omega_)
        # else:
        #     if M > 2:
        #         rho_, omega_, transTheta_, startTheta_ = self.temp_reinit_global_params(M, torch.clone(transStateCount),
        #                                                                                 torch.clone(startStateCount))
        #         # nIters = 4
        #         # for giter in range(nIters):
        #         transTheta_, startTheta_ = self._calcThetaFull(self.cond_cuda(torch.clone(transStateCount)),
        #                                                     self.cond_cuda(torch.clone(startStateCount)), M, rho=rho_)
        #             # rho_, omega_ = self.find_optimum_rhoOmega(startTheta=startTheta_,
        #             #                                           transTheta=transTheta_, rho=rho_, omega=omega_, M=M)
        #     else:
        #         rho_ = torch.clone(self.rho)
        #         omega_ = torch.clone(self.omega)
        #         transTheta_, startTheta_ = self._calcThetaPost(self.cond_cuda(torch.clone(transStateCount)),
        #                                                     self.cond_cuda(torch.clone(startStateCount)), M)


        return self.calcELBO_LinearTerms(rho=self.cond_to_numpy(self.cond_cpu(rho_)),
                                         omega=self.cond_to_numpy(self.cond_cpu(omega_)),
                                         alpha=self.transAlpha,
                                         startAlpha=self.startAlpha,
                                         kappa=self.kappa, gamma=self.gamma,
                                         transTheta=self.cond_to_numpy(self.cond_cpu(transTheta_)),
                                         startTheta=self.cond_to_numpy(self.cond_cpu(startTheta_)),
                                         startStateCount=self.cond_to_numpy(self.cond_cpu(startStateCount)),
                                         transStateCount=self.cond_to_numpy(
                                             torch.clone(self.cond_cpu(transStateCount))))


    def refill(self, resp, respPair, startStateCount, transStateCount, q, q_lat, snr):
        """ Redistribute the responsibility for not letting empty clusters and generate a new model if needed.
        """
        resp_per_group = torch.sum(resp, axis=0)
        print("Group responsability estimated: "+str(resp_per_group.detach().cpu().numpy().astype(np.int64)), flush=True)
        if torch.any(resp_per_group[:-1] < 1.0):
            # Case where last is filled and exists an empty group
            if resp_per_group[-1] >= 1.0:
                resp, respPair = self.refill_resp(resp, respPair)
            else:
                print("Empty group detected, new iteration.\n")
                return resp, respPair, q, q_lat, snr, True
        return resp, respPair, q, q_lat, snr, False

    def reorder(self, resp, respPair, q, q_lat):
        """ Reorder the responsibility to have a sorted assignation.
        """
        resp_per_group = torch.sum(resp, axis=0)
        reorder = torch.argsort(resp_per_group, descending=True)
        resp = resp[:, reorder]
        respPair = respPair[:, reorder, :]
        respPair = respPair[:, :, reorder]
        q = q[:, reorder]
        q_lat = q_lat[:, reorder]
        M = self.M
        gpmodels_temp = [[] * M] * self.n_outputs
        wp_sys = [] * M
        for ld in range(self.n_outputs):
            for i in range(M):
                gpmodels_temp[ld].append(self.gpmodels[ld][reorder[i]])
                wp_sys.append(self.wp_sys[reorder[i]])
        self.gpmodels = gpmodels_temp
        self.wp_sys = wp_sys
        return resp, respPair, q, q_lat

    def new_group(self, resp, respPair, q, q_lat, snr):
        M = resp.shape[1]
        n_samples = self.T
        resp_ = self.cond_cuda(torch.zeros((n_samples, M + 1)))
        resp_[:, :-1] = resp
        resp = resp_
        respPair_ = self.cond_cuda(torch.zeros((n_samples, M + 1, M + 1)))
        respPair_[:, :-1, :-1] = respPair
        respPair = respPair_
        q_ = self.cond_cuda(torch.zeros((n_samples, M + 1, self.n_outputs)))
        q_[:, :-1, :] = q
        q = q_
        q_lat_ = self.cond_cuda(torch.zeros((n_samples, M + 1, self.n_outputs)))
        q_lat_[:, :-1, :] = q_lat
        q_lat = q_lat_
        snr_ = self.cond_cuda(
            torch.zeros((n_samples, M + 1, self.n_outputs)) - np.abs(torch.min(snr, dim=1)[0])[:, np.newaxis] * 2.0)
        snr_[:, :-1, :] = snr
        snr = snr_
        return resp, respPair, q, q_lat, snr

    def refill_resp(self, resp, respPair=None):
        """ Refill HDP parameters if a model is reasignated.
        """
        resp_per_group = torch.sum(resp, axis=0)
        if torch.any(resp_per_group[:-1] < 1.0):
            empty_group_ind = torch.where(resp_per_group < torch.tensor(1.0, device=self.device))[0]
            if empty_group_ind.shape[0] > 1:
                empty_group_ind = empty_group_ind[0]
            resp_last_group = torch.clone(resp[:,-1])
            resp[:,-1] = torch.clone(resp[:,empty_group_ind.item()])
            resp[:,empty_group_ind.item()] = resp_last_group
            if respPair is not None:
                resp_pair_last_group_rows = torch.clone(respPair[:,-1,:])
                resp_pair_last_group_rows_ = torch.clone(resp_pair_last_group_rows[:,-1])
                resp_pair_last_group_rows[:,-1] = torch.clone(resp_pair_last_group_rows[:,empty_group_ind.item()])
                resp_pair_last_group_rows[:, empty_group_ind.item()] = torch.clone(resp_pair_last_group_rows_)
                resp_pair_last_group_cols = torch.clone(respPair[:,:-1,-1])
                resp_pair_last_group_cols_ = torch.clone(resp_pair_last_group_cols[:,empty_group_ind.item()])
                respPair[:, -1, :] = torch.clone(respPair[:, empty_group_ind.item(), :])
                respPair[:, :-1, -1] = torch.clone(respPair[:, :-1, empty_group_ind.item()])
                respPair[:, empty_group_ind.item(), :] = resp_pair_last_group_rows
                access_indexes = tuple(np.delete(np.arange(respPair.shape[1]), empty_group_ind.item()))
                respPair[:, access_indexes, empty_group_ind.item()] = resp_pair_last_group_cols
                respPair[:, -1, empty_group_ind.item()] = resp_pair_last_group_cols_
        if respPair is not None:
            return resp, respPair
        else:
            return resp

    def variational_local_terms_batch(self, M, x_trains, y_trains, y_trains_w, transTheta, startTheta, resp, respPair, q, q_lat, snr, reallocate, reparam=False):
        """
        Compute variational terms of the SLDS for a batch of examples. Iterates over q_first and q_all until convergence is reached.
        Parameters
        ----------
        y : vector double
            observation vector (or warped versions).
        liks : vector, optional
            likelihoods of the warpings.
        """
        # Parameters of model
        t = self.T - 1
        # Compute the initial values
        if transTheta is None:
            transTheta = self.transTheta
        if startTheta is None:
            startTheta = self.startTheta
        digammaSumTransTheta = torch.log(torch.sum(torch.exp(digamma(self.cond_cpu(transTheta[:M, :M + 1]))), axis=1))
        transPi = digamma(self.cond_cpu(transTheta[:M, :M])) - digammaSumTransTheta[:, np.newaxis]
        self.trans_A = transPi
        # Calculate LOG of start state prob vector
        startPi = digamma(self.cond_cpu(startTheta[:M])) - torch.log(
            torch.sum(torch.exp(digamma(self.cond_cpu(startTheta[:M + 1])))))
        i = 0
        reparam = True
        resp_per_group = torch.sum(resp, axis=0)
        if resp_per_group.shape[0] == 1.0 or resp_per_group[-2] >= 1.0 or not self.gpmodels[0][0].fitted:
            resp, respPair, q, q_lat, snr, y_trains_w, reallocate = self.estimate_q_first(M, x_trains=x_trains, y_trains=y_trains,
                                         y_trains_w=y_trains_w, resp=resp, respPair=respPair, q_=q, q_lat_=q_lat, snr_=snr,
                                         startPi=startPi, transPi=transPi, reallocate_=reallocate, reparam=reparam)
            if resp.shape[1] > self.M:
                q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q), self.weight_mean(q_lat),
                                                      self.gpmodels, self.M, snr='saved', post=True)
            else:
                q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q), self.weight_mean(q_lat),
                                                      self.gpmodels, self.M, snr='saved', post=False)
            i = i + 1
            print("First resp: " + str(torch.sum(resp, dim=0).int().numpy()))
        else:
            q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q), self.weight_mean(q_lat), self.gpmodels, self.M, snr='saved', post=False)
            print("Not first estimated q.")
        if not reallocate:
            while True:
                M = resp.shape[1]
                resp, respPair, q, q_lat, snr, y_trains_w, gpmodels = self.estimate_q_all( M, x_trains=x_trains, y_trains=y_trains, y_trains_w=y_trains_w,
                                               resp=resp, respPair=respPair, q_=q, q_lat_=q_lat, snr_=snr, startPi=startPi, transPi=transPi,
                                               reparam=reparam)
                self.gpmodels = gpmodels
                if resp.shape[1] > self.M:
                    q_post, elbo_post = self.compute_q_elbo(resp, respPair, self.weight_mean(q),
                                                            self.weight_mean(q_lat), self.gpmodels, self.M, snr='saved',
                                                            post=True)
                else:
                    q_post, elbo_post = self.compute_q_elbo(resp, respPair, self.weight_mean(q),
                                                            self.weight_mean(q_lat), self.gpmodels, self.M, snr='saved',
                                                            post=False)
                print("ELBO_reduction: "+str(((q_post + elbo_post) - (q_bas + elbo_bas)).item()))
                if (torch.isclose(q_bas + elbo_bas, q_post + elbo_post, rtol=1e-5) and i > 0) or i==10:# or reparam:
                    if not reparam:
                        reparam = True
                    else:
                        break
                q_bas = q_post
                elbo_bas = elbo_post
                i = i + 1
        #q_obs = torch.sum(torch.sum(q_)).item()/len(y_trains)
        return resp, respPair, q, q_lat, snr, y_trains_w, reallocate

    def estimate_q_first(self, M, x_trains, y_trains, y_trains_w, resp, respPair, q_, q_lat_, snr_, startPi, transPi, reallocate_=False, reparam=False):
        """ Smart and complex computation of new group assignation and compare with the last iteration then decides if generate group or not.
            First, it computes if a reallocation of the samples is better, if not, proposes a new birth group and computes the ELBO to measure the goodness.
        """
        y_trains_w_ = torch.clone(y_trains_w)
        if torch.mean(q_) == 0.0:
            snr_ = torch.zeros(y_trains.shape[0], M, self.n_outputs)
            for ld in range(self.n_outputs):
                gp = self.create_gp_default()
                q_[:, 0, ld], q_lat_[:, 0, ld]= gp.full_pass_weighted(x_trains, y_trains[:, :, [ld]], resp[:, 0],
                                                     snr=self.snr_norm[:, ld])
                snr_[:, 0, ld] = self.compute_snr(y_trains[:, :, ld], gp)
                self.gpmodels[ld][0] = gp
        reallocate = False

        q_simple = torch.clone(q_)

        indexes_ = []
        for m in range(M):
            indexes_.append(torch.tensor(self.gpmodels[0][m].indexes).long())
            if indexes_[m].shape[0] == 0:
                indexes_[m] = torch.where(resp[:, m] == self.cond_cuda(torch.tensor(1.0)))[0].long()
        f_ind_old = torch.clone(self.f_ind_old)

        # Compute q with best index of the old model.
        snr_temp = torch.zeros(y_trains.shape[0], M, self.n_outputs)
        for ld in range(self.n_outputs):
            print("\n-----------Lead " + str(ld + 1) + "-----------")
            for m in range(M):
                gp = self.gpmodel_deepcopy(self.gpmodels[ld][m])
                if gp.fitted:
                    gp.reinit_LDS(save_last=False)
                    gp.reinit_GP(save_last=False)
                if len(indexes_[m]) > 0:
                    gp.include_weighted_sample(0, x_trains[f_ind_old[m]], x_trains[f_ind_old[m]], y_trains_w[f_ind_old[m],:,[ld]], h=1.0)
                q_simple[:, m, ld] = gp.compute_sq_err_all(x_trains, y_trains_w[:,:, [ld]])
                snr_temp[:, m, ld] = self.compute_snr(y_trains[:,:,ld], gp)

        if M > 1:
            q_aux = torch.clone(q_simple)
            snr_aux = torch.clone(snr_)
            if torch.sum(resp, dim=0)[-1] == 0:
                q_aux[:, -1, :] = torch.zeros(q_aux[:, -1, :].shape) + torch.min(q_aux) * 2.0
                snr_aux[:, -1, :] = torch.zeros(snr_aux[:, -1, :].shape) + torch.min(snr_aux) * 2.0
            q_norm, _ = self.LogLik(self.weight_mean(q_aux, snr_aux))
            alpha, margprob = self.forward(startPi, transPi, q_norm)
            beta = self.backward(transPi, q_norm, margprob)
            resplog_temp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
            respPairlog_temp, _ = self.LogLik(self.coupled_state_coef(alpha, beta, transPi, q_norm, margprob), axis=1)
            resp_temp = torch.exp(resplog_temp)
            respPair_temp = torch.exp(respPairlog_temp)
            #resp_temp, respPair_temp = self.refill_resp(resp_temp, respPair_temp)

            resp_per_group_temp = torch.sum(resp_temp, axis=0)
            reorder = torch.argsort(resp_per_group_temp, descending=True)
            resp_temp = resp_temp[:, reorder]

            #First, try to reallocate beats, if this not works then generate new group.
            q = torch.clone(q_)
            q_lat = torch.clone(q_lat_)
            gpmodels_temp = [[] for _ in range(self.n_outputs)]
            for ld in range(self.n_outputs):
                print("\n-----------Lead " + str(ld + 1) + "-----------")
                for m in range(M):
                    print("\n   -----------Model " + str(m + 1) + "-----------")
                    gp = self.gpmodel_deepcopy(self.gpmodels[ld][reorder[m]])
                    if not torch.equal(resp[:, reorder[m]].long(), resp_temp[:, m].long()):
                        if gp.fitted:
                            gp.reinit_LDS(save_last=False)
                            gp.reinit_GP(save_last=False)
                        q[:, m, ld], q_lat[:, m, ld] = gp.full_pass_weighted(x_trains, y_trains_w[:,:,[ld]],
                                                            resp_temp[:, m], q=q[:,reorder[m], ld],
                                                            q_lat=q_lat[:, reorder[m], ld],
                                                            snr=self.snr_norm[:,ld])
                        snr_aux[:, m, ld] = self.compute_snr(y_trains_w[:, :, ld], gp)
                    else:
                        q[:, m, ld] = torch.clone(q_[:, reorder[m], ld])
                        snr_aux[:, m, ld] = torch.clone(snr_[:, reorder[m], ld])
                    gpmodels_temp[ld].append(gp)

            # Recompute resp
            q_norm, _ = self.LogLik(self.weight_mean(q, snr_aux))
            alpha, margprob = self.forward(startPi, transPi, q_norm)
            beta = self.backward(transPi, q_norm, margprob)
            resplog_temp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
            respPairlog_temp, _ = self.LogLik(self.coupled_state_coef(alpha, beta, transPi, q_norm, margprob), axis=1)
            resp_temp = torch.exp(resplog_temp)
            respPair_temp = torch.exp(respPairlog_temp)
            #resp_temp, respPair_temp = self.refill_resp(resp_temp, respPair_temp)

            new_indexes = torch.where(torch.sum(np.abs(resp - resp_temp), dim=1) > 1.0)[0]
            print(">>> Prev -------")
            q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q_, snr_), self.weight_mean(q_lat_, snr_),
                                                  self.gpmodels, M, snr=snr_, post=False)
            print(">>> Post -------")
            q_bas_post, elbo_post = self.compute_q_elbo(resp_temp, respPair_temp, self.weight_mean(q, snr_aux), self.weight_mean(q_lat, snr_aux),
                                                        gpmodels_temp, M, snr=snr_aux, post=False)
            update_snr = True
            if torch.all(torch.sum(resp_temp, dim=0)[:-1] >= 1.0):
                if q_bas < q_bas_post:
                    if not q_bas + elbo_bas < q_bas_post + elbo_post:
                        print("Possibly better q_obs but worse elbo.")
                if q_bas + elbo_bas < q_bas_post + elbo_post and q_bas != q_bas_post:
                    print("Reallocating beats into existing groups.")
                    reallocate = True
                    self.gpmodels = gpmodels_temp
                    self.f_ind_old = f_ind_old[reorder]
                    if update_snr:
                        self.snr_norm = self.normalize_snr(snr_aux)
                    else:
                        snr_aux = snr_
                    return resp_temp, respPair_temp, q, q_lat, snr_aux, y_trains_w_, reallocate
                else:
                    print("Not reallocating, trying to generate new group.")
            else:
                print("Bad estimation")
        #f_ind_new_potential = torch.argsort(self.weight_mean(q_simple)[torch.where(resp == 1.0)])
        q_sim_s = self.weight_mean(q_simple)[torch.where(resp == 1.0)]
        q_sim_s = (q_sim_s - torch.max(q_sim_s)) / (torch.max(q_sim_s) - torch.min(q_sim_s))
        q_s = self.weight_mean(q_)[torch.where(resp == 1.0)]
        q_s = (q_s - torch.max(q_s)) / (torch.max(q_s) - torch.min(q_s))
        q_lat_s = self.weight_mean(q_lat_)[torch.where(resp == 1.0)]
        q_lat_s = (q_lat_s - torch.max(q_lat_s)) / (torch.max(q_lat_s) - torch.min(q_lat_s))
        f_ind_new_potential = torch.argsort(q_sim_s)
        q_rank = q_sim_s
        potential_weight = torch.zeros(f_ind_new_potential.shape[0])
        potential_ind = {}
        potential_q = torch.zeros(f_ind_new_potential.shape[0])
        for j, ind in enumerate(f_ind_new_potential):
            potential_ind[ind.item()] = torch.where(torch.isclose(q_rank, q_rank[ind], rtol=0.01))[0]
            potential_weight[ind] = torch.where(torch.isclose(q_rank, q_rank[ind], rtol=0.01))[0].shape[0]
            potential_q[ind] = torch.sum(q_rank[potential_ind[ind.item()]])
        n_steps = self.n_explore_steps
        f_ind_new_potential_def = torch.zeros(n_steps).long()
        last_indexes = torch.tensor([-1])
        j_ = 0
        for j, f_ind_new in enumerate(f_ind_new_potential):
            if j_ == n_steps // 2.0:
                break
            m_chosen = -1
            for m in range(M - 1):
                if f_ind_new in indexes_[m]:
                    m_chosen = m
                    break
            if m_chosen == -1:
                m_chosen = torch.argmax(resp[f_ind_new])
            f_ind_old_chosen = f_ind_old[m_chosen]
            if f_ind_new != f_ind_old_chosen:
                for l_ in last_indexes:
                    if not l_ in potential_ind[f_ind_new.item()]:
                        last_indexes = potential_ind[f_ind_new.item()]
                        f_ind_new_potential_def[j_] = f_ind_new
                        j_ = j_ + 1
                        break

        q_aux = torch.clone(q_simple)
        f_ind_new_q = torch.argsort(q_s + q_lat_s)
        last_indexes = torch.tensor([-1])
        j_ = int(n_steps // 2.0)
        for j, f_ind_new in enumerate(f_ind_new_q):
            if j_ == n_steps:
                break
            m_chosen = -1
            for m in range(M - 1):
                if f_ind_new in indexes_[m]:
                    m_chosen = m
                    break
            if m_chosen == -1:
                m_chosen = torch.argmax(resp[f_ind_new])
            f_ind_old_chosen = f_ind_old[m_chosen]
            if f_ind_new != f_ind_old_chosen:
                for l_ in last_indexes:
                    if not l_ in potential_ind[f_ind_new.item()]:
                        last_indexes = potential_ind[f_ind_new.item()]
                        f_ind_new_potential_def[j_] = f_ind_new
                        j_ = j_ + 1
                        break
        #ord_ = torch.argsort(potential_q[f_ind_new_potential_def[:n_steps]])#, descending=True)
        #f_ind_new_potential_def[:n_steps] = f_ind_new_potential_def[ord_]
        # Adding 5 possible potential indexes not by q.
        #f_ind_new_potential_def = torch.concatenate([f_ind_new_potential_def,f_ind_new_q_def])
        #n_steps = n_steps + 5
        step = 0
        last_indexes = torch.tensor([-1])
        q = torch.clone(q_aux)
        q_lat = torch.clone(q_lat_)
        snr_aux = torch.clone(snr_temp)
        resp_, respPair_, q_def, q_lat_def, snr_aux_def = self.new_group(resp, respPair, torch.clone(q),
                                                                         torch.clone(q_lat), torch.clone(snr_aux))
        _, _, q__def, q_lat__def, snr__def = self.new_group(resp, respPair, torch.clone(q_),
                                                                        torch.clone(q_lat_), torch.clone(snr_))
        M = M + 1
        f_ind_old = torch.zeros(M, device=resp.device).long()
        f_ind_old[:self.f_ind_old.shape[0]] = torch.clone(self.f_ind_old)
        for j, f_ind_new in enumerate(f_ind_new_potential_def):
            if step == n_steps:
                break
            m_chosen = -1
            for m in range(M - 1):
                if f_ind_new in indexes_[m]:
                    m_chosen = m
                    break
            if m_chosen == -1:
                m_chosen = torch.argmax(resp[f_ind_new])
            f_ind_old_chosen = f_ind_old[m_chosen]
            ld_belong = torch.argmax(self.snr_norm[f_ind_new])
            if f_ind_new != f_ind_old_chosen:
                some_new_index = False
                for l_ in last_indexes:
                    if not l_ in potential_ind[f_ind_new.item()]:
                        some_new_index = True
                if some_new_index:
                    q, q_lat, snr_aux = torch.clone(q_def), torch.clone(q_lat_def), torch.clone(snr_aux_def)
                    q__, q_lat__, snr__ = torch.clone(q__def), torch.clone(q_lat__def), torch.clone(snr__def)
                    last_indexes = potential_ind[f_ind_new.item()]
                    print("Step "+str(step+1)+"/"+str(n_steps)+ "- Trying to divide: " + str(m_chosen) + " with beat " + str(f_ind_new.item()))
                    step = step + 1
                    for ld in range(self.n_outputs):
                        gp = self.gpmodel_deepcopy(self.gpmodels[ld][m_chosen])
                        if gp.fitted:
                            gp.reinit_LDS(save_last=False)
                            gp.reinit_GP(save_last=False)
                        gp.include_weighted_sample(0, x_trains[f_ind_new], x_trains[f_ind_new], y_trains_w[f_ind_new,:,[ld]], h=1.0)
                        q[:, -1, ld] = gp.compute_sq_err_all(x_trains, y_trains_w[:,:,[ld]])
                        snr_aux[:, -1, ld] = self.compute_snr(y_trains_w[:, :, ld], gp)
                    # Compute resp
                    q_mean = self.weight_mean(q, snr_aux)
                    q_norm, _ = self.LogLik(q_mean)
                    alpha, margprob = self.forward(startPi, transPi, q_norm)
                    beta = self.backward(transPi, q_norm, margprob)
                    resplog_temp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
                    respPairlog_temp, _ = self.LogLik(
                        self.coupled_state_coef(alpha, beta, transPi, q_norm, margprob), axis=1)
                    resp_temp = torch.exp(resplog_temp)
                    respPair_temp = torch.exp(respPairlog_temp)
                    #resp_temp, respPair_temp = self.refill_resp(resp_temp, respPair_temp)

                    # Reallocating resp to preserve order.
                    resp_per_group_temp = torch.sum(resp_temp, axis=0)
                    reorder = torch.argsort(resp_per_group_temp, descending=True)
                    resp_temp = resp_temp[:, reorder]

                    # Compute chosen model conditioned on new resp
                    # Update all models conditioned on new resp if it has changed
                    gpmodels_temp = [[] for _ in range(self.n_outputs)]
                    for ld in range(self.n_outputs):
                        for m in range(M):
                            if reorder[m] == M - 1:
                                # gp = self.gpmodel_deepcopy(self.gpmodels[ld][m_chosen])
                                # If uncommented then new GP is used for a new model, more expensive but official model.
                                gp = self.create_gp_default()
                                if gp.fitted:
                                    gp.reinit_LDS(save_last=False)
                                    gp.reinit_GP(save_last=False)
                                q[:, m, ld], q_lat[:, m, ld] = gp.full_pass_weighted(x_trains,
                                                                                     y_trains_w[:, :, [ld]],
                                                                                     resp_temp[:, m],
                                                                                     q=q__[:, reorder[m], ld],
                                                                                     q_lat=q_lat__[:, reorder[m], ld],
                                                                                     snr=self.snr_norm[:, ld])
                                snr_aux[:, m, ld] = self.compute_snr(y_trains_w[:, :, ld], gp)
                            else:
                                gp = self.gpmodel_deepcopy(self.gpmodels[ld][reorder[m]])
                                if not torch.equal(resp[:, reorder[m]].long(), resp_temp[:, m].long()):
                                    if gp.fitted:
                                        gp.reinit_LDS(save_last=False)
                                        gp.reinit_GP(save_last=False)
                                    q[:, m, ld], q_lat[:, m, ld] = gp.full_pass_weighted(x_trains,
                                                                                         y_trains_w[:, :, [ld]],
                                                                                         resp_temp[:, m],
                                                                                         q=q__[:, reorder[m], ld],
                                                                                         q_lat=q_lat__[:, reorder[m], ld],
                                                                                         snr=self.snr_norm[:, ld])
                                    snr_aux[:, m, ld] = self.compute_snr(y_trains_w[:, :, ld], gp)
                                else:
                                    q[:, m, ld] = torch.clone(q__[:, reorder[m], ld])
                                    q_lat[:, m, ld] = torch.clone(q_lat__[:, reorder[m], ld])
                                    snr_aux[:, m, ld] = torch.clone(snr__[:, reorder[m], ld])
                            gpmodels_temp[ld].append(gp)

                    # Recompute resp
                    q_mean = self.weight_mean(q, snr_aux)
                    q_norm, _ = self.LogLik(q_mean)
                    alpha, margprob = self.forward(startPi, transPi, q_norm)
                    beta = self.backward(transPi, q_norm, margprob)
                    resplog_temp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
                    respPairlog_temp, _ = self.LogLik(
                        self.coupled_state_coef(alpha, beta, transPi, q_norm, margprob), axis=1)
                    resp_temp = torch.exp(resplog_temp)
                    respPair_temp = torch.exp(respPairlog_temp)
                    #resp_temp, respPair_temp = self.refill_resp(resp_temp, respPair_temp)
                    q_bas, elbo_bas = self.compute_q_elbo(resp_temp, respPair_temp, self.weight_mean(q, snr_aux),
                                                            self.weight_mean(q_lat, snr_aux), gpmodels_temp, M,
                                                            snr=snr_aux, post=True)
                    i__ = 0
                    while True:
                        resp_temp, respPair_temp, q, q_lat, snr_aux, y_trains_w, gpmodels_temp = self.estimate_q_all(M,
                                                                                                                 x_trains=x_trains,
                                                                                                                 y_trains=y_trains,
                                                                                                                 y_trains_w=y_trains_w,
                                                                                                                 resp=resp_temp,
                                                                                                                 respPair=respPair_temp,
                                                                                                                 q_=q,
                                                                                                                 q_lat_=q_lat,
                                                                                                                 snr_=snr_aux,
                                                                                                                 startPi=startPi,
                                                                                                                 transPi=transPi,
                                                                                                                 gpmodels=gpmodels_temp,
                                                                                                                 reparam=reparam)
                        q_post, elbo_post = self.compute_q_elbo(resp_temp, respPair_temp, self.weight_mean(q, snr_aux),
                                                                self.weight_mean(q_lat, snr_aux), gpmodels_temp, M, snr=snr_aux, post=True)
                        print("ELBO_reduction: " + str(((q_post + elbo_post) - (q_bas + elbo_bas)).item()))
                        if (torch.isclose(q_bas + elbo_bas, q_post + elbo_post,
                                          rtol=1e-5) and i__ > 0) or i__ == 10:  # or reparam:
                                break
                        q_bas = q_post
                        elbo_bas = elbo_post
                        i__ = i__ + 1


                    #new_indexes = torch.where(torch.sum(np.abs(resp - resp_temp), dim=1) > 1.0)[0]
                    print(">>> Prev -------")
                    q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q_, snr_), self.weight_mean(q_lat_, snr_), self.gpmodels, self.M, snr=snr_, post=False)
                    print(">>> Post -------")
                    q_bas_post, elbo_post = self.compute_q_elbo(resp_temp, respPair_temp, self.weight_mean(q, snr_aux), self.weight_mean(q_lat, snr_aux), gpmodels_temp, M, snr=snr_aux, post=True)

                    update_snr = True

                    if torch.all(torch.sum(resp_temp, dim=0) >= 1.0):
                        if q_bas < q_bas_post:
                            if not q_bas + elbo_bas < q_bas_post + elbo_post:
                                print("Possibly better q_obs but worse elbo.")
                        if q_bas + elbo_bas < q_bas_post + elbo_post:
                            print("Chosen to divide: "+str(m_chosen)+" with beat "+str(f_ind_new.item()))
                            self.gpmodels = gpmodels_temp
                            pos_new = torch.where(reorder == M - 1)[0].long()
                            indexes = torch.where(resp_temp[:, pos_new] == 1.0)[0]
                            if len(indexes) > 0:
                                f_ind_old[-1] = indexes[torch.argmax(self.weight_mean(q, snr_aux)[indexes, pos_new]).long()]
                            else:
                                f_ind_old[-1] = f_ind_new
                            self.f_ind_old = torch.clone(f_ind_old[reorder])
                            if update_snr:
                                self.snr_norm = self.normalize_snr(snr_aux)
                            else:
                                snr_aux = snr_
                            return resp_temp, respPair_temp, q, q_lat, snr_aux, y_trains_w, reallocate
                    else:
                        print("Bad estimation")
        reallocate = True
        return resp, respPair, q_, q_lat_, snr_, y_trains_w_, reallocate


    def compute_q_elbo(self, resp, respPair, q, q_lat, gpmodels, M, new_indexes=None, snr=None, post=False,
                       one_sample=False, verb=True):
        """ Method to compute ELBO terms.
        """
        n_points = self.x_basis[0].shape[0]
        q_bas = torch.sum(q[torch.where(resp.int() > 0.99)]) * self.static_factor
        elbo_latent = torch.sum(q_lat[torch.where(resp.int() > 0.99)]) * self.dynamic_factor
        if post:
            elbo_bas = self.elbo_Linears(resp, respPair, post=post) * n_points
        else:
            elbo_bas = self.elbo_Linears(resp, respPair) * n_points
        elbo_bas_LDS = 0
        if snr is None:
            frac = torch.ones(self.n_outputs, device=resp.device()) / self.n_outputs#  / self.M
        elif snr == 'saved':
            frac = torch.sum(self.snr_norm, dim=0)
            frac = frac / torch.sum(frac) #* self.n_outputs# / self.M#
        else:
            frac = torch.sum(torch.softmax(torch.max(snr, dim=1)[0], dim=1), dim=0)
            frac = frac / torch.sum(frac) #* self.n_outputs# * self.M#
        for i in range(self.n_outputs):
            elbo_bas_LDS = elbo_bas_LDS + self.full_LDS_elbo(gpmodels[i], torch.sum(resp, dim=0), one_sample=one_sample) * frac[i]

        #elbo_bas = elbo_bas + elbo_latent
        #elbo_bas = 0
        if verb:
            print("Sum resp_temp: " + str(torch.sum(resp, dim=0).int().numpy()))
            print(f"Q_em: {q_bas.item():.2f}, Q_lat: {elbo_latent.item():.2f}, Elbo_linear: {elbo_bas:.2f}, Elbo_LDS: {elbo_bas_LDS.item():.2f}")
        elbo_bas = elbo_bas + elbo_bas_LDS + elbo_latent
        return q_bas, elbo_bas


    def full_LDS_elbo(self, gpmodels, sum_resp, one_sample=False):
        """ Method to accumulate LDS ELBO terms..
        """
        elb = torch.zeros(1)
        M_ = 0
        if one_sample:
            frac = sum_resp / torch.sum(sum_resp)
        else:
            frac = sum_resp / sum_resp
        for i in sum_resp:
            if i > 0:
                M_ = M_ + 1
        gp_temp = gpmodels if one_sample else gpmodels
        for i, gp in enumerate(gp_temp):
            if sum_resp[i] > 0:
                if sum_resp[i] < self.free_deg_MNIV:
                    #elb = elb + gp.return_LDS_param_likelihood(first=True)
                    if one_sample:
                        elb = elb + gp.return_LDS_param_likelihood(first=False) * frac[i] * 1.0
                    else:
                        elb = elb + gp.return_LDS_param_likelihood(first=True) * frac[i] * 1.0
                else:
                    elb = elb + gp.return_LDS_param_likelihood() * frac[i]
        if one_sample:
            return elb / M_
        else:
            return elb / np.min([M_, self.M]) #

    def redefine_default(self, x_trains, y_trains, resp):
        """ Method to compute Sigma and Gamma from a batch of examples and assign it to initial values.
        """
        #if self.estimation_limit_def is None:
        #    n_f = y_trains.shape[0] - 1
        #else:
        #    n_f = min(self.estimation_limit_def, y_trains.shape[0] - 1) if (
        #        self.estimation_limit != np.PINF) else y_trains.shape[0] - 1
        #samples = y_trains[:n_f][:, :, 0].T
        #samples_ = y_trains[1:n_f+1][:, :, 0].T
        #var_y_y = torch.mean(torch.diag(torch.linalg.multi_dot([(samples - torch.mean(samples, dim=1)[:,np.newaxis]), (samples - torch.mean(samples, dim=1)[:,np.newaxis]).T])) / n_f)# * 0.15
        #var_y_y_ = torch.mean(torch.diag(torch.linalg.multi_dot([(samples_ - samples), (samples_ - samples).T])) / n_f)# * 0.15
        #kernel, ini_sigma, ini_gamma, ini_outputscale, bound_sigma, bound_gamma, bound_noise_warp, annealing, method_compute_warp, \
        #    model_type, recursive_warp, warp_updating, inducing_points, estimation_limit, free_deg_MNIV = self.get_default_options()
        print("Redefining default LDS priors.")
        if self.estimation_limit_def is None:
            n_f = y_trains.shape[0] - 1
        else:
            n_f = min(self.estimation_limit_def, y_trains.shape[0] - 1) if (
                self.estimation_limit != np.PINF) else y_trains.shape[0] - 1
        # gp = self.gpmodels[0][0]
        # q__ = torch.zeros(y_trains.shape[0])
        # for j, y in enumerate(y_trains[:,:,[0]]):
        #      q__[j] = gp.log_sq_error(x_trains[j], y)
        # #q__, q_lat__= gp.full_pass_weighted(x_trains, y_trains[:, :, [0]], resp[:, 0], snr=self.snr_norm[:, 0])
        # f_ind = torch.where(q__ == torch.median(q__))[0].item()
        # #f_ind = 0
        # gp = self.create_gp_default()
        # gp.include_weighted_sample(0, x_trains[f_ind], x_trains[f_ind], y_trains[f_ind, :, [0]], h=1.0)
        # q__ = gp.compute_sq_err_all(x_trains, y_trains[:, :, [0]])
        # n_samp = 200
        # selected_beats = torch.argsort(q__, descending=True)[:n_samp]
        # resp_red = torch.zeros(resp.shape)
        # resp_red[selected_beats,0] = torch.ones(n_samp)
        # gp = self.create_gp_default()
        # gp.full_pass_weighted(x_trains, y_trains[:, :, [0]], resp_red[:, 0], snr=self.snr_norm[:, 0])
        # ind_ = -1
        # #noise = (torch.mean(torch.diag(gp.Sigma[ind_])) + torch.mean(torch.diag(gp.Gamma[ind_]))) / 2.0
        # var_y_y_ = torch.max(torch.diag(gp.Gamma[ind_]))
        # var_y_y = torch.max(torch.diag(gp.Sigma[ind_]))
        samples = y_trains[:n_f][:, :, 0].T
        samples_ = y_trains[1:n_f+1][:, :, 0].T
        var_y_y = torch.median(torch.diag(torch.linalg.multi_dot([(samples - torch.mean(samples, dim=1)[:,np.newaxis]), (samples - torch.mean(samples, dim=1)[:,np.newaxis]).T])) / n_f)# * 0.15
        var_y_y_ = torch.median(torch.diag(torch.linalg.multi_dot([(samples_ - samples), (samples_ - samples).T])) / n_f)# * 0.15
        kernel, ini_sigma, ini_gamma, ini_outputscale, bound_sigma, bound_gamma, bound_noise_warp, annealing, method_compute_warp, \
            model_type, recursive_warp, warp_updating, inducing_points, estimation_limit, free_deg_MNIV = self.get_default_options()
        # Good results using 0.012
        # Good results using 0.02
        # Good results using 0.01.
        # Good results using 0.018.
        # ini_Sigma = self.cond_to_torch(np.max([var_y_y, var_y_y_])) * 2.0
        # ini_Gamma = self.cond_to_torch(np.max([var_y_y, var_y_y_])) * 2.0
        ini_Sigma = var_y_y * 0.050
        ini_Gamma = var_y_y * 0.060
        #ini_Gamma = torch.sqrt(self.cond_to_torch(np.min([np.max([var_y_y_,var_y_y * 1.2]), var_y_y * 2.0])) * 0.5)
        #ini_Gamma = var_y_y * 0.012
        #ini_Gamma = self.cond_to_torch(np.min([np.max([var_y_y_,var_y_y * 1.2]), var_y_y * 2.5])) * 2.0
        #ini_Gamma = var_y_y_ * 1.0
        #ini_Sigma = self.cond_to_torch(10.0)
        #ini_Gamma = self.cond_to_torch(25.0)
        # if ini_Sigma > 200.0:
        #      ini_Sigma = ini_Sigma * 0.3
        #      ini_Gamma = ini_Gamma * 0.3
        # ini_Sigma = self.cond_to_torch(np.max([ini_Sigma, 10.0]))
        # ini_Gamma = self.cond_to_torch(np.max(([ini_Gamma, 12.0])))
        #ini_Gamma = self.cond_to_torch(np.max([var_y_y_, var_y_y * 1.5]))
        #ini_Gamma = var_y_y_ * 1.5
        bound_sigma = (ini_Sigma.item() * 1e-7, 1.0)
        bound_gamma = (ini_Gamma.item() * 1e-7, 1.0)
        #bound_sigma = (0.1, 20.0)
        #bound_gamma = (0.1, 20.0)
        print("-----------Reestimated ------------", flush=True)
        print("Sigma: ", ini_Sigma)
        print("Gamma: ", ini_Gamma)
        print("-----------------------------", flush=True)
        kernel = (ConstantKernel(ini_outputscale, (ini_outputscale, ini_outputscale*5.0)) *
                  RBF(self.ini_lengthscale[0], self.bound_lengthscale[0]) + WhiteKernel(bound_sigma[0], bound_sigma))
        self.set_default_options(kernel, ini_Sigma, ini_Gamma, ini_outputscale, bound_sigma, bound_gamma,
                                 bound_noise_warp, annealing, method_compute_warp,
                                 model_type, recursive_warp, warp_updating, inducing_points, estimation_limit, free_deg_MNIV)
        for ld in range(self.n_outputs):
            for m in range(len(self.gpmodels[ld])):
                self.gpmodels[ld][m] = self.create_gp_default()

    def include_sample(self, x_train, y, with_warp=True, force_model=None, minibatch=0, classify=False):
        """
        Include the sample received and restimate all the parameters of the
        full model, including the state-space and the GPS.
        Parameters
        ----------
        y : array sample of shape (s_samples)

        Returns
        -------
        self : returns an instance of self.
        """

        # Adding one sample
        one_sample = True
        t = self.T
        if not classify:
            self.T = self.T + 1
            self.snr_norm = torch.ones(self.T,self.n_outputs)
        M = self.M
        y = self.cond_cuda(self.cond_to_torch(y))
        x_train = self.cond_cuda(self.cond_to_torch(x_train))
        if minibatch == 0 and self.batch is not None:
            minibatch = self.batch
        if minibatch >= t:
            minibatch = 0
        # Compute warp once
        y_mod = []
        liks = np.array([0.0] * (M + 1))
        for m in range(M + 1):
            y_ = self.y.copy()[-1 * minibatch:]
            y_mod.append(y_)
        if not classify:
            self.y.append(y)
            self.x_train.append(x_train)
        if with_warp:
            if t > 0:
                for ld in range(self.n_outputs):
                    y_w, x_w, liks = self.compute_warp_y(x_train, y[:,[ld]], self.method_compute_warp, force_model=force_model, ld=ld)
                    for m in range(M + 1):
                        #TODO: here incorporate n_outputs dimension to y to allow the iteration over the rest of the model.
                        y_mod[m].append(self.cond_cuda(y_w[m]))
                    self.y_w.append(y_w)
                    self.x_w.append(x_w)
                    self.liks.append(liks)
            else:
                for m in range(M + 1):
                    y_mod[m].append(self.cond_cuda(y))
                self.y_w.append([self.cond_cuda(y)] * (M + 1))
                self.x_w.append([self.cond_cuda(torch.atleast_2d(torch.zeros(y.shape[0])).T)] * (M + 1))
                self.liks.append(liks)
        else:
            for m in range(M + 1):
                y_mod[m].append(self.cond_cuda(y))
        elbo = 0.0
        q_all = 0.0
        q_aux = torch.zeros(self.T, self.M+1, self.n_outputs) - np.inf
        q_lat = torch.zeros(self.T, self.M+1, self.n_outputs)
        if t > 0:
            q_aux[:-1, :self.q[-1].shape[1], :] = torch.clone(self.q[-1])
        for ld in range(self.n_outputs):
            for m, gp in enumerate(self.gpmodels[ld]):
                q_lat[:, m, ld] = gp.compute_q_lat_all(torch.from_numpy(np.array(self.x_train)), h_ini=1.0)
                q_aux[-1, m, ld] = gp.log_sq_error(torch.from_numpy(np.array(self.x_train[-1])), y_mod[m][-1], i=-1)
                #q_aux[[-1], m, ld] = self.estimate_new(t, gp, self.x_train[-1], y_mod[m][-1], h=1.0)
        if t > 0:
            resp, resp_log, respPair, respPair_log = self.variational_local_terms(q_aux, self.transTheta, self.startTheta)
            q_all, elbo = self.compute_q_elbo(resp[:-1,:-1], respPair[:-1,:-1,:-1], self.weight_mean(q_aux)[:-1,:-1],
                                                              self.weight_mean(q_lat)[:-1,:-1],
                                                              self.gpmodels, self.M, snr='saved', post=False,
                                                              one_sample=True, verb=self.verbose)
        if t > 0:
            # Define order
            q_ord = torch.argsort(self.weight_mean(q_aux)[-1,:-1], descending=True)
            #m = q_ord[-1].item()
            m = q_ord[0].item()
            q_prev = torch.clone(q_aux)
            q_lat_prev = torch.clone(q_lat)
            # Compute first birth cost
            for ld in range(self.n_outputs):
                prov_gp = self.gpmodel_deepcopy(self.gpmodels[ld][m])
                prov_gp.reinit_GP(save_last=False)
                prov_gp.reinit_LDS(save_last=False)
                q_prev[[-1], -1, ld] = self.estimate_new(t, prov_gp, self.x_train[-1], y_mod[-1][-1], h=1.0)
                prov_gp.include_weighted_sample(t, self.x_train[-1], self.x_train[-1], y_mod[-1][-1], 1.0)
                self.gpmodels[ld].append(prov_gp)
                #self.M = M + 1
                q_lat_prev[:, -1, ld] = prov_gp.compute_q_lat_all(torch.from_numpy(np.array(self.x_train)), h_ini=1.0)
            resp_prev, resp_prev_log, respPair_prev, respPair_prev_log = self.variational_local_terms(q_prev, self.transTheta, self.startTheta, liks)
            q_prev_post, elbo_prev_post = self.compute_q_elbo(resp_prev, respPair_prev, self.weight_mean(q_prev), self.weight_mean(q_lat_prev),
                                                  self.gpmodels, self.M, snr='saved', one_sample=True, post=True, verb=self.verbose)
            elbo_prev_post = elbo_prev_post - elbo#/ np.log(self.T + 1)
            q_prev_post = q_prev_post - q_all
            for ld in range(self.n_outputs):
                self.gpmodels[ld].pop()
                self.M = M
            if torch.argmax(q_prev[-1]) == self.M:
                q_post = torch.clone(q_aux)
                q_lat_post = torch.clone(q_lat)
                for m in q_ord:
                    saved_gps = [self.gpmodels[ld][m] for ld in range(self.n_outputs)]
                    for ld in range(self.n_outputs):
                        post_gp = self.gpmodel_deepcopy(self.gpmodels[ld][m])
                        q_post[[-1], m, ld] = self.estimate_new(t, post_gp, self.x_train[-1], y_mod[m][-1], h=1.0)
                        post_gp.include_weighted_sample(t, self.x_train[-1], self.x_train[-1], y_mod[m][-1], 1.0)
                        self.gpmodels[ld][m] = post_gp
                        post_gp.backwards_pair(1.0)
                        post_gp.bayesian_new_params(1.0)
                        q_lat_post[:, m, ld] = post_gp.compute_q_lat_all(torch.from_numpy(np.array(self.x_train)), h_ini=1.0)
                        #q_post[[-1], m, ld] = self.estimate_new(t, post_gp, self.x_train[-1], y_mod[m][-1], h=1.0)
                    resp_post, resp_post_log, respPair_post, respPair_post_log = self.variational_local_terms(q_post, self.transTheta, self.startTheta, liks)
                    q_bas_post, elbo_bas_post = self.compute_q_elbo(resp_post[:,:-1], respPair_post[:,:-1,:-1], self.weight_mean(q_post)[:,:-1],
                                                          self.weight_mean(q_lat_post)[:,:-1],
                                                          self.gpmodels, self.M, snr='saved', post=False, one_sample=True, verb=self.verbose)
                    elbo_bas_post = elbo_bas_post - elbo#/ np.log(self.T + 1)
                    q_bas_post = q_bas_post - q_all

                    if q_bas_post + elbo_bas_post > q_prev_post + elbo_prev_post:
                        resp, resplog, respPair, respPairlog = self.variational_local_terms(q_post, self.transTheta, self.startTheta, liks)
                        q_chos = q_post
                        q_lat_chos = q_lat_post
                        for ld in range(self.n_outputs):
                            self.gpmodels[ld][m] = saved_gps[ld]
                        break
                    else:
                        for ld in range(self.n_outputs):
                            self.gpmodels[ld][m] = saved_gps[ld]
                        q_chos = q_prev
                        q_lat_chos = q_lat_prev
                        resp, resplog, respPair, respPairlog = resp_prev, resp_prev_log, respPair_prev, respPair_prev_log
            else:
                q_chos = q_aux
                q_lat_chos = q_lat
                resp, resplog, respPair, respPairlog = self.variational_local_terms(q_chos, self.transTheta, self.startTheta, liks)
        else:
            q_chos = q_aux
            q_lat_chos = q_lat
            resp, resplog, respPair, respPairlog = self.variational_local_terms(q_aux, self.transTheta, self.startTheta, liks)

        if len(resp.shape) == 1:
            resp_mod = self.cond_to_numpy(self.cond_cpu(resp))
            resp_modlog = self.cond_to_numpy(self.cond_cpu(resplog))
        else:
            resp_mod = self.cond_to_numpy(self.cond_cpu(resp[-1]))
            resp_modlog = self.cond_to_numpy(self.cond_cpu(resplog[-1]))

        if classify:
            return q_chos[:-1], resp_mod[:-1], liks[:-1]

        # Normalize when responsability is equally shared
        if sum(np.isclose(resp_mod, max(resp_mod), rtol=1e-2)) > 1:
            h_argmax = np.nanargmax(resp_mod)
            resp_mod[:] = 0.0
            resp_mod[h_argmax] = 1.0

        model = np.argmax(resp_mod)
        if self.max_models is not None and model >= self.max_models:
            force_model = np.argmax(resp_modlog[:-1])
            model = force_model

        # When converged apply all transformations to real models.
        if not force_model is None:
            resp_mod[:] = 0.0
            resp_mod[force_model] = 1.0
            model = np.argmax(resp_mod)
            # resp = self.normalize_log(resp_mod)

        # Birth of new model
        birth = False
        if model == self.M:
            birth = True
        if birth:
            print("Birth of new model: ", self.M + 1)
            self.M = self.M + 1
            M = self.M
            y_mod.append(self.y.copy())
            for ld in range(self.n_outputs):
                self.gpmodels[ld].append(self.create_gp_default())
            self.x_basis.append(self.x_basis_ini)
            resp, respPair, q_chos, q_lat_chos = self.reorder(resp, respPair, q_chos, q_lat_chos)
            startStateCount = resp[0]
            transStateCount = torch.sum(respPair, axis=0)
            if M > 2:
                self.reinit_global_params(M - 1, transStateCount, startStateCount)
            if M >= 2:
                nIters = 4
                for giter in range(nIters):
                    self.transTheta, self.startTheta = self._calcThetaFull(self.cond_cuda(transStateCount),
                                                                           self.cond_cuda(startStateCount), M)
                    self.rho, self.omega = self.find_optimum_rhoOmega()

            # Update transition matrix
            digammaSumTransTheta = torch.log(
                torch.sum(torch.exp(digamma(self.cond_cpu(self.transTheta[:M, :M + 1]))), axis=1))
            transPi = digamma(self.cond_cpu(self.transTheta[:M, :M])) - digammaSumTransTheta[:, np.newaxis]
            self.trans_A = transPi
        else:
            resp, respPair, q_chos, q_lat_chos = self.reorder(resp, respPair, q_chos, q_lat_chos)
            startStateCount = resp[0,:M]
            transStateCount = torch.sum(respPair[:,:M,:M], axis=0)
            if M > 2:
                self.reinit_global_params(M - 1, transStateCount, startStateCount)
            if M >= 2:
                nIters = 4
                for giter in range(nIters):
                    self.transTheta, self.startTheta = self._calcThetaFull(self.cond_cuda(transStateCount),
                                                                           self.cond_cuda(startStateCount), M)
                    self.rho, self.omega = self.find_optimum_rhoOmega()

            # Update transition matrix
            digammaSumTransTheta = torch.log(
                torch.sum(torch.exp(digamma(self.cond_cpu(self.transTheta[:M, :M + 1]))), axis=1))
            transPi = digamma(self.cond_cpu(self.transTheta[:M, :M])) - digammaSumTransTheta[:, np.newaxis]
            self.trans_A = transPi

        if len(resp.shape) == 1:
            resp_mod = self.cond_to_numpy(self.cond_cpu(resp))
            resp_modlog = self.cond_to_numpy(self.cond_cpu(resplog))
        else:
            resp_mod = self.cond_to_numpy(self.cond_cpu(resp[-1]))
            resp_modlog = self.cond_to_numpy(self.cond_cpu(resplog[-1]))

        # Normalize when responsability is equally shared
        if sum(np.isclose(resp_mod, max(resp_mod), rtol=1e-2)) > 1:
            h_argmax = np.nanargmax(resp_mod)
            resp_mod[:] = 0.0
            resp_mod[h_argmax] = 1.0

        model = np.argmax(resp_mod)
        if self.max_models is not None and model >= self.max_models:
            force_model = np.argmax(resp_modlog[:-1])
            model = force_model

        # When converged apply all transformations to real models.
        if not force_model is None:
            resp_mod[:] = 0.0
            resp_mod[force_model] = 1.0
            model = np.argmax(resp_mod)

        # Assign and update GP model
        self.actual_state = model
        print("Main model chosen:", model + 1)
        if minibatch == 0:
            minibatch = None
        for ld in range(self.n_outputs):
            for m in range(M):
                new_x_basis = self.gpmodels[ld][m].include_weighted_sample(t, self.x_train[-1], self.x_train[-1], y_mod[m][-1], resp_mod[m])
                self.x_basis[m] = self.cond_cuda(self.cond_to_torch(new_x_basis.detach()))
                if resp_mod[m] > 0.9:
                    self.y_train = torch.concatenate([self.y_train, torch.atleast_2d(y_mod[m][-1]).T[:, :, np.newaxis]])
                if self.bayesian_params:
                    self.gpmodels[ld][m].bayesian_new_params(resp_mod[m], model_type=self.model_type[m])
                else:
                    self.gpmodels[ld][m].new_params_weighted(resp_mod[m], batch=minibatch, min_samples=self.min_samples,
                                                         max_samples=self.max_samples,
                                                         div_samples=self.div_samples, verbose=False,
                                                         model_type=self.model_type[m], check_var=self.check_var)
        self.compute_q_elbo(resp[:,:M], respPair[:,:M,:M], self.weight_mean(q_chos)[:,:M],
                            self.weight_mean(q_lat_chos)[:,:M],
                            self.gpmodels, self.M, snr='saved', post=False, one_sample=True)
        q_ = torch.zeros((self.T, self.M, self.n_outputs))
        for ld in range(self.n_outputs):
            q_[:,:,ld] = self.compute_q(y=y_mod, gpmodels=self.gpmodels[ld], ld=ld)
        self.q.append(q_)
        for ld in range(self.n_outputs):
            if len(self.gpmodels[ld][model].indexes) > 1 and self.warp_updating[model] and with_warp:
                self.wp_sys[ld][model].update_warp(x_train, self.x_w[-1][model])

    def update_initial_sigma(self, model):
        """ Method to iteratively update initial sigma (not works so well)
        """
        gp = self.gpmodels[model]
        estimator = (torch.mean(torch.diag(gp.Sigma[-1])) + torch.mean(torch.diag(gp.Gamma[-1]))) \
                    * self.ini_sigma_def/torch.mean(torch.diag(gp.Sigma[-1]))
        self.ini_sigma_def = (self.ini_sigma_def * self.T + estimator)/(self.T + 1)
        self.gpmodels[-1] = self.create_gp_default()


    def calcELBO_LinearTerms(self, rho, omega, alpha, startAlpha, kappa, gamma, transTheta, startTheta, startStateCount,
                             transStateCount):
        """ Method to compute HPD linear terms.
        """
        transStateCount_ = transStateCount
        Ltop = self.L_top(rho=rho, omega=omega, alpha=alpha, startAlpha=startAlpha, kappa=kappa, gamma=gamma)
        LdiffcDir = - self.c_Dir(transTheta) - self.c_Dir(startTheta)
        K = transStateCount_.shape[0]
        if startTheta.shape[0] == rho.size:
            Ebeta = self.cond_to_numpy(self.cond_cpu(self.rho2beta(rho, returnSize='K')))
        else:
            Ebeta = self.cond_to_numpy(self.cond_cpu(self.rho2beta(rho, returnSize='K + 1')))
        LstartSlack = np.inner(
            startStateCount + startAlpha * Ebeta - startTheta,
            digamma(startTheta) - np.log(np.sum(np.exp(digamma(startTheta))))
        )
        alphaEbetaPlusKappa = alpha * np.tile(Ebeta, (K, 1))
        alphaEbetaPlusKappa[:, :K] += kappa * np.eye(K)
        digammaSum = np.log(np.sum(np.exp(digamma(transTheta)), axis=1))
        transStateCount_[:K, :] = transStateCount_[:K, :] + alphaEbetaPlusKappa
        LtransSlack = np.sum(
            (transStateCount_ - transTheta) *
            (digamma(transTheta) - digammaSum[:, np.newaxis])
        )
        return Ltop + LdiffcDir + LstartSlack + LtransSlack

    def calcELBO_NonlinearTerms(self, resp, respPair):
        """ Method to compute H[q].
        """
        Htable = self.calc_Htable(respPair)
        Hstart = self.calc_Hstart(resp)
        Lentropy = Htable.sum() + Hstart.sum()
        return Lentropy

    def calc_Hstart(self, resp, eps=1e-100):
        firstHvec = -1 * np.sum(resp * np.log(resp + eps), axis=0)
        return firstHvec

    def calc_Htable(self, respPair, eps=1e-100):
        sigma = respPair / (respPair.sum(axis=2)[:, :, np.newaxis] + eps)
        sigma += eps  # make it safe for taking logs!
        logsigma = sigma  # alias
        np.log(logsigma, out=logsigma)  # use fast numexpr library if possible
        H_KxK = -1 * np.sum(respPair * logsigma, axis=0)
        return H_KxK

    def L_top(self, rho, omega, alpha, startAlpha, kappa, gamma):
        K = rho.size
        eta1 = rho * omega
        eta0 = (1 - rho) * omega
        digamma_omega = digamma(omega)
        ElogU = digamma(eta1) - digamma_omega
        Elog1mU = digamma(eta0) - digamma_omega

        def c_Beta(a1, a0):
            return np.sum(gammaln(a1 + a0)) - np.sum(gammaln(a1)) - np.sum(gammaln(a0))

        diff_cBeta = K * c_Beta(1.0, gamma) - c_Beta(eta1, eta0)

        tAlpha = K * K * np.log(alpha) + K * np.log(startAlpha)
        if kappa > 0:
            coefU = K + 1.0 + eta1
            coef1mU = K * kvec(K) + 1.9 + gamma - eta0
            sumEbeta = np.sum(self.cond_to_numpy(self.cond_cpu(self.rho2beta(rho, returnSize='K'))))
            tBeta = sumEbeta * (np.log(alpha + kappa) - np.log(kappa))
            tKappa = K * (np.log(kappa) - np.log(alpha + kappa))
        else:
            coefU = (K + 1) + 1.0 - eta1
            coef1mU = (K + 1) * kvec(K) + gamma - eta0
            tBeta = 0.0
            tKappa = 0.0

        diff_logU = np.inner(coefU, ElogU) \
                    + np.inner(coef1mU, Elog1mU)
        return tAlpha + tKappa + tBeta + diff_cBeta + diff_logU

    def c_Dir(self, AMat, arem=None):
        ''' Evaluate cumulant function of the Dir distribution
        When input is vectorized, we compute sum over all entries.
        Returns
        -------
        c : scalar real
        '''
        AMat = np.asarray(AMat)
        D = AMat.shape[0]
        if arem is None:
            if AMat.ndim == 1:
                return gammaln(np.sum(AMat)) - np.sum(gammaln(AMat))
            else:
                return np.sum(gammaln(np.sum(AMat, axis=1))) \
                       - np.sum(gammaln(AMat))

        return np.sum(gammaln(np.sum(AMat, axis=1) + arem)) \
               - np.sum(gammaln(AMat)) \
               - D * np.sum(gammaln(arem))

    def find_optimum_rhoOmega(self, startTheta=None, transTheta=None, rho=None, omega=None, M=None):
        ''' Performs numerical optimization of rho and omega for M-step update.

                Note that the optimizer forces rho to be in [EPS, 1-EPS] for
                the sake of numerical stability

                Returns
                -------
                rho : 1D array, size K
                omega : 1D array, size K
                Info : dict of information about optimization.
                '''

        # Calculate expected log transition probability
        # using theta vectors for all K states plus initial state
        if startTheta is None:
            startTheta = self.startTheta
        if transTheta is None:
            transTheta = self.transTheta
        ELogPi = digamma(self.cond_to_numpy(self.cond_cpu(transTheta))) \
                 - np.log(np.sum(np.exp(digamma(self.cond_to_numpy(self.cond_cpu(transTheta)))), axis=1))[:,
                   np.newaxis]
        sumELogPi = np.sum(ELogPi, axis=0)
        startELogPi = digamma(self.cond_to_numpy(self.cond_cpu(startTheta))) \
                      - np.log(np.sum(np.exp(digamma(self.cond_to_numpy(self.cond_cpu(startTheta))))))

        # Select initial rho, omega values for gradient descent
        if not rho is None:
            initRho = self.cond_to_numpy(self.cond_cpu(rho))
        else:
            if hasattr(self, 'rho'):
                initRho = self.cond_to_numpy(self.cond_cpu(self.rho))
            else:
                initRho = None
        if not omega is None:
            initOmega = self.cond_to_numpy(self.cond_cpu(omega))
        else:
            if hasattr(self, 'omega'):
                initOmega = self.cond_to_numpy(self.cond_cpu(self.omega))
            else:
                initOmega = None
        if not M is None:
            M_ = M
        else:
            M_ = self.M + 1

        # Do the optimization
        try:
            rho, omega, fofu, Info = \
                find_optimum_multiple_tries(
                    sumLogPi=sumELogPi,
                    sumLogPiActiveVec=None,
                    sumLogPiRemVec=None,
                    startAlphaLogPi=self.startAlpha * startELogPi,
                    nDoc=M_,
                    gamma=self.gamma,
                    alpha=self.transAlpha,
                    kappa=self.kappa,
                    initrho=initRho,
                    initomega=initOmega)
            self.OptimizerInfo = Info
            self.OptimizerInfo['fval'] = fofu

        except ValueError as error:
            if hasattr(self, 'rho') and self.rho.size()[0] == self.M:
                print(
                    '***** Optim failed. Remain at cur val. ' +
                    str(error))
                rho = self.rho
                omega = self.omega
            else:
                print('***** Optim failed. Set to prior. ' + str(error))
                omega = (self.gamma + 1) * self.cond_cuda(torch.ones(self.M))
                rho = 1 / float(1 + self.gamma) * self.cond_cuda(torch.ones(self.M))

        return self.cond_cuda(self.cond_to_torch(rho)), self.cond_cuda(self.cond_to_torch(omega))

    def estimate_new(self, t, gpmodel, x_train, y, h=1.0):
        """
        Auxiliary function, reestimate new variational parameters.

        """
        mean_, cov_, C_, Sigma_ = gpmodel.smoother_weighted(x_train, y, h)
        if len(gpmodel.indexes) == 1:
            q_new = gpmodel.log_sq_error(x_train, y, mean=mean_[-1], cov=cov_[-1], C=C_[-1], Sigma=Sigma_[-1],
                                         i=-1, first=True)
        else:
            q_new = gpmodel.log_sq_error(x_train, y, mean=mean_[-1], cov=cov_[-1], C=C_[-1], Sigma=Sigma_[-1], i=-1)#, first=True)
        return q_new

    def estimate_q_all(self, M, x_trains, y_trains, y_trains_w, resp, respPair, q_, q_lat_, snr_, startPi, transPi,
                       gpmodels=None, reparam=False):
        """ Internal method to converge the ELBO using the most recent assignation of the examples.
        """
        if gpmodels is None:
            gpmodels = self.gpmodels
        q = torch.zeros((len(x_trains), M, self.n_outputs), device=self.device) + torch.min(q_) * 2.0
        q_lat = torch.zeros((len(x_trains), M, self.n_outputs), device=self.device)  # + torch.min(q_lat_) * 2.0
        snr_aux = torch.clone(snr_)
        resp_per_group_temp = torch.sum(resp, axis=0)
        reorder = torch.argsort(resp_per_group_temp, descending=True)
        resp_temp = torch.clone(resp[:, reorder])
        q__ = None
        indexes_ = [[] for _ in range(self.n_outputs)]
        gpmodels_temp = [[] for _ in range(self.n_outputs)]
        for ld in range(self.n_outputs):
            print("\n-----------Lead " + str(ld + 1) + "-----------")
            for m in range(M):
                print("\n   -----------Model " + str(m + 1) + "-----------")
                indexes_[ld].append(torch.where(resp_temp[:, m] == self.cond_cuda(torch.tensor(1.0)))[0].int())
                if len(gpmodels[ld]) > reorder[m]:
                    gp = self.gpmodel_deepcopy(gpmodels[ld][reorder[m]])
                    gp_indexes = torch.tensor(gp.indexes, device=indexes_[ld][m].device).int()
                    if not torch.equal(indexes_[ld][m], gp_indexes):
                        if gp.fitted:
                            if reparam:
                                gp.reinit_LDS(save_last=False)
                                gp.reinit_GP(save_last=False)
                            else:
                                gp.reinit_LDS(save_last=True)
                                gp.reinit_GP(save_last=False)
                        else:
                            # No added indexes so createtemp
                            gp = self.create_gp_default(i=reorder[m])
                        q[:, m, ld], q_lat[:, m, ld] = gp.full_pass_weighted(x_trains, y_trains_w[:, :, [ld]],
                                                                             resp_temp[:, m],
                                                                             q=q_[:, reorder[m], ld],
                                                                             q_lat=q_lat[:, reorder[m], ld],
                                                                             snr=self.snr_norm[:, ld])
                        snr_aux[:, m, ld] = self.compute_snr(y_trains_w[:, :, ld], gp)
                    else:
                        q[:, m, ld] = q_[:, reorder[m], ld]
                        q_lat[:, m, ld] = q_lat_[:, reorder[m], ld]
                        snr_aux[:, m, ld] = torch.clone(snr_[:, m, ld])
                else:
                    # New GP to add
                    gp = self.create_gp_default(i=reorder[m])
                    if len(indexes_[ld][m]) > 0.0:
                        q[:, m, ld], q_lat[:, m, ld] = gp.full_pass_weighted(x_trains, y_trains_w[:, :, [ld]],
                                                                             resp_temp[:, m],
                                                                             q=q_[:, reorder[m], ld],
                                                                             q_lat=q_lat[:, reorder[m], ld],
                                                                             snr=self.snr_norm[:, ld])
                        snr_aux[:, m, ld] = self.compute_snr(y_trains_w[:, :, ld], gp)
                    else:
                        q[:, m, ld] = q_[:, m, ld]
                        q_lat[:, m, ld] = q_lat_[:, m, ld]
                        snr_aux[:, m, ld] = torch.zeros(snr_.shape[0])
                gpmodels_temp[ld].append(gp)
        q_norm, _ = self.LogLik(self.weight_mean(q, snr_aux))
        alpha, margprob = self.forward(startPi, transPi, q_norm)
        beta = self.backward(transPi, q_norm, margprob)
        logresp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
        logrespPair, _ = self.LogLik(self.coupled_state_coef(alpha, beta, transPi, q_norm, margprob), axis=1)
        resp_temp = torch.exp(logresp)
        respPair_temp = torch.exp(logrespPair)
        #resp_temp, respPair_temp = self.refill_resp(resp_temp, respPair_temp)
        # Finally see if it is worthy to birth new cluster
        #new_indexes = torch.where(torch.sum(np.abs(resp - resp_temp), dim=1) > 1.0)[0]
        print(">>> Q_all_loop -------")
        q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q_, snr_),
                                              self.weight_mean(q_lat_, snr_), gpmodels, self.M, snr=snr_, post=True)
        q_bas_post, elbo_post = self.compute_q_elbo(resp_temp, respPair_temp, self.weight_mean(q, snr_aux),
                                                    self.weight_mean(q_lat, snr_aux), gpmodels_temp, M, snr=snr_aux, post=True)
        update_snr = True
        if torch.all(torch.sum(resp_temp, dim=0) >= 1.0):
            if q_bas + elbo_bas < q_bas_post + elbo_post:
                # self.gpmodels = gpmodels_temp
                if reorder.shape[0] == self.f_ind_old.shape[0]:
                    self.f_ind_old = self.f_ind_old[reorder]
                if update_snr:
                    self.snr_norm = self.normalize_snr(snr_aux)
                else:
                    snr_aux = snr_
                return resp_temp, respPair_temp, q, q_lat, snr_aux, y_trains_w, gpmodels_temp
            else:
                return resp, respPair, q_, q_lat_, snr_, y_trains_w, gpmodels
        else:
            print("Bad estimation")
            return resp, respPair, q_, q_lat_, snr_, y_trains_w, gpmodels


    def compute_warp_y(self, x_train, y, strategie='standard', force_model=None, gpmodel=None, i=None, ld=0):
        """
        Computation of warps following some strategies and functions proposed on warp_models

        Parameters
        ----------
        x_train : Array
            Domain of new observations.
        y : Array
            New observations vector.
        strategie : String, optional
            Describes the strategie to compute the warps of the model. The default is 'standard'.

        Returns
        -------
        y_w : Array.
            Vector describing warped sample respect to the models.
        x_w : Array.
            Vector describing the applied warp.
        liks : Double
            Likelihood of the array assuming some hyperparameters of noise of the GP warp.

        """
        M = self.M + 1
        l = len(x_train)
        x_w = [self.cond_cuda(self.cond_to_torch(np.atleast_2d(np.zeros(l)).T))] * M
        y_w = [self.cond_cuda(self.cond_to_torch(y))] * M
        base = self.wp_sys[ld][-1].warp_gp.log_sq_error(x_train, self.cond_cuda(self.cond_to_torch(x_w[-1]))).item()
        liks = [base] * M
        if strategie=='greedy' or strategie=='greedy_bound':
            q_C = self.estimate_new(gpmodels=self.gpmodels, x_train=self.x_train[-1], y=y)[-1,:,ld]
        def trans_noise(noise):
            if torch.mean(noise) > 1.0:
                noise = noise
            else:
                noise = noise
            if self.model_type[m] == 'static':
                noise = noise * 0.5
            else:
                noise = noise * 1.0
            return noise
        if not force_model is None:
            m = force_model
            if gpmodel is None:
                model = self.gpmodels[ld][m]
            else:
                model = gpmodel
            wp_sys_ = self.wp_sys[ld][m]
            if len(model.indexes) != 0:
                if self.verbose and False:
                    print("Warp respect model: ", m + 1)
                if i is None:
                    mean, cov = model.observe_last(x_train)
                else:
                    mean, cov = model.observe(x_train, t=i)
                noise = torch.diag(cov)
                noise = trans_noise(noise)
                x_, y_, lik_, loss_ = wp_sys_.compute_warp(self.cond_detach(x_train), self.cond_detach(y),
                                                           self.cond_detach(mean),
                                                           model.gp.kernel.get_params()["k1__k2__length_scale"],
                                                           noise=noise,
                                                           visualize=False, verbose=False,
                                                           train_iter=150)
                y_w[m] = y_
                x_w[m] = x_
                liks[m] = lik_ + self.wp_sys[ld][-1].warp_gp.log_sq_error(x_train,
                                                                      self.cond_cuda(self.cond_to_torch(x_))).item()
            else:
                liks[m] = base
        elif strategie == 'standard':
            for m in range(M):
                model = self.gpmodels[ld][m]
                wp_sys_ = self.wp_sys[ld][m]
                if len(model.indexes) != 0:
                    print("Warp respect model: ", m + 1)
                    mean, cov = model.observe_last(x_train)
                    noise = torch.diag(cov)
                    noise = trans_noise(noise)
                    x_, y_, lik_, loss_ = wp_sys_.compute_warp(self.cond_detach(x_train), self.cond_detach(y),
                                                           self.cond_detach(mean),
                                                           model.gp.kernel.get_params()["k1__k2__length_scale"],
                                                           noise=noise, visualize=False, verbose=self.verbose,
                                                           train_iter=150)
                    y_w[m] = y_
                    x_w[m] = x_
                    liks[m] = lik_ + self.wp_sys[ld][-1].warp_gp.log_sq_error(x_train,
                                                                          self.cond_cuda(self.cond_to_torch(x_))).item()
                else:
                    liks[m] = base
        elif strategie == 'greedy_bound':
            order_C = np.argsort(-q_C)
            for i in range(len(order_C)):
                m = order_C[i]
                model = self.gpmodels[ld][m]
                wp_sys_ = self.wp_sys[ld][m]
                if len(model.indexes) != 0:
                    print("Warp respect model: ", m + 1)
                    mean, cov = model.observe_last(x_train)
                    noise = torch.diag(cov)
                    noise = trans_noise(noise)
                    x_, y_, lik_, loss_ = wp_sys_.compute_warp(self.cond_detach(x_train), self.cond_detach(y),
                                                               self.cond_detach(mean),
                                                               model.gp.kernel.get_params()["k1__k2__length_scale"],
                                                               noise=noise, visualize=False, verbose=self.verbose,
                                                               train_iter=150)
                    y_w[m] = y_
                    x_w[m] = x_
                    # Normalize over log_sq from 0 knowledge model
                    liks[m] = lik_ + self.wp_sys[ld][-1].warp_gp.log_sq_error(x_train,
                                                                          self.cond_cuda(self.cond_to_torch(x_))).item()
                else:
                    liks[m] = base
                    print("Warp respect zero-knowledge model:", m + 1)
                    print("----Warping computed----")
                if i >= 3:
                    break
        elif strategie == 'greedy':
            order_C = np.argsort(-q_C)
            for i in range(len(order_C)):
                m = order_C[i]
                model = self.gpmodels[ld][m]
                wp_sys_ = self.wp_sys[ld][m]
                if len(model.indexes) != 0:
                    print("Warp respect model: ", m + 1)
                    mean, cov = model.observe_last(x_train)
                    noise = torch.diag(cov)
                    noise = trans_noise(noise)
                    x_, y_, lik_, loss_ = wp_sys_.compute_warp(self.cond_detach(x_train), self.cond_detach(y),
                                                           self.cond_detach(mean),
                                                           model.gp.kernel.get_params()["k1__k2__length_scale"],
                                                           noise=noise,
                                                           visualize=False, verbose=self.verbose,
                                                           train_iter=150)

                    y_w[m] = y_
                    x_w[m] = x_
                    # Normalize over log_sq from 0 knowledge model
                    basis_lik = self.wp_sys[ld][-1].warp_gp.log_sq_error(x_train, self.cond_cuda(self.cond_to_torch(x_))).item()
                    liks[m] = lik_ + basis_lik
                    liks_ = np.array(liks[m]) * 0.5
                    if i < len(order_C) - 1 and i < 8:
                        first = order_C[0]
                        if ((q_C[m] + liks_ - q_C[order_C[i + 1]]) / (q_C[m] - q_C[order_C[i + 1]]) > 0.3 / (
                                np.log(max(model.N, 1) + 1))) or i == 5:
                            break
                    else:
                        break
                else:
                    liks[m] = base
                    print("Warp respect zero-knowledge model:", m + 1)
                    print("----Warping computed----")

        else:
            print('Only two strategies implemented: standard and greedy')
        return y_w, x_w, liks

    def compute_trans_A(self, M):
        digammaSumTransTheta = torch.log(
            torch.sum(torch.exp(digamma(self.cond_cpu(self.transTheta[:M, :M + 1]))), axis=1))
        transPi = digamma(self.cond_cpu(self.transTheta[:M, :M])) - digammaSumTransTheta[:, np.newaxis]
        if transPi.shape[0] == M:
            return transPi
        else:
            transPi_ = torch.zeros(M,M) - np.inf
            transPi_[:M-1,:M-1] = transPi
            return transPi_

    def compute_trans_pi(self, M, pi):
        if pi.shape[0] == M:
            return pi
        else:
            pi_ = torch.zeros(M) - np.inf
            pi_[:M-1] = pi
            return pi_

    # Now methods to compute the forward and backward iterations for the state-space model of the switching variable.
    def forward(self, pi=None, trans_A=None, q=None):
        """
        Forward-pass messages to compute alphas.

        Parameters
        ----------
        pi : array like of shape (m_samples) vector of initial distribution

        A : matrix like of shape (mxm_samples) transition matrix of the model

        q : matrix like of shape (txm_samples) variational observations of the example t

        Returns
        -------
        alphas : matrix of shape txm with alphas.

        """
        if pi is None:
            pi = self.pi[-1]
        if trans_A is None:
            trans_A = self.trans_A
        if q is None:
            q = self.q[-1]
        M = self.M
        T = q.shape[0]
        K = q.shape[1]
        fmsg = torch.zeros((T, K), device=self.device)
        margPrObs = torch.zeros(T, device=self.device)
        pi_ = self.compute_trans_pi(K, pi)
        pi_ = torch.exp(pi_)
        trans_A = self.compute_trans_A(K)
        PiTMat = torch.exp(trans_A.T)
        q_ = q
        def safe_exp(x):
            return torch.exp(torch.sub(x, torch.atleast_2d(torch.max(x, axis=1)[0]).T))
        q_ = safe_exp(q_)
        PiTMat[PiTMat < 1e-6] += 1e-4
        pi_[pi_ < 1e-10] += 1e-4
        if self.fmsg is not None:
            if self.fmsg.shape[1] < K:
                fmsg[:T - 1,:K - 1] = torch.clone(self.fmsg)
            else:
                fmsg[:T - 1, :K] = torch.clone(self.fmsg)
            margPrObs[:T-1] = torch.clone(self.margPrObs)
            fmsg[-1] = torch.matmul(PiTMat, fmsg[-2]) * q_[-1]
            margPrObs[-1] = torch.sum(fmsg[-1])
            fmsg[-1] /= margPrObs[-1]
        else:
            for t in range(0, T):
                if t == 0:
                    fmsg[t] = pi_ * q_[0]
                else:
                    fmsg[t] = torch.matmul(PiTMat, fmsg[t - 1]) * q_[t]
                margPrObs[t] = torch.sum(fmsg[t])
                fmsg[t] /= margPrObs[t]
                if margPrObs[t] == 0.0:
                    print("Forward message ill conditioned")
        if torch.any(torch.isnan(torch.log(fmsg))):
            print("Error nan")
            return None, None
        if torch.argmax(fmsg[-1]) != torch.argmax(q[-1]) and self.verbose:
            print("Miss")
        return fmsg, margPrObs

    def backward(self, trans_A=None, q=None, margprob=None):
        """
        Backward-pass messages to compute betas.

        Parameters
        ----------
        A : matrix like of shape (mxm_samples) transition matrix of the model

        q : matrix like of shape (txm_samples) variational observations of the example t

        Returns
        -------
        betas : matrix of shape txm with betas.

        """
        if trans_A is None:
            trans_A = self.trans_A
        if q is None:
            q = self.q[-1]
        M = q.shape[1]
        q_ = q
        def safe_exp(x):
            return torch.exp(torch.sub(x, torch.atleast_2d(torch.max(x, axis=1)[0]).T))
        trans_A = self.compute_trans_A(M)
        PiMat = self.cond_cuda(torch.exp(trans_A))
        q_ = safe_exp(q_)
        T = q.shape[0]
        K = M
        bmsg = torch.ones((T, K), device=self.device)
        PiMat[PiMat < 1e-5] += 1e-4
        for t in range(T - 2, -1, -1):
            bmsg[t] = torch.matmul(PiMat, bmsg[t + 1] * q_[t + 1])
            bmsg[t] /= torch.sum(bmsg[t][:-1])
        if torch.any(torch.isnan(torch.log(bmsg))):
            print("Error nan")
        return bmsg

    def coupled_state_coef(self, alpha=None, beta=None, trans_A=None, q=None, margprobs=None):
        """
        Compute psi coefficients for ease reestimation of parameters on state-model.

        Parameters
        ----------
        alpha : matrix of shape txm with alphas, optional
             The default is compute alphas.
        beta : matrix of shape txm with betas, optional
             The default is compute betas.
        A : matrix like of shape (mxm_samples) transition matrix of the model
            The default is last saved.
        q : matrix like of shape (txm_samples) variational observations of the example t
            The default is last saved.
        Returns
        -------
        psi : array of matrix of shape txmxm
            Psi coefficients.

        """
        if alpha is None or margprobs is None:
            alpha, margprobs = self.forward()
        if beta is None:
            beta = self.backward()
        if trans_A is None:
            trans_A = self.trans_A
        if q is None:
            q = self.q[-1]
        def safe_exp(x):
            return torch.exp(torch.sub(x, torch.atleast_2d(torch.max(x, axis=1)[0]).T))
        alpha = self.cond_cuda(self.cond_to_torch(alpha))
        beta = self.cond_cuda(self.cond_to_torch(beta))
        q = self.cond_cuda(self.cond_to_torch(q))

        bmsgSoftEv = safe_exp(q)
        bmsgSoftEv *= beta
        # respPair[1:] = alpha[:-1][:, :, np.newaxis] * \
        #               bmsgSoftEv[1:][:, np.newaxis, :]
        respPair = alpha[:, :, np.newaxis] * \
                      bmsgSoftEv[:, np.newaxis, :]
        den = torch.sum(respPair, dim=(1,2))[:, np.newaxis, np.newaxis]
        den[den == 0] = 1e-10
        respPair /= den
        return torch.log(respPair)

        # return psi

    def sum_coupled_state_coef(self, alpha=None, beta=None, trans_A=None, q=None):
        """
        Compute psi coefficients for ease reestimation of parameters on state-model.

        Parameters
        ----------
        alpha : matrix of shape txm with alphas, optional
             The default is compute alphas.
        beta : matrix of shape txm with betas, optional
             The default is compute betas.
        A : matrix like of shape (mxm_samples) transition matrix of the model
            The default is last saved.
        q : matrix like of shape (txm_samples) variational observations of the example t
            The default is last saved.
        Returns
        -------
        psi : array of matrix of shape txmxm
            Psi coefficients.

        """
        if alpha is None:
            alpha = self.forward()
        if beta is None:
            beta = self.backward()
        if trans_A is None:
            trans_A = self.trans_A
        if q is None:
            q = self.q[-1]

        alpha = self.cond_cuda(self.cond_to_torch(alpha))
        beta = self.cond_cuda(self.cond_to_torch(beta))
        trans_A = self.cond_cuda(self.cond_to_torch(trans_A))
        q = self.cond_cuda(self.cond_to_torch(q))

        T = alpha.shape[0]
        M = self.M + 1
        psi = self.cond_cuda(self.cond_to_torch(np.matrix([[-np.inf] * M] * M)))
        for i in range(M - 1):
            for j in range(M):
                psi[i, j] = torch.sum(alpha[:, i]) + trans_A[i, j] + torch.sum(q[:, j]) + torch.sum(beta[:, j])# - den
            den = torch.logsumexp(psi.view(-1), 0)
            psi = psi - den
        return psi

    def compute_q(self, x_train=None, y=None, gpmodels=None, mean=None, cov=None, C=None, Sigma=None, liks=None,
                  batch=None, ld=None):
        """
        Compute variational parameters related with the observations. (Only for online scheme)
        Parameters
        ----------
        y : array sample of shape (s_samples)

        Returns
        -------
        q : array of shape m with q
        """

        M = self.M
        if x_train is None:
            x_train = self.x_train[-1]
        if y is None:
            y = self.y_w
        if isinstance(y, np.ndarray):
            T = len(y)
        else:
            T = len(y[0])
        if gpmodels is None:
            gpmodels = self.gpmodels
        if mean is None:
            mean = []
            for m in range(M):
                mean.append(gpmodels[m].f_star)
        if cov is None:
            cov = []
            for m in range(M):
                cov.append(gpmodels[m].cov_f)
        if C is None:
            C = []
            for m in range(M):
                C.append(gpmodels[m].C)
        if Sigma is None:
            Sigma = []
            for m in range(M):
                Sigma.append(gpmodels[m].Sigma)
        if type(y) is torch.Tensor:
            q = np.repeat(-np.inf, M)
            for m in range(M):
                q[m] = gpmodels[m].log_sq_error(x_train, y, mean[m][-1], cov[m][-1], C[m][-1], Sigma[m][-1], -1)
        else:
            if len(self.q) > 0:
                if ld is not None:
                    q = torch.clone(self.q[-1][:,:,ld])
                else:
                    q = torch.clone(self.q[-1])
                if len(q[-1]) < M:
                    sh = q.shape
                    q_ = torch.zeros((sh[0], sh[1] + 1))
                    q_[:,:-1] = q
                    q = q_
                    for t in range(T - 1):
                        first = True if gpmodels[-1].N == 1 else False
                        q_ = gpmodels[-1].log_sq_error(x_train, y[-1][t], mean[-1][0], cov[-1][0],
                                                       C[-1][0], Sigma[-1][0], 0, first=first)
                        q[t,-1] = q_[0]
                q_ = torch.from_numpy(np.repeat(-np.inf, M))
                for m in range(M):
                    first = True if gpmodels[m].N == 1 else False
                    q_[m] = gpmodels[m].log_sq_error(x_train, y[m][-1], mean[m][-1], cov[m][-1],
                                                     C[m][-1], Sigma[m][-1], T - 1, first=first)
                q = torch.concatenate([q, q_[np.newaxis,:]])
            else:
                q = torch.zeros((self.T, M, self.n_outputs))
                q_ = torch.from_numpy(np.repeat(-np.inf, M))
                for m in range(M):
                    first = True if gpmodels[m].N == 1 else False
                    q_[m] = gpmodels[m].log_sq_error(x_train, y[m][-1], mean[m][-1], cov[m][-1],
                                                     C[m][-1], Sigma[m][-1], T - 1, first=first)
                q = q_

        return self.cond_to_torch(q)

    def compute_h(self, alpha=None, beta=None, time=None):
        """
        Compute variational parameters related with relevance.

        Parameters
        ----------
        alpha : array os shpae m with alphas (forward msg), optional
            The default is compute alphas.
        beta : array of shape m with betas (forward msg), optional
            The default is compute betas.
        t : int time where is wanted to compute h_t(i) for all states
            The default is last.
        Returns
        -------
        h_ : array of shape m with h

        """
        T = alpha.shape[0]
        M = self.M + 1
        if alpha is None:
            alpha = self.forward()
        if beta is None:
            beta = self.backward()
        alpha = self.cond_cuda(self.cond_to_torch(alpha))
        beta = self.cond_cuda(self.cond_to_torch(beta))
        h = self.cond_cuda(self.cond_to_torch(np.matrix([[-np.inf] * M] * T)))
        for t in range(T):
            den = self.cond_cuda(self.cond_to_torch(-np.inf))
            for i in range(M):
                den = torch.logaddexp(den, alpha[t, i] + beta[t, i])
            h_ = self.cond_cuda(self.cond_to_torch(np.array([-np.inf] * M)))
            for i in range(M):
                h_[i] = alpha[t, i] + beta[t, i] - den
            h[t] = h_

        if time is None:
            return self.cond_cuda(self.cond_to_torch(h))
        else:
            return self.cond_cuda(self.cond_to_torch(h[time]))

    def baum_welch(self, alpha=None, beta=None, trans_A=None, q=None):
        """
        Re-estimate parameters of HMM on terms of baum-welch algorithm

        Parameters
        ----------
        alpha : matrix of shape txm with alphas, optional
             The default is compute alphas.
        beta : matrix of shape txm with betas, optional
             The default is compute betas.
        A : matrix like of shape (mxm_samples) transition matrix of the model
            The default is last saved.
        q : matrix like of shape (txm_samples) variational observations of the example t
            The default is last saved.

        Returns
        -------
        pi : array of size m, initial distribution vector.

        trans_A : matrix if size mxm, transition matrix


        """
        if self.hmm_switch:
            M = self.M
            T = self.T

            if alpha is None:
                alpha = self.forward()
            if beta is None:
                beta = self.backward()
            if trans_A is None:
                trans_A = self.trans_A
            if q is None:
                q = self.q[-1]

            alpha = self.cond_to_torch(alpha)
            beta = self.cond_to_torch(beta)
            trans_A = self.cond_to_torch(trans_A)
            q = self.cond_to_torch(q)

            h = self.cond_to_torch(self.compute_h(alpha, beta))
            pi_ = self.cond_to_torch(np.ravel(h[0]))
            trans_A_ = self.cond_to_torch(np.matrix([[0.0] * M] * M))
            # Compute numerator and denominator using specific functions as described in Rabiner
            psi = self.cond_to_torch(self.coupled_state_coef(alpha, beta, trans_A, q))
            for i in range(M):
                den = self.cond_to_torch(-np.inf)
                for t in range(T - 1):
                    for j in range(M):
                        den = torch.logaddexp(den, psi[t][i, j])
                for j in range(M):
                    num = self.cond_to_torch(-np.inf)
                    for t in range(T - 1):
                        num = torch.logaddexp(num, psi[t][i, j])
                    # Case where zero expected times of transitions are detected
                    if num == -np.inf:
                        trans_A_[i, j] = -np.inf
                    else:
                        trans_A_[i, j] = num - den
                # To maintain properties of the transition matrix (sum of row = 1)
                aux = self.normalize_log(np.ravel(trans_A_[i]))
                for j in range(M):
                    trans_A_[i, j] = aux[j]

            return pi_, trans_A_
        else:
            return self.pi[-1], self.trans_A[-1]

    def return_model_of_sample(self, n_sample):
        for i, g in enumerate(self.gpmodels[0]):
            if n_sample in g.indexes:
                return i

    #Methods for the internal management and conversion of the model.
    def selected_gpmodels(self):
        selgp = 0
        for gp in self.gpmodels[0]:
            if len(gp.indexes) > 0:
                selgp = selgp + 1
        return list(range(selgp))

    def save_swgp(self, st):
        self.keep_last_all()
        self.full_model_to_cpu()
        with open(st, 'wb') as inp:
            plk.dump(self, inp)

    def gpmodel_deepcopy(self, gpmodel):
        gp_ = gp.GPI_model(gpmodel.gp.kernel.clone_with_theta(gpmodel.gp.kernel.theta), gpmodel.x_basis.clone(), verbose=self.verbose)
        gp_.y_train = gpmodel.y_train.copy()
        gp_.x_train = gpmodel.x_train.copy()
        gp_.f_star = gpmodel.f_star.copy()
        gp_.f_star_sm = gpmodel.f_star_sm.copy()
        gp_.cov_f = gpmodel.cov_f.copy()
        gp_.cov_f_sm = gpmodel.cov_f_sm.copy()
        gp_.y_var = gpmodel.y_var.copy()
        gp_.var = gpmodel.var.copy()
        gp_.A = gpmodel.A.copy()
        gp_.Gamma = gpmodel.Gamma.copy()
        gp_.C = gpmodel.C.copy()
        gp_.Sigma = gpmodel.Sigma.copy()
        gp_.gp.assign_alpha_ini(gp_.Sigma[0], gp_.Gamma[0])
        gp_.likelihood = gpmodel.likelihood.copy()
        gp_.N = gpmodel.N
        gp_.indexes = gpmodel.indexes.copy()
        gp_.fitted = gpmodel.fitted
        gp_.ini_cov_def = gpmodel.ini_cov_def
        gp_.A_def, gp_.Gamma_def = gpmodel.A_def, gpmodel.Gamma_def
        gp_.C_def, gp_.Sigma_def = gpmodel.C_def, gpmodel.Sigma_def
        gp_.internal_params = gpmodel.internal_params
        gp_.observation_params = gpmodel.observation_params
        gp_.ini_kernel_theta = gpmodel.ini_kernel_theta
        gp_.free_deg_MNIV = gpmodel.free_deg_MNIV
        return gp_

    def normalize_log(self, x):
        # Special case when maximum bound has been reached
        bound = 1e-50
        log_bound = np.log(bound)
        if np.max(x) == -np.inf:
            x = np.repeat(log_bound, len(x))
        if not np.isclose(np.max(x), 0):
            aux = np.abs(x) / np.max(np.abs(x))
            aux = -aux + np.ones(len(aux))
            aux = [bound if i == 0 else i for i in aux]
            # aux = np.exp(aux)
            den = np.sum(aux)
            x = np.log(aux / den)
        else:
            ind = np.argmax(x)
            x = np.repeat(log_bound, len(x))
            x[ind] = 0.0
        return x

    def cond_to_torch(self, x):
        if type(x) is list and type(x[0]) is np.ndarray:
            x = torch.from_numpy(np.array(x))
            x.requires_grad = False
        elif type(x) is not torch.Tensor:
            x = torch.from_numpy(np.array(x))
            x.requires_grad = False
        else:
            x.requires_grad = False
        return x

    def cond_to_numpy(self, x):
        if type(x) is torch.Tensor:
            x = x.detach().numpy()
        return x

    def cond_cuda(self, x):
        if self.cuda:
            if type(x) is list or type(x) is np.ndarray:
                for i in x:
                    if not i.is_cuda and torch.cuda.is_available():
                        i = i.cuda()
                        i.requires_grad = False
            elif not x.is_cuda and torch.cuda.is_available():
                x = x.cuda()
                x.requires_grad = False
        return x

    def cond_cpu(self, x):
        if type(x) is torch.Tensor:
            x = x.cpu()
        return x

    def cond_detach(self, x):
        if type(x) is torch.Tensor:
            x = x.detach()
        return x

    def full_model_to_numpy(self):
        def recursive_numpy(x):
            if type(x) is torch.Tensor:
                x = self.cond_to_numpy(x)
            elif type(x) is list:
                if len(x) > 0:
                    if type(x[0]) is torch.Tensor:
                        for j, i in enumerate(x):
                            x[j] = self.cond_to_numpy(i)
                    elif len(x[0]) > 0:
                        if type(x[0][0]) is torch.Tensor:
                            for i in x:
                                for j, t in enumerate(i):
                                    i[j] = self.cond_to_numpy(t)
            else:
                x = x
            return x

        self.x_basis = recursive_numpy(self.x_basis)
        self.y = recursive_numpy(self.y)
        self.y_w = recursive_numpy(self.y_w)
        self.x_w = recursive_numpy(self.x_w)
        self.liks = recursive_numpy(self.liks)
        self.rho = recursive_numpy(self.rho)
        self.omega = recursive_numpy(self.omega)
        self.theta = recursive_numpy(self.theta)
        self.transTheta = recursive_numpy(self.transTheta)
        self.startTheta = recursive_numpy(self.startTheta)
        for gp in self.gpmodels:
            gp.model_to_numpy()
        for w_gp in self.wp_sys:
            w_gp.warp_gp.model_to_numpy()

    def full_model_to_torch(self):
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
        self.y = recursive_torch(self.y)
        self.y_w = recursive_torch(self.y_w)
        self.x_w = recursive_torch(self.x_w)
        self.liks = recursive_torch(self.liks)
        self.rho = recursive_torch(self.rho)
        self.omega = recursive_torch(self.omega)
        self.theta = recursive_torch(self.theta)
        self.transTheta = recursive_torch(self.transTheta)
        self.startTheta = recursive_torch(self.startTheta)
        for gp in self.gpmodels:
            gp.model_to_torch()
        for w_gp in self.wp_sys:
            w_gp.warp_gp.model_to_torch()

    def full_model_to_cuda(self):
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
        self.y = recursive_cuda(self.y)
        self.y_w = recursive_cuda(self.y_w)
        self.x_w = recursive_cuda(self.x_w)
        self.liks = recursive_cuda(self.liks)
        self.rho = recursive_cuda(self.rho)
        self.omega = recursive_cuda(self.omega)
        self.theta = recursive_cuda(self.theta)
        self.transTheta = recursive_cuda(self.transTheta)
        self.startTheta = recursive_cuda(self.startTheta)
        for gp in self.gpmodels:
            gp.model_to_cuda()
        for w_gp in self.wp_sys:
            w_gp.warp_gp.model_to_cuda()

    def full_model_to_cpu(self):
        if self.cuda:
            def recursive_cpu(x):
                if type(x) is torch.Tensor:
                    x = x.cpu()
                elif type(x) is list:
                    if len(x) > 0:
                        if type(x[0]) is torch.Tensor or type(x[0]) is torch.float64:
                            for j, i in enumerate(x):
                                x[j] = i.cpu()
                        elif type(x[0]) is list:
                            if type(x[0][0]) is torch.Tensor:
                                for x_ in x:
                                    for j, i in enumerate(x_):
                                        x_[j] = i.cpu()
                else:
                    x = x
                return x

            self.x_basis = recursive_cpu(self.x_basis)
            self.y = recursive_cpu(self.y)
            self.y_w = recursive_cpu(self.y_w)
            self.x_w = recursive_cpu(self.x_w)
            self.liks = recursive_cpu(self.liks)
            self.rho = recursive_cpu(self.rho)
            self.omega = recursive_cpu(self.omega)
            self.theta = recursive_cpu(self.theta)
            self.transTheta = recursive_cpu(self.transTheta)
            self.startTheta = recursive_cpu(self.startTheta)
            for gp in self.gpmodels:
                gp.model_to_cpu()
            for w_gp in self.wp_sys:
                w_gp.warp_gp.model_to_cpu()

