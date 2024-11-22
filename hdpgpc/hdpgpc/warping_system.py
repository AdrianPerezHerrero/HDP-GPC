## A package to implement all compressed functions to compute a warp between a model and some observations.
import torch
import pyro
import pyro.distributions as dist
import gpytorch

import numpy as np

import matplotlib.pyplot as plt

dtype = torch.float64
torch.set_default_dtype(dtype)

from hdpgpc.util_plots import plot_gp_gpytorch, plot_gp_pyro, print_hyperparams
from hdpgpc.GPI_models_pytorch import ExactGPModel, AlignmentGPModel, LinearExactGPModel
from hdpgpc.GPI_model import GPI_model
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import warnings

class Warping_system():
    
    def __init__(self, x_basis_warp, noise_warp, bound_noise_warp, recursive=False, cuda=False, bayesian=True, mode='balanced'):
        self.cuda = cuda
        self.recursive = recursive
        self.prior = torch.Tensor()
        self.T = 0
        self.mode = mode
        self.bayesian = bayesian
        N = len(x_basis_warp)
        #Gaussian Kernel prior
        self.warp_gp = self.create_GP(x_basis_warp, noise_warp, bound_noise_warp)
        self.warp_lik = None
        self.mll_warp = None
        self.x_warps = []
        self.x_fixed = x_basis_warp
        if type(self.x_fixed) is torch.Tensor:
            if self.cuda and torch.cuda.is_available() and self.x_fixed.is_cuda:
                self.x_fixed = self.x_fixed.cpu()
            self.x_fixed = self.x_fixed.numpy()
        # self.x_post = torch.atleast_2d(torch.arange(0, N, dtype=dtype)).T
        self.kernel = self.warp_gp.gp.kernel.clone_with_theta(self.warp_gp.gp.kernel.theta)
        self.jitter = 1e-4 * self.warp_gp.Sigma[-1]
        K_X_X = torch.from_numpy(self.kernel(self.x_fixed, self.x_fixed)) + self.jitter
        self.x_fixed = torch.from_numpy(self.x_fixed)
        # self.K_Xs_X = torch.tensor(kernel(self.x_post, self.x_fixed))
        self.L = torch.linalg.cholesky(K_X_X)
        if torch.cuda.is_available() and self.cuda:
            self.x_fixed = self.x_fixed.cuda()
            # self.x_post = self.x_post.cuda()
            # self.K_Xs_X = self.K_Xs_X.cuda()
            self.L = self.L.cuda()

    def create_GP(self, x_basis, noise_warp, bound_noise_warp):
        x_basis = x_basis
        noise_bounds = bound_noise_warp
        if self.mode == 'fine':
            len = 1.0
        elif self.mode == 'rough':
            len = 9.0
        else:
            len = 4.0
        warp_gp = GPI_model(RBF(len, (0.1, len))
                            + WhiteKernel(noise_warp, noise_bounds), x_basis,
                            annealing=False, bayesian=self.bayesian, cuda=self.cuda,
                            inducing_points=False, estimation_limit=20)
        cond = warp_gp.GPR_static(noise_warp)
        # cond = warp_gp.GPR_dynamic(noise_warp, 0.001)
        warp_gp.initial_conditions(ini_mean=None, ini_cov=None,
                       ini_A=cond[0], ini_Gamma=cond[1], ini_C=cond[2], ini_Sigma=cond[3])
        return warp_gp

    def model_to_cpu(self):
        if torch.cuda.is_available() and self.cuda:
            self.x_fixed = self.x_fixed.cpu()
            self.L = self.L.cpu()
        
    #Function to construct a monotonic sequence from a bunch of variables.
    def monotonic_sequence(self, x, x_post):
        N = x_post[-1]
        N_0 = x_post[0]
        x = torch.nn.functional.softmax(x, dim=0)
        x = torch.atleast_2d((N - N_0 + 2) * torch.cumsum(x, dim=0) + N_0 - 2).T
        if x.detach().shape[0] != x_post.detach().shape[0]:
            x = torch.cholesky_solve(x - self.x_fixed, self.L)
            if self.cuda and torch.cuda.is_available():
                x_post_ = x_post.cpu()
                x_fix = self.x_fixed.cpu()
            else:
                x_post_ = x_post
                x_fix = self.x_fixed
            K_Xs_X = torch.from_numpy(self.kernel(x_post_.numpy(), x_fix.numpy()))
            if self.cuda and torch.cuda.is_available():
                K_Xs_X = K_Xs_X.cuda()
            x = x_post + torch.matmul(K_Xs_X, x)
        x = (x - min(x)) / (max(x) - min(x)) * N
        x = x.T[0]
        return x
    
    def compute_gp_warp(self, x_model, noise_warp, verbose):
        warp_lik = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(noise_warp, noise_warp*1.1))
        warp_gp = LinearExactGPModel(x_model, x_model, warp_lik)
        warp_gp.covar_module.base_kernel.raw_lengthscale_constraint = gpytorch.constraints.Interval(1.0, 1.01)
        warp_lik.train()
        warp_gp.train()
    
        training_iter = 1500
    
        optimizer_warp = torch.optim.Adam(warp_gp.parameters(), lr=0.5)
        mll_warp = gpytorch.mlls.ExactMarginalLogLikelihood(warp_lik, warp_gp)
    
        losses_warp = []
        
        #CUDA OPTION
        if torch.cuda.is_available() and self.cuda:
            warp_gp = warp_gp.cuda()
            warp_lik = warp_lik.cuda()
            
            
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer_warp.zero_grad()
            # Output from model
            output = warp_gp(x_model)
            # Calc loss and backprop gradients
            loss = -mll_warp(output, x_model)
            loss.backward()
            losses_warp.append(loss.item())
            if verbose and i%10==0:
                print('Iter %d/%d - Loss: %.3f' % (
                    i + 1, training_iter, loss.item()))
            optimizer_warp.step()
            if len(losses_warp)>20:
                if np.isclose(losses_warp[-1],losses_warp[-2],rtol=1e-5) and np.isclose(losses_warp[-2],
                                                                                        losses_warp[-3], rtol=1e-5):
                    break
        if verbose:
            print_hyperparams(warp_gp)
     
        warp_gp.eval()
        warp_lik.eval()
        
        return warp_gp, warp_lik, mll_warp
    
    def update_warp(self, x_model, x_warp):
        if self.T == 0:
            ini_sigma = self.warp_gp.Sigma[-1]
            x_fixed, _ = self.warp_gp.fit_kernel_params(x_model, x_warp, ini_sigma, torch.zeros(ini_sigma.shape))
            self.x_fixed = x_fixed
            if type(self.x_fixed) is torch.Tensor:
                if self.cuda and torch.cuda.is_available() and self.x_fixed.is_cuda:
                    self.x_fixed = self.x_fixed.cpu()
                self.x_fixed = self.x_fixed.numpy()
            self.kernel = self.warp_gp.gp.kernel.clone_with_theta(self.warp_gp.gp.kernel.theta)
            K_X_X = torch.from_numpy(self.kernel(self.x_fixed, self.x_fixed))
            if torch.cuda.is_available() and self.cuda:
                K_X_X = K_X_X.cuda()
            K_X_X = K_X_X + self.jitter
            self.x_fixed = torch.from_numpy(self.x_fixed)
            self.L = torch.linalg.cholesky(K_X_X)
            if torch.cuda.is_available() and self.cuda:
                self.x_fixed = self.x_fixed.cuda()
                self.L = self.L.cuda()
            self.warp_gp.include_sample(self.T, x_model, x_warp, embedding=False)
        else:
            self.warp_gp.include_sample(self.T, x_model, x_warp, embedding=False)
            self.warp_gp.backwards()
            if self.bayesian:
                self.warp_gp.bayesian_new_params(1.0, model_type='static')
            else:
                self.warp_gp.new_params(model_type='static', verbose=False)
        self.x_warps.append(x_warp)
        self.T = self.T+1
        
    def compute_warp(self, x_model, y_model, y_target, theta, noise=None, visualize=False, verbose=False, train_iter=100):
        try:
            N = len(x_model)
            #Pytorch transforming
            if type(x_model) is not torch.Tensor:
                x_model = torch.tensor(x_model.T[0], dtype = dtype)
            else:
                x_model = x_model.T[0]
            if type(y_model) is not torch.Tensor:
                y_model = torch.tensor(y_model.T[0], dtype = dtype)
            else:
                y_model = y_model.T[0]
            if type(y_target) is not torch.Tensor:
                y_target = torch.tensor(y_target.T[0], dtype = dtype)
            else:
                y_target = y_target.T[0]
            if type(theta) is not torch.Tensor:
                theta = torch.tensor(theta, dtype = dtype)
            zero_noise_model = False
            if noise is None:
                zero_noise_model = True
            elif type(noise) is np.float64:
                noise = torch.tensor([noise]*len(y_target), dtype = dtype)
            elif type(noise) is np.ndarray:
                noise = torch.from_numpy(noise)
            else:
                noise = noise.clone().detach()
            if self.mode == 'fine':
                noise = noise * 0.01
            if zero_noise_model:
                likelihood_fixed = gpytorch.likelihoods.GaussianLikelihood()
            else:
                #Reduce a bit the noise just for a major precision of warping.
                likelihood_fixed = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise)
            fixed_model = ExactGPModel(x_model, y_model, likelihood_fixed)
            
            if not zero_noise_model:
                # fixed_model.covar_module.base_kernel.raw_lengthscale_constraint = gpytorch.constraints.Interval(theta*0.99, theta*1.1)
                fixed_model.covar_module.base_kernel.raw_lengthscale_constraint = gpytorch.constraints.Interval(0.99, 1.1)
            else:
                fixed_model.covar_module.base_kernel.raw_lengthscale_constraint = gpytorch.constraints.Interval(0.99, 1.1)
                likelihood_fixed.noise_covar.noise_constraint = gpytorch.constraints.Interval(0.0, 0.1)
            likelihood_fixed.train()
            fixed_model.train()
            if zero_noise_model:
                training_iter=train_iter
            else:
                training_iter=550
            
            # optimizer_fixed = torch.optim.Adam(fixed_model.parameters(), lr=0.1)
            optimizer_fixed = torch.optim.Rprop(fixed_model.parameters(), lr=0.5)
            mll_fixed = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_fixed, fixed_model)
            losses_fixed = []
            
            #CUDA option
            if torch.cuda.is_available() and self.cuda:
                fixed_model = fixed_model.cuda()
                likelihood_fixed = likelihood_fixed.cuda()
                x_model = x_model.cuda()
                y_model = y_model.cuda()
                y_target = y_target.cuda()

            if self.mode == 'fine':
                min_tr = 100
            else:
                min_tr = 20

            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer_fixed.zero_grad()
                # Output from model
                output = fixed_model(x_model)
                # Calc loss and backprop gradients
                loss = -mll_fixed(output, y_model)
                loss.backward()
                losses_fixed.append(loss.item())
                if verbose and i%10==0:
                    print('Iter %d/%d - Loss: %.3f' % (
                        i + 1, training_iter, loss.item()))
                optimizer_fixed.step()
                if len(losses_fixed)>min_tr:
                    if np.isclose(losses_fixed[-1],losses_fixed[-2],rtol=1e-5)\
                            and np.isclose(losses_fixed[-2], losses_fixed[-3],rtol=1e-5):
                        break
            
            fixed_model.eval()
            likelihood_fixed.eval()
            
            if verbose:
                print_hyperparams(fixed_model, cuda=self.cuda)

            if visualize:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    obs_fixed = likelihood_fixed(fixed_model(x_model))
                plot_gp_gpytorch(x_model.cpu(), y_model.cpu(), x_model.cpu(), obs_fixed, title = "Basis model")
            
            size_var_warp = self.x_fixed.shape[0]
            if not self.recursive or self.T < 2:
                U_dist = pyro.nn.PyroSample(dist.LogNormal(0.0, 1e-4))
                U_var = U_dist[0].sample((size_var_warp,)).detach()
                self.prior = torch.cat((self.prior, U_var.clone().view(-1, size_var_warp)), 0).view(-1, size_var_warp)
            else:
                U_var = self.prior[-1]

            if torch.cuda.is_available() and self.cuda:
                U_var = U_var.cuda()

            U_var.requires_grad = True
            G_prior = self.monotonic_sequence(U_var, torch.atleast_2d(x_model).T)

            #CUDA OPTION
            if torch.cuda.is_available() and self.cuda:
                G_prior = G_prior.cuda()

            f, cov = self.warp_gp.observe_last(torch.atleast_2d(x_model).T)
            noise_warp = torch.diag(cov)# * 0.5
            warp_lik = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise_warp)
            warp_gp = LinearExactGPModel(x_model, G_prior, warp_lik)
            warp_gp.covar_module.base_kernel.raw_lengthscale_constraint = \
               gpytorch.constraints.Interval(1.0, 1.01)
            if torch.cuda.is_available() and self.cuda:
                warp_gp = warp_gp.cuda()
                warp_lik = warp_lik.cuda()

            training_iter = 1500
            if self.mode == 'fine' or self.mode == 'balanced':
                lr_ = 0.01
                lr_p = 0.3
            elif self.mode == 'rough':
                lr_ = 0.06
                lr_p = 0.5
            else:
                lr_ = 0.01
                lr_p = 0.3
            optimizer = torch.optim.Adam([{'params': U_var}, {'params': warp_gp.parameters(), 'lr': lr_p}], lr=lr_)
            models = gpytorch.models.IndependentModelList(fixed_model, warp_gp)
            mll_warp = gpytorch.mlls.ExactMarginalLogLikelihood(warp_lik, warp_gp)
            likelihoods = gpytorch.likelihoods.LikelihoodList(likelihood_fixed, warp_lik)
            mll_join = gpytorch.mlls.SumMarginalLogLikelihood(likelihoods, models)
            losses = []

            if self.mode == 'fine':
                lim = 2000
                training_iter = 5000
                tol_ = 1e-5
            elif self.mode == 'rough':
                lim = 100
                training_iter = 700
                tol_ = 5e-2
            else:
                lim = 200
                training_iter = 1500
                tol_ = 1e-2

            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                G_prior = self.monotonic_sequence(U_var, torch.atleast_2d(x_model).T)
                #Working one
                output = models(G_prior, x_model)
                # Calc loss and backprop gradients
                loss = -mll_join(output, [y_target, G_prior]) + torch.sum(torch.square(U_var))
                # loss.backward(retain_graph=True)
                loss.backward()
                losses.append(loss.item())
                if verbose and i%10==0:
                    with torch.no_grad():
                        print('Iter %d/%d - Loss: %.3f - Sq.U_var: %.3f' % (
                            i + 1, training_iter, loss.item(), torch.sum(torch.square(U_var))))
                optimizer.step()
                if len(losses) > lim:
                    if np.isclose(np.sum(np.subtract(losses[-10:], losses[-11:-1])), 0, atol=tol_):
                        break
                        
            if visualize:
                plt.figure(77, figsize=(12,6))
                plt.title("Losses of warp computing")
                plt.plot(losses)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    warp_sample = warp_lik(warp_gp(x_model))
                plot_gp_gpytorch(x_model.cpu(), G_prior.cpu().detach(), x_model.cpu(), warp_sample, title = "Warp GP model")
                
                plt.figure(79, figsize=(12,6))
                plt.title("Warping function")
                plt.plot(x_model.cpu(), x_model.cpu())
                plt.plot(x_model.cpu(), G_prior.cpu().detach())
                
                aligned_example = likelihood_fixed(fixed_model(G_prior))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    aligned_example1 = likelihood_fixed(fixed_model(x_model))
         
                plot_gp_gpytorch(torch.cat([x_model.cpu(),x_model.cpu()]), torch.cat([y_model.cpu(),y_target.cpu()]), x_model.cpu(), aligned_example1, 
                                 title = "Model and target observations")
                plot_gp_gpytorch(x_model.cpu(), y_target.cpu(), x_model.cpu(), aligned_example, title = "Warped model")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if torch.cuda.is_available() and self.cuda:
                    wp_ = G_prior.detach() - x_model.detach()
                else:
                    wp_ = np.atleast_2d(G_prior.detach().numpy()-x_model.detach().numpy()).T
                lik_warp = self.warp_gp.log_sq_error(torch.atleast_2d(x_model).T, wp_).item()
                
            if torch.cuda.is_available() and self.cuda:
                x_warp = G_prior - x_model
                x_warp = x_warp[:, None].detach()
                y_warp = fixed_model(G_prior).loc
                y_warp = y_warp[:, None].detach()
            else:
                x_warp = torch.atleast_2d(G_prior.detach()-x_model.detach()).T
                y_warp = torch.atleast_2d(fixed_model(G_prior).loc.detach()).T

            if verbose:
                print("----Warping computed----")
        except ValueError as e:
            print("Model so tight, trying reduce train_iter")
            if train_iter < 20:
                raise(e)
            if torch.cuda.is_available() and self.cuda:
                x_model = np.atleast_2d(x_model.detach().cpu()).T
                y_model = np.atleast_2d(y_model.detach().cpu()).T
                y_target = np.atleast_2d(y_target.detach().cpu()).T
                theta = theta.detach().numpy()
            else:
                x_model = np.atleast_2d(x_model.detach()).T
                y_model = np.atleast_2d(y_model.detach()).T
                y_target = np.atleast_2d(y_target.detach()).T
                theta = theta.detach().numpy()
            if noise is None:
                noise = noise
            elif len(noise.detach())==1:
                noise = noise.detach()
            else:
                noise = np.atleast_2d(noise.detach()).T
            x_warp, y_warp, lik_warp, losses = self.compute_warp(x_model, y_model, y_target, theta,
                                                            noise=noise, visualize=visualize,
                                                            verbose=verbose, train_iter=train_iter-5)
        if self.recursive:
            if torch.cuda.is_available() and self.cuda:
                self.prior = torch.cat((self.prior,U_var.detach().cpu().clone().view(-1,size_var_warp)),0).view(-1,size_var_warp)
            else:
                self.prior = torch.cat((self.prior,U_var.detach().clone().view(-1,size_var_warp)),0).view(-1,size_var_warp)
        return x_warp, y_warp, lik_warp, losses
