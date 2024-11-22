# -*- coding: utf-8 -*-

# %% Directory check cell

import os

homedir = os.getenv('HOME')
if homedir is None:
    os.chdir("D:/Programs/Workspaces/spyder-workspace/HDP-GPC")
else:
    try:
        os.chdir(homedir + "/Documents/HDP-GPC")
    except FileNotFoundError:
        aux_dir = input("Please specify directory:\n")
        os.chdir(aux_dir)

cwd = os.getcwd()

# %% Import cell and config cell

import GPI_HDP as hdpgp
import numpy as np
import torch

dtype = torch.float64
torch.set_default_dtype(dtype)

from get_data import get_data, take_standard_labels
from util_plots import plot_models, print_results

import time
start_ini_time = time.time()
# %% Get data cell
#samples = [50, 130]
samples = [60, 140]
M = 2
#samples = [0, -1]
rec = "102"
data, labels = get_data(database="mitdb", record=rec, deriv=0, test=False,
                        scale_data=True, scale_type="mean", d2_data=False, samples=samples, ann='atr')
dat_ = data
data, data_2d, labels = take_standard_labels(data, labels, filter=labels)
n_f = 20
# if len(data.shape) == 3:
#     std = np.sqrt(np.mean(np.min(np.var(dat_, axis=0), axis=1)))
#     std_dif = np.sqrt(np.mean(np.sqrt(np.min(np.sum((np.array(dat_[1:]) - np.array(dat_[:-1])) ** 2, axis=0), axis=1))))
# else:
#     std = np.sqrt(np.mean(np.var(dat_, axis=0)))
#     std_dif = np.sqrt(np.mean(np.sqrt(np.sum((np.array(dat_[1:]) - np.array(dat_[:-1])) ** 2, axis=0))))
data_2d_t = torch.from_numpy(np.array(data_2d))
samples = data_2d_t[:n_f][:, :, 0].T
samples_ = data_2d_t[1:n_f+1][:, :, 0].T
std = torch.sqrt(torch.mean(torch.diag(torch.linalg.multi_dot([(samples - torch.mean(samples, dim=1)[:,np.newaxis]), (samples - torch.mean(samples, dim=1)[:,np.newaxis]).T])) / n_f)).item()# * 0.15
std_dif = torch.sqrt(torch.mean(torch.diag(torch.linalg.multi_dot([(samples_ - samples), (samples_ - samples).T])) / n_f)).item()
bound_sigma_, sig_, outputscale_ = (std * 1.0, std * 1.1), std * 3.0, 300.0 #v42
gamma = [std_dif * 5.0] * M
ini_lengthscale = 10.0
bound_lengthscale = (2.0, 20.0)
sigma = [sig_ * 1.0] * M
noise_warp = bound_sigma_[0] * 0.5
bound_noise_warp = (bound_sigma_[0] * 0.1, bound_sigma_[0] * 0.2)
bound_gamma = (0.1 ** 2, 20.0 ** 2)

#Print hyperparams
print("Bound Sigma: ", bound_sigma_)
print("Sigma: ", sigma)
print("Outputscale: ", outputscale_)
print("Gamma: ", gamma)

N_0 = 0
N = 2228
# N_0 = 30
# N = 4
l = 0
L = len(data[0])


# data, data_2d, labels = take_standard_labels(data, labels)


N = len(data) - 1
#N = len(data)
# N = 165
x_basis = np.atleast_2d(np.arange(l, L, 1, dtype=np.float64)).T
# ini_lengthscale = 5.0
# x_basis = np.atleast_2d([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0]).T
x_basis_warp = np.atleast_2d(np.arange(l, L, 2, dtype=np.float64)).T
x_train = np.atleast_2d(np.arange(l, L, dtype=np.float64)).T

# kernels =[]
# for i in range(M):
# kernels.append(ExpSineSquared(1, 1, (1e-10, 1e15),(1e-10, 1e15)))
# kernels.append(Matern())
warp = True

sw_gp = hdpgp.GPI_HDP(x_basis, x_basis_warp=x_basis_warp, kernels=None, model_type='dynamic',
                          ini_lengthscale=ini_lengthscale, bound_lengthscale=bound_lengthscale,
                          ini_gamma=gamma, ini_sigma=sigma, ini_outputscale=outputscale_, noise_warp=noise_warp,
                          bound_sigma=bound_sigma_, bound_gamma=bound_gamma, bound_noise_warp=bound_noise_warp,
                          warp_updating=True, method_compute_warp='greedy', verbose=True,
                          annealing=False, hmm_switch=True, max_models=100, batch=0, mode_warp='rough',
                          check_var=False, bayesian_params=True, cuda=False, inducing_points=False,
                          reestimate_initial_params=True, estimation_limit=50)

categories = np.unique(labels)
x_trains = [x_basis] * (N-N_0)
y_trains = np.array(data_2d)[N_0:N]
print(categories)
for i in range(N_0, min(N, len(data)), 1):
    start_time = time.time()
    print("Sample:", i, "/", str(N-1), "label:", labels[i])
    if i ==192:
        print("Stop")
    sw_gp.include_sample(x_trains[i], y_trains[i], with_warp=warp)#, force_model=labels[i]-1)# force_model=0 if labels[i]==-1 else 1)#
    print("Time --- %s seconds ---" % (time.time() - start_time))

print("Time --- %s mins ---" % str((time.time() - start_ini_time)/60.0))
out = cwd + "/plots/Run_" + time.asctime().replace(" ", "_").replace(":", "-") + "_Rec" + rec + "_"
#labels_full = labels + labels_test
#labels_full = np.array(labels, dtype=np.int32)
labels_full = labels[N_0:N]
main_model = print_results(sw_gp, labels_full, N_0, error=False)
selected_gpmodels = 0
for gp in sw_gp.gpmodels[0]:
    if len(gp.indexes) > 0:
        selected_gpmodels = selected_gpmodels + 1
selected_gpmodels = list(range(selected_gpmodels))
#selected_gpmodels = [0]
#selected_gpmodels = list(range(10))
plot_models(sw_gp, selected_gpmodels, main_model, labels_full, N_0, save=out+"Models_lead_1.html", lead=0, step=0.5, plot_latent=True)
#plot_models(sw_gp, selected_gpmodels, main_model, labels_full, N_0, save=out+"Models_lead_2.html", lead=1, step=0.5, plot_latent=True)
#plot_warp(sw_gp, selected_gpmodels, main_model, labels_full, N_0, save=out+"Warps.html")
#plot_MDS(sw_gp, main_model, labels_full, N_0, save=out+"MDS.html")

