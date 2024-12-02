# -*- coding: utf-8 -*-

#Directory check cell

import os

cwd = os.path.join(os.getcwd(), "hdpgpc")
data_dir = os.path.join(os.path.join(cwd, "data"), "mitbih")
#Import cell and config cell

import hdpgpc.GPI_HDP as hdpgp
import numpy as np
import torch
import time
import sys

dtype = torch.float64
torch.set_default_dtype(dtype)

from hdpgpc.get_data import compute_estimators_LDS
from hdpgpc.util_plots import plot_models_plotly, print_results


#Get data cell
#Select record to work with
if len(sys.argv) > 1:
    rec = sys.argv[1]
    #rec = "100"
else:
    rec = "102"

#Data should have the shape [num_samples, num_obs_per_sample, num_outputs]
data = np.load(os.path.join(data_dir, rec+".npy"))
labels = np.load(os.path.join(data_dir, rec+"_labels.npy"))

#Select the lead of the data to work with
lead = 0
data = data[:,:,[lead]]
num_samples, num_obs_per_sample, num_outputs = data.shape
#Take a small batch to estimate the priors.
n_f = 20
std, std_dif, bound_sigma, bound_gamma = compute_estimators_LDS(data, n_f)
#Define the priors
#Bound_sigma refers to bound for the observation noise of the initial GP
#Ini_lengthscale and bound_lengthscale refers to initial lengthscale of the initial GP
#Sigma is the diag value of the observation noise of the LDS
#Gamma is the diag value of the latent noise of the LDS
#Outputscale is the max amplitude of the records (if data is standardized it can be set to 1.0)
M = 2
sigma = std * 1.0
gamma = std_dif * 1.0
outputscale_ = 300.0
ini_lengthscale = 3.0
bound_lengthscale = (1.0, 20.0)
#Warp priors
noise_warp = std * 0.1
bound_noise_warp = (noise_warp * 0.1, noise_warp * 0.2)

#Now define time index support. Data in this case is taken on samples [60,140]
samples = [0, num_obs_per_sample]
l, L = samples[0], samples[1]
x_basis = np.atleast_2d(np.arange(l, L, 1, dtype=np.float64)).T

#x_warp can be defined on smaller set to ensure smoothness of the warp
x_basis_warp = np.atleast_2d(np.arange(l, L, 2, dtype=np.float64)).T
x_train = np.atleast_2d(np.arange(l, L, dtype=np.float64)).T

#Define the model
warp = False
sw_gp = hdpgp.GPI_HDP(x_basis, x_basis_warp=x_basis_warp, n_outputs=num_outputs, kernels=None, model_type='dynamic',
                          ini_lengthscale=ini_lengthscale, bound_lengthscale=bound_lengthscale,
                          ini_gamma=gamma, ini_sigma=sigma, ini_outputscale=outputscale_, noise_warp=noise_warp,
                          bound_sigma=bound_sigma, bound_gamma=bound_gamma, bound_noise_warp=bound_noise_warp,
                          warp_updating=warp, method_compute_warp='greedy', verbose=True,
                          hmm_switch=True, max_models=100, mode_warp='rough',
                          bayesian_params=True, inducing_points=False, estimation_limit=30)


start_ini_time = time.time()
for i in range(data.shape[0]):
    start_time = time.time()
    print("Sample:", i, "/", str(data.shape[0]-1), "label:", labels[i])
    sw_gp.include_sample(x_train, data[i], with_warp=warp)
    print("Time --- %s seconds ---" % (time.time() - start_time))

#Print results
print("Time --- %s mins ---" % str((time.time() - start_ini_time)/60.0))
out = os.path.join(os.path.join(cwd,"results"), "Rec" + rec + "_")
main_model = print_results(sw_gp, labels, 0, error=False)
selected_gpmodels = sw_gp.selected_gpmodels()
plot_models_plotly(sw_gp, selected_gpmodels, main_model, labels, 0, save=out+"Online_Clusters.png",
                lead=0, step=0.5, plot_latent=True)