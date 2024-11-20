# -*- coding: utf-8 -*-

# %% Directory check cell

import os
cwd = os.getcwd()

# %% Import cell and config cell

import hdpgpc.GPI_HDP as hdpgp
import numpy as np
import torch

dtype = torch.float64
torch.set_default_dtype(dtype)

from hdpgpc.get_data import get_data, take_standard_labels
from hdpgpc.util_plots import plot_models, print_results

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