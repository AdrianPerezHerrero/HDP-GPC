#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os

import xarray
from scipy.signal import sweep_poly

from hdpgpc.get_data import compute_estimators_LDS
import math
from scipy.fft import fft
import h5py
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import wavespectra as wv
import xarray as xr
import importlib
import pickle
import hdpgpc.GPI_HDP as hdpgp
import re
from datetime import datetime, timedelta
import time
#%%
from hdpgpc.util_plots import print_results
import hdpgpc.util_plots as up
importlib.reload(up)
from hdpgpc.buoy_utils import interpolate_spectral_coefficients


#Config
interpolate_dense=True
#%%
from datetime import datetime, timedelta

def matlab_datenum_to_datetime(matlab_datenum):
    """
    Convert MATLAB datenum to Python datetime.
    """
    days = int(matlab_datenum)
    frac = matlab_datenum % 1
    python_datetime = datetime.fromordinal(days - 366) + timedelta(days=frac)
    return python_datetime
#%% md
# ---
#%% md
# ### 1. Lectura de datos
#%%
cwd = os.path.dirname(os.getcwd())
data_path = os.path.join(cwd, 'data', 'ocean', 'wawaves')
f_hillary = h5py.File(os.path.join(data_path, 'Hillarys_202407.mat'),'r')
f_hillary_2 = h5py.File(os.path.join(data_path, 'Hillarys_202408.mat'),'r')
#f = h5py.File(os.path.join(data_path, 'TIDE_SouthAfricaDrifting03_202406.mat'),'r')
print(f_hillary.get('SpotData').keys())
direction = np.concatenate([np.array(f_hillary.get('SpotData/direction')), np.array(f_hillary_2.get('SpotData/direction'))], axis=1)
variance_density = np.concatenate([np.array(f_hillary.get('SpotData/varianceDensity')), np.array(f_hillary_2.get('SpotData/varianceDensity'))], axis=1)
spec_time_hillarys = np.concatenate([np.array(f_hillary.get('SpotData/spec_time')), np.array(f_hillary_2.get('SpotData/spec_time'))], axis=1)
time_hillarys = np.concatenate([np.array(f_hillary.get('SpotData/time')), np.array(f_hillary_2.get('SpotData/time'))], axis=1)
frequency_hillarys = np.concatenate([np.array(f_hillary.get('SpotData/frequency')), np.array(f_hillary_2.get('SpotData/frequency'))], axis=1)
dm_h = np.concatenate([np.array(f_hillary.get('SpotData/dm')), np.array(f_hillary_2.get('SpotData/dm'))], axis=1)
dp_h = np.concatenate([np.array(f_hillary.get('SpotData/dp')), np.array(f_hillary_2.get('SpotData/dp'))], axis=1)
a1 = np.concatenate([np.array(f_hillary.get('SpotData/a1')), np.array(f_hillary_2.get('SpotData/a1'))], axis=1)
a2 = np.concatenate([np.array(f_hillary.get('SpotData/a2')), np.array(f_hillary_2.get('SpotData/a2'))], axis=1)
b1 = np.concatenate([np.array(f_hillary.get('SpotData/b1')), np.array(f_hillary_2.get('SpotData/b1'))], axis=1)
b2 = np.concatenate([np.array(f_hillary.get('SpotData/b2')), np.array(f_hillary_2.get('SpotData/b2'))], axis=1)

print(variance_density.shape)
print(direction.shape)
print(a1.shape)
print(np.array(f_hillary.get('SpotData/dmspr')).shape)


#%%
if interpolate_dense:
    result = interpolate_spectral_coefficients(
        frequency_hillarys[:,0],
        variance_density,
        a1, b1, a2, b2,
        n_points=200,
        enforce_constraints=True
    )
    frequency_hillarys_dense = np.array([result['f_dense']]*frequency_hillarys.shape[1]).T
    S_dense = result['variance_density_dense']
    plt.plot(frequency_hillarys[:,0],variance_density[:,0])
    plt.plot(frequency_hillarys_dense[:,0], S_dense[:,0])

    frequency_hillarys = frequency_hillarys_dense
    variance_density = S_dense
    a1, b1, a2, b2 = result['a1_dense'], result['b1_dense'], result['a2_dense'], result['b2_dense']
#%%
startFreq_sea = np.array(f_hillary.get('SpotData/startFreq_sea'))[0][0]
startFreq_swell = np.array(f_hillary.get('SpotData/startFreq_swell'))[0][0]
print(frequency_hillarys.shape)
#%%
print("Shape of the data: ")
print(variance_density.shape)
num_obs_per_sample, num_samples  = variance_density.shape
#%%
S_theta =  np.zeros((variance_density.shape[1], variance_density.shape[0], 37))
directions = np.deg2rad(np.linspace(0, 360.0, 37))
delta_theta = np.deg2rad(10.0)
for t in range(S_theta.shape[0]):
    for f in range(S_theta.shape[1]):
        # Load S(f), a1, b1, a2, b2 for this time and frequency
        S = variance_density[f, t]# Omnidirectional spectrum
        a1_, b1_, a2_, b2_ = a1[f,t], b1[f,t], a2[f,t], b2[f,t]  # Directional moments

        for i, theta in enumerate(directions):
            # Compute D(f, theta)
            D = (1 / 2 * np.pi) * (
                    1 + (a1_ * np.cos(theta) + b1_ * np.sin(theta))
                    + (a2_ * np.cos(2 * theta) + b2_ * np.sin(2 * theta))
            )

            # Ensure non-negativity
            D = max(D, 0)

            # Compute S(f, theta)
            S_theta[t, f, i] = S * D

        # Optional: Renormalize to ensure sum(S_theta * delta_theta) ≈ S
        integral = np.sum(S_theta[t, f, :]) * delta_theta
        integral = integral if integral > 0 else 1.0
        S_theta[t, f, :] *= S / integral  # Adjust if integral != S
#%%
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import hdpgpc.buoy_utils as buoy_utils

dirs = np.linspace(0, 360.0, 37)
dirs_rad= np.deg2rad(dirs)

#ind_ = 0
S_theta_rotated = np.zeros_like(S_theta)

for ind_ in range(0, num_samples, 1):
    time_h = matlab_datenum_to_datetime(spec_time_hillarys[:, ind_].item())
    ind_dp = np.argmin(np.abs(time_hillarys - spec_time_hillarys[:, ind_]))
    error_mean_, error_peak_ = buoy_utils.compute_rotation_deviation(S_theta[ind_], dirs, a1[:,ind_], b1[:,ind_], dm_h[:,ind_dp], dp_h[:,ind_dp])
    error = -np.mean([error_mean_, error_peak_])
    dirs_rotated = (dirs + error) % 360.0
    f_interp = interp1d(
        dirs,
        S_theta[ind_,:,:],
        kind='linear', # 'cubic' is smoother but can overshoot negatively
        axis=-1,       # Interpolate along direction axis
        assume_sorted=True
    )
    #indexes_directions_new = np.floor(dirs_rotated // 10).astype(np.int32)
    S_theta_rotated[ind_, :, :] = f_interp(dirs_rotated)

dataset = xr.Dataset(
    data_vars=dict(
        efth=(["time", "freq", "dir"], S_theta_rotated)
    ),
    coords=dict(
        time=(["time"], spec_time_hillarys[0]),
        freq=(["freq"], frequency_hillarys[:,0]),
        dir=(["dir"], dirs)
    )
)
dts_hillarys = wv.SpecDataset(dataset)
new_dirs = np.linspace(0, 360, 37)# Cada 10° incluyendo 360°
rotated_dirs = (180 + np.linspace(0, 360, 37))%360
directions_hillarys = new_dirs
data_ = dts_hillarys.interp(dir=rotated_dirs)
freq = data_.freq
freq_hillarys = data_.freq

time_select_datetime = datetime(2024, 8, 2, 0)
time_select_ordinal = datetime.toordinal(time_select_datetime + timedelta(days = 366))
index_hillary = np.argmin(np.abs(spec_time_hillarys - time_select_ordinal))
time_h = matlab_datenum_to_datetime(spec_time_hillarys[:, index_hillary].item())
ind_dp = np.argmin(np.abs(time_hillarys - spec_time_hillarys[:, index_hillary]))
print(f"Main direction on dataset: {dm_h[:, ind_dp]}")
print("Time event")
print(time_select_datetime)
print(index_hillary)
fig, ax = buoy_utils.plot_directional_spectrum(freq[:40], directions_hillarys, data_[index_hillary,:40],
                                        title='Hillary Buoy - Directional Spectrum of time '+ str(time_select_datetime))

#%%
# data_shoal = np.load("/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/data/ocean/wawaves/Hillarys_0708_shoaled.npy")
# print(data_shoal.shape)
# data = data_shoal
data = data_.to_numpy()
freq_ = np.array(freq)
if not interpolate_dense:
    freq = np.array(freq)
#%%
 #Here we are going to compute the Heighs and try to filter following Hamiltons rule.
interv = np.repeat(freq_[1]-freq_[0], num_obs_per_sample)[:, np.newaxis]
hs_h = 4 * np.sqrt(np.sum(data, axis=2) * delta_theta @ interv)
print("Shape of hs: "+str(hs_h.shape))
plt.plot(hs_h)
chosen_indexes = np.where((hs_h > 0.5) & (hs_h < 1.5))[0]
chosen_indexes = np.arange(num_samples)
print("How much spectra falls in the range 0.5-1.5 Hs: " + str(chosen_indexes.shape[0]))
plt.show()
plt.plot(freq, data[:,:,0][chosen_indexes].T)
plt.show()
lognorm_data = np.log(data + 1e-6) - np.mean(np.log(data+ 1e-6), axis=1)[:,np.newaxis,:]
plt.plot(freq, lognorm_data[:,:,0][chosen_indexes].T)
plt.show()

#%% md
# ### Se cogen solo los datos de un año natural
#%%
#Select indexes
data = data[chosen_indexes]
print("Shape of the data: ")
print(data.shape)

num_samples, num_obs_per_sample, num_outputs = data.shape
#%%
lim_freq = startFreq_sea
index_freq = np.where(freq>lim_freq)[0][0]+1
std, std_dif, bound_sigma, bound_gamma = compute_estimators_LDS(data[:,0:index_freq,:], n_f=50, dim=25)
#Seem like these estimators are so big for this data, let's reduce them
# std = std * 0.5
# std_dif = std_dif * 0.48
std = std * 0.5
std_dif = std_dif * 0.40
bound_sigma = (std * 1e-7, std * 1e-6)
bound_gamma = (std_dif * 1e-9, std_dif * 1e-8)

print("Final sigma:", std)
print("Final gamma:", std_dif)
print("Final sigma bound:", bound_sigma)
print("Final gamma bound:", bound_gamma)

M = 2
sigma = [std * 1.0] * M
gamma = [std_dif * 1.0] * M
outputscale_ = 0.1
ini_lengthscale = 1e-3
bound_lengthscale = (1e-7, 5e-3)
samples = [0, num_obs_per_sample]
l, L = samples[0], samples[1]
# x_basis has to have the same dimension structure as data[0], in this case (171,1) but it could be (n_inducing_points, 1).
lim_freq = startFreq_sea
ini_lim_freq = 0.040
index_freq = np.where(freq>lim_freq)[0][0]+1
index_ini = np.where(freq>ini_lim_freq)[0][0]+1
x_basis = np.atleast_2d(freq[index_ini:index_freq]).T
x_train = np.atleast_2d(freq[index_ini:index_freq]).T
print(x_train.shape)
#If x_basis is wanted to be smaller than the observations length, then the inducing points approach can be applied setting this parameter to True.
inducing_points = False
#Choose if warp is going to be applied. (In the most recent version is optimized to work with online inference, but it can be used in offline as an additional step at the end of the clustering).
warp = False
#Warp priors
noise_warp = std * 0.1
bound_noise_warp = (noise_warp * 0.1, noise_warp * 0.2)
#Warp time indexes
x_basis_warp = np.atleast_2d(np.arange(freq[0], freq[-1], freq.shape[0] / 2.0, dtype=np.float64)).T
#%%
sw_gp_hillary = hdpgp.GPI_HDP(x_basis=x_basis, x_basis_warp=x_basis_warp, n_outputs=37,
                          ini_lengthscale=ini_lengthscale, bound_lengthscale=bound_lengthscale,
                          ini_gamma=gamma, ini_sigma=sigma, ini_outputscale=outputscale_, noise_warp=noise_warp,
                          bound_sigma=bound_sigma, bound_gamma=bound_gamma, bound_noise_warp=bound_noise_warp,
                          verbose=False, max_models=100, inducing_points=inducing_points, reestimate_initial_params=False,
                          n_explore_steps=15, free_deg_MNIV=8, share_gp=True, use_snr=False, reduce_outputs=True, reduce_outputs_ratio=0.05)
#Good labels: 5
cluster_labels_h = np.load('/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/cluster_labels/resampled_low_freq/cluster_labels_hillary_202407_dynamic_5.npy')
#cluster_labels_h = np.load('/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/cluster_labels/cluster_labels_hillary_202407_dynamic_7.npy')
# ------ Cluster grouping on dynamic_3 ------
# cluster_labels_h[np.where(cluster_labels_h == 1)] = 0
# #cluster_labels_h[np.where(cluster_labels_h == 5)] = 0
# cluster_labels_h[np.where(cluster_labels_h == 2)] = 1
# cluster_labels_h[np.where(cluster_labels_h == 3)] = 2
# cluster_labels_h[np.where(cluster_labels_h == 4)] = 3
# cluster_labels_h[np.where(cluster_labels_h == 6)] = 4
#
# #After-transformations
# cluster_labels_h[np.where(cluster_labels_h == 4)] = 3
# cluster_labels_h[np.where(cluster_labels_h == 5)] = 4

# ------ Cluster grouping on dynamic ------
# cluster_labels_h[np.where(cluster_labels_h == 3)] = 0
# cluster_labels_h[np.where(cluster_labels_h == 5)] = 0
# cluster_labels_h[np.where(cluster_labels_h == 6)] = 0
# cluster_labels_h[np.where(cluster_labels_h == 7)] = 0
#
# cluster_labels_h[np.where(cluster_labels_h == 4)] = 3
# cluster_labels_h[np.where(cluster_labels_h == 8)] = 4
# cluster_labels_h[np.where(cluster_labels_h == 9)] = 5
#
# #After transform
# cluster_labels_h[np.where(cluster_labels_h == 3)] = 2
# cluster_labels_h[np.where(cluster_labels_h == 4)] = 2
# cluster_labels_h[np.where(cluster_labels_h == 5)] = 2

# ------ Cluster grouping on dynamic 4 ------

# cluster_labels_h[np.where(cluster_labels_h == 2)] = 1
M = np.unique(cluster_labels_h).shape[0]
num_samples = data.shape[0]
#logdata = np.log(data + 1e-6)
x_trains = np.array([x_train] * num_samples)
sw_gp_hillary.reload_model_from_labels(x_trains, data[:,index_ini:index_freq,:], cluster_labels_h, M)
#%%
sw_gp_hillary.gpmodels[0][0].x_train[0].shape
#%%
labels = np.array(['N'] * 6500)
main_model = print_results(sw_gp_hillary, labels, 0, error=False)
selected_gpmodels = sw_gp_hillary.selected_gpmodels()
up.plot_models_plotly(sw_gp_hillary, selected_gpmodels, main_model, labels, N_0=0, lead=26, step=(freq[1]-freq[0])/1,
                   plot_latent=False, title='WAVE CLUSTER',ticks=True, yscale=True, save="/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/clusters_hillary.png", line_max=True)
#%% md
# Now with drifting 03.
#%%

cwd = os.path.dirname(os.getcwd())
data_path = os.path.join(cwd, 'data', 'ocean', 'wawaves')
#f = h5py.File(os.path.join(data_path, 'Hillarys_202405.mat'), 'r')
f_drift03 = h5py.File(os.path.join(data_path, 'TIDE_SouthAfricaDrifting03_202407.mat'),'r')
f_drift03_2 = h5py.File(os.path.join(data_path, 'TIDE_SouthAfricaDrifting03_202408.mat'),'r')
print(f_drift03.get('SpotData').keys())
direction = np.concatenate([np.array(f_drift03.get('SpotData/direction')), np.array(f_drift03_2.get('SpotData/direction'))], axis=1)
variance_density = np.concatenate([np.array(f_drift03.get('SpotData/varianceDensity')), np.array(f_drift03_2.get('SpotData/varianceDensity'))], axis=1)
spec_time_drift03 = np.concatenate([np.array(f_drift03.get('SpotData/spec_time')), np.array(f_drift03_2.get('SpotData/spec_time'))], axis=1)
time_drift03 = np.concatenate([np.array(f_drift03.get('SpotData/time')), np.array(f_drift03_2.get('SpotData/time'))], axis=1)
frequency = np.concatenate([np.array(f_drift03.get('SpotData/frequency')), np.array(f_drift03_2.get('SpotData/frequency'))], axis=1)
dm_d03 = np.concatenate([np.array(f_drift03.get('SpotData/dm')), np.array(f_drift03_2.get('SpotData/dm'))], axis=1)
dp_d03 = np.concatenate([np.array(f_drift03.get('SpotData/dp')), np.array(f_drift03_2.get('SpotData/dp'))], axis=1)
a1 = np.concatenate([np.array(f_drift03.get('SpotData/a1')), np.array(f_drift03_2.get('SpotData/a1'))], axis=1)
a2 = np.concatenate([np.array(f_drift03.get('SpotData/a2')), np.array(f_drift03_2.get('SpotData/a2'))], axis=1)
b1 = np.concatenate([np.array(f_drift03.get('SpotData/b1')), np.array(f_drift03_2.get('SpotData/b1'))], axis=1)
b2 = np.concatenate([np.array(f_drift03.get('SpotData/b2')), np.array(f_drift03_2.get('SpotData/b2'))], axis=1)
print("Shape of the data: ")
print(variance_density.shape)

if interpolate_dense:
    result = interpolate_spectral_coefficients(
        frequency[:,0],
        variance_density,
        a1, b1, a2, b2,
        n_points=200,
        enforce_constraints=True
    )
    frequency_drift03_dense = np.array([result['f_dense']]*frequency.shape[1]).T
    S_dense = result['variance_density_dense']
    plt.plot(frequency[:,0],variance_density[:,0])
    plt.plot(frequency_drift03_dense[:,0], S_dense[:,0])

    frequency = frequency_drift03_dense
    variance_density = S_dense
    a1, b1, a2, b2 = result['a1_dense'], result['b1_dense'], result['a2_dense'], result['b2_dense']
#%%
num_obs_per_sample, num_samples = variance_density.shape
S_theta = np.zeros((variance_density.shape[1], variance_density.shape[0], 37))
directions = np.deg2rad(np.linspace(0, 360.0, 37))
delta_theta = np.deg2rad(10.0)
#%%
for t in range(S_theta.shape[0]):
    for f in range(S_theta.shape[1]):
        # Load S(f), a1, b1, a2, b2 for this time and frequency
        S = variance_density[f, t]  # Omnidirectional spectrum
        a1_, b1_, a2_, b2_ = a1[f, t], b1[f, t], a2[f, t], b2[f, t]  # Directional moments
        if a1_ == -9999.0:
            a1_, b1_, a2_, b2_ = 0.0, 0.0, 0.0, 0.0
        for i, theta in enumerate(directions):
            # Compute D(f, theta)
            D = (1 / 2 * np.pi) * (
                    1 + (a1_ * np.cos(theta) + b1_ * np.sin(theta))
                    + (a2_ * np.cos(2 * theta) + b2_ * np.sin(2 * theta))
            )

            # Ensure non-negativity
            D = max(D, 0)

            # Compute S(f, theta)
            S_theta[t, f, i] = S * D

        # Optional: Renormalize to ensure sum(S_theta * delta_theta) ≈ S
        integral = np.sum(S_theta[t, f, :]) * delta_theta
        integral = integral if integral > 0 else 1.0
        S_theta[t, f, :] *= S / integral  # Adjust if integral != S

#%%

import hdpgpc.buoy_utils as buoy_utils
dirs = np.linspace(0, 360.0, 37)

S_theta_rotated = np.zeros_like(S_theta)

for ind_ in range(0, num_samples, 1):
    ind_dp = np.argmin(np.abs(time_drift03 - spec_time_drift03[:, ind_]))
    error_mean_, error_peak_ = buoy_utils.compute_rotation_deviation(S_theta[ind_], dirs, a1[:,ind_], b1[:,ind_], dm_d03[:,ind_dp], dp_d03[:,ind_dp])
    error = -np.mean([error_mean_, error_peak_])
    dirs_rotated = (dirs + error) % 360.0
    f_interp = interp1d(
        dirs,
        S_theta[ind_,:,:],
        kind='linear', # 'cubic' is smoother but can overshoot negatively
        axis=-1,       # Interpolate along direction axis
        assume_sorted=True
    )
    #indexes_directions_new = np.floor(dirs_rotated // 10).astype(np.int32)
    S_theta_rotated[ind_, :, :] = f_interp(dirs_rotated)

dataset = xr.Dataset(
    data_vars=dict(
        efth=(["time", "freq", "dir"], S_theta_rotated)
    ),
    coords=dict(
        time=(["time"], spec_time_drift03[0]),
        freq=(["freq"], frequency[:, 0]),
        dir=(["dir"], dirs)
    )
)
dts_drift03 = wv.SpecDataset(dataset)
new_dirs = np.linspace(0, 360, 37)# Cada 10° incluyendo 360°
rotated_dirs = (180 + new_dirs) % 360
data_drifting03 = dts_drift03.interp(dir=rotated_dirs)
freq = data_drifting03.freq
data_drifting03 = data_drifting03.to_numpy()
freq_ = np.array(freq)
if not interpolate_dense:
    freq = np.array(freq)
#%%
#Here we are going to compute the Heighs and try to filter following Hamiltons rule.
interv = np.repeat(freq_[1] - freq_[0], num_obs_per_sample)[:, np.newaxis]
print(data_drifting03.shape)
print(interv.shape)
hs_d03 = 4 * np.sqrt(np.sum(data_drifting03, axis=2) * delta_theta @ interv)
print("Shape of hs: " + str(hs_d03.shape))
plt.plot(hs_d03)
chosen_indexes = np.where((hs_d03 > 0.5) & (hs_d03 < 1.5))[0]
chosen_indexes = np.arange(num_samples)
print("How much spectra falls in the range 0.5-1.5 Hs: " + str(chosen_indexes.shape[0]))
plt.show()
plt.plot(freq, data_drifting03[:, :, 0][chosen_indexes].T)
plt.show()
lognorm_data = np.log(data_drifting03 + 1e-6) - np.mean(np.log(data_drifting03 + 1e-6), axis=1)[:, np.newaxis, :]
plt.plot(freq, lognorm_data[:, :, 0][chosen_indexes].T)
plt.show()
#%%
data_drifting03 = data_drifting03[chosen_indexes]
print("Shape of the data: ")
print(data_drifting03.shape)

lim_freq = startFreq_sea
index_freq = np.where(freq>lim_freq)[0][0]+1
num_samples, num_obs_per_sample, num_outputs = data_drifting03.shape
std, std_dif, bound_sigma, bound_gamma = compute_estimators_LDS(data_drifting03[:,0:index_freq,:], n_f=50, dim=25)
#Seem like these estimators are so big for this data, let's reduce them
# std = std * 1.0
# std_dif = std_dif * 0.1
std = std * 0.40
std_dif = std_dif * 0.15
bound_sigma = (std * 1e-7, std * 1e-6)
bound_gamma = (std_dif * 1e-9, std_dif * 1e-8)

print("Final sigma:", std)
print("Final gamma:", std_dif)
print("Final sigma bound:", bound_sigma)
print("Final gamma bound:", bound_gamma)

M = 2
sigma = [std * 1.0] * M
gamma = [std_dif * 1.0] * M
outputscale_ = 0.1
ini_lengthscale = 1e-3
bound_lengthscale = (1e-7, 5e-3)
samples = [0, num_obs_per_sample]
l, L = samples[0], samples[1]
# x_basis has to have the same dimension structure as data[0], in this case (171,1) but it could be (n_inducing_points, 1).
lim_freq = startFreq_sea
ini_lim_freq = 0.040
index_freq = np.where(freq>lim_freq)[0][0]+1
index_ini = np.where(freq>ini_lim_freq)[0][0]+1
x_basis = np.atleast_2d(freq[index_ini:index_freq]).T
x_train = np.atleast_2d(freq[index_ini:index_freq]).T
print(x_train.shape)
#If x_basis is wanted to be smaller than the observations length, then the inducing points approach can be applied setting this parameter to True.
inducing_points = False
#Choose if warp is going to be applied. (In the most recent version is optimized to work with online inference, but it can be used in offline as an additional step at the end of the clustering).
warp = False
#Warp priors
noise_warp = std * 0.1
bound_noise_warp = (noise_warp * 0.1, noise_warp * 0.2)
#Warp time indexes
x_basis_warp = np.atleast_2d(np.arange(freq[0], freq[-1], freq.shape[0] / 2.0, dtype=np.float64)).T
#%%
sw_gp_drift03 = hdpgp.GPI_HDP(x_basis=x_basis, x_basis_warp=x_basis_warp, n_outputs=37,
                          ini_lengthscale=ini_lengthscale, bound_lengthscale=bound_lengthscale,
                          ini_gamma=gamma, ini_sigma=sigma, ini_outputscale=outputscale_, noise_warp=noise_warp,
                          bound_sigma=bound_sigma, bound_gamma=bound_gamma, bound_noise_warp=bound_noise_warp,
                          verbose=False, max_models=100, inducing_points=inducing_points, reestimate_initial_params=False,
                          n_explore_steps=15, free_deg_MNIV=8, share_gp=True, use_snr=False, reduce_outputs=True, reduce_outputs_ratio=0.2)
#Good labels: 10
cluster_labels_d03 = np.load('/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/cluster_labels/resampled_low_freq/cluster_labels_drift03_202407_dynamic_10.npy')
#cluster_labels_d03 = np.load('/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/cluster_labels/cluster_labels_drift03_202407_dynamic_10.npy')


M = np.unique(cluster_labels_d03).shape[0]
num_samples = data_drifting03.shape[0]
#logdata = np.log(data + 1e-6)
x_trains = np.array([x_train] * num_samples)
sw_gp_drift03.reload_model_from_labels(x_trains, data_drifting03[:,index_ini:index_freq,:], cluster_labels_d03, M)
#%%
labels = np.array(['N'] * 6500)
main_model = print_results(sw_gp_drift03, labels, 0, error=False)
selected_gpmodels = sw_gp_drift03.selected_gpmodels()
up.plot_models_plotly(sw_gp_drift03, selected_gpmodels, main_model, labels, N_0=0, lead=26, step=(freq[1]-freq[0])/1,
                   plot_latent=False, title='WAVE CLUSTER',ticks=True, yscale=True, save="/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/clusters_drift03.png", line_max=True)
#%% md
# Now with drifting 06.
#%%

cwd = os.path.dirname(os.getcwd())
data_path = os.path.join(cwd, 'data', 'ocean', 'wawaves')
#f = h5py.File(os.path.join(data_path, 'Hillarys_202405.mat'), 'r')
f_drift06 = h5py.File(os.path.join(data_path, 'TIDE_SouthAfricaDrifting08_202407.mat'), 'r')
f_drift06_2 = h5py.File(os.path.join(data_path, 'TIDE_SouthAfricaDrifting08_202408.mat'), 'r')
print(f_drift06.get('SpotData').keys())
direction = np.concatenate([np.array(f_drift06.get('SpotData/direction')), np.array(f_drift06_2.get('SpotData/direction'))], axis=1)
variance_density = np.concatenate([np.array(f_drift06.get('SpotData/varianceDensity')), np.array(f_drift06_2.get('SpotData/varianceDensity'))], axis=1)
spec_time_drift06 = np.concatenate([np.array(f_drift06.get('SpotData/spec_time')), np.array(f_drift06_2.get('SpotData/spec_time'))], axis=1)
time_drift06 = np.concatenate([np.array(f_drift06.get('SpotData/time')), np.array(f_drift06_2.get('SpotData/time'))], axis=1)
frequency = np.concatenate([np.array(f_drift06.get('SpotData/frequency')), np.array(f_drift06_2.get('SpotData/frequency'))], axis=1)
dm_d06 = np.concatenate([np.array(f_drift06.get('SpotData/dm')), np.array(f_drift06_2.get('SpotData/dm'))], axis=1)
dp_d06 = np.concatenate([np.array(f_drift06.get('SpotData/dp')), np.array(f_drift06_2.get('SpotData/dp'))], axis=1)
a1 = np.concatenate([np.array(f_drift06.get('SpotData/a1')), np.array(f_drift06_2.get('SpotData/a1'))], axis=1)
a2 = np.concatenate([np.array(f_drift06.get('SpotData/a2')), np.array(f_drift06_2.get('SpotData/a2'))], axis=1)
b1 = np.concatenate([np.array(f_drift06.get('SpotData/b1')), np.array(f_drift06_2.get('SpotData/b1'))], axis=1)
b2 = np.concatenate([np.array(f_drift06.get('SpotData/b2')), np.array(f_drift06_2.get('SpotData/b2'))], axis=1)
print("Shape of the data: ")
print(variance_density.shape)

if interpolate_dense:
    result = interpolate_spectral_coefficients(
        frequency[:,0],
        variance_density,
        a1, b1, a2, b2,
        n_points=200,
        enforce_constraints=True
    )
    frequency_drift06_dense = np.array([result['f_dense']]*frequency.shape[1]).T
    S_dense = result['variance_density_dense']
    plt.plot(frequency[:,0],variance_density[:,0])
    plt.plot(frequency_drift06_dense[:,0], S_dense[:,0])

    frequency = frequency_drift06_dense
    variance_density = S_dense
    a1, b1, a2, b2 = result['a1_dense'], result['b1_dense'], result['a2_dense'], result['b2_dense']
#%%

num_obs_per_sample, num_samples = variance_density.shape
S_theta = np.zeros((variance_density.shape[1], variance_density.shape[0], 37))
directions = np.deg2rad(np.linspace(0, 360.0, 37))
delta_theta = np.deg2rad(10.0)
for t in range(S_theta.shape[0]):
    for f in range(S_theta.shape[1]):
        # Load S(f), a1, b1, a2, b2 for this time and frequency
        S = variance_density[f, t]  # Omnidirectional spectrum
        a1_, b1_, a2_, b2_ = a1[f, t], b1[f, t], a2[f, t], b2[f, t]  # Directional moments

        for i, theta in enumerate(directions):
            # Compute D(f, theta)
            D = (1 / 2 * np.pi) * (
                    1 + (a1_ * np.cos(theta) + b1_ * np.sin(theta))
                    + (a2_ * np.cos(2 * theta) + b2_ * np.sin(2 * theta))
            )

            # Ensure non-negativity
            D = max(D, 0)

            # Compute S(f, theta)
            S_theta[t, f, i] = S * D

        # Optional: Renormalize to ensure sum(S_theta * delta_theta) ≈ S
        integral = np.sum(S_theta[t, f, :]) * delta_theta
        integral = integral if integral > 0 else 1.0
        S_theta[t, f, :] *= S / integral  # Adjust if integral != S
#%%
import hdpgpc.buoy_utils as buoy_utils
dirs = np.linspace(0, 360.0, 37)
S_theta_rotated = np.zeros_like(S_theta)

for ind_ in range(0, num_samples, 1):
    ind_dp = np.argmin(np.abs(time_drift06 - spec_time_drift06[:, ind_]))
    error_mean_, error_peak_ = buoy_utils.compute_rotation_deviation(S_theta[ind_], dirs, a1[:,ind_], b1[:,ind_], dm_d06[:,ind_dp], dp_d06[:,ind_dp])
    error = -np.mean([error_mean_, error_peak_])
    dirs_rotated = (dirs + error) % 360.0
    f_interp = interp1d(
        dirs,
        S_theta[ind_,:,:],
        kind='linear', # 'cubic' is smoother but can overshoot negatively
        axis=-1,       # Interpolate along direction axis
        assume_sorted=True
    )
    #indexes_directions_new = np.floor(dirs_rotated // 10).astype(np.int32)
    S_theta_rotated[ind_, :, :] = f_interp(dirs_rotated)

dataset = xr.Dataset(
    data_vars=dict(
        efth=(["time", "freq", "dir"], S_theta_rotated)
    ),
    coords=dict(
        time=(["time"], spec_time_drift06[0]),
        freq=(["freq"], frequency[:, 0]),
        dir=(["dir"], dirs)
    )
)
dts_drift06 = wv.SpecDataset(dataset)
new_dirs = np.linspace(0, 360, 37)  # Cada 10° incluyendo 360°
rotated_dirs = (180 + new_dirs) % 360
data_drifting06 = dts_drift06.interp(dir=rotated_dirs)
freq = data_drifting06.freq
data_drifting06 = data_drifting06.to_numpy()
freq_ = np.array(freq)
if not interpolate_dense:
    freq = np.array(freq)
#Here we are going to compute the Heighs and try to filter following Hamiltons rule.
interv = np.repeat(freq_[1] - freq_[0], num_obs_per_sample)[:, np.newaxis]
hs_d06 = 4 * np.sqrt(np.sum(data_drifting06, axis=2) * delta_theta @ interv)
print("Shape of hs: " + str(hs_d06.shape))
plt.plot(hs_d06)
chosen_indexes = np.where((hs_d06 > 0.5) & (hs_d06 < 1.5))[0]
chosen_indexes = np.arange(num_samples)
print("How much spectra falls in the range 0.5-1.5 Hs: " + str(chosen_indexes.shape[0]))
plt.show()
plt.plot(freq, data_drifting06[:, :, 0][chosen_indexes].T)
plt.show()
lognorm_data = np.log(data_drifting06 + 1e-6) - np.mean(np.log(data_drifting06 + 1e-6), axis=1)[:, np.newaxis, :]
plt.plot(freq, lognorm_data[:, :, 0][chosen_indexes].T)
plt.show()
data_drifting06 = data_drifting06[chosen_indexes]
print("Shape of the data: ")
print(data_drifting06.shape)

lim_freq = startFreq_sea
index_freq = np.where(freq>lim_freq)[0][0]+1
num_samples, num_obs_per_sample, num_outputs = data_drifting06.shape
std, std_dif, bound_sigma, bound_gamma = compute_estimators_LDS(data_drifting06[:, 0:index_freq, :], n_f=50, dim=25)
#Seem like these estimators are so big for this data, let's reduce them
# std = std * 0.025
# std_dif = std_dif * 0.008
std = std * 0.50
std_dif = std_dif * 0.10
bound_sigma = (std * 1e-7, std * 1e-6)
bound_gamma = (std_dif * 1e-9, std_dif * 1e-8)

print("Final sigma:", std)
print("Final gamma:", std_dif)
print("Final sigma bound:", bound_sigma)
print("Final gamma bound:", bound_gamma)
#%%

M = 2
sigma = [std * 1.0] * M
gamma = [std_dif * 1.0] * M
outputscale_ = 0.1
ini_lengthscale = 1e-3
bound_lengthscale = (1e-7, 5e-3)
samples = [0, num_obs_per_sample]
l, L = samples[0], samples[1]
# x_basis has to have the same dimension structure as data[0], in this case (171,1) but it could be (n_inducing_points, 1).
lim_freq = startFreq_sea
ini_lim_freq = 0.040
index_freq = np.where(freq>lim_freq)[0][0]+1
index_ini = np.where(freq>ini_lim_freq)[0][0]+1
x_basis = np.atleast_2d(freq[index_ini:index_freq]).T
x_train = np.atleast_2d(freq[index_ini:index_freq]).T
print(x_train.shape)
#If x_basis is wanted to be smaller than the observations length, then the inducing points approach can be applied setting this parameter to True.
inducing_points = False
#Choose if warp is going to be applied. (In the most recent version is optimized to work with online inference, but it can be used in offline as an additional step at the end of the clustering).
warp = False
#Warp priors
noise_warp = std * 0.1
bound_noise_warp = (noise_warp * 0.1, noise_warp * 0.2)
#Warp time indexes
x_basis_warp = np.atleast_2d(np.arange(freq[0], freq[-1], freq.shape[0] / 2.0, dtype=np.float64)).T
sw_gp_drift06 = hdpgp.GPI_HDP(x_basis=x_basis, x_basis_warp=x_basis_warp, n_outputs=37,
                              ini_lengthscale=ini_lengthscale, bound_lengthscale=bound_lengthscale,
                              ini_gamma=gamma, ini_sigma=sigma, ini_outputscale=outputscale_, noise_warp=noise_warp,
                              bound_sigma=bound_sigma, bound_gamma=bound_gamma, bound_noise_warp=bound_noise_warp,
                              verbose=False, max_models=100, inducing_points=inducing_points,
                              reestimate_initial_params=False,
                              n_explore_steps=15, free_deg_MNIV=8, share_gp=True, use_snr=False, reduce_outputs=True,
                              reduce_outputs_ratio=0.2)
#Good labels: 11
cluster_labels_d08 = np.load('/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/cluster_labels/resampled_low_freq/cluster_labels_drift08_202407_dynamic_11.npy')
#cluster_labels_d08 = np.load('/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/cluster_labels/cluster_labels_drift08_202407_dynamic_11.npy')


M = np.unique(cluster_labels_d08).shape[0]
num_samples = data_drifting06.shape[0]
#logdata = np.log(data + 1e-6)
x_trains = np.array([x_train] * num_samples)
sw_gp_drift06.reload_model_from_labels(x_trains, data_drifting06[:, index_ini:index_freq, :], cluster_labels_d08, M)
labels = np.array(['N'] * 6500)
main_model = print_results(sw_gp_drift06, labels, 0, error=False)
selected_gpmodels = sw_gp_drift06.selected_gpmodels()
up.plot_models_plotly(sw_gp_drift06, selected_gpmodels, main_model, labels, N_0=0, lead=26, step=(freq[1] - freq[0]) / 1,
                      plot_latent=False, title='WAVE CLUSTER', ticks=True, yscale=True,
                      save="/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/clusters_drift06.png", line_max=True)
#%%
print(spec_time_drift03.shape)


data_time_drift03 = [matlab_datenum_to_datetime(t) for t in spec_time_drift03[0]]
data_time_drift06 = [matlab_datenum_to_datetime(t) for t in spec_time_drift06[0]]
data_time_hillarys = [matlab_datenum_to_datetime(t) for t in spec_time_hillarys[0]]
print(np.min(data_time_hillarys), np.max(data_time_hillarys))
print(np.min(data_time_drift03), np.max(data_time_drift03))
print(np.min(data_time_drift06), np.max(data_time_drift06))

#%% md
# # Gráfica distribución por meses y clusters
#%%

df_hillarys = pd.DataFrame({
    'time': data_time_hillarys,
    'cluster': sw_gp_hillary.resp_assigned[-1]
})

# Crear columna cos días do ano (usado como eixo X)
df_hillarys['min'] = df_hillarys['time'].dt.minute
df_hillarys['hour'] = df_hillarys['time'].dt.hour
df_hillarys['day'] = df_hillarys['time'].dt.dayofyear  # Día do ano (1-365)
df_hillarys['month'] = df_hillarys['time'].dt.month    # Para ticks e estética
df_hillarys['date'] = df_hillarys['time'].dt.date

print(len(data_time_drift03))
print(len(sw_gp_drift03.resp_assigned[-1]))
df_drift03 = pd.DataFrame({
    'time': data_time_drift03,
    'cluster': sw_gp_drift03.resp_assigned[-1]
})

# Crear columna cos días do ano (usado como eixo X)
df_drift03['min'] = df_drift03['time'].dt.minute
df_drift03['hour'] = df_drift03['time'].dt.hour
df_drift03['day'] = df_drift03['time'].dt.dayofyear  # Día do ano (1-365)
df_drift03['month'] = df_drift03['time'].dt.month    # Para ticks e estética
df_drift03['date'] = df_drift03['time'].dt.date  # Por se queres filtrar por data concreta


df_drift06 = pd.DataFrame({
    'time': data_time_drift06,
    'cluster': sw_gp_drift06.resp_assigned[-1]
})

# Crear columna cos días do ano (usado como eixo X)
df_drift06['min'] = df_drift06['time'].dt.minute
df_drift06['hour'] = df_drift06['time'].dt.hour
df_drift06['day'] = df_drift06['time'].dt.dayofyear  # Día do ano (1-365)
df_drift06['month'] = df_drift06['time'].dt.month    # Para ticks e estética
df_drift06['date'] = df_drift06['time'].dt.date

frequencies_hillarys = freq_hillarys
directions_hillarys = np.rad2deg(directions)
energy_spectral = data
tensors_list_hillarys = []
for gp in sw_gp_hillary.gpmodels[0]:
    subdata = np.mean(data[gp.indexes], axis=0)
    tensors_list_hillarys.append(subdata)

frequencies_drift03 = freq
directions_drift03 = np.rad2deg(directions)
energy_spectral = data_drifting03
tensors_list_drift03 = []
for gp in sw_gp_drift03.gpmodels[0]:
    subdata = np.mean(data_drifting03[gp.indexes], axis=0)
    tensors_list_drift03.append(subdata)

frequencies_drift06 = freq
directions_drift06 = np.rad2deg(directions)
energy_spectral = data_drifting06
tensors_list_drift06 = []
for gp in sw_gp_drift06.gpmodels[0]:
    subdata = np.mean(data_drifting06[gp.indexes], axis=0)
    tensors_list_drift06.append(subdata)

clust_h, indexes_h = np.unique(sw_gp_hillary.resp_assigned[-1], return_index=True)

clust_d, indexes_d = np.unique(sw_gp_drift03.resp_assigned[-1], return_index=True)

clust_d6, indexes_d = np.unique(sw_gp_drift06.resp_assigned[-1], return_index=True)

tensors_list_hillarys = np.array(tensors_list_hillarys)
tensors_list_drift03 = np.array(tensors_list_drift03)
tensors_list_drift06 = np.array(tensors_list_drift06)

n_clusters_hillarys = clust_h.shape[0]
n_clusters_drift03 = clust_d.shape[0]
n_clusters_drift06 = clust_d6.shape[0]

ds_cluster_means_hillarys = xr.Dataset(
    data_vars=dict(
        efth=(["cluster", "freq", "dir"], tensors_list_hillarys),
        std_mat = (["cluster", "freq", ])
    ),
    coords=dict(
        cluster=(["cluster"], clust_h),
        freq=(["freq"], frequencies_hillarys.to_numpy()),
        dir=(["dir"], directions_hillarys)
))

ds_cluster_means_drift03 = xr.Dataset(
    data_vars=dict(
        efth=(["cluster", "freq", "dir"], tensors_list_drift03),
    ),
    coords=dict(
        cluster=(["cluster"], clust_d),
        freq=(["freq"], np.array(frequencies_drift03)),
        dir=(["dir"], directions_drift03)
))

ds_cluster_means_drift06 = xr.Dataset(
    data_vars=dict(
        efth=(["cluster", "freq", "dir"], tensors_list_drift06),
    ),
    coords=dict(
        cluster=(["cluster"], clust_d6),
        freq=(["freq"], np.array(frequencies_drift06)),
        dir=(["dir"], directions_drift06)
))

efth_ordered_hillarys = ds_cluster_means_hillarys['efth'].transpose('cluster', 'dir', 'freq')

efth_ordered_drift03 = ds_cluster_means_drift03['efth'].transpose('cluster', 'dir', 'freq')

efth_ordered_drift06 = ds_cluster_means_drift06['efth'].transpose('cluster', 'dir', 'freq')

#%%

gp_b = sw_gp_hillary.gpmodels[0][0]
x_bas = gp_b.x_basis
dist_clusters_hillarys = []
dist_clusters_drift03 = []
dist_clusters_drift06 = []
order='sum_energy'
##Order of clusters from similar to disimilar with respect to Hillarys 0.
if order == 'KL_similarity':
    for i in range(n_clusters_hillarys):
        dist = 0
        for j in range(sw_gp_hillary.n_outputs):
            gp_b = sw_gp_hillary.gpmodels[j][0]
            gp = sw_gp_hillary.gpmodels[j][i]
            x_bas = gp.x_basis
            dist = dist + gp_b.KL_divergence(len(gp_b.indexes)-1, gp,len(gp.indexes)-1, x_bas=x_bas)
        dist_clusters_hillarys.append(dist)
    for i in range(n_clusters_drift03):
        dist = 0
        for j in range(sw_gp_hillary.n_outputs):
            gp_b = sw_gp_hillary.gpmodels[j][0]
            gp = sw_gp_drift03.gpmodels[j][i]
            x_bas = gp.x_basis
            dist = dist + gp_b.KL_divergence(len(gp_b.indexes)-1, gp,len(gp.indexes)-1, x_bas=x_bas)
        dist_clusters_drift03.append(dist)
    for i in range(n_clusters_drift06):
        dist = 0
        for j in range(sw_gp_hillary.n_outputs):
            gp_b = sw_gp_hillary.gpmodels[j][0]
            gp = sw_gp_drift06.gpmodels[j][i]
            x_bas = gp.x_basis
            dist = dist + gp_b.KL_divergence(len(gp_b.indexes)-1, gp,len(gp.indexes)-1, x_bas=x_bas)
        dist_clusters_drift06.append(dist)
elif order == 'ocurrence':
    ## Order by occurrence
    dist_clusters_hillarys = [gp.indexes[0] for gp in sw_gp_hillary.gpmodels[0]]
    dist_clusters_drift03 = [gp.indexes[0] for gp in sw_gp_drift03.gpmodels[0]]
    dist_clusters_drift06 = [gp.indexes[0] for gp in sw_gp_drift06.gpmodels[0]]
elif order == 'sum_energy':
    #Order by energy
    dist_clusters_hillarys = np.zeros(n_clusters_hillarys)
    for i in range(sw_gp_hillary.n_outputs):
        dist_clusters_hillarys = dist_clusters_hillarys + np.array([torch.sum(gp.f_star_sm[-1]) for gp in sw_gp_hillary.gpmodels[i]])
    dist_clusters_drift03 = np.zeros(n_clusters_drift03)
    for i in range(sw_gp_drift03.n_outputs):
        dist_clusters_drift03 = dist_clusters_drift03 + np.array([torch.sum(gp.f_star_sm[-1]) for gp in sw_gp_drift03.gpmodels[i]])
    dist_clusters_drift06 = np.zeros(n_clusters_drift06)
    for i in range(sw_gp_drift06.n_outputs):
        dist_clusters_drift06 = dist_clusters_drift06 + np.array([torch.sum(gp.f_star_sm[-1]) for gp in sw_gp_drift06.gpmodels[i]])
elif order == 'max_energy_frequency':
    dist_clusters_hillarys = np.zeros(n_clusters_hillarys)
    for i in range(n_clusters_hillarys):
        dist_clusters_hillarys[i] = sw_gp_hillary.gpmodels[0][i].x_train[0][np.argmax(np.sum(np.array([gps[i].f_star_sm[-1][:,0] for gps in sw_gp_hillary.gpmodels]), axis=0))]
    dist_clusters_drift03 = np.zeros(n_clusters_drift03)
    for i in range(n_clusters_drift03):
        dist_clusters_drift03[i] = sw_gp_drift03.gpmodels[0][i].x_train[0][np.argmax(np.sum(np.array([gps[i].f_star_sm[-1] for gps in sw_gp_drift03.gpmodels]), axis=0))]
    dist_clusters_drift06 = np.zeros(n_clusters_drift06)
    for i in range(n_clusters_drift06):
        dist_clusters_drift06[i] = sw_gp_drift06.gpmodels[0][i].x_train[0][np.argmax(np.sum(np.array([gps[i].f_star_sm[-1] for gps in sw_gp_drift06.gpmodels]), axis=0))]
n_clusters_hillarys_ord = np.argsort(dist_clusters_hillarys)
n_clusters_drift03_ord = np.argsort(dist_clusters_drift03)
n_clusters_drift06_ord = np.argsort(dist_clusters_drift06)

#n_clusters_hillarys_ord = np.array([0,1,2,3])
print(dist_clusters_hillarys)
print(n_clusters_hillarys_ord)

#%%
from matplotlib import gridspec
from wavespectra import specarray


def plot_cluster_spectrum_and_timeline_list(df_, efth_ordered_, n_clusters_, title="", save=None):
    cols = ['b', 'r', 'g', 'y']
    num_models = len(df_)
    fig = plt.figure(figsize=(30, 8))
    width_rad = [0.1/num_models] * num_models
    width_rad = width_rad.append(0.9)
    width_rad = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]
    num_clust_max = np.max(np.array([n.shape for n in n_clusters_]))
    gs = gridspec.GridSpec(num_clust_max, num_models*2+1, width_ratios=width_rad, hspace = 0.3,)
    efth_max_int = 0.0
    efth_min_int = 0.0
    for mod in range(num_models):
        efth_max_int = np.max([efth_max_int, np.max(np.sum(efth_ordered_[mod].values, axis=1))])
        efth_min_int = np.min([efth_min_int, np.min(np.sum(efth_ordered_[mod].values, axis=1))])

    # Panel de dispersión
    ax_scatter = plt.subplot(gs[:, -1])
    for mod in range(num_models):
        col = cols[mod]
        df = df_[mod]
        efth_ordered = efth_ordered_[mod]
        n_clusters = n_clusters_[mod]

        efth_max = efth_ordered.max().item()
        efth_min = efth_ordered.min().item()

        for i, j in enumerate(n_clusters):
            ax = plt.subplot(gs[n_clusters.shape[0]-i-1 +(num_clust_max-n_clusters.shape[0]), mod*2], projection='polar')
            plt.sca(ax)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)

            da = efth_ordered.isel(cluster=j)
            spec = specarray.SpecArray(da)

            spec.plot(
                kind="contourf",
                cmap='Spectral_r',
                add_colorbar=False,
                normalised=False,
                show_theta_labels=False,
                show_radii_labels=False,
                vmin=efth_min,
                vmax=efth_max,
            )

            max_freq = da['freq'][np.argmax(np.sum(da.values, axis=0)[np.where(da['freq']<0.30)])]
            g = 9.81  # m/s^2
            wave_speed = (g / (4 * np.pi * max_freq)) / 1000 * 3600
            ax.set_title(f"Cluster {j}, Per {np.round(1/max_freq,3).item()}, Time {np.round(4000/ (wave_speed*24.0), 2).item()}.", fontsize=5, y = 0.96)

            ax_ = plt.subplot(gs[n_clusters.shape[0]-i-1 +(num_clust_max-n_clusters.shape[0]), mod * 2 + 1])
            ax_.plot(da['freq'][np.where(da['freq']<0.30)], np.sum(da.values, axis=0)[np.where(da['freq']<0.30)])
            ax_.axvline(da['freq'][np.argmax(np.sum(da.values, axis=0)[np.where(da['freq']<0.30)])], color='r')
            ax_.set_ylim((efth_min_int, efth_max_int))
            ax_.set_yticklabels([])
            x_ticks = np.append(ax_.get_xticks()[2:], np.round(max_freq, 3))
            ax_.set_xticks(x_ticks)
            ax_.tick_params(axis='both', which='major', labelsize=6)
            #ax_.set_xticklabels(x_ticks)

        #x_jittered = rand_jitter(df['day'].values)
        x_jittered = df['day'].values + df['hour'].values/24.0 + df['min'].values/1440.0
        #y_jittered = rand_jitter(df['cluster'].values)

        ord_clust = np.array([np.where(n_clusters==cl)[0] for cl in df['cluster'].values])
        ax_scatter.scatter(x_jittered, ord_clust - 0.1*mod, s=70, color=col, alpha=0.5)
        if mod==0:
            month_start_days = df.groupby('month')['day'].min()
            month_labels_presentes = ['Xan', 'Feb', 'Mar', 'Abr', 'Mai', 'Xuñ', 'Xul', 'Ago', 'Set', 'Out', 'Nov', 'Dec']
            month_labels_filtrados = [month_labels_presentes[m - 1] for m in month_start_days.index]

            days = df['day'].values
            labels_days = df['time'].dt.day.values

            ax_scatter.set_xticks(month_start_days.values)
            ax_scatter.set_xticklabels(month_labels_filtrados)
            ax_scatter.set_xticks(days, minor=True)
            ax_scatter.set_xticklabels(labels_days, minor=True)

            #ax_scatter.set_yticks([])
            ax_scatter.set_xlabel("Mes")
            ax_scatter.set_title("Daily distribution of cluster spectra "+title)
            ax_scatter.grid(which='major', alpha=0.5)
            ax_scatter.grid(which='minor', linestyle="--", alpha=0.1)
            #ax_scatter.legend()

    if save:
        dirname = "/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/"
        os.makedirs(dirname, exist_ok=True)
        fig.savefig(dirname + save, dpi=300)
#%%
plot_cluster_spectrum_and_timeline_list([df_hillarys, df_drift03, df_drift06], [efth_ordered_hillarys, efth_ordered_drift03, efth_ordered_drift06], [n_clusters_hillarys_ord, n_clusters_drift03_ord, n_clusters_drift06_ord], title="Hillarys (first, blue), Drift03 (second, red) and Drift08 (third, green).", save="Buoy_cluster_comparison.png")
#%%
fig = plt.figure(figsize=(20,4))
plt.plot(data_time_hillarys, hs_h)
plt.plot(data_time_drift03, hs_d03)
plt.plot(data_time_drift06, hs_d06)
#plt.vlines([datetime(year=2024, month=8, day=11, hour=9), datetime(year=2024, month=8, day=11, hour=20)], 0, 12, 'r', linestyle='dashed', alpha=0.5)
#plt.vlines([datetime(year=2024, month=8, day=6, hour=4), datetime(year=2024, month=8, day=6, hour=9)], 0, 12, 'r', linestyle='dashed', alpha=0.5)
plt.grid()
#%% md
# ## Estimate a co-occurrence frequency matrix for cluster identification
# 
# Implementation of co-occurrence frequency matrix to estimate the probability of happening cluster i at drift then observe cluster j after a lag l.
#%%
hsig_d03 = np.concatenate([np.array(f_drift03.get('SpotData/hsig')), np.array(f_drift03_2.get('SpotData/hsig'))], axis=1)
dp_d03 = np.concatenate([np.array(f_drift03.get('SpotData/dp')), np.array(f_drift03_2.get('SpotData/dp'))], axis=1)
spread_d03_ = np.concatenate([np.array(f_drift03.get('SpotData/directionalSpread')), np.array(f_drift03_2.get('SpotData/directionalSpread'))], axis=1)
lon_d03 = np.concatenate([np.array(f_drift03.get('SpotData/lon')), np.array(f_drift03_2.get('SpotData/lon'))], axis=1)[0]
lat_d03 = np.concatenate([np.array(f_drift03.get('SpotData/lat')), np.array(f_drift03_2.get('SpotData/lat'))], axis=1)[0]
tp_d03 = np.concatenate([np.array(f_drift03.get('SpotData/tp')), np.array(f_drift03_2.get('SpotData/tp'))], axis=1)[0]
#peak_freq_d03 = np.array([1/tp for tp in tp_d03])
peak_freq_d03 = frequencies_drift03[np.argmax(np.sum(data_drifting03, axis=2), axis=1)]

hsig_d08 = np.concatenate([np.array(f_drift06.get('SpotData/hsig')), np.array(f_drift06_2.get('SpotData/hsig'))], axis=1)
dp_d08 = np.concatenate([np.array(f_drift06.get('SpotData/dp')), np.array(f_drift06_2.get('SpotData/dp'))], axis=1)
spread_d06_ = np.concatenate([np.array(f_drift06.get('SpotData/directionalSpread')), np.array(f_drift06_2.get('SpotData/directionalSpread'))], axis=1)
lon_d06 = np.concatenate([np.array(f_drift06.get('SpotData/lon')), np.array(f_drift06_2.get('SpotData/lon'))], axis=1)[0]
lat_d06 = np.concatenate([np.array(f_drift06.get('SpotData/lat')), np.array(f_drift06_2.get('SpotData/lat'))], axis=1)[0]
tp_d06 = np.concatenate([np.array(f_drift06.get('SpotData/tp')), np.array(f_drift06_2.get('SpotData/tp'))], axis=1)[0]


peak_freq_d03 = np.array([1/tp for tp in tp_d03])

# Now compute co-occurrence with physics
cooccurrence, prob_matrix, metadata = buoy_utils.compute_cooccurrence_matrix_with_physics(
    labels_offshore=cluster_labels_d03,
    labels_coastal=cluster_labels_d08,
    timestamps_offshore=np.array(data_time_drift03),
    timestamps_coastal=np.array(data_time_drift06),
    peak_frequencies_offshore=peak_freq_d03,
    peak_directions_offshore=dp_d03[0],
    offshore_location=(lat_d03, lon_d03),
    coastal_location=(lat_d06, lon_d06),
    time_window_hours=6
)

plt.imshow(prob_matrix)
plt.title('Physics-Based Co-occurrence with clusters')
plt.yticks(np.arange(0, np.max(cluster_labels_d03), 1))
plt.xticks(np.arange(0, np.max(cluster_labels_d08), 1))
plt.grid(alpha=0.2)
plt.colorbar()
plt.show()
#%% md
# ---
#%% md
# Analise the segmentation provided by the clustering method by separating each label change as a starting event.
#%%
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


def plot_h_with_cluster(hs, time, events_ids, label='Drift03', colors=None):
    fig, ax = plt.subplots(figsize=(20,4))
    plt.plot(time, hs, label=label)
    unique_events = np.unique(events_ids)
    unique_events = unique_events[unique_events != -1]  # Exclude "no event" label if using -1

    # Create a colormap for events
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_events)))
    else:
        colors = colors
    for idx, event_id in enumerate(unique_events):
            # Get rows for this event
            event_mask = events_ids == event_id

            # Get start and end times for this event
            t_start = time[event_mask].min()
            t_end = time[event_mask].max()

            # Add vertical span (rectangular region)
            ax.axvspan(t_start, t_end,
                       alpha=0.2,
                       color=colors[idx],
                       label=f'Event {int(event_id)}',
                       zorder=0)
            ax.axvline(t_start, 0, 1, c='k', alpha=0.5, lw=0.5)
            ax.axvline(t_end, 0, 1, c='k', alpha=0.5, lw=0.5)
            ax.text(t_start+timedelta(hours=1), np.max(hs), str(int(event_id)))

    # Format the plot
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Wave Height (m)', fontsize=12)
    ax.set_title(label +' Wave Height with Event Detection', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Format x-axis dates
    ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45)
    plt.show()
#%%
from hdpgpc.buoy_utils import ordinal_float_to_datetime, identify_swell_events
from hdpgpc.util_plots import color
time_hillary_dt = np.array([ordinal_float_to_datetime(t)-timedelta(days=366) for t in spec_time_hillarys[0]])
time_drift03_dt = np.array([ordinal_float_to_datetime(t)-timedelta(days=366) for t in spec_time_drift03[0]])
time_drift06_dt = np.array([ordinal_float_to_datetime(t)-timedelta(days=366) for t in spec_time_drift06[0]])

events_ids_h, event_infoh, smoothed_labelsh = identify_swell_events(time_hillary_dt, cluster_labels_h,
                                                                    min_interruption_hours=24, min_gap_hours=72,
                                                                    smoother=False)
print(len(event_infoh))
color_h = [color[event_infoh[j]['cluster']+1] for j in range(len(event_infoh))]
events_ids_d03, event_info1, smoothed_labels1 = identify_swell_events(time_drift03_dt, cluster_labels_d03,
                                                                      min_interruption_hours=24, min_gap_hours=72,
                                                                      smoother=False)
color_d03 = [color[event_info1[j]['cluster']] for j in range(len(event_info1))]
events_ids_d06, event_info2, smoothed_labels2 = identify_swell_events(time_drift06_dt, cluster_labels_d08,
                                                                      min_interruption_hours=24, min_gap_hours=72,
                                                                      smoother=False)
color_d06 = [color[event_info2[j]['cluster']+1] for j in range(len(event_info2))]

ax= plt.figure(figsize=(10,5))
plt.scatter(time_drift03_dt, cluster_labels_d03 - 10, s=0.4, c='darkred', marker='x', alpha=0.3)
plt.scatter(time_drift06_dt, cluster_labels_d08 - 10, s=0.4, c='darkgreen', marker='x', alpha=0.3)
plt.scatter(time_hillary_dt, events_ids_h, s=0.1, c='b')
plt.scatter(time_drift03_dt, events_ids_d03, s=0.1, c='r')
plt.scatter(time_drift06_dt, events_ids_d06, s=0.1, c='g')
plt.yticks(np.arange(0, np.max(events_ids_h), 4))
plt.grid()

#events_ids1 = cluster_labels_d03
plot_h_with_cluster(hs_h, time_hillary_dt, events_ids_h, label='Hillary', colors=color_h)
plot_h_with_cluster(hs_d03, time_drift03_dt, events_ids_d03, label='Drift03', colors=color_d03)
plot_h_with_cluster(hs_d06, time_drift06_dt, events_ids_d06, label='Drift06', colors=color_d06)
#%%
from hdpgpc.buoy_utils import BuoyObservation, compute_cluster_representative
# Now compute co-occurrence with physics
peak_freq_d03 = np.array([1/tp for tp in tp_d03])

cooccurrence, prob_matrix, metadata = buoy_utils.compute_cooccurrence_matrix_with_physics(
    labels_offshore=events_ids_d03,
    labels_coastal=events_ids_d06,
    timestamps_offshore=np.array(data_time_drift03),
    timestamps_coastal=np.array(data_time_drift06),
    peak_frequencies_offshore=peak_freq_d03,
    peak_directions_offshore=dp_d03[0],
    offshore_location=(lat_d03, lon_d03),
    coastal_location=(lat_d06, lon_d06),
    time_window_hours=4
)

plt.imshow(prob_matrix)
plt.title('Physics-Based Co-occurrence with events')
plt.yticks(np.arange(0, np.max(events_ids_d03), 2))
plt.xticks(np.arange(0, np.max(events_ids_d06), 2))
plt.grid(alpha=0.2)
plt.colorbar()
plt.show()
#%%
from hdpgpc.buoy_utils import extract_frequency_evolution, compute_source_distance_from_dispersion

def compute_mean_frequency(frequencies, spectrum_1d):
    """
    Compute energy-weighted mean (centroid) frequency
    """
    total_energy = np.sum(spectrum_1d)
    if total_energy > 0:
        mean_freq = np.sum(frequencies * spectrum_1d) / total_energy
    else:
        mean_freq = np.nan
    return mean_freq

print(data_drifting03.shape)
peak_freq_h = frequencies_hillarys[index_ini + np.argmax(np.sum(data[:, index_ini:index_freq, :], axis=2), axis=1)]
#peak_freq_d03 = np.array([1/tp for tp in tp_d03])
peak_freq_d03 = frequencies_drift03[index_ini + np.argmax(np.sum(data_drifting03[:, index_ini:index_freq, :], axis=2), axis=1)]
#peak_freq_d03 = np.array([compute_mean_frequency(frequencies_drift03, d.T) for d in np.sum(data_drifting03, axis=2)])
#peak_freq_d06 = np.array([1/tp for tp in tp_d06])
peak_freq_d06 = frequencies_drift06[index_ini + np.argmax(np.sum(data_drifting06[:, index_ini:index_freq, :], axis=2), axis=1)]


#These are the consecutive indexes identified in the event i
event_d03 = 28
print(f"Average height on the event {np.mean(hsig_d03[0][np.where(events_ids_d03==event_d03)]):.2f}")
times_cluster, freqs_cluster = extract_frequency_evolution(event_d03, spec_time_drift03[0], peak_freq_d03, events_ids_d03)
num_obs_event = times_cluster.shape[0]
print(f"Number of observations for the event {times_cluster.shape[0]}")

times_cluster_datetime = np.array([ordinal_float_to_datetime(t) for t in times_cluster])
plt.plot(times_cluster_datetime, freqs_cluster)
print("\nEstimated source data from frequency evolution:")
num_obs_used = np.min([num_obs_event, np.max([int(num_obs_event*0.5), 5])])
#source_dispersion = compute_source_distance_from_dispersion(times_cluster_datetime, freqs_cluster)
#print(f"Distance estimated: {source_dispersion['distance_km']}")
print(prob_matrix[event_d03])
event_d06 = np.argmax(prob_matrix[event_d03,:])
#event_d06 = 14
#event_d06 = 35
#event_d06 = 28
print("Associated event for d06: "+str(event_d06))

plt.figure()
for idx in np.where(events_ids_d03==event_d03)[0]:
    plt.plot(frequencies_drift03[index_ini:index_freq], np.sum(data_drifting03[idx, index_ini:index_freq, :], axis=1))
plt.show()
#%%
from hdpgpc.buoy_utils import BuoyObservation, plot_directional_spectrum, plot_position_event_full, compute_cluster_representative
lon_h = np.concatenate([np.array(f_hillary.get('SpotData/lon')), np.array(f_hillary_2.get('SpotData/lon'))], axis=1)[0]
lat_h = np.concatenate([np.array(f_hillary.get('SpotData/lat')), np.array(f_hillary_2.get('SpotData/lat'))], axis=1)[0]
dp_h = np.concatenate([np.array(f_hillary.get('SpotData/dp')), np.array(f_hillary_2.get('SpotData/dp'))], axis=1)
freq__ = np.array(f_hillary.get('SpotData/frequency'))[:,0]
plot=True

# event_d03 = 15
# event_d06 = 10
energy_event_estimate = []
time_event_estimate = []
duration_event_estimate_hours = []
for event_d03 in np.unique(events_ids_d03):
    event_d06 = np.argmax(prob_matrix[event_d03,:])
    cluster_members_d03 = events_ids_d03 == event_d03
    idx_d03_ev = np.where(cluster_members_d03)[0][0]
    cluster_observations_d03 = []
    cluster_members_d06 = events_ids_d06 == event_d06
    idx_d06_ev = np.where(cluster_members_d06)[0][0]
    cluster_observations_d06 = []

    print(np.where(cluster_members_d03)[0])
    print(np.where(cluster_members_d06)[0])
    for idx in np.where(cluster_members_d03)[0]:
        time_d03 = matlab_datenum_to_datetime(spec_time_drift03[:, idx].item())
        idx_dp = np.argmin(np.abs(time_drift03 - spec_time_drift03[:, idx]))
        peak_energy_d03 = np.max(data_drifting03[idx])
        sp_d03 = spread_d03_[np.argmin(np.abs(freq__ - frequencies_drift03[np.argmax(np.sum(data_drifting03[idx], axis=1))].item())), idx_dp]
        cluster_observations_d03.append(BuoyObservation(lat_d03[idx], lon_d03[idx],
                                                    time_d03, data_drifting03[idx], peak_freq_d03[idx], dp_d03[:, idx_dp],
                                                    peak_energy_d03, sp_d03))

    for idx in np.where(cluster_members_d06)[0]:
        time_d06 = matlab_datenum_to_datetime(spec_time_drift06[:, idx].item())
        idx_dp = np.argmin(np.abs(time_drift06 - spec_time_drift06[:, idx]))
        peak_energy_d06 = np.max(data_drifting06[idx])
        sp_d06 = spread_d06_[np.argmin(np.abs(freq__ - frequencies_drift06[np.argmax(np.sum(data_drifting06[idx], axis=1))].item())), idx_dp]
        cluster_observations_d06.append(BuoyObservation(lat_d06[idx], lon_d06[idx],
                                                    time_d06, data_drifting06[idx], peak_freq_d06[idx], dp_d08[:, idx_dp],
                                                    peak_energy_d06, sp_d06))

    representer_d03 = compute_cluster_representative(cluster_observations_d03, frequencies_drift03)
    representer_d06 = compute_cluster_representative(cluster_observations_d06, frequencies_drift06)
    #plot_directional_spectrum(frequencies_drift03, directions_drift03, cluster_observations_d03[0].mean_spectrum)
    #print(cluster_observations_d03[0].peak_direction)
    #plot_directional_spectrum(frequencies_drift03, directions_drift03, representer_d03.mean_spectrum)
    #plot_directional_spectrum(frequencies_drift06, directions_drift06, representer_d06.mean_spectrum)

    time_selected = representer_d03.time
    print(f"Start of event {event_d03} on Drift03: {representer_d03.time} with peak direction: {representer_d03.peak_direction}")
    print(f"Start of event {event_d06} on Drift06: {representer_d06.time} with peak direction: {representer_d06.peak_direction}")
    #print(time_drift03_dt[idx_d03_ev])
    triangulation = buoy_utils.SwellBackTriangulation(representer_d03, representer_d06, [lat_h[0], lon_h[0]])
    #source_pre = triangulation.triangulate_source_location(method='weighted_centroid')
    source_bayesian = triangulation.triangulate_source_location_bayesian(method='grid_search')
    source = triangulation.triangulate_source_asynchronous(method='time_corrected')
    #source_iterative = triangulation.triangulate_source_asynchronous(method='iterative')
    #source = source_bayesian
    #source = source_asynchronous

    #arrival_pre = triangulation.predict_coastal_arrival(source_pre)
    arrival = triangulation.predict_coastal_arrival(source)
    #arrival_iterative = triangulation.predict_coastal_arrival(source_iterative)
    arrival_asynchronous = triangulation.predict_coastal_arrival(source)
    arrival_bayesian = triangulation.predict_coastal_arrival(source_bayesian)

    #arrival_sparse = triangulation.predict_coastal_arrival_spectral(source)
    # print(f"First arrival time {arrival_sparse['first_arrival_time']}, last arrival time {arrival_sparse['last_arrival_time']}")
    # print(f"Estimated arrival time {arrival['arrival_time']}")
    duration_ev_ = np.mean([(representer_d03.time_end-representer_d03.time).total_seconds() / 3600.0, (representer_d06.time_end-representer_d06.time).total_seconds() / 3600.0])
    energy_event_estimate.append(arrival['arrival_energy'].item())
    time_event_estimate.append(arrival['arrival_time'])
    duration_event_estimate_hours.append(duration_ev_)

    print(f"End of event {event_d03} on Drift03: {representer_d03.time_end}")
    print(f"End of event {event_d06} on Drift06: {representer_d06.time_end}")
    fractional_day_arrival = (arrival['arrival_time'].hour * 3600 + arrival['arrival_time'].minute * 60 + arrival['arrival_time'].second) / 86400.0

    time_arrival_ordinal = datetime.toordinal(arrival['arrival_time'] + timedelta(days = 366)) + fractional_day_arrival
    index_hillary_arrival = np.argmin(np.abs(spec_time_hillarys - time_arrival_ordinal))
    index_hillary_dp_arrival = np.argmin(np.abs(time_hillarys - time_arrival_ordinal))
    peak_freq_h_arrival = frequencies_hillarys[np.argmax(np.sum(data[index_hillary_arrival], axis=1))]
    peak_energy_h_arrival = np.max(data[index_hillary_arrival])

    b_lat_1 = np.array([c['lat'] for c in source['buoy1_backtrack']['all_candidates']]).squeeze()
    b_lon_1 = np.array([c['lon'] for c in source['buoy1_backtrack']['all_candidates']]).squeeze()
    b_lat_2 = np.array([c['lat'] for c in source['buoy2_backtrack']['all_candidates']]).squeeze()
    b_lon_2 = np.array([c['lon'] for c in source['buoy2_backtrack']['all_candidates']]).squeeze()

    #b_lat_1, b_lon_1 = None, None
    #b_lat_2, b_lon_2 = None, None

    min_lat = np.min(np.concatenate([lat_h, lat_d03, lat_d06, [source['source_lat']]]))#, b_lat_1, b_lat_2]))
    max_lat = np.max(np.concatenate([lat_h, lat_d03, lat_d06, [source['source_lat']]]))#, b_lat_1, b_lat_2]))
    min_lon = np.min(np.concatenate([lon_h, lon_d03, lon_d06, [source['source_lon']]]))#, b_lon_1, b_lon_2]))
    max_lon = np.max(np.concatenate([lon_h, lon_d03, lon_d06, [source['source_lon']]]))#, b_lon_1, b_lon_2]))
    min_lat, max_lat, min_lon, max_lon = -62, 0, 10, 125
    # Create figure with grid layout
    if plot:
        plot_position_event_full(source, arrival, (min_lat, max_lat), (min_lon, max_lon), peak_energy_h_arrival, peak_freq_h_arrival, dp_h,
                                 index_hillary_dp_arrival, time_selected, lat_h, lon_h, lat_d03, lon_d03, lat_d06, lon_d06,
                                 idx_d03_ev, idx_d06_ev, index_hillary_arrival,
                                triangulation.DISPERSION_RATE,cluster_labels_d03, cluster_labels_d08, cluster_labels_h, b_lon_1, b_lat_1, b_lon_2, b_lat_2, title=f"direction_corrected/representers_events_{event_d03}_{event_d06}_",
                                 end_of_event = representer_d03.time_end)

    if plot:
        fig, ax = plt.subplots(figsize=(20,4))
        plt.plot(time_hillary_dt, hs_h, label='Hillarys', alpha=0.8)
        plt.plot(time_drift03_dt, hs_d03, label='Drift03', alpha=0.6)
        plt.plot(time_drift06_dt, hs_d06, label='Drift06', alpha=0.6)

        plt.vlines([representer_d03.time], 0, 12, 'red', linestyle='dashed', alpha=0.5, label='Event time on drift 03')
        plt.vlines([representer_d06.time], 0, 12, 'green', linestyle='dashed', alpha=0.5, label='Event time on drift 03')
        plt.vlines([arrival['arrival_time']], 0, 12, 'orange', linestyle='dashed', alpha=0.5, label='Cluster prediction asynchronous')
        ax.axvspan(arrival['arrival_time'], arrival['arrival_time']+timedelta(hours=duration_ev_),
                       alpha=0.2,
                       color='r',
                       zorder=0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        plt.title(f"Prediction for event at Drift03 on {representer_d03.time} and Drift06 on {representer_d06.time}")
        plt.legend()
        plt.grid(True, which='major', color='k', linestyle='-', linewidth=0.3)
        plt.grid(True, which='minor', color='k', linestyle='-', alpha=0.2, linewidth=0.3)
        plt.savefig(f"/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/plots/height_wave_event_pred_2/event_{event_d03}_{event_d06}_{representer_d03.time}.png", dpi=300)
        plt.close()

#%%
from scipy import stats
dates_ = []
energies_ = []
for ev in np.unique(events_ids_d03):
    for d in range(int(duration_event_estimate_hours[ev])):
        dates_.append(time_event_estimate[ev]+timedelta(hours=d))
        energies_.append(np.log(energy_event_estimate[ev]+1))
fig, ax = plt.subplots(figsize=(15,4))
ord_idx = np.argsort(time_event_estimate)
#plt.scatter(np.array(time_event_estimate), np.array(energy_event_estimate)*0.2, c='red', s=2.0, marker='x')
ax.scatter(np.array(dates_), np.array(energies_), c='red', s=2.0, marker='x')
#plt.plot(np.array(time_event_estimate)[ord_idx], np.array(energy_event_estimate)[ord_idx]*0.2, c='red')
ax.plot(data_time_hillarys, hs_h)
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.title(f"Prediction for event at Drift03 on {representer_d03.time} and Drift06 on {representer_d06.time}")
plt.legend()
plt.grid(True, which='major', color='k', linestyle='-', linewidth=0.3)
plt.grid(True, which='minor', color='k', linestyle='-', alpha=0.2, linewidth=0.3)
plt.savefig("/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/plots/events_point_prediction_comparison.png")


measured_series = pd.Series(hs_h[:,0], index=data_time_hillarys)
predicted_series = pd.Series(np.array(energy_event_estimate)[ord_idx], index=np.array(time_event_estimate)[ord_idx])
ord_idx_ = np.argsort(dates_)
predicted_series_durations_series = pd.Series(np.array(energies_)[ord_idx_], index=np.array(dates_)[ord_idx_])

predicted_series_durations = pd.DataFrame({
    'date':np.array(dates_),
    'energy': np.array(energies_)})


predicted_series_agg = predicted_series_durations.groupby('date')['energy'].max().reset_index()
predicted_series_durations_indexed = predicted_series_agg.set_index('date')

TOLERANCE = pd.Timedelta(minutes=60)
predicted_series_binned = predicted_series_durations_indexed.groupby(pd.Grouper(freq=TOLERANCE)).max().reset_index()
predicted_series_binned = predicted_series_binned[predicted_series_binned['energy'] > 0]
unique_dates = predicted_series_binned['date'].tolist()
summed_energies = predicted_series_binned['energy'].tolist()

fig = plt.figure(figsize=(15,4))
#plt.scatter(np.array(time_event_estimate), np.array(energy_event_estimate)*0.2, c='red', s=2.0, marker='x')
#plt.scatter(np.array(unique_dates), np.array(summed_energies), c='red', s=2.0, marker='x')
plt.plot(np.array(unique_dates), np.array(summed_energies), c='red')
plt.plot(data_time_hillarys, hs_h)
plt.savefig("/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/plots/events_point_prediction_comparison_binned.png")

tolerance = pd.Timedelta(minutes=120)

aligned_measured = []
aligned_predicted = []
matched_times = []

for pred_time, pred_val in predicted_series_durations_series.items():
    # Find measured observation within tolerance
    time_diffs = np.abs(measured_series.index - pred_time)
    min_diff_idx = time_diffs.argmin()
    min_diff = time_diffs[min_diff_idx]

    if min_diff <= tolerance:
        meas_time = measured_series.index[min_diff_idx]
        meas_val = measured_series.iloc[min_diff_idx]

        aligned_measured.append(meas_val)
        aligned_predicted.append(pred_val)
        matched_times.append(pred_time)

measured_clean = np.array(aligned_measured)
predicted_clean = np.array(aligned_predicted)
matched_times = np.array(matched_times)
r, p_value = stats.pearsonr(measured_clean, predicted_clean)

plt.figure(figsize=(15,4))
plt.plot(matched_times, measured_clean, c='blue', label='Measured Wave Height at Hillarys')
plt.plot(matched_times, predicted_clean, c='red', label='Predicted Wave Energy at Hillarys')
plt.title(f"Comparison predicted energy vs measured height. Pearson correlation 95 ({r:.2f}), P-value ({p_value:.2E})")
plt.legend()
plt.savefig("/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/plots/events_point_prediction_comparison_axisXsampled_2.png")
#%%
# print("\n Hillarys")
# event_h = 53
# cluster_members_h = events_ids_h == event_h
# for idx in np.where(cluster_members_h)[0]:
#     idx_dp = np.argmin(np.abs(time_hillarys - spec_time_hillarys[:, idx]))
#     print(dp_h[:,idx_dp], np.max(data[idx]), peak_freq_h[idx].item())
#
# print("\n Drift 03")
# for idx in np.where(cluster_members_d03)[0]:
#     idx_dp = np.argmin(np.abs(time_drift03 - spec_time_drift03[:, idx]))
#     print(dp_d03[:,idx_dp], np.max(data_drifting03[idx]), peak_freq_d03[idx].item())
# print("\n Drift 06")
# for idx in np.where(cluster_members_d06)[0]:
#     idx_dp = np.argmin(np.abs(time_drift06 - spec_time_drift06[:, idx]))
#     print(dp_d08[:,idx_dp], np.max(data_drifting06[idx]), peak_freq_d06[idx].item())
#%%
pe_h = np.max(data, axis=(1,2))
pe_d03 = np.max(data_drifting03, axis=(1,2))
pe_d06 = np.max(data_drifting06, axis=(1,2))

fig, ax = plt.subplots(figsize=(20,4))
plt.plot(time_hillary_dt, hs_h, label='Hillarys', alpha=0.8)
plt.plot(time_drift03_dt, hs_d03, label='Drift03', alpha=0.6)
plt.plot(time_drift06_dt, hs_d06, label='Drift06', alpha=0.6)

plt.vlines([representer_d03.time], 0, 12, 'orange', linestyle='dashed', alpha=0.5, label='Event time on drift 03')
plt.vlines([representer_d06.time], 0, 12, 'green', linestyle='dashed', alpha=0.5, label='Event time on drift 03')
plt.vlines([arrival['arrival_time']], 0, 12, 'r', linestyle='dashed', alpha=0.5, label='Cluster prediction asynchronous')
#plt.vlines([arrival_iterative['arrival_time']], 0, 12, 'darkred', linestyle='dashed', alpha=0.5, label='Cluster prediction asynchronous iterative')
#plt.vlines([arrival_asynchronous['arrival_time']], 0, 12, 'darkred', linestyle='dashed', alpha=0.5, label='Cluster prediction asynchron')
#plt.vlines([arrival_bayesian['arrival_time']], 0, 12, 'pink', linestyle='dashed', alpha=0.5, label='Cluster prediction bayesian')
plt.vlines([datetime(year=2024, month=8, day=16, hour=14)], 0, 12, 'y', linestyle='dashed', alpha=0.5, label='Point prediction')
#plt.vlines([datetime(year=2024, month=8, day=11, hour=15)], 0, 12, 'y', linestyle='dashed', alpha=0.5, label='Point prediction')
#plt.vlines([datetime(year=2024, month=7, day=14, hour=20)], 0, 12, 'y', linestyle='dashed', alpha=0.5, label='Point prediction')
#plt.vlines([datetime(year=2024, month=8, day=20, hour=0)], 0, 12, 'y', linestyle='dashed', alpha=0.5, label='Point prediction')
#plt.vlines([datetime(year=2024, month=8, day=22, hour=17)], 0, 12, 'y', linestyle='dashed', alpha=0.5, label='Point prediction')
#plt.vlines([datetime(year=2024, month=8, day=5, hour=19)], 0, 12, 'y', linestyle='dashed', alpha=0.5, label='Point prediction')
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.title(f"Prediction for event at Drift03 on {representer_d03.time} and Drift06 on {representer_d06.time}")
plt.legend()
plt.grid(True, which='major', color='k', linestyle='-', linewidth=0.3)
plt.grid(True, which='minor', color='k', linestyle='-', alpha=0.2, linewidth=0.3)
plt.savefig(f"/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/plots/height_wave_event_pred/event_{event_d03}_{representer_d03.time}.png")
#%%

print("END")