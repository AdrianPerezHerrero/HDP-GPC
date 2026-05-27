#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDP-GPC clustering for NDBC directional wave spectra.

This script mirrors the CLARA preprocessing script and the HDP-GPC
configuration used in EjemploCompleto.ipynb:
    1. Read NDBC directional spectra for 2017 and 2018.
    2. Merge wind and depth metadata when available.
    3. Compute Hs from the non-directional spectrum.
    4. Keep spectra with hs_min < Hs < hs_max.
    5. Average consecutive blocks of block_step spectra.
    6. Use frequency bins np.arange(low_freq_index, high_freq_index, freq_index_step).
    7. Fit multi-output offline HDP-GPC to 2017 spectra by default.
    8. Reconstruct the model from the learned 2017 labels.
    9. Assign 2018 spectra to the learned 2017 regimes.
   10. Optionally fit the model directly on the concatenated 2017+2018 data.
   11. Save labelled NetCDF/CSV outputs and diagnostic figures.

Dependencies:
    numpy, pandas, xarray, matplotlib, wavespectra, torch, hdpgpc

Example:
    python EjemploCompletoHDPGPC.py \
        --data-dir ../data/NDBC \
        --output-dir ./hdpgpc_wave_results
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

try:
    from wavespectra import read_ndbc
except ImportError as exc:
    raise ImportError(
        "wavespectra is required. Install it with `pip install wavespectra` "
        "or run the script in the same environment as the notebook."
    ) from exc

try:
    import hdpgpc.GPI_HDP as hdpgp
    from hdpgpc.get_data import compute_estimators_LDS
except ImportError as exc:
    raise ImportError(
        "The local `hdpgpc` package is required. Run this script in the same "
        "environment used by EjemploCompleto.ipynb."
    ) from exc


warnings.filterwarnings("ignore", message="Can't decode floating point timedelta to 's'")
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class HDPGPCWaveConfig:
    data_dir: Path
    output_dir: Path

    dir2017_raw: str = "dir2017.nc"
    dir2018_raw: str = "dir2018.nc"
    meteo2017: str = "meteo2017.nc"
    meteo2018: str = "meteo2018.nc"
    ocean2017: str = "ocean2017.nc"

    dir2017_converted: str = "dir2017_dir.nc"
    dir2018_converted: str = "dir2018_dir.nc"

    hs_min: float = 0.5
    hs_max: float = 2.5
    block_step: int = 3

    low_freq_index: int = 3
    high_freq_index: int = 35
    freq_index_step: int = 2

    # HDP-GPC parameters from the notebook
    max_models: int = 50
    n_explore_steps: int = 10
    outputscale: float = 1.0
    ini_lengthscale: float = 1e-2
    bound_lengthscale_min: float = 1e-7
    bound_lengthscale_max: float = 5e-1
    inducing_points: bool = False
    warp: bool = False
    cuda: bool = False
    verbose: bool = False
    share_gp: bool = True
    use_snr: bool = False
    reduce_outputs: bool = True
    reduce_outputs_ratio: float = 0.3
    reestimate_initial_params: bool = False

    # Stage-specific LDS-scale multipliers from the notebook
    fit_sigma_multiplier: float = 1e-2
    fit_gamma_multiplier: float = 1e-3
    reload_sigma_multiplier: float = 1e-3
    reload_gamma_multiplier: float = 1e-4

    fit_free_deg_mniv: int = 3
    reload_free_deg_mniv: int = 5

    sigma_bound_low_factor: float = 1e-7
    sigma_bound_high_factor: float = 1e-5
    gamma_bound_low_factor: float = 1e-9
    gamma_bound_high_factor: float = 1e-5

    noise_warp_factor: float = 0.1
    noise_warp_bound_low_factor: float = 0.1
    noise_warp_bound_high_factor: float = 0.2

    default_depth_2018: float = 33.0
    random_state: int = 7
    max_curves_per_cluster_plot: int = 1000

    save_model_pickle: bool = False

    # If True, fit/reload HDP-GPC on the concatenated 2017+2018 data
    # instead of fitting on 2017 and assigning 2018 afterwards.
    fit_combined_2017_2018: bool = False


# ---------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------

def squeeze_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    """Remove singleton lat/lon dimensions when present."""
    for dim in ["lat", "lon", "latitude", "longitude"]:
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.isel({dim: 0}, drop=True)
    return ds


def convert_ndbc_to_directional(raw_path: Path, converted_path: Path) -> xr.Dataset:
    """Convert NDBC directional file with wavespectra if needed."""
    if converted_path.exists():
        return xr.open_dataset(converted_path, decode_timedelta=True)

    print(f"Converting {raw_path} -> {converted_path}")
    ds = read_ndbc(raw_path, directional=True, weight_coeff=True)
    ds = squeeze_lat_lon(ds)
    ds.to_netcdf(converted_path)
    return xr.open_dataset(converted_path, decode_timedelta=True)


def read_nondirectional_spectrum(raw_path: Path) -> xr.Dataset:
    """Read non-directional NDBC spectrum for Hs computation."""
    ds = read_ndbc(raw_path, directional=False)
    ds = squeeze_lat_lon(ds)
    return ds


def merge_meteo_and_depth(
    ds_wave: xr.Dataset,
    meteo_path: Optional[Path],
    ocean_path: Optional[Path] = None,
    default_depth: Optional[float] = None,
) -> xr.Dataset:
    """Merge wind speed, wind direction and depth as in the notebook."""
    ds = ds_wave.copy()

    if meteo_path is not None and meteo_path.exists():
        ds_meteo = xr.open_dataset(meteo_path, decode_timedelta=True)
        ds_meteo = squeeze_lat_lon(ds_meteo)

        for var in ds_meteo.data_vars:
            ds_meteo[var].data = ds_meteo[var].values

        meteo_interp = ds_meteo.interp(time=ds_wave.time)

        if "wind_spd" in meteo_interp:
            ds = ds.assign({"wspd": meteo_interp["wind_spd"]})
        if "wind_dir" in meteo_interp:
            ds = ds.assign({"wdir": meteo_interp["wind_dir"]})

    depth_value = default_depth
    if ocean_path is not None and ocean_path.exists():
        ds_ocean = xr.open_dataset(ocean_path, decode_timedelta=True)
        ds_ocean = squeeze_lat_lon(ds_ocean)
        if "depth" in ds_ocean:
            depth_value = float(np.ravel(ds_ocean["depth"].values)[0])

    if depth_value is not None:
        dpt_da = xr.DataArray(
            data=np.full(ds.sizes["time"], depth_value, dtype=np.float32),
            coords={"time": ds.time},
            dims=["time"],
            name="dpt",
        )
        ds["dpt"] = dpt_da

    return ds


def compute_hs_from_nondirectional(data_no_direct: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Notebook formula: Hs = 4 * sqrt(S @ delta_f)."""
    delta_f = float(freq[1] - freq[0])
    energy = np.asarray(data_no_direct) @ np.full(data_no_direct.shape[1], delta_f)
    energy = np.maximum(energy, 0.0)
    return 4.0 * np.sqrt(energy)


def filter_by_hs(
    data_direct: np.ndarray,
    data_no_direct: np.ndarray,
    freq: np.ndarray,
    hs_min: float,
    hs_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the Hs range filtering used in the notebook."""
    hs = compute_hs_from_nondirectional(data_no_direct, freq)
    chosen = np.where((hs > hs_min) & (hs < hs_max))[0]
    print(f"Spectra with {hs_min} < Hs < {hs_max}: {chosen.size}")
    return data_direct[chosen], data_no_direct[chosen], chosen, hs[chosen]


def block_average(data: np.ndarray, step: int) -> np.ndarray:
    """Average consecutive non-overlapping blocks."""
    n_blocks = data.shape[0] // step
    trimmed = data[: n_blocks * step]

    if data.ndim == 2:
        grouped = trimmed.reshape(n_blocks, step, data.shape[1])
    elif data.ndim == 3:
        grouped = trimmed.reshape(n_blocks, step, data.shape[1], data.shape[2])
    else:
        raise ValueError(f"Unsupported data ndim: {data.ndim}")

    return grouped.mean(axis=1)




def get_freq_indices(cfg: HDPGPCWaveConfig) -> np.ndarray:
    """Return the frequency-bin indices selected for HDP-GPC fitting."""
    if cfg.freq_index_step <= 0:
        raise ValueError("freq_index_step must be a positive integer.")
    if cfg.high_freq_index <= cfg.low_freq_index:
        raise ValueError("high_freq_index must be greater than low_freq_index.")
    return np.arange(cfg.low_freq_index, cfg.high_freq_index, cfg.freq_index_step, dtype=int)


def select_frequency_bins(data: np.ndarray, cfg: HDPGPCWaveConfig) -> np.ndarray:
    """Select frequency bins using np.arange(low_freq_index, high_freq_index, freq_index_step)."""
    return data[:, get_freq_indices(cfg), :]


def prepare_year(year: int, cfg: HDPGPCWaveConfig) -> Dict[str, object]:
    """Load, merge, filter and block-average one year of NDBC spectra."""
    if year == 2017:
        raw_file = cfg.dir2017_raw
        converted_file = cfg.dir2017_converted
        meteo_file = cfg.meteo2017
        ocean_file = cfg.ocean2017
        default_depth = None
    elif year == 2018:
        raw_file = cfg.dir2018_raw
        converted_file = cfg.dir2018_converted
        meteo_file = cfg.meteo2018
        ocean_file = None
        default_depth = cfg.default_depth_2018
    else:
        raise ValueError("Only 2017 and 2018 are configured in this script.")

    raw_path = cfg.data_dir / raw_file
    converted_path = cfg.data_dir / converted_file
    meteo_path = cfg.data_dir / meteo_file
    ocean_path = cfg.data_dir / ocean_file if ocean_file is not None else None

    ds_wave = convert_ndbc_to_directional(raw_path, converted_path)
    ds = merge_meteo_and_depth(
        ds_wave,
        meteo_path=meteo_path,
        ocean_path=ocean_path,
        default_depth=default_depth,
    )

    ds_nondir = read_nondirectional_spectrum(raw_path)

    data_direct = ds["efth"].to_numpy()
    data_no_direct = ds_nondir["efth"].to_numpy()
    freq = ds["freq"].to_numpy()
    directions = ds["dir"].to_numpy()

    data_filtered, nondir_filtered, chosen_indexes, hs_filtered = filter_by_hs(
        data_direct=data_direct,
        data_no_direct=data_no_direct,
        freq=freq,
        hs_min=cfg.hs_min,
        hs_max=cfg.hs_max,
    )

    data_avg = block_average(data_filtered, cfg.block_step)
    nondir_avg = block_average(nondir_filtered, cfg.block_step)

    ds_subset = ds.isel(time=chosen_indexes).coarsen(time=cfg.block_step, boundary="trim").mean()

    return {
        "ds": ds,
        "ds_subset": ds_subset,
        "data_filtered": data_filtered,
        "nondir_filtered": nondir_filtered,
        "data_avg": data_avg,
        "nondir_avg": nondir_avg,
        "chosen_indexes": chosen_indexes,
        "hs_filtered": hs_filtered,
        "freq": freq,
        "freq_used": freq[get_freq_indices(cfg)],
        "directions": directions,
    }


# ---------------------------------------------------------------------
# HDP-GPC utilities
# ---------------------------------------------------------------------

def compute_wave_l_scale(
    data_avg: np.ndarray,
    cfg: HDPGPCWaveConfig,
    stage: str,
) -> Dict[str, object]:
    """Compute the LDS initialisation scales exactly as in the notebook."""
    y = select_frequency_bins(data_avg, cfg)
    n_samples = y.shape[0]

    std, std_dif, _, _ = compute_estimators_LDS(y, n_f=n_samples - 1)

    if stage == "fit":
        std = std * cfg.fit_sigma_multiplier
        std_dif = std_dif * cfg.fit_gamma_multiplier
    elif stage == "reload":
        std = std * cfg.reload_sigma_multiplier
        std_dif = std_dif * cfg.reload_gamma_multiplier
    else:
        raise ValueError("stage must be 'fit' or 'reload'")

    bound_sigma = (
        std * cfg.sigma_bound_low_factor,
        std * cfg.sigma_bound_high_factor,
    )
    bound_gamma = (
        std_dif * cfg.gamma_bound_low_factor,
        std_dif * cfg.gamma_bound_high_factor,
    )

    noise_warp = std * cfg.noise_warp_factor
    bound_noise_warp = (
        noise_warp * cfg.noise_warp_bound_low_factor,
        noise_warp * cfg.noise_warp_bound_high_factor,
    )

    print(f"[{stage}] Final sigma: {std}")
    print(f"[{stage}] Final gamma: {std_dif}")
    print(f"[{stage}] Final sigma bound: {bound_sigma}")
    print(f"[{stage}] Final gamma bound: {bound_gamma}")

    return {
        "std": std,
        "std_dif": std_dif,
        "sigma": [std * 1.0] * 2,
        "gamma": [std_dif * 1.0] * 2,
        "bound_sigma": bound_sigma,
        "bound_gamma": bound_gamma,
        "noise_warp": noise_warp,
        "bound_noise_warp": bound_noise_warp,
    }


def build_hdpgpc_model(
    data_avg: np.ndarray,
    freq: np.ndarray,
    cfg: HDPGPCWaveConfig,
    stage: str,
) -> Tuple[object, np.ndarray, np.ndarray, Dict[str, object]]:
    """Build a GPI_HDP object with the notebook hyperparameters."""
    n_outputs = data_avg.shape[2]
    #n_outputs = 5
    scales = compute_wave_l_scale(data_avg, cfg, stage=stage)

    x_basis = np.atleast_2d(freq[get_freq_indices(cfg)]).T
    x_train = np.atleast_2d(freq[get_freq_indices(cfg)]).T
    x_basis_warp = x_basis

    free_deg = cfg.fit_free_deg_mniv if stage == "fit" else cfg.reload_free_deg_mniv

    sw_gp = hdpgp.GPI_HDP(
        x_basis=x_basis,
        x_basis_warp=x_basis_warp,
        n_outputs=n_outputs,
        cuda=cfg.cuda,
        ini_lengthscale=cfg.ini_lengthscale,
        bound_lengthscale=(cfg.bound_lengthscale_min, cfg.bound_lengthscale_max),
        ini_gamma=scales["gamma"],
        ini_sigma=scales["sigma"],
        ini_outputscale=cfg.outputscale,
        noise_warp=scales["noise_warp"],
        bound_sigma=scales["bound_sigma"],
        bound_gamma=scales["bound_gamma"],
        bound_noise_warp=scales["bound_noise_warp"],
        verbose=cfg.verbose,
        max_models=cfg.max_models,
        inducing_points=cfg.inducing_points,
        reestimate_initial_params=cfg.reestimate_initial_params,
        n_explore_steps=cfg.n_explore_steps,
        free_deg_MNIV=free_deg,
        share_gp=cfg.share_gp,
        use_snr=cfg.use_snr,
        reduce_outputs=cfg.reduce_outputs,
        reduce_outputs_ratio=cfg.reduce_outputs_ratio,
    )

    return sw_gp, x_basis, x_train, scales


def make_x_trains(x_train: np.ndarray, n_samples: int) -> np.ndarray:
    """Replicate the support array for all samples."""
    return np.array([x_train] * n_samples)


def extract_labels_from_swgp(sw_gp: object, n_samples: int, reference_output: int = 0) -> np.ndarray:
    """Extract labels from the first output model list, as in the notebook."""
    labels = np.full(n_samples, fill_value=-1, dtype=int)

    for cluster_id, gp in enumerate(sw_gp.gpmodels[reference_output]):
        labels[np.asarray(gp.indexes, dtype=int)] = cluster_id

    if np.any(labels < 0):
        missing = int(np.sum(labels < 0))
        raise RuntimeError(f"Could not assign labels for {missing} samples.")

    return labels


def remap_labels_to_consecutive(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map arbitrary cluster ids to 0,...,M-1, matching the notebook."""
    vals = np.unique(labels)
    labels_trans = np.array([np.where(vals == lab)[0][0] for lab in labels], dtype=int)
    return labels_trans, vals


# ---------------------------------------------------------------------
# Output datasets and plots
# ---------------------------------------------------------------------

def add_labels_to_subset(ds_subset: xr.Dataset, labels: np.ndarray) -> xr.Dataset:
    """Attach cluster labels to the coarsened xarray dataset."""
    if ds_subset.sizes["time"] != labels.shape[0]:
        raise ValueError(
            f"Time size mismatch: ds_subset has {ds_subset.sizes['time']} "
            f"but labels has {labels.shape[0]}."
        )
    return ds_subset.assign(cluster_label=(("time",), labels.astype(int)))


def make_cluster_mean_dataset(ds_labeled: xr.Dataset) -> xr.Dataset:
    """Compute mean efth, wind speed and wind direction per cluster."""
    cluster_means = []
    cluster_labels = []
    wspd_means = []
    wdir_means = []

    for c in np.unique(ds_labeled.cluster_label.values):
        ds_cluster = ds_labeled.sel(time=ds_labeled.cluster_label == c)
        if ds_cluster.time.size == 0:
            continue

        cluster_means.append(ds_cluster.efth.mean(dim="time").expand_dims(time=[0]))
        cluster_labels.append(int(c))

        if "wspd" in ds_cluster:
            wspd_means.append(ds_cluster.wspd.mean(dim="time").expand_dims(time=[0]))
        if "wdir" in ds_cluster:
            wdir_means.append(ds_cluster.wdir.mean(dim="time").expand_dims(time=[0]))

    data_vars = {
        "efth": xr.concat(cluster_means, dim="time"),
        "freq": ds_labeled.freq,
        "dir": ds_labeled.dir,
    }

    if wspd_means:
        data_vars["wspd"] = xr.concat(wspd_means, dim="time")
    if wdir_means:
        data_vars["wdir"] = xr.concat(wdir_means, dim="time")

    ds_means = xr.Dataset(data_vars)
    ds_means = ds_means.assign_coords(cluster=("time", cluster_labels))
    ds_means = ds_means.swap_dims({"time": "cluster"})

    ds_means.efth.attrs = ds_labeled.efth.attrs
    ds_means.freq.attrs = ds_labeled.freq.attrs
    ds_means.dir.attrs = ds_labeled.dir.attrs

    return ds_means


def make_label_dataframe(ds_labeled: xr.Dataset, year_label: str) -> pd.DataFrame:
    """Create a time/cluster dataframe for plotting and export."""
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(ds_labeled.time.values),
            "cluster": ds_labeled.cluster_label.values.astype(int),
            "set": year_label,
        }
    )
    df["year"] = df["time"].dt.year
    df["day"] = df["time"].dt.dayofyear
    df["month"] = df["time"].dt.month
    df["date"] = df["time"].dt.date
    return df


def drop_if_present(ds: xr.Dataset, var: str) -> xr.Dataset:
    return ds.drop_vars(var) if var in ds.variables else ds


def plot_cluster_timeline(df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save a simple cluster-label timeline."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, group in df.groupby("set"):
        ax.scatter(group["time"], group["cluster"], s=10, alpha=0.65, label=name)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cluster")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_hdpgpc_frequency_overplots(
    x2017_tensor: np.ndarray,
    x2018_tensor: Optional[np.ndarray],
    freq_used: np.ndarray,
    labels2017: np.ndarray,
    labels2018: Optional[np.ndarray],
    output_path: Path,
    max_curves_per_cluster: int,
    random_state: int,
    representative: str = "mean",
) -> None:
    """
    Plot frequency-marginal cluster overplots.

    Since the HDP-GPC representative lives inside the model object, this plot
    uses empirical cluster means/medians for a lightweight diagnostic, while
    the NetCDF outputs keep the original directional spectra. When 2018 labels
    are provided, clusters appearing only in 2018 are also shown and the red
    representative is computed from all displayed years assigned to the cluster.
    """
    rng = np.random.default_rng(random_state)

    labels2017 = np.asarray(labels2017, dtype=int).reshape(-1)
    labels2018 = None if labels2018 is None else np.asarray(labels2018, dtype=int).reshape(-1)

    if labels2018 is None:
        clusters = np.unique(labels2017)
    else:
        clusters = np.union1d(np.unique(labels2017), np.unique(labels2018))
    k = len(clusters)

    spectra2017_plot = x2017_tensor.sum(axis=2)
    spectra2018_plot = x2018_tensor.sum(axis=2) if x2018_tensor is not None else None

    ncols = 4
    nrows = int(math.ceil(k / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.0 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for panel, c in enumerate(clusters):
        ax = axes[panel]

        idx17_full = np.where(labels2017 == c)[0]
        idx17 = idx17_full.copy()
        if idx17.size > max_curves_per_cluster:
            idx17 = rng.choice(idx17, size=max_curves_per_cluster, replace=False)

        if idx17.size > 0:
            ax.plot(freq_used, spectra2017_plot[idx17].T, color="0.75", alpha=0.25, linewidth=0.8)

        n18 = 0
        idx18_full = np.array([], dtype=int)
        if labels2018 is not None and spectra2018_plot is not None:
            idx18_full = np.where(labels2018 == c)[0]
            n18 = idx18_full.size
            idx18 = idx18_full.copy()
            if idx18.size > max_curves_per_cluster:
                idx18 = rng.choice(idx18, size=max_curves_per_cluster, replace=False)
            if idx18.size > 0:
                ax.plot(freq_used, spectra2018_plot[idx18].T, color="tab:blue", alpha=0.30, linewidth=0.8)

        # Compute an empirical representative from all available years for this
        # cluster. This matters in joint 2017+2018 fitting, where a cluster may
        # contain only 2018 spectra.
        rep_sources = []
        if idx17_full.size > 0:
            rep_sources.append(spectra2017_plot[idx17_full])
        if idx18_full.size > 0 and spectra2018_plot is not None:
            rep_sources.append(spectra2018_plot[idx18_full])
        if rep_sources:
            rep_data = np.vstack(rep_sources)
            if representative == "median":
                rep = np.median(rep_data, axis=0)
            else:
                rep = np.mean(rep_data, axis=0)
            ax.plot(freq_used, rep, color="red", linewidth=2.0)

        ax.set_title(f"Cluster {int(c)} (2017 n={idx17_full.size}, 2018 n={n18})", fontsize=9)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Spectral density")
        ax.grid(alpha=0.2)

    for ax in axes[k:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



# ---------------------------------------------------------------------
# Joint 2017+2018 workflow
# ---------------------------------------------------------------------

def run_combined_2017_2018(cfg: HDPGPCWaveConfig) -> None:
    """
    Fit HDP-GPC directly on the concatenated 2017+2018 spectra.

    This mode differs from the default workflow, where 2017 is used to learn
    regimes and 2018 is assigned afterwards. Here both years participate in
    the same include_batch call, so the inferred states are obtained from the
    full 2017+2018 data set. The resulting labels are then split again into
    year-specific outputs for plotting and export.
    """
    np.random.seed(cfg.random_state)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing 2017 data...")
    y2017 = prepare_year(2017, cfg)
    data2017 = y2017["data_avg"]
    freq = y2017["freq"]
    freq_used = y2017["freq_used"]
    n_samples2017 = data2017.shape[0]

    print("Preparing 2018 data...")
    y2018 = prepare_year(2018, cfg)
    data2018 = y2018["data_avg"]
    n_samples2018 = data2018.shape[0]

    data_joint = np.concatenate([data2017, data2018], axis=0)
    n_samples_joint = data_joint.shape[0]

    print(
        "Joint HDP-GPC tensor shape: "
        f"{select_frequency_bins(data_joint, cfg).shape}"
    )

    # -----------------------------------------------------------------
    # 1) Fit directly on the concatenated 2017+2018 data.
    # -----------------------------------------------------------------
    print("\nFitting HDP-GPC model on concatenated 2017+2018 data...")
    data_joint = data_joint[:,:, [4,11,18,24]]
    sw_gp_fit, x_basis, x_train, fit_scales = build_hdpgpc_model(data_joint, freq, cfg, stage="fit")
    x_trains_joint = make_x_trains(x_train, n_samples_joint)

    start = time.time()
    sw_gp_fit.include_batch(
        x_trains_joint,
        select_frequency_bins(data_joint, cfg),
    )
    fit_minutes = (time.time() - start) / 60.0
    print(f"Joint include_batch time: {fit_minutes:.3f} min")

    labels_joint_raw = extract_labels_from_swgp(sw_gp_fit, n_samples_joint, reference_output=0)
    labels_joint, original_label_values = remap_labels_to_consecutive(labels_joint_raw)
    n_clusters_joint = len(np.unique(labels_joint))
    print(f"Inferred joint clusters: K={n_clusters_joint}")

    labels2017 = labels_joint[:n_samples2017]
    labels2018 = labels_joint[n_samples2017:]

    ds2017_labeled = add_labels_to_subset(y2017["ds_subset"], labels2017)
    ds2018_labeled = add_labels_to_subset(y2018["ds_subset"], labels2018)

    ds2017_labeled.to_netcdf(cfg.output_dir / "ds_hdpgpc_2017_joint.nc")
    ds2018_labeled.to_netcdf(cfg.output_dir / "ds_hdpgpc_2018_joint.nc")

    df2017 = make_label_dataframe(ds2017_labeled, year_label="2017 joint fit")
    df2018 = make_label_dataframe(ds2018_labeled, year_label="2018 joint fit")
    df2017.to_csv(cfg.output_dir / "hdpgpc_labels_2017_joint.csv", index=False)
    df2018.to_csv(cfg.output_dir / "hdpgpc_labels_2018_joint.csv", index=False)

    np.save(cfg.output_dir / "hdpgpc_labels_2017_joint.npy", labels2017)
    np.save(cfg.output_dir / "hdpgpc_labels_2018_joint.npy", labels2018)
    np.save(cfg.output_dir / "hdpgpc_labels_2017_2018_joint.npy", labels_joint)

    if cfg.save_model_pickle:
        with open(cfg.output_dir / "sw_gp_fit_2017_2018_joint.pkl", "wb") as f:
            pickle.dump(sw_gp_fit, f)

    # -----------------------------------------------------------------
    # 2) Reconstruct model from the joint labels, matching the notebook
    #    reload_model_from_labels workflow.
    # -----------------------------------------------------------------
    print("\nReconstructing HDP-GPC model from joint 2017+2018 labels...")
    sw_gp, _, x_train_reload, reload_scales = build_hdpgpc_model(data_joint, freq, cfg, stage="reload")
    x_trains_joint_reload = make_x_trains(x_train_reload, n_samples_joint)

    start = time.time()
    sw_gp.reload_model_from_labels(
        x_trains_joint_reload,
        select_frequency_bins(data_joint, cfg),
        labels_joint,
        n_clusters_joint,
    )
    reload_minutes = (time.time() - start) / 60.0
    print(f"Joint reload_model_from_labels time: {reload_minutes:.3f} min")

    if cfg.save_model_pickle:
        with open(cfg.output_dir / "sw_gp_reloaded_2017_2018_joint.pkl", "wb") as f:
            pickle.dump(sw_gp, f)

    # Combined dataset, preserving the 2017-then-2018 order used by the model.
    ds2017_for_concat = ds2017_labeled
    ds2018_for_concat = ds2018_labeled
    if "dpt" in ds2017_for_concat.variables and "dpt" in ds2018_for_concat.variables:
        pass
    else:
        ds2017_for_concat = drop_if_present(ds2017_for_concat, "dpt")
        ds2018_for_concat = drop_if_present(ds2018_for_concat, "dpt")

    ds_joint = xr.concat([ds2017_for_concat, ds2018_for_concat], dim="time")
    ds_joint.to_netcdf(cfg.output_dir / "ds_hdpgpc_2017_2018_joint.nc")

    df_joint = pd.concat([df2017, df2018], ignore_index=True)
    df_joint.to_csv(cfg.output_dir / "hdpgpc_labels_2017_2018_joint.csv", index=False)

    ds_joint_means = make_cluster_mean_dataset(ds_joint)
    ds_joint_means.to_netcdf(cfg.output_dir / "ds_hdpgpc_cluster_means_2017_2018_joint.nc")

    all_clusters = np.unique(labels_joint)
    counts = pd.DataFrame(
        {
            "cluster": all_clusters.astype(int),
            "n_2017": [int(np.sum(labels2017 == c)) for c in all_clusters],
            "n_2018": [int(np.sum(labels2018 == c)) for c in all_clusters],
            "n_total": [int(np.sum(labels_joint == c)) for c in all_clusters],
        }
    )
    counts.to_csv(cfg.output_dir / "hdpgpc_cluster_counts_2017_2018_joint.csv", index=False)

    summary = {
        "mode": "joint_2017_2018_fit",
        "n_2017": int(n_samples2017),
        "n_2018": int(n_samples2018),
        "n_joint": int(n_samples_joint),
        "k_joint_inferred": int(n_clusters_joint),
        "clusters_joint": [int(c) for c in np.unique(labels_joint)],
        "fit_minutes": float(fit_minutes),
        "reload_minutes": float(reload_minutes),
        "low_freq_index": int(cfg.low_freq_index),
        "high_freq_index": int(cfg.high_freq_index),
        "freq_index_step": int(cfg.freq_index_step),
        "freq_indices": [int(i) for i in get_freq_indices(cfg)],
        "hs_min": float(cfg.hs_min),
        "hs_max": float(cfg.hs_max),
        "block_step": int(cfg.block_step),
        "max_models": int(cfg.max_models),
        "n_explore_steps": int(cfg.n_explore_steps),
        "reduce_outputs": bool(cfg.reduce_outputs),
        "reduce_outputs_ratio": float(cfg.reduce_outputs_ratio),
        "fit_scales": {k: str(v) for k, v in fit_scales.items()},
        "reload_scales": {k: str(v) for k, v in reload_scales.items()},
    }
    with open(cfg.output_dir / "hdpgpc_run_summary_2017_2018_joint.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_cluster_timeline(
        df_joint,
        output_path=cfg.output_dir / "hdpgpc_timeline_2017_2018_joint.png",
        title="HDP-GPC wave-spectra regimes: joint 2017+2018 fit",
    )

    plot_hdpgpc_frequency_overplots(
        x2017_tensor=select_frequency_bins(data2017, cfg),
        x2018_tensor=select_frequency_bins(data2018, cfg),
        freq_used=freq_used,
        labels2017=labels2017,
        labels2018=labels2018,
        output_path=cfg.output_dir / "hdpgpc_frequency_overplots_2017_2018_joint.png",
        max_curves_per_cluster=cfg.max_curves_per_cluster_plot,
        random_state=cfg.random_state,
        representative="mean",
    )

    print("\nDone.")
    print(f"Outputs saved to: {cfg.output_dir.resolve()}")
    print(f"Inferred K from joint 2017+2018 data: {n_clusters_joint}")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def run(cfg: HDPGPCWaveConfig) -> None:
    if cfg.fit_combined_2017_2018:
        return run_combined_2017_2018(cfg)

    np.random.seed(cfg.random_state)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing 2017 data...")
    y2017 = prepare_year(2017, cfg)
    data2017 = y2017["data_avg"]
    freq = y2017["freq"]
    freq_used = y2017["freq_used"]
    n_samples2017 = data2017.shape[0]

    print(f"2017 HDP-GPC tensor shape: {select_frequency_bins(data2017, cfg).shape}")

    # -----------------------------------------------------------------
    # 1) Fit 2017 model, as in notebook cells 33--37.
    # -----------------------------------------------------------------
    print("\nFitting 2017 HDP-GPC model...")
    sw_gp_fit, x_basis, x_train, fit_scales = build_hdpgpc_model(data2017, freq, cfg, stage="fit")
    x_trains2017 = make_x_trains(x_train, n_samples2017)

    start = time.time()
    sw_gp_fit.include_batch(
        x_trains2017,
        select_frequency_bins(data2017, cfg),
    )
    fit_minutes = (time.time() - start) / 60.0
    print(f"2017 include_batch time: {fit_minutes:.3f} min")

    labels2017_raw = extract_labels_from_swgp(sw_gp_fit, n_samples2017, reference_output=0)
    labels2017, original_label_values = remap_labels_to_consecutive(labels2017_raw)
    n_clusters2017 = len(np.unique(labels2017))
    print(f"Inferred 2017 clusters: K={n_clusters2017}")

    ds2017_labeled = add_labels_to_subset(y2017["ds_subset"], labels2017)
    ds2017_labeled.to_netcdf(cfg.output_dir / "ds_hdpgpc_2017.nc")

    df2017 = make_label_dataframe(ds2017_labeled, year_label="2017 fitted")
    df2017.to_csv(cfg.output_dir / "hdpgpc_labels_2017.csv", index=False)

    ds2017_means = make_cluster_mean_dataset(ds2017_labeled)
    ds2017_means.to_netcdf(cfg.output_dir / "ds_hdpgpc_cluster_means_2017.nc")

    np.save(cfg.output_dir / "hdpgpc_labels_2017.npy", labels2017)

    if cfg.save_model_pickle:
        with open(cfg.output_dir / "sw_gp_fit_2017.pkl", "wb") as f:
            pickle.dump(sw_gp_fit, f)

    # -----------------------------------------------------------------
    # 2) Reconstruct model from 2017 labels, as in notebook cells 40--45.
    # -----------------------------------------------------------------
    print("\nReconstructing HDP-GPC model from 2017 labels...")
    sw_gp, _, x_train_reload, reload_scales = build_hdpgpc_model(data2017, freq, cfg, stage="reload")
    x_trains2017_reload = make_x_trains(x_train_reload, n_samples2017)

    start = time.time()
    sw_gp.reload_model_from_labels(
        x_trains2017_reload,
        select_frequency_bins(data2017, cfg),
        labels2017,
        n_clusters2017,
    )
    reload_minutes = (time.time() - start) / 60.0
    print(f"2017 reload_model_from_labels time: {reload_minutes:.3f} min")

    if cfg.save_model_pickle:
        with open(cfg.output_dir / "sw_gp_reloaded_2017.pkl", "wb") as f:
            pickle.dump(sw_gp, f)

    # -----------------------------------------------------------------
    # 3) Assign 2018 spectra to learned regimes, as in notebook cells 46--52.
    # -----------------------------------------------------------------
    print("\nPreparing 2018 data...")
    y2018 = prepare_year(2018, cfg)
    data2018 = y2018["data_avg"]
    n_samples2018 = data2018.shape[0]
    x_trains2018 = make_x_trains(x_train_reload, n_samples2018)

    print(f"2018 HDP-GPC assignment tensor shape: {select_frequency_bins(data2018, cfg).shape}")

    print("Assigning 2018 spectra to learned 2017 regimes...")
    start = time.time()
    labels2018 = sw_gp.cluster_new_batch(
        x_trains2018,
        select_frequency_bins(data2018, cfg),
    )
    assign_minutes = (time.time() - start) / 60.0
    labels2018 = np.asarray(labels2018, dtype=int)
    print(f"2018 cluster_new_batch time: {assign_minutes:.3f} min")

    # In case cluster_new_batch returns labels in another integer dtype/shape
    labels2018 = labels2018.reshape(-1)

    ds2018_labeled = add_labels_to_subset(y2018["ds_subset"], labels2018)
    ds2018_labeled.to_netcdf(cfg.output_dir / "ds_hdpgpc_2018_assigned.nc")

    df2018 = make_label_dataframe(ds2018_labeled, year_label="2018 assigned")
    df2018.to_csv(cfg.output_dir / "hdpgpc_labels_2018_assigned.csv", index=False)
    np.save(cfg.output_dir / "hdpgpc_labels_2018_assigned.npy", labels2018)

    # Combined dataset, matching the notebook's ds_def.nc idea.
    ds2017_for_concat = ds2017_labeled
    ds2018_for_concat = ds2018_labeled
    # Drop depth if one year has it and the other has incompatible coordinates/encoding.
    if "dpt" in ds2017_for_concat.variables and "dpt" in ds2018_for_concat.variables:
        pass
    else:
        ds2017_for_concat = drop_if_present(ds2017_for_concat, "dpt")
        ds2018_for_concat = drop_if_present(ds2018_for_concat, "dpt")

    ds_combined = xr.concat([ds2017_for_concat, ds2018_for_concat], dim="time").sortby("time")
    ds_combined.to_netcdf(cfg.output_dir / "ds_hdpgpc_2017_2018_combined.nc")

    df_combined = pd.concat([df2017, df2018], ignore_index=True).sort_values("time")
    df_combined.to_csv(cfg.output_dir / "hdpgpc_labels_2017_2018_combined.csv", index=False)

    ds_combined_means = make_cluster_mean_dataset(ds_combined)
    ds_combined_means.to_netcdf(cfg.output_dir / "ds_hdpgpc_cluster_means_2017_2018.nc")

    # Summary table
    all_clusters = np.union1d(np.unique(labels2017), np.unique(labels2018))
    counts = pd.DataFrame(
        {
            "cluster": all_clusters.astype(int),
            "n_2017": [int(np.sum(labels2017 == c)) for c in all_clusters],
            "n_2018_assigned": [int(np.sum(labels2018 == c)) for c in all_clusters],
        }
    )
    counts.to_csv(cfg.output_dir / "hdpgpc_cluster_counts_2017_2018.csv", index=False)

    summary = {
        "n_2017": int(n_samples2017),
        "n_2018": int(n_samples2018),
        "k_2017_inferred": int(n_clusters2017),
        "clusters_2017": [int(c) for c in np.unique(labels2017)],
        "clusters_2018_assigned": [int(c) for c in np.unique(labels2018)],
        "fit_minutes": float(fit_minutes),
        "reload_minutes": float(reload_minutes),
        "assign_minutes": float(assign_minutes),
        "low_freq_index": int(cfg.low_freq_index),
        "high_freq_index": int(cfg.high_freq_index),
        "freq_index_step": int(cfg.freq_index_step),
        "freq_indices": [int(i) for i in get_freq_indices(cfg)],
        "hs_min": float(cfg.hs_min),
        "hs_max": float(cfg.hs_max),
        "block_step": int(cfg.block_step),
        "max_models": int(cfg.max_models),
        "n_explore_steps": int(cfg.n_explore_steps),
        "reduce_outputs": bool(cfg.reduce_outputs),
        "reduce_outputs_ratio": float(cfg.reduce_outputs_ratio),
        "fit_scales": {k: str(v) for k, v in fit_scales.items()},
        "reload_scales": {k: str(v) for k, v in reload_scales.items()},
    }
    with open(cfg.output_dir / "hdpgpc_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Diagnostic plots
    plot_cluster_timeline(
        df_combined,
        output_path=cfg.output_dir / "hdpgpc_timeline_2017_2018.png",
        title="HDP-GPC wave-spectra regimes: 2017 fitted and 2018 assigned",
    )

    plot_hdpgpc_frequency_overplots(
        x2017_tensor=select_frequency_bins(data2017, cfg),
        x2018_tensor=select_frequency_bins(data2018, cfg),
        freq_used=freq_used,
        labels2017=labels2017,
        labels2018=labels2018,
        output_path=cfg.output_dir / "hdpgpc_frequency_overplots_2017_with_2018.png",
        max_curves_per_cluster=cfg.max_curves_per_cluster_plot,
        random_state=cfg.random_state,
        representative="mean",
    )

    print("\nDone.")
    print(f"Outputs saved to: {cfg.output_dir.resolve()}")
    print(f"Inferred K from 2017: {n_clusters2017}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> HDPGPCWaveConfig:
    parser = argparse.ArgumentParser(description="HDP-GPC clustering for NDBC directional wave spectra.")

    parser.add_argument("--data-dir", type=Path, default=Path("../data/NDBC"))
    parser.add_argument("--output-dir", type=Path, default=Path("./hdpgpc_wave_results"))

    parser.add_argument("--hs-min", type=float, default=0.5)
    parser.add_argument("--hs-max", type=float, default=2.5)
    parser.add_argument("--block-step", type=int, default=3)

    parser.add_argument("--low-freq-index", type=int, default=8)
    parser.add_argument("--high-freq-index", type=int, default=32)
    parser.add_argument("--freq-index-step", type=int, default=3, help="Subsample frequency bins with np.arange(low_freq_index, high_freq_index, freq_index_step).")

    parser.add_argument("--max-models", type=int, default=50)
    parser.add_argument("--n-explore-steps", type=int, default=5)
    parser.add_argument("--ini-lengthscale", type=float, default=1e-2)
    parser.add_argument("--bound-lengthscale-min", type=float, default=1e-7)
    parser.add_argument("--bound-lengthscale-max", type=float, default=5e-1)
    parser.add_argument("--outputscale", type=float, default=1.0)

    parser.add_argument("--fit-sigma-multiplier", type=float, default=2e-2)
    parser.add_argument("--fit-gamma-multiplier", type=float, default=5e-4)
    parser.add_argument("--reload-sigma-multiplier", type=float, default=1e-3)
    parser.add_argument("--reload-gamma-multiplier", type=float, default=1e-4)

    parser.add_argument("--fit-free-deg-mniv", type=int, default=5)
    parser.add_argument("--reload-free-deg-mniv", type=int, default=10)

    parser.add_argument("--reduce-outputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reduce-outputs-ratio", type=float, default=0.3)
    parser.add_argument("--share-gp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-snr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inducing-points", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reestimate-initial-params", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--max-curves-per-cluster-plot", type=int, default=1000)
    parser.add_argument("--default-depth-2018", type=float, default=33.0)
    parser.add_argument("--save-model-pickle", action="store_true")
    parser.add_argument(
        "--fit-combined-2017-2018",
        action="store_true",
        help=(
            "Fit HDP-GPC directly on the concatenated 2017+2018 spectra. "
            "If omitted, the script fits on 2017 and assigns 2018 afterwards."
        ),
    )

    args = parser.parse_args()

    return HDPGPCWaveConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hs_min=args.hs_min,
        hs_max=args.hs_max,
        block_step=args.block_step,
        low_freq_index=args.low_freq_index,
        high_freq_index=args.high_freq_index,
        freq_index_step=args.freq_index_step,
        max_models=args.max_models,
        n_explore_steps=args.n_explore_steps,
        ini_lengthscale=args.ini_lengthscale,
        bound_lengthscale_min=args.bound_lengthscale_min,
        bound_lengthscale_max=args.bound_lengthscale_max,
        outputscale=args.outputscale,
        fit_sigma_multiplier=args.fit_sigma_multiplier,
        fit_gamma_multiplier=args.fit_gamma_multiplier,
        reload_sigma_multiplier=args.reload_sigma_multiplier,
        reload_gamma_multiplier=args.reload_gamma_multiplier,
        fit_free_deg_mniv=args.fit_free_deg_mniv,
        reload_free_deg_mniv=args.reload_free_deg_mniv,
        reduce_outputs=args.reduce_outputs,
        reduce_outputs_ratio=args.reduce_outputs_ratio,
        share_gp=args.share_gp,
        use_snr=args.use_snr,
        cuda=args.cuda,
        verbose=args.verbose,
        inducing_points=args.inducing_points,
        reestimate_initial_params=args.reestimate_initial_params,
        random_state=args.random_state,
        max_curves_per_cluster_plot=args.max_curves_per_cluster_plot,
        default_depth_2018=args.default_depth_2018,
        save_model_pickle=args.save_model_pickle,
        fit_combined_2017_2018=args.fit_combined_2017_2018,
    )


if __name__ == "__main__":
    config = parse_args()
    run(config)
