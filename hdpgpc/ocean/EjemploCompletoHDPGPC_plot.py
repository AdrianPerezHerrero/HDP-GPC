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
    6. Use frequency bins low_freq_index:high_freq_index.
    7. Fit multi-output offline HDP-GPC to 2017 spectra.
    8. Reconstruct the model from the learned 2017 labels.
    9. Assign 2018 spectra to the learned 2017 regimes.
   10. Save labelled NetCDF/CSV outputs and diagnostic figures.

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

# Notebook plotting helper. Place func_plot.py in the same folder as this script.
# The function below is the one used in EjemploCompleto.ipynb to generate
# the polar-directional / integrated-frequency cluster summary.
try:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from func_plot import plot_cluster_spectrum_and_timeline
except ImportError:
    plot_cluster_spectrum_and_timeline = None



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

    # Optional label reloading. If reload_labels_2017 is provided, the
    # initial HDP-GPC fitting step is skipped and the model is reconstructed
    # with reload_model_from_labels, matching the notebook workflow.
    reload_labels_2017: Optional[Path] = None
    reload_labels_2018: Optional[Path] = None
    reload_labels_joint: Optional[Path] = None
    label_column: str = "cluster"

    # If true, the notebook-style cluster summary is generated using only
    # the 2017 fitted spectra. The model can still assign/export 2018 data.
    plot_2017_only: bool = False

    # Match EjemploCompleto.ipynb: the notebook calls
    # plot_cluster_spectrum_and_timeline(..., norm=False). If True,
    # wavespectra normalises the polar spectra, which changes the colour/radial
    # scaling and can make the directional panels look like rings.
    plot_normalised: bool = False

    # Diagnostic cluster-overplot layout. `plot_ncols` controls how many
    # cluster panels are placed per row. If `plot_include_direction` is True,
    # each cluster panel contains both the direction-integrated frequency
    # spectra and the frequency-integrated directional distribution.
    plot_ncols: int = 4
    plot_include_direction: bool = True


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
        "freq_used": freq[cfg.low_freq_index:cfg.high_freq_index],
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
    y = data_avg[:, cfg.low_freq_index:cfg.high_freq_index, :]
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
    scales = compute_wave_l_scale(data_avg, cfg, stage=stage)

    x_basis = np.atleast_2d(freq[cfg.low_freq_index:cfg.high_freq_index]).T
    x_train = np.atleast_2d(freq[cfg.low_freq_index:cfg.high_freq_index]).T
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


def _extract_labels_from_netcdf(
    path: Path,
    expected_length: int,
    label_column: str = "cluster",
) -> np.ndarray:
    """
    Extract cluster labels from a labelled NetCDF dataset.

    This mirrors the notebook workflow, where a dataset produced as
    `ds_cluster_def.nc` or `ds_def.nc` is opened with xarray and labels are
    read from `ds.cluster_label.values` before calling
    `reload_model_from_labels`.
    """
    ds = xr.open_dataset(path, decode_timedelta=True)

    # In the notebook the variable is called `cluster_label`.  Since the
    # command-line default is `cluster`, prefer `cluster_label` for NetCDF
    # files unless the user explicitly provides another existing variable.
    if label_column != "cluster" and label_column in ds.variables:
        label_var = label_column
    else:
        candidate_vars = ["cluster_label", "labels", "label", "cluster"]
        label_var = next((name for name in candidate_vars if name in ds.variables), None)

    if label_var is None:
        raise ValueError(
            f"Could not find a label variable in {path}. Expected one of "
            "`cluster_label`, `labels`, `label`, `cluster`, or the variable "
            f"specified with --label-column={label_column!r}."
        )

    label_da = ds[label_var]

    # If a combined 2017+2018 dataset is provided, try to select the 2017
    # subset automatically when that gives the expected length.
    if label_da.size != expected_length and "time" in label_da.dims and "time" in ds.coords:
        years = pd.to_datetime(ds["time"].values).year
        for year in [2017, 2018]:
            mask = years == year
            if int(mask.sum()) == expected_length:
                label_da = label_da.isel(time=mask)
                print(
                    f"Loaded labels from {path.name}:{label_var} after selecting year {year} "
                    f"({expected_length} samples)."
                )
                break

    labels = np.asarray(label_da.values).reshape(-1)
    if labels.size != expected_length:
        raise ValueError(
            f"Loaded {labels.size} labels from {path}:{label_var}, but expected "
            f"{expected_length}. If this is a combined NetCDF, provide a file whose "
            "time dimension matches the filtered/coarsened 2017 spectra or adjust "
            "the preprocessing arguments."
        )

    return labels


def load_labels_from_file(path: Path, expected_length: int, label_column: str = "cluster") -> np.ndarray:
    """
    Load labels from .npy, .npz, .csv, .txt, .dat, .nc or .netcdf files.

    For NetCDF files, labels are read from `cluster_label` by default, matching
    the notebook pattern `labels_num = ds.cluster_label.values`. For CSV files,
    the function first tries `label_column`; if that column is absent and there
    is a single column, that column is used. For NPZ files, it tries common keys
    such as `labels`, `labels2017`, and `cluster`. Labels are returned as a
    one-dimensional integer array and must match expected_length.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label file does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        labels = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        candidate_keys = ["labels", "labels2017", "labels_2017", "cluster", "clusters"]
        found_key = next((key for key in candidate_keys if key in data.files), None)
        if found_key is None:
            if len(data.files) == 1:
                found_key = data.files[0]
            else:
                raise ValueError(
                    f"Could not infer label array from {path}. Available keys: {data.files}. "
                    f"Use one of {candidate_keys} or provide a single-array NPZ."
                )
        labels = data[found_key]
    elif suffix in {".nc", ".netcdf"}:
        labels = _extract_labels_from_netcdf(
            path=path,
            expected_length=expected_length,
            label_column=label_column,
        )
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if label_column in df.columns:
            labels = df[label_column].to_numpy()
        elif len(df.columns) == 1:
            labels = df.iloc[:, 0].to_numpy()
        else:
            raise ValueError(
                f"Column `{label_column}` not found in {path}. Available columns: {list(df.columns)}"
            )
    elif suffix in {".txt", ".dat"}:
        labels = np.loadtxt(path)
    else:
        raise ValueError(
            f"Unsupported label file extension `{suffix}`. Use .npy, .npz, "
            ".csv, .tsv, .txt, .dat, .nc or .netcdf."
        )

    labels = np.asarray(labels).reshape(-1)
    if labels.size != expected_length:
        raise ValueError(
            f"Loaded {labels.size} labels from {path}, but expected {expected_length}. "
            "Check that labels correspond to the same filtered/coarsened spectra."
        )

    if not np.all(np.isfinite(labels)):
        raise ValueError(f"Labels in {path} contain NaN or infinite values.")

    return labels.astype(int)



def _find_label_variable(ds: xr.Dataset, label_column: str = "cluster") -> str:
    """Find the label variable in a labelled NetCDF dataset."""
    if label_column != "cluster" and label_column in ds.variables:
        return label_column

    candidate_vars = ["cluster_label", "labels", "label", "cluster"]
    label_var = next((name for name in candidate_vars if name in ds.variables), None)
    if label_var is None:
        raise ValueError(
            "Could not find a label variable in the NetCDF file. Expected one of "
            "`cluster_label`, `labels`, `label`, `cluster`, or the variable "
            f"specified with --label-column={label_column!r}."
        )
    return label_var


def load_joint_labels_from_file(
    path: Path,
    expected_length_2017: int,
    expected_length_2018: int,
    label_column: str = "cluster",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single set of labels covering the preprocessed 2017 and 2018 spectra.

    The function accepts the same formats as `load_labels_from_file`. If a
    NetCDF/CSV/TSV file contains a time column or coordinate, the labels are
    split by calendar year when the 2017 and 2018 counts match the preprocessed
    tensors. Otherwise, the labels are assumed to be ordered as
    [2017 samples, 2018 samples], which is the order used by this script for
    the notebook-style combined plots.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Joint label file does not exist: {path}")

    expected_total = expected_length_2017 + expected_length_2018
    suffix = path.suffix.lower()

    labels = None
    years = None

    if suffix in {".nc", ".netcdf"}:
        ds = xr.open_dataset(path, decode_timedelta=True)
        label_var = _find_label_variable(ds, label_column=label_column)
        labels = np.asarray(ds[label_var].values).reshape(-1)

        if "time" in ds.coords and ds[label_var].size == ds.sizes.get("time", ds[label_var].size):
            try:
                years = pd.to_datetime(ds["time"].values).year
            except Exception:
                years = None

    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if label_column in df.columns:
            labels = df[label_column].to_numpy()
        elif "cluster_label" in df.columns:
            labels = df["cluster_label"].to_numpy()
        elif len(df.columns) == 1:
            labels = df.iloc[:, 0].to_numpy()
        else:
            raise ValueError(
                f"Column `{label_column}` not found in {path}. Available columns: {list(df.columns)}"
            )

        if "time" in df.columns:
            try:
                years = pd.to_datetime(df["time"]).dt.year.to_numpy()
            except Exception:
                years = None
        elif "date" in df.columns:
            try:
                years = pd.to_datetime(df["date"]).dt.year.to_numpy()
            except Exception:
                years = None

    else:
        labels = load_labels_from_file(
            path=path,
            expected_length=expected_total,
            label_column=label_column,
        )

    labels = np.asarray(labels).reshape(-1)
    if labels.size != expected_total:
        raise ValueError(
            f"Loaded {labels.size} joint labels from {path}, but expected "
            f"{expected_total} = {expected_length_2017} + {expected_length_2018}. "
            "Check that the labels correspond to the same filtered/coarsened spectra."
        )

    if not np.all(np.isfinite(labels)):
        raise ValueError(f"Labels in {path} contain NaN or infinite values.")

    labels = labels.astype(int)

    if years is not None and len(years) == labels.size:
        mask2017 = years == 2017
        mask2018 = years == 2018
        if int(mask2017.sum()) == expected_length_2017 and int(mask2018.sum()) == expected_length_2018:
            print(
                f"Splitting joint labels from {path.name} by time coordinate/column "
                f"({expected_length_2017} samples in 2017, {expected_length_2018} in 2018)."
            )
            return labels[mask2017], labels[mask2018]

    print(
        f"Splitting joint labels from {path.name} by concatenated order: "
        f"first {expected_length_2017} samples as 2017 and remaining "
        f"{expected_length_2018} samples as 2018."
    )
    return labels[:expected_length_2017], labels[expected_length_2017:]



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
    ncols: int = 4,
    include_direction: bool = True,
    directions: Optional[np.ndarray] = None,
) -> None:
    """
    Plot cluster overplots ordered from the smallest cluster id to the largest.

    The first panel is therefore cluster 0 in the top-left position and the
    last cluster appears in the bottom-right direction of the grid. If
    include_direction=True, each cluster cell contains two internal panels:
    the direction-integrated frequency spectra and the frequency-integrated
    directional distribution. If include_direction=False, only the frequency
    spectra are plotted.
    """
    rng = np.random.default_rng(random_state)

    labels2017 = np.asarray(labels2017, dtype=int).reshape(-1)
    labels2018 = None if labels2018 is None else np.asarray(labels2018, dtype=int).reshape(-1)

    if labels2018 is not None:
        clusters = np.union1d(np.unique(labels2017), np.unique(labels2018))
    else:
        clusters = np.unique(labels2017)
    clusters = np.sort(clusters.astype(int))
    k = len(clusters)

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(k / ncols))

    spectra2017_freq = x2017_tensor.sum(axis=2)
    spectra2018_freq = x2018_tensor.sum(axis=2) if x2018_tensor is not None else None

    spectra2017_dir = x2017_tensor.sum(axis=1)
    spectra2018_dir = x2018_tensor.sum(axis=1) if x2018_tensor is not None else None

    if include_direction and directions is None:
        directions = np.arange(x2017_tensor.shape[2])

    cell_height = 3.2 if not include_direction else 5.6
    fig = plt.figure(figsize=(4.2 * ncols, cell_height * nrows))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.30, hspace=0.45)

    for panel, c in enumerate(clusters):
        row = panel // ncols
        col = panel % ncols

        if include_direction:
            inner = outer[row, col].subgridspec(2, 1, height_ratios=[2.4, 1.5], hspace=0.22)
            ax_freq = fig.add_subplot(inner[0, 0])
            ax_dir = fig.add_subplot(inner[1, 0])
        else:
            ax_freq = fig.add_subplot(outer[row, col])
            ax_dir = None

        idx17_full = np.where(labels2017 == c)[0]
        idx17 = idx17_full.copy()
        if idx17.size > max_curves_per_cluster:
            idx17 = rng.choice(idx17, size=max_curves_per_cluster, replace=False)

        idx18_full = np.array([], dtype=int)
        idx18 = idx18_full
        if labels2018 is not None and spectra2018_freq is not None:
            idx18_full = np.where(labels2018 == c)[0]
            idx18 = idx18_full.copy()
            if idx18.size > max_curves_per_cluster:
                idx18 = rng.choice(idx18, size=max_curves_per_cluster, replace=False)

        # -------------------------
        # Frequency-marginal spectra
        # -------------------------
        if idx17.size > 0:
            ax_freq.plot(
                freq_used,
                spectra2017_freq[idx17].T,
                color="0.75",
                alpha=0.25,
                linewidth=0.8,
            )
        if idx18.size > 0 and spectra2018_freq is not None:
            ax_freq.plot(
                freq_used,
                spectra2018_freq[idx18].T,
                color="tab:blue",
                alpha=0.30,
                linewidth=0.8,
            )

        freq_rep_sources = []
        if idx17_full.size > 0:
            freq_rep_sources.append(spectra2017_freq[idx17_full])
        if idx18_full.size > 0 and spectra2018_freq is not None:
            freq_rep_sources.append(spectra2018_freq[idx18_full])
        if freq_rep_sources:
            freq_rep_data = np.vstack(freq_rep_sources)
            if representative == "median":
                freq_rep = np.median(freq_rep_data, axis=0)
            else:
                freq_rep = np.mean(freq_rep_data, axis=0)
            ax_freq.plot(freq_used, freq_rep, color="red", linewidth=2.0)

        ax_freq.set_title(
            f"Cluster {int(c)} (2017 n={idx17_full.size}, 2018 n={idx18_full.size})",
            fontsize=9,
        )
        ax_freq.set_xlabel("Frequency [Hz]")
        ax_freq.set_ylabel("Spectral density")
        ax_freq.grid(alpha=0.2)

        # -------------------------
        # Direction-marginal spectra
        # -------------------------
        if include_direction and ax_dir is not None:
            if idx17.size > 0:
                ax_dir.plot(
                    directions,
                    spectra2017_dir[idx17].T,
                    color="0.75",
                    alpha=0.25,
                    linewidth=0.8,
                )
            if idx18.size > 0 and spectra2018_dir is not None:
                ax_dir.plot(
                    directions,
                    spectra2018_dir[idx18].T,
                    color="tab:blue",
                    alpha=0.30,
                    linewidth=0.8,
                )

            dir_rep_sources = []
            if idx17_full.size > 0:
                dir_rep_sources.append(spectra2017_dir[idx17_full])
            if idx18_full.size > 0 and spectra2018_dir is not None:
                dir_rep_sources.append(spectra2018_dir[idx18_full])
            if dir_rep_sources:
                dir_rep_data = np.vstack(dir_rep_sources)
                if representative == "median":
                    dir_rep = np.median(dir_rep_data, axis=0)
                else:
                    dir_rep = np.mean(dir_rep_data, axis=0)
                ax_dir.plot(directions, dir_rep, color="red", linewidth=2.0)

            ax_dir.set_xlabel("Direction [deg]")
            ax_dir.set_ylabel("Energy")
            ax_dir.grid(alpha=0.2)

    # Hide unused cells, if any.
    for empty_panel in range(k, nrows * ncols):
        row = empty_panel // ncols
        col = empty_panel % ncols
        ax = fig.add_subplot(outer[row, col])
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_notebook_style_cluster_summary(
    ds_subset: xr.Dataset,
    df: pd.DataFrame,
    ds_cluster_means: xr.Dataset,
    windowed_data_no_direct: np.ndarray,
    freq: np.ndarray,
    output_path: Path,
    ncols = 1,
    include_direction: bool = True,
    norm: bool = False,
    clusters_to_plot: Optional[list] = None,
) -> None:
    """
    Generate the same cluster-summary figure used in EjemploCompleto.ipynb.

    This is a thin wrapper around func_plot.plot_cluster_spectrum_and_timeline.
    The input dataframe index is reset deliberately because the notebook
    function uses df.index[df["cluster"] == k] to index rows of
    windowed_data_no_direct. Therefore, df and windowed_data_no_direct must
    have exactly the same order.
    """
    if plot_cluster_spectrum_and_timeline is None:
        raise ImportError(
            "Could not import plot_cluster_spectrum_and_timeline from func_plot.py. "
            "Place func_plot.py in the same directory as EjemploCompletoHDPGPC.py."
        )

    df_plot = df.reset_index(drop=True).copy()

    if len(df_plot) != windowed_data_no_direct.shape[0]:
        raise ValueError(
            f"Plot dataframe has {len(df_plot)} rows but windowed_data_no_direct "
            f"has {windowed_data_no_direct.shape[0]} rows. They must match."
        )

    # The notebook passes `efth_ordered = ds_cluster_means["efth"].transpose("cluster", "dir", "freq")`.
    # The plotting helper expects this DataArray, not the full Dataset.
    if isinstance(ds_cluster_means, xr.Dataset):
        if "efth" not in ds_cluster_means:
            raise ValueError("ds_cluster_means must contain variable `efth`.")
        efth_ordered = ds_cluster_means["efth"]
    else:
        efth_ordered = ds_cluster_means

    if {"cluster", "dir", "freq"}.issubset(set(efth_ordered.dims)):
        efth_ordered = efth_ordered.transpose("cluster", "dir", "freq")

    plt.close("all")
    plt.ioff()

    plot_cluster_spectrum_and_timeline(
        ds_subset=ds_subset,
        df=df_plot,
        efth_ordered=efth_ordered,
        windowed_data_no_direct=windowed_data_no_direct,
        freq=freq,
        n_clusters=None,
        norm=norm,
        clusters_to_plot=clusters_to_plot,
        ncols=ncols,
        include_direction=include_direction
    )

    fig = plt.gcf()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def run(cfg: HDPGPCWaveConfig) -> None:
    np.random.seed(cfg.random_state)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing 2017 data...")
    y2017 = prepare_year(2017, cfg)
    data2017 = y2017["data_avg"]
    freq = y2017["freq"]
    freq_used = y2017["freq_used"]
    n_samples2017 = data2017.shape[0]

    print(f"2017 HDP-GPC tensor shape: {data2017[:, cfg.low_freq_index:cfg.high_freq_index, :].shape}")

    # -----------------------------------------------------------------
    # 1) Obtain labels. By default, fit the model on 2017 and then assign
    #    2018. If --reload-labels-joint is provided, skip the model fit,
    #    split the supplied joint labels into 2017 and 2018, and use them
    #    directly for exports and plots.
    # -----------------------------------------------------------------
    joint_label_mode = cfg.reload_labels_joint is not None
    y2018 = None
    data2018 = None
    n_samples2018 = None
    labels2018 = None
    sw_gp_fit = None
    fit_scales = {}
    reload_scales = {}
    fit_minutes = 0.0
    reload_minutes = 0.0
    assign_minutes = 0.0

    if joint_label_mode and (cfg.reload_labels_2017 is not None or cfg.reload_labels_2018 is not None):
        raise ValueError(
            "Use either --reload-labels-joint or the pair "
            "--reload-labels-2017/--reload-labels-2018, not both."
        )

    if joint_label_mode:
        print("\nPreparing 2018 data to split joint labels...")
        y2018 = prepare_year(2018, cfg)
        data2018 = y2018["data_avg"]
        n_samples2018 = data2018.shape[0]

        print(f"2018 HDP-GPC tensor shape: {data2018[:, cfg.low_freq_index:cfg.high_freq_index, :].shape}")
        print(f"\nLoading joint 2017+2018 labels from {cfg.reload_labels_joint} ...")

        labels2017_loaded, labels2018_loaded = load_joint_labels_from_file(
            cfg.reload_labels_joint,
            expected_length_2017=n_samples2017,
            expected_length_2018=n_samples2018,
            label_column=cfg.label_column,
        )

        labels_joint_loaded = np.concatenate([labels2017_loaded, labels2018_loaded], axis=0)
        labels_joint, original_label_values = remap_labels_to_consecutive(labels_joint_loaded)

        labels2017 = labels_joint[:n_samples2017]
        labels2018 = labels_joint[n_samples2017:]
        n_clusters2017 = len(np.unique(labels_joint))

        print(
            f"Reloaded joint labels: K={n_clusters2017} "
            f"({n_samples2017} samples in 2017, {n_samples2018} samples in 2018)."
        )

    elif cfg.reload_labels_2017 is not None:
        print(f"\nLoading 2017 labels from {cfg.reload_labels_2017} ...")
        labels2017_loaded = load_labels_from_file(
            cfg.reload_labels_2017,
            expected_length=n_samples2017,
            label_column=cfg.label_column,
        )
        labels2017, original_label_values = remap_labels_to_consecutive(labels2017_loaded)
        n_clusters2017 = len(np.unique(labels2017))
        print(f"Reloaded 2017 labels: K={n_clusters2017}")

    else:
        print("\nFitting 2017 HDP-GPC model...")
        sw_gp_fit, x_basis, x_train, fit_scales = build_hdpgpc_model(data2017, freq, cfg, stage="fit")
        x_trains2017 = make_x_trains(x_train, n_samples2017)

        start = time.time()
        sw_gp_fit.include_batch(
            x_trains2017,
            data2017[:, cfg.low_freq_index:cfg.high_freq_index, :],
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

    if cfg.save_model_pickle and sw_gp_fit is not None:
        with open(cfg.output_dir / "sw_gp_fit_2017.pkl", "wb") as f:
            pickle.dump(sw_gp_fit, f)

    # -----------------------------------------------------------------
    # 2) Reconstruct model from 2017 labels and assign 2018 spectra, unless
    #    joint 2017+2018 labels were provided. In joint-label mode the script
    #    acts as a plotting/export utility and avoids the expensive model
    #    reconstruction and assignment steps.
    # -----------------------------------------------------------------
    if not joint_label_mode:
        print("\nReconstructing HDP-GPC model from 2017 labels...")
        sw_gp, _, x_train_reload, reload_scales = build_hdpgpc_model(data2017, freq, cfg, stage="reload")
        x_trains2017_reload = make_x_trains(x_train_reload, n_samples2017)

        start = time.time()
        sw_gp.reload_model_from_labels(
            x_trains2017_reload,
            data2017[:, cfg.low_freq_index:cfg.high_freq_index, :],
            labels2017,
            n_clusters2017,
        )
        reload_minutes = (time.time() - start) / 60.0
        print(f"2017 reload_model_from_labels time: {reload_minutes:.3f} min")

        if cfg.save_model_pickle:
            with open(cfg.output_dir / "sw_gp_reloaded_2017.pkl", "wb") as f:
                pickle.dump(sw_gp, f)

        print("\nPreparing 2018 data...")
        y2018 = prepare_year(2018, cfg)
        data2018 = y2018["data_avg"]
        n_samples2018 = data2018.shape[0]
        x_trains2018 = make_x_trains(x_train_reload, n_samples2018)

        print(f"2018 HDP-GPC assignment tensor shape: {data2018[:, cfg.low_freq_index:cfg.high_freq_index, :].shape}")

        if cfg.reload_labels_2018 is not None:
            print(f"Loading 2018 labels from {cfg.reload_labels_2018} ...")
            labels2018_loaded = load_labels_from_file(
                cfg.reload_labels_2018,
                expected_length=n_samples2018,
                label_column=cfg.label_column,
            )
            # Map original 2017 label values to the consecutive ids used after
            # reloading 2017. Unknown 2018 labels are kept only if they already
            # match the consecutive id space.
            original_to_new = {int(old): int(new) for new, old in enumerate(original_label_values)}
            labels2018 = np.array(
                [original_to_new.get(int(label), int(label)) for label in labels2018_loaded],
                dtype=int,
            )
            assign_minutes = 0.0
            print("Using provided 2018 labels; cluster_new_batch was skipped.")
        else:
            print("Assigning 2018 spectra to learned 2017 regimes...")
            start = time.time()
            labels2018 = sw_gp.cluster_new_batch(
                x_trains2018,
                data2018[:, cfg.low_freq_index:cfg.high_freq_index, :],
            )
            assign_minutes = (time.time() - start) / 60.0
            labels2018 = np.asarray(labels2018, dtype=int)
            print(f"2018 cluster_new_batch time: {assign_minutes:.3f} min")
    else:
        print("\nUsing provided joint labels; reload_model_from_labels and cluster_new_batch were skipped.")

    # In case cluster_new_batch or a label loader returns labels in another integer dtype/shape
    labels2018 = np.asarray(labels2018, dtype=int).reshape(-1)

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
        output_path=cfg.output_dir / "hdpgpc_timeline_2017_2018.pdf",
        title="HDP-GPC wave-spectra regimes: 2017 fitted and 2018 assigned",
    )

    plot_hdpgpc_frequency_overplots(
        x2017_tensor=data2017[:, cfg.low_freq_index:cfg.high_freq_index, :],
        x2018_tensor=data2018[:, cfg.low_freq_index:cfg.high_freq_index, :],
        freq_used=freq_used,
        labels2017=labels2017,
        labels2018=labels2018,
        output_path=cfg.output_dir / "hdpgpc_frequency_overplots_2017_with_2018.png",
        max_curves_per_cluster=cfg.max_curves_per_cluster_plot,
        random_state=cfg.random_state,
        representative="mean",
        ncols=cfg.plot_ncols,
        include_direction=cfg.plot_include_direction,
        directions=y2017["directions"],
    )

    # Notebook-style figure from func_plot.py: polar directional spectra +
    # direction-integrated frequency spectra, using the same function as
    # EjemploCompleto.ipynb. The dataframe index must match rows of
    # windowed_data_no_direct because func_plot.py indexes spectra by df.index.
    if cfg.plot_2017_only:
        print("Generating notebook-style cluster summary using only 2017 samples...")
        df_for_notebook_plot = df2017.reset_index(drop=True)
        ds_for_notebook_plot = ds2017_labeled.assign_coords(
            time=df_for_notebook_plot["time"].to_numpy()
        )
        windowed_data_no_direct_for_plot = y2017["nondir_avg"]
        notebook_plot_output = cfg.output_dir / "hdpgpc_notebook_style_cluster_summary_2017.png"
    else:
        print("Generating notebook-style cluster summary using 2017 fitted + 2018 assigned samples...")
        df_for_notebook_plot = pd.concat([df2017, df2018], ignore_index=True)
        ds_for_notebook_plot = xr.concat([ds2017_for_concat, ds2018_for_concat], dim="time")
        ds_for_notebook_plot = ds_for_notebook_plot.assign_coords(
            time=df_for_notebook_plot["time"].to_numpy()
        )
        windowed_data_no_direct_for_plot = np.concatenate(
            [y2017["nondir_avg"], y2018["nondir_avg"]],
            axis=0,
        )
        notebook_plot_output = cfg.output_dir / "hdpgpc_notebook_style_cluster_summary.pdf"

    print(f"Notebook-style polar plot normalised={cfg.plot_normalised} (EjemploCompleto.ipynb uses False).")

    plot_notebook_style_cluster_summary(
        ds_subset=ds_for_notebook_plot,
        df=df_for_notebook_plot,
        ds_cluster_means=make_cluster_mean_dataset(ds_for_notebook_plot),
        windowed_data_no_direct=windowed_data_no_direct_for_plot,
        freq=freq,
        output_path=notebook_plot_output,
        norm=cfg.plot_normalised,
        ncols=cfg.plot_ncols,
        include_direction=cfg.plot_include_direction,
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

    parser.add_argument("--low-freq-index", type=int, default=3)
    parser.add_argument("--high-freq-index", type=int, default=35)

    parser.add_argument("--max-models", type=int, default=50)
    parser.add_argument("--n-explore-steps", type=int, default=10)
    parser.add_argument("--ini-lengthscale", type=float, default=1e-2)
    parser.add_argument("--bound-lengthscale-min", type=float, default=1e-7)
    parser.add_argument("--bound-lengthscale-max", type=float, default=5e-1)
    parser.add_argument("--outputscale", type=float, default=1.0)

    parser.add_argument("--fit-sigma-multiplier", type=float, default=1e-2)
    parser.add_argument("--fit-gamma-multiplier", type=float, default=1e-3)
    parser.add_argument("--reload-sigma-multiplier", type=float, default=1e-3)
    parser.add_argument("--reload-gamma-multiplier", type=float, default=1e-4)

    parser.add_argument("--fit-free-deg-mniv", type=int, default=3)
    parser.add_argument("--reload-free-deg-mniv", type=int, default=5)

    parser.add_argument("--reduce-outputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reduce-outputs-ratio", type=float, default=0.3)
    parser.add_argument("--share-gp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-snr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inducing-points", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reestimate-initial-params", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--max-curves-per-cluster-plot", type=int, default=6000)
    parser.add_argument("--default-depth-2018", type=float, default=33.0)
    parser.add_argument("--save-model-pickle", action="store_true")
    parser.add_argument(
        "--plot-2017-only",
        action="store_true",
        help=(
            "Generate the notebook-style polar/frequency cluster summary using only "
            "the 2017 fitted spectra. The script still reloads/reconstructs the model "
            "and assigns/exports 2018 data unless other code is changed."
        ),
    )
    parser.add_argument(
        "--plot-normalised",
        action="store_true",
        help=(
            "Pass norm=True to func_plot.plot_cluster_spectrum_and_timeline. "
            "By default this is disabled to reproduce EjemploCompleto.ipynb, "
            "which uses norm=False for the polar spectra."
        ),
    )
    parser.add_argument(
        "--plot-ncols",
        type=int,
        default=4,
        help="Number of cluster panels per row in the diagnostic overplot figure.",
    )
    parser.add_argument(
        "--plot-include-direction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled, each cluster panel in the diagnostic overplot includes "
            "both frequency-marginal and direction-marginal spectra. Use "
            "--no-plot-include-direction to plot only frequency spectra."
        ),
    )

    parser.add_argument(
        "--reload-labels-2017",
        type=Path,
        default=None,
        help=(
            "Path to previously computed 2017 labels (.npy, .npz, .csv, .tsv, .txt, .nc). "
            "If provided, the initial include_batch fit is skipped and the model is "
            "rebuilt with reload_model_from_labels, as in EjemploCompleto.ipynb."
        ),
    )
    parser.add_argument(
        "--reload-labels-2018",
        type=Path,
        default=None,
        help=(
            "Optional path to 2018 labels (.npy, .npz, .csv, .tsv, .txt, .nc) to use for plotting/exports. If omitted, "
            "2018 spectra are assigned with cluster_new_batch after reloading the 2017 model."
        ),
    )
    parser.add_argument(
        "--reload-labels-joint",
        type=Path,
        default=None,
        help=(
            "Path to one label file covering both preprocessed 2017 and 2018 samples "
            "(.npy, .npz, .csv, .tsv, .txt, .nc). If the file has a time column or "
            "coordinate, labels are split by year; otherwise they are split in "
            "concatenated order [2017, 2018]. This skips model fitting, model reload, "
            "and 2018 assignment, and only exports/plots the provided labels."
        ),
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="cluster",
        help="Column/variable used when reloading labels from CSV/TSV/NetCDF files.",
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
        reload_labels_2017=args.reload_labels_2017,
        reload_labels_2018=args.reload_labels_2018,
        reload_labels_joint=args.reload_labels_joint,
        label_column=args.label_column,
        plot_2017_only=args.plot_2017_only,
        plot_normalised=args.plot_normalised,
        plot_ncols=args.plot_ncols,
        plot_include_direction=args.plot_include_direction,
    )


if __name__ == "__main__":
    config = parse_args()
    run(config)
