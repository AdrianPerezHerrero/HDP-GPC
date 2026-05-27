#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLARA clustering for NDBC directional wave spectra.

This script follows the preprocessing used in EjemploCompleto.ipynb:
    1. Read NDBC directional spectra for 2017 and 2018.
    2. Merge wind and depth metadata when available.
    3. Compute Hs from the non-directional spectrum.
    4. Keep spectra with 0.5 < Hs < 2.5 m.
    5. Average consecutive blocks of 3 spectra.
    6. Use frequency bins 3:35.
    7. Apply CLARA using Manhattan distance and no standardisation.
    8. Scan K progressively and select a Hamilton-style exploratory K.
    9. By default, fit CLARA on 2017 and assign 2018 spectra to the 2017 medoids.
       With --fit-combined-2017-2018, fit CLARA jointly on the 3-hour averaged
       spectra from 2017 and 2018.

Dependencies:
    numpy, pandas, xarray, matplotlib, scikit-learn, wavespectra

Example:
    python clara_wave_spectra.py \
        --data-dir ../data/NDBC \
        --output-dir ./clara_wave_results \
        --k-min 2 \
        --k-max 30
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import pairwise_distances, silhouette_score

try:
    from wavespectra import read_ndbc
except ImportError as exc:
    raise ImportError(
        "wavespectra is required. Install it with `pip install wavespectra` "
        "or run the script in the same environment as the notebook."
    ) from exc


warnings.filterwarnings("ignore", message="Can't decode floating point timedelta to 's'")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class ClaraConfig:
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

    use_directional_spectra: bool = True
    fit_combined_2017_2018: bool = False

    k_min: int = 2
    k_max: int = 30
    selected_k: Optional[int] = None

    clara_samples: int = 5
    clara_sample_size: int = 333
    silhouette_sample_size: int = 1200
    random_state: int = 7

    min_relative_improvement: float = 0.02
    elbow_patience: int = 2

    max_curves_per_cluster_plot: int = 100
    default_depth_2018: float = 33.0


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
    """
    Convert NDBC directional file with wavespectra if the converted file
    does not already exist.
    """
    if converted_path.exists():
        return xr.open_dataset(converted_path, decode_timedelta=True)

    print(f"Converting {raw_path} -> {converted_path}")
    ds = read_ndbc(raw_path, directional=True, weight_coeff=True)
    ds = squeeze_lat_lon(ds)
    ds.to_netcdf(converted_path)
    return xr.open_dataset(converted_path, decode_timedelta=True)


def read_nondirectional_spectrum(raw_path: Path) -> xr.Dataset:
    """Read the non-directional NDBC spectrum used to compute Hs."""
    ds = read_ndbc(raw_path, directional=False)
    ds = squeeze_lat_lon(ds)
    return ds


def merge_meteo_and_depth(
    ds_wave: xr.Dataset,
    meteo_path: Optional[Path],
    ocean_path: Optional[Path] = None,
    default_depth: Optional[float] = None,
) -> xr.Dataset:
    """
    Replicates the notebook merge of wind speed, wind direction and depth.
    Wind/depth are not needed for CLARA, but they are useful for later
    interpretation and saved outputs.
    """
    ds = ds_wave.copy()

    if meteo_path is not None and meteo_path.exists():
        ds_meteo = xr.open_dataset(meteo_path, decode_timedelta=True)
        ds_meteo = squeeze_lat_lon(ds_meteo)

        # Avoid interpolation issues with encoded/cftime-like variables.
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
    """
    Notebook formula:
        Hs = 4 * sqrt(S @ delta_f)
    where S is the non-directional spectrum.
    """
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the Hamilton-style Hs range filtering used in the notebook."""
    hs = compute_hs_from_nondirectional(data_no_direct, freq)
    chosen = np.where((hs > hs_min) & (hs < hs_max))[0]
    print(f"Spectra with {hs_min} < Hs < {hs_max}: {chosen.size}")
    return data_direct[chosen], data_no_direct[chosen], chosen


def block_average(data: np.ndarray, step: int) -> np.ndarray:
    """Average consecutive non-overlapping blocks, as in the notebook."""
    n_blocks = data.shape[0] // step
    trimmed = data[: n_blocks * step]

    if data.ndim == 2:
        grouped = trimmed.reshape(n_blocks, step, data.shape[1])
    elif data.ndim == 3:
        grouped = trimmed.reshape(n_blocks, step, data.shape[1], data.shape[2])
    else:
        raise ValueError(f"Unsupported data ndim: {data.ndim}")

    return grouped.mean(axis=1)


def build_clara_matrix(
    data_avg: np.ndarray,
    low_freq_index: int,
    high_freq_index: int,
    use_directional_spectra: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the matrix clustered by CLARA.

    If use_directional_spectra=True:
        data_avg has shape (N, F, D), and we flatten F x D.
    Otherwise:
        the directional dimension is summed first, giving a frequency marginal.
    """
    data_sel = np.asarray(data_avg[:, low_freq_index:high_freq_index, :], dtype=float)

    if use_directional_spectra:
        x_tensor = data_sel
        x_clara = x_tensor.reshape(x_tensor.shape[0], -1)
    else:
        x_tensor = data_sel.sum(axis=2)
        x_clara = x_tensor.copy()

    x_clara = np.nan_to_num(x_clara, nan=0.0, posinf=0.0, neginf=0.0)
    x_clara[x_clara < 0.0] = 0.0

    return x_clara, x_tensor


def prepare_year(
    year: int,
    cfg: ClaraConfig,
) -> Dict[str, object]:
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

    data_direct_filtered, data_no_direct_filtered, chosen_indexes = filter_by_hs(
        data_direct=data_direct,
        data_no_direct=data_no_direct,
        freq=freq,
        hs_min=cfg.hs_min,
        hs_max=cfg.hs_max,
    )

    data_avg = block_average(data_direct_filtered, cfg.block_step)
    nondir_avg = block_average(data_no_direct_filtered, cfg.block_step)

    x_clara, x_tensor = build_clara_matrix(
        data_avg=data_avg,
        low_freq_index=cfg.low_freq_index,
        high_freq_index=cfg.high_freq_index,
        use_directional_spectra=cfg.use_directional_spectra,
    )

    ds_subset = ds.isel(time=chosen_indexes).coarsen(time=cfg.block_step, boundary="trim").mean()

    return {
        "ds": ds,
        "ds_subset": ds_subset,
        "data_direct_filtered": data_direct_filtered,
        "data_no_direct_filtered": data_no_direct_filtered,
        "data_avg": data_avg,
        "nondir_avg": nondir_avg,
        "x_clara": x_clara,
        "x_tensor": x_tensor,
        "chosen_indexes": chosen_indexes,
        "freq": freq,
        "freq_used": freq[cfg.low_freq_index:cfg.high_freq_index],
        "directions": directions,
    }


# ---------------------------------------------------------------------
# CLARA implementation
# ---------------------------------------------------------------------

def pam_build_initialisation(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Greedy BUILD initialisation for PAM on a precomputed distance matrix.
    """
    n = distance_matrix.shape[0]
    medoids = [int(np.argmin(distance_matrix.sum(axis=1)))]

    while len(medoids) < k:
        current = distance_matrix[:, medoids].min(axis=1)
        best_gain = -np.inf
        best_candidate = None

        medoid_set = set(medoids)
        for candidate in range(n):
            if candidate in medoid_set:
                continue

            new_cost = np.minimum(current, distance_matrix[:, candidate]).sum()
            gain = current.sum() - new_cost

            if gain > best_gain:
                best_gain = gain
                best_candidate = candidate

        medoids.append(int(best_candidate))

    return np.array(medoids, dtype=int)


def pam_improve(
    distance_matrix: np.ndarray,
    medoids: np.ndarray,
    max_iter: int = 20,
    max_swap_candidates: int = 8000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    PAM swap improvement on a subsample.

    For speed, if the number of possible swaps is too large, a random
    subset of swaps is evaluated. This keeps the CLARA scan practical.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = distance_matrix.shape[0]
    medoids = np.array(medoids, dtype=int)

    for _ in range(max_iter):
        current_cost = distance_matrix[:, medoids].min(axis=1).sum()
        best_cost = current_cost
        best_swap = None

        medoid_set = set(medoids.tolist())
        non_medoids = np.array([i for i in range(n) if i not in medoid_set], dtype=int)

        swaps = [(mi, h) for mi in range(len(medoids)) for h in non_medoids]

        if len(swaps) > max_swap_candidates:
            chosen = rng.choice(len(swaps), size=max_swap_candidates, replace=False)
            swaps = [swaps[i] for i in chosen]

        for mi, h in swaps:
            trial = medoids.copy()
            trial[mi] = h
            cost = distance_matrix[:, trial].min(axis=1).sum()

            if cost < best_cost:
                best_cost = cost
                best_swap = (mi, h)

        if best_swap is None:
            break

        medoids[best_swap[0]] = best_swap[1]

    return medoids


def clara_manhattan(
    x: np.ndarray,
    k: int,
    n_sampling: int,
    sample_size: int,
    random_state: int,
) -> Dict[str, np.ndarray]:
    """
    CLARA with Manhattan distance and no standardisation.

    Returns labels for the full dataset, medoid indices in the full dataset,
    point-to-medoid distances, and total Manhattan assignment cost.
    """
    rng = np.random.default_rng(random_state)
    n = x.shape[0]
    sample_size = int(min(max(sample_size, 40 + 2 * k), n))

    best = None

    for s in range(n_sampling):
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        xs = x[sample_idx]

        d_sub = pairwise_distances(xs, metric="manhattan")

        medoids_pos = pam_build_initialisation(d_sub, k)
        medoids_pos = pam_improve(
            d_sub,
            medoids_pos,
            max_iter=20,
            rng=np.random.default_rng(random_state + 1000 * s + k),
        )

        medoid_idx = sample_idx[medoids_pos]

        d_all = pairwise_distances(x, x[medoid_idx], metric="manhattan")
        labels = d_all.argmin(axis=1)
        distances = d_all.min(axis=1)
        cost = distances.sum()

        if best is None or cost < best["cost"]:
            best = {
                "labels": labels,
                "medoid_indices": medoid_idx,
                "distances": distances,
                "cost": cost,
            }

    return best


def upper_tercile_representatives(x: np.ndarray, labels: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hamilton-style representative spectrum:
    mean of the upper one-third values in each spectral bin.
    Also returns the median spectrum for comparison.
    """
    representatives = []
    medians = []

    for c in range(k):
        xc = x[labels == c]
        if xc.shape[0] == 0:
            representatives.append(np.full(x.shape[1], np.nan))
            medians.append(np.full(x.shape[1], np.nan))
            continue

        n_top = max(1, int(np.ceil(xc.shape[0] / 3)))
        sorted_vals = np.sort(xc, axis=0)

        representatives.append(sorted_vals[-n_top:].mean(axis=0))
        medians.append(np.median(xc, axis=0))

    return np.vstack(representatives), np.vstack(medians)


def medoid_separation_ratio(x: np.ndarray, labels: np.ndarray, medoid_indices: np.ndarray) -> float:
    """
    A simple distinctness diagnostic:
        minimum medoid-to-medoid distance / median within-cluster medoid distance.
    """
    k = len(medoid_indices)
    if k <= 1:
        return float("nan")

    d_med = pairwise_distances(x[medoid_indices], metric="manhattan")
    d_med[d_med == 0.0] = np.nan
    min_medoid_sep = np.nanmin(d_med)

    within = []
    for c in range(k):
        xc = x[labels == c]
        if xc.shape[0] == 0:
            continue
        d = pairwise_distances(xc, x[medoid_indices[c]][None, :], metric="manhattan")
        within.append(float(np.median(d)))

    median_within = float(np.nanmedian(within))
    return min_medoid_sep / (median_within + 1e-12)


def scan_k_with_clara(x: np.ndarray, cfg: ClaraConfig) -> Tuple[pd.DataFrame, Dict[int, Dict[str, object]]]:
    """Run CLARA for K in [k_min, k_max] and compute Hamilton-style diagnostics."""
    rows = []
    models = {}

    for k in range(cfg.k_min, cfg.k_max + 1):
        print(f"Running CLARA for K={k}...")

        model = clara_manhattan(
            x,
            k=k,
            n_sampling=cfg.clara_samples,
            sample_size=cfg.clara_sample_size,
            random_state=cfg.random_state + k,
        )

        labels = model["labels"]
        medoids = model["medoid_indices"]
        cluster_sizes = np.bincount(labels, minlength=k)

        if len(np.unique(labels)) > 1 and x.shape[0] > 2:
            sil = silhouette_score(
                x,
                labels,
                metric="manhattan",
                sample_size=min(cfg.silhouette_sample_size, x.shape[0]),
                random_state=cfg.random_state,
            )
        else:
            sil = np.nan

        sep_ratio = medoid_separation_ratio(x, labels, medoids)
        reps, medians = upper_tercile_representatives(x, labels, k)

        model.update(
            {
                "cluster_sizes": cluster_sizes,
                "silhouette": sil,
                "medoid_separation_ratio": sep_ratio,
                "representatives": reps,
                "medians": medians,
            }
        )
        models[k] = model

        rows.append(
            {
                "K": k,
                "total_manhattan_cost": model["cost"],
                "mean_manhattan_cost": model["cost"] / x.shape[0],
                "silhouette": sil,
                "min_cluster_size": int(cluster_sizes.min()),
                "max_cluster_size": int(cluster_sizes.max()),
                "n_singletons": int(np.sum(cluster_sizes == 1)),
                "medoid_separation_ratio": sep_ratio,
            }
        )

    results = pd.DataFrame(rows)
    results["rel_cost_improvement"] = (
        -results["mean_manhattan_cost"].diff()
        / results["mean_manhattan_cost"].shift(1)
    )

    return results, models


def choose_hamilton_k(results: pd.DataFrame, cfg: ClaraConfig) -> Tuple[int, int, int]:
    """
    Automatic guide:
      - k_silhouette: maximum silhouette coefficient.
      - k_elbow: first K at/after k_silhouette where relative cost improvement
        is small for cfg.elbow_patience consecutive steps.
      - k_hamilton: max(k_silhouette, k_elbow), unless cfg.selected_k is given.

    This is not a substitute for visual inspection; it implements a reproducible
    version of Hamilton's scan-and-inspect approach.
    """
    k_silhouette = int(results.loc[results["silhouette"].idxmax(), "K"])

    small_gain_count = 0
    k_elbow = None

    for _, row in results.iterrows():
        k = int(row["K"])
        if k < k_silhouette:
            continue

        gain = row["rel_cost_improvement"]
        if np.isfinite(gain) and gain < cfg.min_relative_improvement:
            small_gain_count += 1
        else:
            small_gain_count = 0

        if small_gain_count >= cfg.elbow_patience:
            k_elbow = k - cfg.elbow_patience + 1
            break

    if k_elbow is None:
        k_elbow = k_silhouette

    k_hamilton = max(k_silhouette, k_elbow)

    if cfg.selected_k is not None:
        k_hamilton = int(cfg.selected_k)

    return k_hamilton, k_silhouette, int(k_elbow)


# ---------------------------------------------------------------------
# Outputs and plots
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


def make_label_dataframe(ds_labeled: xr.Dataset) -> pd.DataFrame:
    """Create a time/cluster dataframe for plotting and export."""
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(ds_labeled.time.values),
            "cluster": ds_labeled.cluster_label.values.astype(int),
        }
    )
    df["year"] = df["time"].dt.year
    df["day"] = df["time"].dt.dayofyear
    df["month"] = df["time"].dt.month
    df["date"] = df["time"].dt.date
    return df


def plot_k_diagnostics(results: pd.DataFrame, k_selected: int, k_silhouette: int, output_path: Path) -> None:
    """Save diagnostics for K selection."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(results["K"], results["silhouette"], marker="o")
    axes[0].axvline(k_silhouette, linestyle="--")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_title("Silhouette guide")

    axes[1].plot(results["K"], results["mean_manhattan_cost"], marker="o")
    axes[1].axvline(k_selected, linestyle="--")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Mean Manhattan cost")
    axes[1].set_title("Within-cluster cost")

    axes[2].plot(results["K"], results["medoid_separation_ratio"], marker="o")
    axes[2].axhline(1.0, linestyle="--")
    axes[2].axvline(k_selected, linestyle="--")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Medoid separation / within spread")
    axes[2].set_title("Representative distinctness")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def reshape_flat_representatives(
    flat: np.ndarray,
    x_tensor: np.ndarray,
    use_directional_spectra: bool,
) -> np.ndarray:
    """Reshape flattened representatives to the tensor shape used for plotting."""
    k = flat.shape[0]
    if use_directional_spectra:
        f = x_tensor.shape[1]
        d = x_tensor.shape[2]
        return flat.reshape(k, f, d)
    return flat.reshape(k, x_tensor.shape[1])


def plot_clara_overplots(
    x_tensor: np.ndarray,
    freq_used: np.ndarray,
    labels: np.ndarray,
    medoid_indices: np.ndarray,
    representatives_flat: np.ndarray,
    medians_flat: np.ndarray,
    use_directional_spectra: bool,
    output_path: Path,
    max_curves_per_cluster: int,
    random_state: int,
) -> None:
    """
    Save Hamilton-style cluster overplots.

    For directional spectra, plots the frequency marginal for readability.
    """
    rng = np.random.default_rng(random_state)
    k = len(medoid_indices)

    reps = reshape_flat_representatives(representatives_flat, x_tensor, use_directional_spectra)
    meds = reshape_flat_representatives(medians_flat, x_tensor, use_directional_spectra)

    if use_directional_spectra:
        spectra_plot = x_tensor.sum(axis=2)
        reps_plot = reps.sum(axis=2)
        meds_plot = meds.sum(axis=2)
        medoids_plot = x_tensor[medoid_indices].sum(axis=2)
    else:
        spectra_plot = x_tensor
        reps_plot = reps
        meds_plot = meds
        medoids_plot = x_tensor[medoid_indices]

    ncols = 4
    nrows = int(math.ceil(k / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.0 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for c in range(k):
        ax = axes[c]
        idx = np.where(labels == c)[0]

        if idx.size > max_curves_per_cluster:
            idx_plot = rng.choice(idx, size=max_curves_per_cluster, replace=False)
        else:
            idx_plot = idx

        ax.plot(freq_used, spectra_plot[idx_plot].T, color="0.75", alpha=0.35, linewidth=0.8)
        ax.plot(freq_used, medoids_plot[c], color="black", linewidth=1.0, label="medoid")
        ax.plot(freq_used, meds_plot[c], color="white", linewidth=1.8, label="median")
        ax.plot(freq_used, reps_plot[c], color="red", linewidth=2.0, label="upper-tercile rep.")

        ax.set_title(f"Cluster {c} (n={idx.size})", fontsize=9)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Spectral density")
        ax.grid(alpha=0.2)

    for ax in axes[k:]:
        ax.axis("off")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3)
    fig.suptitle(
        "CLARA clusters: grey=spectra, black=medoid, red=upper-tercile representative",
        y=1.01,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

from matplotlib.lines import Line2D

def plot_clara_overplots_with_2018_assignments(
    x2017_tensor,
    x2018_tensor,
    freq_used,
    labels2017,
    labels2018,
    medoid_indices,
    representatives_flat,
    use_directional_spectra,
    output_path,
    max_curves_per_cluster=80,
    random_state=7,
    show_legend=True,
):
    """
    Plot 2017 spectra used to infer CLARA clusters together with
    2018 spectra assigned to the learned 2017 medoids.

    Grey curves: 2017 spectra used for fitting.
    Blue curves: 2018 spectra assigned to the 2017 medoids.
    Black curve: 2017 medoid.
    Red curve: upper-tercile representative.
    """
    rng = np.random.default_rng(random_state)
    k = len(medoid_indices)

    reps = reshape_flat_representatives(
        representatives_flat,
        x2017_tensor,
        use_directional_spectra,
    )

    if use_directional_spectra:
        spectra2017_plot = x2017_tensor.sum(axis=2)
        spectra2018_plot = x2018_tensor.sum(axis=2)
        reps_plot = reps.sum(axis=2)
        medoids_plot = x2017_tensor[medoid_indices].sum(axis=2)
    else:
        spectra2017_plot = x2017_tensor
        spectra2018_plot = x2018_tensor
        reps_plot = reps
        medoids_plot = x2017_tensor[medoid_indices]

    ncols = 4
    nrows = int(np.ceil(k / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.0 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for c in range(k):
        ax = axes[c]

        idx17_full = np.where(labels2017 == c)[0]
        idx18_full = np.where(labels2018 == c)[0]

        idx17 = idx17_full.copy()
        idx18 = idx18_full.copy()

        if idx17.size > max_curves_per_cluster:
            idx17 = rng.choice(idx17, size=max_curves_per_cluster, replace=False)
        if idx18.size > max_curves_per_cluster:
            idx18 = rng.choice(idx18, size=max_curves_per_cluster, replace=False)

        if idx17.size > 0:
            ax.plot(
                freq_used,
                spectra2017_plot[idx17].T,
                color="0.75",
                alpha=0.25,
                linewidth=0.8,
                label="_nolegend_",
            )

        if idx18.size > 0:
            ax.plot(
                freq_used,
                spectra2018_plot[idx18].T,
                color="tab:blue",
                alpha=0.30,
                linewidth=0.8,
                label="_nolegend_",
            )

        ax.plot(
            freq_used,
            medoids_plot[c],
            color="black",
            linewidth=1.1,
            label="_nolegend_",
        )

        ax.plot(
            freq_used,
            reps_plot[c],
            color="red",
            linewidth=2.0,
            label="_nolegend_",
        )

        ax.set_title(
            f"Cluster {c} "
            f"(2017 n={idx17_full.size}, 2018 n={idx18_full.size})",
            fontsize=9,
        )
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Spectral density")
        ax.grid(alpha=0.2)

    for ax in axes[k:]:
        ax.axis("off")

    if show_legend:
        legend_handles = [
            Line2D([0], [0], color="0.75", linewidth=1.2, alpha=0.8,
                   label="2017 fitted spectra"),
            Line2D([0], [0], color="tab:blue", linewidth=1.2, alpha=0.8,
                   label="2018 assigned spectra"),
            Line2D([0], [0], color="black", linewidth=1.2,
                   label="2017 medoid"),
            Line2D([0], [0], color="red", linewidth=2.0,
                   label="upper-tercile representative"),
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=4,
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )

        fig.tight_layout(rect=(0, 0, 1, 0.965))
    else:
        fig.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_cluster_timeline(df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save a simple cluster-label timeline."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(df["time"], df["cluster"], s=10, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cluster")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_timeline_by_year(df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save a combined 2017/2018 cluster-label timeline with year-specific colours."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for year, label, alpha in [(2017, "2017", 0.65), (2018, "2018", 0.65)]:
        dfi = df[df["year"] == year]
        if dfi.empty:
            continue
        ax.scatter(
            dfi["time"],
            dfi["cluster"],
            s=10,
            alpha=alpha,
            label=label,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Cluster")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def assign_to_medoids(x_new: np.ndarray, x_train: np.ndarray, medoid_indices: np.ndarray) -> np.ndarray:
    """Assign new spectra to the nearest 2017 CLARA medoid."""
    d = pairwise_distances(x_new, x_train[medoid_indices], metric="manhattan")
    return d.argmin(axis=1)



def concat_labeled_years(ds2017_labeled: xr.Dataset, ds2018_labeled: xr.Dataset) -> xr.Dataset:
    """
    Concatenate the labelled 2017 and 2018 datasets along time.

    The two yearly files usually have the same spectral coordinates and metadata,
    but xarray can be strict with identical non-time coordinates. The settings
    below keep the time-varying data and avoid unnecessary conflicts in static
    metadata.
    """
    return xr.concat(
        [ds2017_labeled, ds2018_labeled],
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="override",
    )


def run_combined_2017_2018(cfg: ClaraConfig) -> None:
    """
    Fit CLARA directly on the concatenated 3-hour averaged spectra from 2017
    and 2018, then split the resulting labels back by year for export.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing 2017 data...")
    y2017 = prepare_year(2017, cfg)
    x2017 = y2017["x_clara"]
    x2017_tensor = y2017["x_tensor"]
    freq_used = y2017["freq_used"]

    print("Preparing 2018 data...")
    y2018 = prepare_year(2018, cfg)
    x2018 = y2018["x_clara"]
    x2018_tensor = y2018["x_tensor"]

    if x2017.shape[1] != x2018.shape[1]:
        raise ValueError(
            f"2017 and 2018 CLARA matrices have different feature counts: "
            f"{x2017.shape[1]} and {x2018.shape[1]}."
        )

    x_joint = np.concatenate([x2017, x2018], axis=0)
    x_joint_tensor = np.concatenate([x2017_tensor, x2018_tensor], axis=0)

    n2017 = x2017.shape[0]
    n2018 = x2018.shape[0]

    print(f"2017 CLARA matrix shape: {x2017.shape}")
    print(f"2018 CLARA matrix shape: {x2018.shape}")
    print(f"Joint 2017+2018 CLARA matrix shape: {x_joint.shape}")

    print("\nScanning K with CLARA on joint 2017+2018 spectra...")
    scan_results, clara_models = scan_k_with_clara(x_joint, cfg)

    k_selected, k_silhouette, k_elbow = choose_hamilton_k(scan_results, cfg)
    print("\nSuggested K values")
    print("------------------")
    print(f"Best K by silhouette coefficient: {k_silhouette}")
    print(f"Cost-elbow K after silhouette:     {k_elbow}")
    print(f"Selected Hamilton-style K:         {k_selected}")
    print(
        "Inspect the overplot grid before finalising K. If representatives are "
        "visually redundant, choose a smaller K or amalgamate similar clusters."
    )

    scan_results.to_csv(cfg.output_dir / "clara_k_scan_2017_2018_joint.csv", index=False)

    with open(cfg.output_dir / "clara_selection_summary_2017_2018_joint.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "k_silhouette": int(k_silhouette),
                "k_elbow": int(k_elbow),
                "k_selected": int(k_selected),
                "fit_combined_2017_2018": True,
                "n_2017": int(n2017),
                "n_2018": int(n2018),
                "n_joint": int(x_joint.shape[0]),
                "use_directional_spectra": bool(cfg.use_directional_spectra),
                "low_freq_index": int(cfg.low_freq_index),
                "high_freq_index": int(cfg.high_freq_index),
                "hs_min": float(cfg.hs_min),
                "hs_max": float(cfg.hs_max),
                "block_step": int(cfg.block_step),
            },
            f,
            indent=2,
        )

    selected_model = clara_models[k_selected]
    labels_joint = selected_model["labels"].astype(int)
    medoids_joint = selected_model["medoid_indices"].astype(int)

    labels2017 = labels_joint[:n2017]
    labels2018 = labels_joint[n2017:]

    ds2017_labeled = add_labels_to_subset(y2017["ds_subset"], labels2017)
    ds2018_labeled = add_labels_to_subset(y2018["ds_subset"], labels2018)
    ds_joint_labeled = concat_labeled_years(ds2017_labeled, ds2018_labeled)

    ds2017_labeled.to_netcdf(cfg.output_dir / "ds_clara_2017_joint_labels.nc")
    ds2018_labeled.to_netcdf(cfg.output_dir / "ds_clara_2018_joint_labels.nc")
    ds_joint_labeled.to_netcdf(cfg.output_dir / "ds_clara_2017_2018_joint.nc")

    ds_joint_means = make_cluster_mean_dataset(ds_joint_labeled)
    ds_joint_means.to_netcdf(cfg.output_dir / "ds_clara_cluster_means_2017_2018_joint.nc")

    df2017 = make_label_dataframe(ds2017_labeled)
    df2018 = make_label_dataframe(ds2018_labeled)
    df_joint = pd.concat([df2017, df2018], ignore_index=True).sort_values("time")

    df2017.to_csv(cfg.output_dir / "clara_labels_2017_joint.csv", index=False)
    df2018.to_csv(cfg.output_dir / "clara_labels_2018_joint.csv", index=False)
    df_joint.to_csv(cfg.output_dir / "clara_labels_2017_2018_joint.csv", index=False)

    np.savez(
        cfg.output_dir / "clara_model_2017_2018_joint.npz",
        labels_joint=labels_joint,
        labels2017=labels2017,
        labels2018=labels2018,
        medoid_indices=medoids_joint,
        medoid_spectra=x_joint[medoids_joint],
        representatives=selected_model["representatives"],
        medians=selected_model["medians"],
        cluster_sizes=selected_model["cluster_sizes"],
        freq_used=freq_used,
        directions=y2017["directions"],
        n2017=n2017,
        n2018=n2018,
    )

    plot_k_diagnostics(
        scan_results,
        k_selected=k_selected,
        k_silhouette=k_silhouette,
        output_path=cfg.output_dir / "clara_k_diagnostics_2017_2018_joint.png",
    )

    plot_clara_overplots(
        x_tensor=x_joint_tensor,
        freq_used=freq_used,
        labels=labels_joint,
        medoid_indices=medoids_joint,
        representatives_flat=selected_model["representatives"],
        medians_flat=selected_model["medians"],
        use_directional_spectra=cfg.use_directional_spectra,
        output_path=cfg.output_dir / f"clara_overplots_2017_2018_joint_K{k_selected}.png",
        max_curves_per_cluster=cfg.max_curves_per_cluster_plot,
        random_state=cfg.random_state,
    )

    plot_cluster_timeline_by_year(
        df_joint,
        output_path=cfg.output_dir / "clara_timeline_2017_2018_joint.png",
        title=f"Joint 2017+2018 CLARA cluster timeline, K={k_selected}",
    )

    assignment_counts = pd.DataFrame(
        {
            "cluster": np.arange(k_selected),
            "n_2017": np.bincount(labels2017, minlength=k_selected),
            "n_2018": np.bincount(labels2018, minlength=k_selected),
            "n_total": np.bincount(labels_joint, minlength=k_selected),
        }
    )
    assignment_counts.to_csv(cfg.output_dir / "clara_cluster_counts_2017_2018_joint.csv", index=False)

    print("\nDone.")
    print(f"Joint 2017+2018 outputs saved to: {cfg.output_dir.resolve()}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(cfg: ClaraConfig) -> None:
    if cfg.fit_combined_2017_2018:
        run_combined_2017_2018(cfg)
        return

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing 2017 data...")
    y2017 = prepare_year(2017, cfg)
    x2017 = y2017["x_clara"]
    x2017_tensor = y2017["x_tensor"]
    freq_used = y2017["freq_used"]

    print(f"2017 CLARA matrix shape: {x2017.shape}")

    print("\nScanning K with CLARA...")
    scan_results, clara_models = scan_k_with_clara(x2017, cfg)

    k_selected, k_silhouette, k_elbow = choose_hamilton_k(scan_results, cfg)
    print("\nSuggested K values")
    print("------------------")
    print(f"Best K by silhouette coefficient: {k_silhouette}")
    print(f"Cost-elbow K after silhouette:     {k_elbow}")
    print(f"Selected Hamilton-style K:         {k_selected}")
    print(
        "Inspect the overplot grid before finalising K. If representatives are "
        "visually redundant, choose a smaller K or amalgamate similar clusters."
    )

    scan_results.to_csv(cfg.output_dir / "clara_k_scan_2017.csv", index=False)

    with open(cfg.output_dir / "clara_selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "k_silhouette": int(k_silhouette),
                "k_elbow": int(k_elbow),
                "k_selected": int(k_selected),
                "use_directional_spectra": bool(cfg.use_directional_spectra),
                "low_freq_index": int(cfg.low_freq_index),
                "high_freq_index": int(cfg.high_freq_index),
                "hs_min": float(cfg.hs_min),
                "hs_max": float(cfg.hs_max),
                "block_step": int(cfg.block_step),
            },
            f,
            indent=2,
        )

    selected_model = clara_models[k_selected]
    labels2017 = selected_model["labels"].astype(int)
    medoids2017 = selected_model["medoid_indices"].astype(int)

    ds2017_labeled = add_labels_to_subset(y2017["ds_subset"], labels2017)
    ds2017_labeled.to_netcdf(cfg.output_dir / "ds_clara_2017.nc")

    ds2017_means = make_cluster_mean_dataset(ds2017_labeled)
    ds2017_means.to_netcdf(cfg.output_dir / "ds_clara_cluster_means_2017.nc")

    df2017 = make_label_dataframe(ds2017_labeled)
    df2017.to_csv(cfg.output_dir / "clara_labels_2017.csv", index=False)

    np.savez(
        cfg.output_dir / "clara_model_2017.npz",
        labels=labels2017,
        medoid_indices=medoids2017,
        medoid_spectra=x2017[medoids2017],
        representatives=selected_model["representatives"],
        medians=selected_model["medians"],
        cluster_sizes=selected_model["cluster_sizes"],
        freq_used=freq_used,
        directions=y2017["directions"],
    )

    plot_k_diagnostics(
        scan_results,
        k_selected=k_selected,
        k_silhouette=k_silhouette,
        output_path=cfg.output_dir / "clara_k_diagnostics_2017.png",
    )

    plot_clara_overplots(
        x_tensor=x2017_tensor,
        freq_used=freq_used,
        labels=labels2017,
        medoid_indices=medoids2017,
        representatives_flat=selected_model["representatives"],
        medians_flat=selected_model["medians"],
        use_directional_spectra=cfg.use_directional_spectra,
        output_path=cfg.output_dir / f"clara_overplots_2017_K{k_selected}.png",
        max_curves_per_cluster=cfg.max_curves_per_cluster_plot,
        random_state=cfg.random_state,
    )

    plot_cluster_timeline(
        df2017,
        output_path=cfg.output_dir / "clara_timeline_2017.png",
        title=f"2017 CLARA cluster timeline, K={k_selected}",
    )

    print("\nPreparing 2018 data...")
    y2018 = prepare_year(2018, cfg)
    x2018 = y2018["x_clara"]

    print(f"2018 CLARA assignment matrix shape: {x2018.shape}")

    labels2018 = assign_to_medoids(x2018, x2017, medoids2017)

    plot_clara_overplots_with_2018_assignments(
        x2017_tensor=x2017_tensor,
        x2018_tensor=y2018["x_tensor"],
        freq_used=freq_used,
        labels2017=labels2017,
        labels2018=labels2018,
        medoid_indices=medoids2017,
        representatives_flat=selected_model["representatives"],
        use_directional_spectra=cfg.use_directional_spectra,
        output_path=cfg.output_dir / f"clara_overplots_2017_with_2018_K{k_selected}.png",
        max_curves_per_cluster=cfg.max_curves_per_cluster_plot,
        random_state=cfg.random_state,
        show_legend=False,
    )

    ds2018_labeled = add_labels_to_subset(y2018["ds_subset"], labels2018)
    ds2018_labeled.to_netcdf(cfg.output_dir / "ds_clara_2018_assigned.nc")

    df2018 = make_label_dataframe(ds2018_labeled)
    df2018.to_csv(cfg.output_dir / "clara_labels_2018_assigned.csv", index=False)

    plot_cluster_timeline(
        df2018,
        output_path=cfg.output_dir / "clara_timeline_2018_assigned.png",
        title=f"2018 spectra assigned to 2017 CLARA medoids, K={k_selected}",
    )

    assignment_counts = pd.DataFrame(
        {
            "cluster": np.arange(k_selected),
            "n_2017": np.bincount(labels2017, minlength=k_selected),
            "n_2018_assigned": np.bincount(labels2018, minlength=k_selected),
        }
    )
    assignment_counts.to_csv(cfg.output_dir / "clara_cluster_counts_2017_2018.csv", index=False)

    print("\nDone.")
    print(f"Outputs saved to: {cfg.output_dir.resolve()}")


def parse_args() -> ClaraConfig:
    parser = argparse.ArgumentParser(description="CLARA clustering for NDBC wave spectra.")

    parser.add_argument("--data-dir", type=Path, default=Path("../data/NDBC"))
    parser.add_argument("--output-dir", type=Path, default=Path("./clara_wave_results_17_18_nondir"))

    parser.add_argument("--hs-min", type=float, default=0.5)
    parser.add_argument("--hs-max", type=float, default=2.5)
    parser.add_argument("--block-step", type=int, default=3)

    parser.add_argument("--low-freq-index", type=int, default=3)
    parser.add_argument("--high-freq-index", type=int, default=35)

    parser.add_argument(
        "--nondirectional-clara",
        action="store_true",
        help=(
            "Cluster frequency marginals instead of flattened directional spectra. "
            "Use this for a closer reproduction of Hamilton's original non-directional setting."
        ),
    )
    parser.add_argument(
        "--fit-combined-2017-2018",
        action="store_true",
        help=(
            "Fit CLARA jointly on the 3-hour averaged spectra from 2017 and 2018. "
            "If omitted, the script keeps the original workflow: fit on 2017 and "
            "assign 2018 spectra to the 2017 medoids."
        ),
    )

    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=100)
    parser.add_argument(
        "--selected-k",
        type=int,
        default=None,
        help="Force a selected K after inspection. If omitted, the script uses the automatic guide.",
    )

    parser.add_argument("--clara-samples", type=int, default=5)
    parser.add_argument("--clara-sample-size", type=int, default=333)
    parser.add_argument("--silhouette-sample-size", type=int, default=1200)
    parser.add_argument("--random-state", type=int, default=7)

    parser.add_argument("--min-relative-improvement", type=float, default=0.001)
    parser.add_argument("--elbow-patience", type=int, default=2)

    parser.add_argument("--max-curves-per-cluster-plot", type=int, default=1000)
    parser.add_argument("--default-depth-2018", type=float, default=33.0)

    args = parser.parse_args()

    return ClaraConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hs_min=args.hs_min,
        hs_max=args.hs_max,
        block_step=args.block_step,
        low_freq_index=args.low_freq_index,
        high_freq_index=args.high_freq_index,
        use_directional_spectra=not args.nondirectional_clara,
        fit_combined_2017_2018=args.fit_combined_2017_2018,
        k_min=args.k_min,
        k_max=args.k_max,
        selected_k=args.selected_k,
        clara_samples=args.clara_samples,
        clara_sample_size=args.clara_sample_size,
        silhouette_sample_size=args.silhouette_sample_size,
        random_state=args.random_state,
        min_relative_improvement=args.min_relative_improvement,
        elbow_patience=args.elbow_patience,
        max_curves_per_cluster_plot=args.max_curves_per_cluster_plot,
        default_depth_2018=args.default_depth_2018,
    )


if __name__ == "__main__":
    config = parse_args()
    run(config)