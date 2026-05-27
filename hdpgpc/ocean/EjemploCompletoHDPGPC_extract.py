#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract an ordered HDP-GPC wave-spectra transition subdataset.

This script reads a labelled HDP-GPC NetCDF file, such as

    ds_hdpgpc_2017_2018_combined.nc

and extracts an artificial ordered sequence formed by:

    1. n samples from cluster_from, preserving their original order;
    2. n samples from cluster_to, preserving their original order.

The intended use is to build a small sequence in which the morphology moves
from one wave-spectral regime to another, for example from cluster 2 to
cluster 4 when both are similar but differ mainly by directional rotation.

The output dataset uses a new dimension called `sample`, while preserving the
original timestamp and original position in the source file.

Example
-------
python extract_hdpgpc_rotation_transition.py \
    --input-nc ./hdpgpc_wave_results/ds_hdpgpc_2017_2018_combined.nc \
    --output-nc ./hdpgpc_wave_results/rotation_transition_c2_c4.nc \
    --output-csv ./hdpgpc_wave_results/rotation_transition_c2_c4_metadata.csv \
    --cluster-from 2 \
    --cluster-to 4 \
    --n-per-cluster 100

If you want the script to search for a pair of 100-sample windows whose
frequency spectra are similar but whose directional distributions are rotated,
add:

    --auto-match-windows

By default, "consecutive" means consecutive occurrences after filtering by
cluster label, not necessarily consecutive calendar-time samples. This usually
works better for wave-regime data, because cluster assignments often alternate.
Use --require-temporal-contiguity only if you need strict uninterrupted runs in
the original time series.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def _find_label_variable(ds: xr.Dataset, preferred: str = "cluster_label") -> str:
    """Find the cluster-label variable in a labelled xarray dataset."""
    if preferred in ds.variables:
        return preferred

    for candidate in ("cluster_label", "cluster", "label", "labels"):
        if candidate in ds.variables:
            return candidate

    raise ValueError(
        "Could not find a label variable. Expected one of "
        "`cluster_label`, `cluster`, `label`, or `labels`."
    )


def _validate_dataset(ds: xr.Dataset, label_var: str) -> None:
    """Validate the minimal structure needed by the extractor."""
    if "efth" not in ds.variables:
        raise ValueError("Input NetCDF must contain variable `efth`.")

    if label_var not in ds.variables:
        raise ValueError(f"Input NetCDF does not contain label variable `{label_var}`.")

    if "time" not in ds.dims and "time" not in ds.coords:
        raise ValueError("Input NetCDF must contain a `time` dimension or coordinate.")

    if "freq" not in ds.dims and "freq" not in ds.coords:
        raise ValueError("Input NetCDF must contain a `freq` dimension or coordinate.")

    if "dir" not in ds.dims and "dir" not in ds.coords:
        raise ValueError("Input NetCDF must contain a `dir` dimension or coordinate.")


def select_order_preserving_cluster_samples(
    labels: np.ndarray,
    cluster_id: int,
    n_samples: int,
    occurrence_start: int = 0,
    require_temporal_contiguity: bool = False,
) -> np.ndarray:
    """
    Select n samples from a cluster while preserving original order.

    If require_temporal_contiguity is False, the function selects consecutive
    occurrences after filtering by the cluster label. For example, if cluster 2
    appears at original positions [4, 9, 13, 20, ...], the returned indices are
    the first n positions of that ordered list, unless occurrence_start is set.

    If require_temporal_contiguity is True, the function requires a strict run
    of n adjacent time indices all assigned to cluster_id.
    """
    labels = np.asarray(labels, dtype=int).reshape(-1)

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if occurrence_start < 0:
        raise ValueError("occurrence_start must be non-negative.")

    if not require_temporal_contiguity:
        idx = np.flatnonzero(labels == cluster_id)
        needed = occurrence_start + n_samples

        if idx.size < needed:
            raise ValueError(
                f"Cluster {cluster_id} has only {idx.size} samples, but "
                f"occurrence_start={occurrence_start} and n_samples={n_samples} "
                f"require at least {needed} samples."
            )

        return idx[occurrence_start:needed]

    mask = labels == cluster_id
    run_start: Optional[int] = None
    run_length = 0
    skipped_runs = 0

    for i, is_cluster in enumerate(mask):
        if is_cluster:
            if run_start is None:
                run_start = i
                run_length = 1
            else:
                run_length += 1

            if run_length >= n_samples and skipped_runs >= occurrence_start:
                return np.arange(run_start, run_start + n_samples)
        else:
            if run_length >= n_samples:
                skipped_runs += 1
            run_start = None
            run_length = 0

    if run_length >= n_samples and skipped_runs >= occurrence_start and run_start is not None:
        return np.arange(run_start, run_start + n_samples)

    raise ValueError(
        f"Could not find a strict temporally contiguous run of {n_samples} "
        f"samples for cluster {cluster_id}. Try removing "
        "`--require-temporal-contiguity`."
    )


def _normalise(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a non-negative vector normalised to sum one."""
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v[v < 0.0] = 0.0
    total = float(v.sum())
    if total <= eps:
        return np.zeros_like(v)
    return v / total


def _mean_frequency_signature(ds: xr.Dataset, idx: np.ndarray) -> np.ndarray:
    """Mean frequency-marginal spectrum of selected samples."""
    da = ds["efth"].isel(time=idx)
    freq_sig = da.sum(dim="dir").mean(dim="time").values
    return _normalise(freq_sig)


def _mean_direction_signature(ds: xr.Dataset, idx: np.ndarray) -> np.ndarray:
    """Mean direction-marginal spectrum of selected samples."""
    da = ds["efth"].isel(time=idx)
    dir_sig = da.sum(dim="freq").mean(dim="time").values
    return _normalise(dir_sig)


def circular_shift_distance(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    """
    Minimum L1 distance between two circular direction signatures after shifting b.

    Returns
    -------
    distance : float
        Minimum L1 distance.
    shift_bins : int
        Number of bins by which b must be rolled to best match a.
        Positive values correspond to np.roll(b, shift_bins).
    """
    a = _normalise(a)
    b = _normalise(b)

    if a.size != b.size:
        raise ValueError("Direction signatures must have the same length.")

    best_dist = np.inf
    best_shift = 0

    for shift in range(a.size):
        d = float(np.sum(np.abs(a - np.roll(b, shift))))
        if d < best_dist:
            best_dist = d
            best_shift = shift

    return best_dist, best_shift


def dominant_direction_deg(ds: xr.Dataset, sample_dim: str = "sample") -> float:
    """Estimate the dominant direction of a selected dataset from integrated energy."""
    if sample_dim not in ds["efth"].dims:
        sample_dim = "time"

    dir_energy = ds["efth"].sum(dim="freq").mean(dim=sample_dim)
    directions = ds["dir"].values
    return float(directions[int(np.argmax(dir_energy.values))])


def circular_difference_deg(a: float, b: float) -> float:
    """Signed circular difference b - a, in degrees."""
    return float((b - a + 180.0) % 360.0 - 180.0)


def _windowed_occurrence_indices(
    labels: np.ndarray,
    cluster_id: int,
    n_samples: int,
    step: int,
) -> list[np.ndarray]:
    """Build candidate windows over consecutive cluster occurrences."""
    occurrence_idx = np.flatnonzero(labels == cluster_id)
    if occurrence_idx.size < n_samples:
        return []

    step = max(1, int(step))
    return [
        occurrence_idx[start:start + n_samples]
        for start in range(0, occurrence_idx.size - n_samples + 1, step)
    ]


def auto_match_cluster_windows(
    ds: xr.Dataset,
    labels: np.ndarray,
    cluster_from: int,
    cluster_to: int,
    n_samples: int,
    window_step: int = 25,
    frequency_weight: float = 1.0,
    rotation_weight: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Choose two order-preserving cluster windows that are frequency-similar and
    directionally rotation-related.

    The score is

        frequency_weight * L1(freq_from, freq_to)
        +
        rotation_weight * circular_L1(dir_from, dir_to)

    where circular_L1 is computed after allowing a circular shift of the second
    directional signature. Lower is better.
    """
    labels = np.asarray(labels, dtype=int).reshape(-1)

    from_windows = _windowed_occurrence_indices(labels, cluster_from, n_samples, window_step)
    to_windows = _windowed_occurrence_indices(labels, cluster_to, n_samples, window_step)

    if not from_windows:
        raise ValueError(f"Cluster {cluster_from} does not have {n_samples} samples.")
    if not to_windows:
        raise ValueError(f"Cluster {cluster_to} does not have {n_samples} samples.")

    best_score = np.inf
    best_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None
    best_info: dict = {}

    from_signatures = [
        (idx, _mean_frequency_signature(ds, idx), _mean_direction_signature(ds, idx))
        for idx in from_windows
    ]
    to_signatures = [
        (idx, _mean_frequency_signature(ds, idx), _mean_direction_signature(ds, idx))
        for idx in to_windows
    ]

    for idx_from, freq_from, dir_from in from_signatures:
        for idx_to, freq_to, dir_to in to_signatures:
            freq_dist = float(np.sum(np.abs(freq_from - freq_to)))
            dir_dist, shift_bins = circular_shift_distance(dir_from, dir_to)
            score = frequency_weight * freq_dist + rotation_weight * dir_dist

            if score < best_score:
                best_score = score
                best_pair = (idx_from, idx_to)
                best_info = {
                    "score": float(score),
                    "frequency_l1_distance": float(freq_dist),
                    "direction_circular_l1_distance": float(dir_dist),
                    "best_shift_bins": int(shift_bins),
                    "n_direction_bins": int(dir_from.size),
                }

    if best_pair is None:
        raise RuntimeError("No valid pair of windows found.")

    idx_from, idx_to = best_pair

    if "dir" in ds.coords:
        directions = np.asarray(ds["dir"].values, dtype=float)
        if directions.size > 1:
            diffs = np.diff(np.r_[directions, directions[0] + 360.0])
            spacing = float(np.nanmedian(np.abs(diffs)))
            best_info["best_shift_degrees"] = float(best_info["best_shift_bins"] * spacing)

    return idx_from, idx_to, best_info


def build_transition_dataset(
    ds_labeled: xr.Dataset,
    label_var: str,
    cluster_from: int,
    cluster_to: int,
    n_per_cluster: int,
    start_from: int = 0,
    start_to: int = 0,
    require_temporal_contiguity: bool = False,
    auto_match_windows: bool = False,
    auto_window_step: int = 25,
) -> xr.Dataset:
    """
    Build the ordered transition dataset.

    Output order:
        sample 0 ... n-1: cluster_from
        sample n ... 2n-1: cluster_to
    """
    ds = ds_labeled.sortby("time")
    labels = ds[label_var].values.astype(int).reshape(-1)

    if auto_match_windows:
        idx_from, idx_to, match_info = auto_match_cluster_windows(
            ds=ds,
            labels=labels,
            cluster_from=cluster_from,
            cluster_to=cluster_to,
            n_samples=n_per_cluster,
            window_step=auto_window_step,
        )
    else:
        idx_from = select_order_preserving_cluster_samples(
            labels=labels,
            cluster_id=cluster_from,
            n_samples=n_per_cluster,
            occurrence_start=start_from,
            require_temporal_contiguity=require_temporal_contiguity,
        )
        idx_to = select_order_preserving_cluster_samples(
            labels=labels,
            cluster_id=cluster_to,
            n_samples=n_per_cluster,
            occurrence_start=start_to,
            require_temporal_contiguity=require_temporal_contiguity,
        )
        match_info = {}

    selected_idx = np.concatenate([idx_from, idx_to])
    ds_seq = ds.isel(time=selected_idx).copy()

    original_position = selected_idx.astype(int)

    ds_seq = ds_seq.assign_coords(sample=("time", np.arange(selected_idx.size)))
    ds_seq = ds_seq.swap_dims({"time": "sample"})
    ds_seq = ds_seq.rename({"time": "original_time"})

    source_cluster = np.concatenate([
        np.full(n_per_cluster, cluster_from, dtype=int),
        np.full(n_per_cluster, cluster_to, dtype=int),
    ])
    transition_phase = np.concatenate([
        np.zeros(n_per_cluster, dtype=int),
        np.ones(n_per_cluster, dtype=int),
    ])

    ds_seq["original_position"] = ("sample", original_position)
    ds_seq["source_cluster"] = ("sample", source_cluster)
    ds_seq["transition_phase"] = ("sample", transition_phase)

    if label_var != "cluster_label" and "cluster_label" not in ds_seq.variables:
        ds_seq = ds_seq.rename({label_var: "cluster_label"})

    ds_from = ds_seq.isel(sample=slice(0, n_per_cluster))
    ds_to = ds_seq.isel(sample=slice(n_per_cluster, 2 * n_per_cluster))

    dir_from = dominant_direction_deg(ds_from)
    dir_to = dominant_direction_deg(ds_to)
    rotation = circular_difference_deg(dir_from, dir_to)

    ds_seq.attrs.update({
        "description": (
            "Order-preserving HDP-GPC wave-spectra transition subdataset. "
            f"Samples 0-{n_per_cluster - 1} come from cluster {cluster_from}; "
            f"samples {n_per_cluster}-{2 * n_per_cluster - 1} come from "
            f"cluster {cluster_to}."
        ),
        "cluster_from": int(cluster_from),
        "cluster_to": int(cluster_to),
        "n_per_cluster": int(n_per_cluster),
        "selection_mode": (
            "auto_match_windows"
            if auto_match_windows
            else (
                "strict_temporal_contiguity"
                if require_temporal_contiguity
                else "consecutive_cluster_occurrences"
            )
        ),
        "dominant_direction_from_deg": float(dir_from),
        "dominant_direction_to_deg": float(dir_to),
        "estimated_rotation_deg": float(rotation),
    })

    for key, value in match_info.items():
        ds_seq.attrs[f"auto_match_{key}"] = value

    print("")
    print("Selected transition dataset")
    print("---------------------------")
    print(f"Cluster from: {cluster_from}")
    print(f"Cluster to:   {cluster_to}")
    print(f"Samples each: {n_per_cluster}")
    print(f"Total samples: {2 * n_per_cluster}")
    print(f"Dominant direction from: {dir_from:.1f} deg")
    print(f"Dominant direction to:   {dir_to:.1f} deg")
    print(f"Estimated rotation:      {rotation:+.1f} deg")

    if match_info:
        print("")
        print("Auto-matching diagnostics")
        print("-------------------------")
        for key, value in match_info.items():
            print(f"{key}: {value}")

    return ds_seq



def _axis_edges_from_centers(values: np.ndarray) -> np.ndarray:
    """Build plotting edges from regularly or irregularly spaced bin centres."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Need at least two coordinate values to build bin edges.")

    mid = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (mid[0] - values[0])
    last = values[-1] + (values[-1] - mid[-1])
    return np.r_[first, mid, last]


def _direction_edges_deg(directions: np.ndarray) -> np.ndarray:
    """Build angular bin edges for directions expressed in degrees."""
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 1 or directions.size < 2:
        raise ValueError("Need at least two directions to build angular edges.")

    # Most NDBC reconstructions use a regular 0, 10, ..., 350 grid.  This also
    # works for shifted regular grids.
    spacing = float(np.nanmedian(np.diff(np.sort(directions))))
    return np.r_[directions - 0.5 * spacing, directions[-1] + 0.5 * spacing]


def _append_circular_point(x_deg: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Append the first point at +360 degrees for clean circular line plots."""
    x_deg = np.asarray(x_deg, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.r_[x_deg, x_deg[0] + 360.0], np.r_[y, y[0]]


def _phase_subset(ds_seq: xr.Dataset, phase: int) -> xr.Dataset:
    """Return selected samples belonging to one transition phase."""
    return ds_seq.where(ds_seq["transition_phase"] == phase, drop=True)


def _mean_efth_for_phase(ds_seq: xr.Dataset, phase: int) -> xr.DataArray:
    """Mean 2D spectrum for one transition phase."""
    ds_phase = _phase_subset(ds_seq, phase)
    if ds_phase.sizes.get("sample", 0) == 0:
        raise ValueError(f"No samples found for transition_phase={phase}.")
    return ds_phase["efth"].mean(dim="sample").transpose("freq", "dir")


def _plot_mean_spectrum_polar(ax, mean_efth: xr.DataArray, title: str) -> None:
    """Plot a mean directional spectrum on a polar axis without wavespectra."""
    freq = np.asarray(mean_efth["freq"].values, dtype=float)
    directions = np.asarray(mean_efth["dir"].values, dtype=float)
    z = np.asarray(mean_efth.values, dtype=float)

    theta_edges = np.deg2rad(_direction_edges_deg(directions))
    freq_edges = _axis_edges_from_centers(freq)
    theta_grid, freq_grid = np.meshgrid(theta_edges, freq_edges)

    mesh = ax.pcolormesh(theta_grid, freq_grid, z, shading="auto")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids([0, 90, 180, 270], labels=["N", "E", "S", "W"])
    ax.set_title(title, fontsize=10, fontweight="bold")
    return mesh


def _sample_dominant_direction_series(ds_seq: xr.Dataset) -> np.ndarray:
    """Dominant direction per sample, obtained from the frequency-integrated spectrum."""
    dir_energy = ds_seq["efth"].sum(dim="freq")
    directions = np.asarray(ds_seq["dir"].values, dtype=float)
    argmax = np.argmax(np.asarray(dir_energy.values), axis=1)
    return directions[argmax]


def _sample_direction_resultant_series(ds_seq: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Circular mean direction and resultant length per sample.

    The resultant length is a rough concentration measure: values close to 1
    indicate a concentrated directional distribution; values close to 0 indicate
    broad or multi-directional energy.
    """
    dir_energy = np.asarray(ds_seq["efth"].sum(dim="freq").values, dtype=float)
    directions = np.deg2rad(np.asarray(ds_seq["dir"].values, dtype=float))

    weights = np.nan_to_num(dir_energy, nan=0.0, posinf=0.0, neginf=0.0)
    weights[weights < 0.0] = 0.0
    total = weights.sum(axis=1)
    total_safe = np.where(total > 0.0, total, 1.0)

    c = (weights * np.cos(directions)[None, :]).sum(axis=1) / total_safe
    s = (weights * np.sin(directions)[None, :]).sum(axis=1) / total_safe

    mean_dir = (np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0
    resultant = np.sqrt(c ** 2 + s ** 2)
    mean_dir[total <= 0.0] = np.nan
    resultant[total <= 0.0] = np.nan
    return mean_dir, resultant


def _rolling_nanmean(x: np.ndarray, window: int) -> np.ndarray:
    """Simple centred rolling mean that ignores NaNs."""
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x

    out = np.full_like(x, np.nan, dtype=float)
    half = window // 2
    for i in range(x.size):
        lo = max(0, i - half)
        hi = min(x.size, i + half + 1)
        out[i] = np.nanmean(x[lo:hi])
    return out


def plot_rotation_diagnostics(
    ds_seq: xr.Dataset,
    output_path: Path,
    rolling_window: int = 11,
) -> None:
    """
    Save a diagnostic figure to assess whether the selected subdataset is mainly
    affected by a directional rotation.

    The figure contains:
        1. mean 2D directional spectrum for the first cluster block;
        2. mean 2D directional spectrum for the second cluster block;
        3. direction-integrated frequency spectra for both blocks;
        4. frequency-integrated directional distributions, including the shifted
           second distribution that best aligns with the first;
        5. sample-wise dominant/mean direction along the synthetic sequence.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "sample" not in ds_seq.dims:
        raise ValueError("Diagnostic plot expects the extracted dataset with dimension `sample`.")

    cluster_from = int(ds_seq.attrs.get("cluster_from", 0))
    cluster_to = int(ds_seq.attrs.get("cluster_to", 1))
    n_per_cluster = int(ds_seq.attrs.get("n_per_cluster", ds_seq.sizes["sample"] // 2))

    ds_from = _phase_subset(ds_seq, 0)
    ds_to = _phase_subset(ds_seq, 1)

    mean_from = _mean_efth_for_phase(ds_seq, 0)
    mean_to = _mean_efth_for_phase(ds_seq, 1)

    freq = np.asarray(ds_seq["freq"].values, dtype=float)
    directions = np.asarray(ds_seq["dir"].values, dtype=float)

    # Integrated signatures.
    freq_from = np.asarray(ds_from["efth"].sum(dim="dir").mean(dim="sample").values, dtype=float)
    freq_to = np.asarray(ds_to["efth"].sum(dim="dir").mean(dim="sample").values, dtype=float)
    dir_from_raw = np.asarray(ds_from["efth"].sum(dim="freq").mean(dim="sample").values, dtype=float)
    dir_to_raw = np.asarray(ds_to["efth"].sum(dim="freq").mean(dim="sample").values, dtype=float)

    dir_from = _normalise(dir_from_raw)
    dir_to = _normalise(dir_to_raw)
    circular_dist, shift_bins = circular_shift_distance(dir_from, dir_to)
    dir_to_shifted = np.roll(dir_to, shift_bins)

    if directions.size > 1:
        spacing = float(np.nanmedian(np.diff(np.sort(directions))))
    else:
        spacing = 0.0
    shift_deg = shift_bins * spacing

    dom_dir = _sample_dominant_direction_series(ds_seq)
    mean_dir, resultant = _sample_direction_resultant_series(ds_seq)

    # Unwrap for visualising the evolution over the artificial sample index.
    dom_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(dom_dir)))
    mean_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(mean_dir)))
    mean_smooth = _rolling_nanmean(mean_unwrapped, rolling_window)

    rotation_attr = float(ds_seq.attrs.get("estimated_rotation_deg", np.nan))

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], wspace=0.35, hspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0], projection="polar")
    ax1 = fig.add_subplot(gs[0, 1], projection="polar")
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1:])

    mesh0 = _plot_mean_spectrum_polar(
        ax0,
        mean_from,
        f"Cluster {cluster_from}: mean 2D spectrum",
    )
    mesh1 = _plot_mean_spectrum_polar(
        ax1,
        mean_to,
        f"Cluster {cluster_to}: mean 2D spectrum",
    )
    fig.colorbar(mesh1, ax=[ax0, ax1], shrink=0.75, pad=0.08, label="Spectral density")

    # Directional distributions.
    x_dir, y_from = _append_circular_point(directions, dir_from)
    _, y_to = _append_circular_point(directions, dir_to)
    _, y_to_shifted = _append_circular_point(directions, dir_to_shifted)

    ax2.plot(x_dir, y_from, label=f"cluster {cluster_from}")
    ax2.plot(x_dir, y_to, label=f"cluster {cluster_to}")
    ax2.plot(
        x_dir,
        y_to_shifted,
        linestyle="--",
        label=f"cluster {cluster_to} shifted {shift_deg:.1f}°",
    )
    ax2.set_xlabel("Direction (deg)")
    ax2.set_ylabel("Normalised energy")
    ax2.set_title("Direction marginal and best circular shift")
    ax2.set_xlim(float(directions.min()), float(directions.min() + 360.0))
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, frameon=False)

    # Frequency marginals.
    ax3.plot(freq, freq_from, label=f"cluster {cluster_from}")
    ax3.plot(freq, freq_to, label=f"cluster {cluster_to}")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Spectral density")
    ax3.set_title("Frequency marginal")
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=8, frameon=False)

    # Sample-wise direction evolution.
    sample = np.asarray(ds_seq["sample"].values, dtype=int)
    ax4.scatter(sample, dom_unwrapped, s=12, alpha=0.45, label="dominant direction")
    ax4.plot(sample, mean_unwrapped, linewidth=1.0, alpha=0.65, label="circular mean direction")
    ax4.plot(sample, mean_smooth, linewidth=2.0, label=f"rolling mean ({rolling_window})")
    ax4.axvline(n_per_cluster - 0.5, linestyle="--", linewidth=1.2)
    ax4.set_xlabel("Synthetic sample index")
    ax4.set_ylabel("Unwrapped direction (deg)")
    ax4.set_title("Direction along the ordered subdataset")
    ax4.grid(alpha=0.25)
    ax4.legend(fontsize=8, frameon=False, loc="best")

    # # Add a small concentration axis on top of ax4 for context.
    # ax4b = ax4.twinx()
    # ax4b.plot(sample, resultant, linewidth=0.8, alpha=0.35, label="directional concentration")
    # ax4b.set_ylabel("Resultant length")
    # ax4b.set_ylim(0, 1.05)

    fig.suptitle(
        "Rotation diagnostic for extracted HDP-GPC wave-spectra sequence\n"
        f"estimated dominant-direction rotation = {rotation_attr:+.1f}°, "
        f"best distributional shift = {shift_deg:.1f}°, "
        f"circular L1 distance after shift = {circular_dist:.3f}",
        fontsize=12,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_metadata_csv(ds_seq: xr.Dataset, output_csv: Path) -> None:
    """Write a compact metadata table for the selected samples."""
    df = pd.DataFrame({
        "sample": ds_seq["sample"].values.astype(int),
        "original_time": pd.to_datetime(ds_seq["original_time"].values),
        "original_position": ds_seq["original_position"].values.astype(int),
        "cluster_label": ds_seq["cluster_label"].values.astype(int),
        "source_cluster": ds_seq["source_cluster"].values.astype(int),
        "transition_phase": ds_seq["transition_phase"].values.astype(int),
    })

    if "wspd" in ds_seq.variables:
        df["wspd"] = np.asarray(ds_seq["wspd"].values).reshape(-1)

    if "wdir" in ds_seq.variables:
        df["wdir"] = np.asarray(ds_seq["wdir"].values).reshape(-1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def write_numpy_array(ds_seq: xr.Dataset, output_npy: Path) -> None:
    """
    Save the selected spectra as a plain NumPy array with shape
    (num_samples, frequency, direction).
    """
    if "efth" not in ds_seq.variables:
        raise ValueError("Selected dataset must contain variable `efth`.")

    arr = (
        ds_seq["efth"]
        .transpose("sample", "freq", "dir")
        .values
        .astype(np.float32, copy=False)
    )

    output_npy = Path(output_npy)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, arr)
    print(f"Saved NumPy:   {output_npy} with shape {arr.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract an ordered transition subdataset from a labelled "
            "HDP-GPC wave-spectra NetCDF file."
        )
    )

    parser.add_argument(
        "--input-nc",
        type=Path,
        required=True,
        help=(
            "Labelled HDP-GPC NetCDF file, usually "
            "`ds_hdpgpc_2017_2018_combined.nc`."
        ),
    )
    parser.add_argument(
        "--output-nc",
        type=Path,
        required=True,
        help="Output NetCDF path for the selected transition dataset.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for selected-sample metadata.",
    )
    parser.add_argument(
        "--output-npy",
        type=Path,
        default=None,
        help=(
            "Optional output .npy path. Saves only the spectra as a NumPy "
            "array with shape (num_samples, frequency, direction). If omitted, "
            "a .npy file with the same stem as --output-nc is saved."
        ),
    )
    parser.add_argument(
        "--label-var",
        type=str,
        default="cluster_label",
        help="Name of the label variable in the NetCDF file.",
    )
    parser.add_argument("--cluster-from", type=int, default=2)
    parser.add_argument("--cluster-to", type=int, default=4)
    parser.add_argument("--n-per-cluster", type=int, default=100)
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help=(
            "Occurrence offset inside cluster-from. Ignored when "
            "--auto-match-windows is used."
        ),
    )
    parser.add_argument(
        "--start-to",
        type=int,
        default=0,
        help=(
            "Occurrence offset inside cluster-to. Ignored when "
            "--auto-match-windows is used."
        ),
    )
    parser.add_argument(
        "--require-temporal-contiguity",
        action="store_true",
        help=(
            "Require strict uninterrupted runs in the original time series. "
            "Without this flag, the script selects consecutive occurrences "
            "after filtering by cluster label."
        ),
    )
    parser.add_argument(
        "--auto-match-windows",
        action="store_true",
        help=(
            "Search for n-sample windows in the two clusters whose frequency "
            "marginals are similar and whose direction marginals match after "
            "a circular shift. This is useful when searching for a rotation-"
            "affected pair of regimes."
        ),
    )
    parser.add_argument(
        "--auto-window-step",
        type=int,
        default=25,
        help="Step size, in cluster occurrences, used by --auto-match-windows.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help=(
            "Optional diagnostic figure path. If omitted, a PNG with suffix "
            "`_rotation_diagnostic.png` is saved next to --output-nc."
        ),
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=11,
        help="Rolling window used to smooth the direction series in the diagnostic plot.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ds = xr.open_dataset(args.input_nc, decode_timedelta=True)
    label_var = _find_label_variable(ds, preferred=args.label_var)
    _validate_dataset(ds, label_var)

    ds_seq = build_transition_dataset(
        ds_labeled=ds,
        label_var=label_var,
        cluster_from=args.cluster_from,
        cluster_to=args.cluster_to,
        n_per_cluster=args.n_per_cluster,
        start_from=args.start_from,
        start_to=args.start_to,
        require_temporal_contiguity=args.require_temporal_contiguity,
        auto_match_windows=args.auto_match_windows,
        auto_window_step=args.auto_window_step,
    )

    args.output_nc.parent.mkdir(parents=True, exist_ok=True)
    ds_seq.to_netcdf(args.output_nc)
    print(f"\nSaved NetCDF: {args.output_nc}")

    output_npy = args.output_npy
    if output_npy is None:
        output_npy = args.output_nc.with_suffix(".npy")
    write_numpy_array(ds_seq, output_npy)

    output_plot = args.output_plot
    if output_plot is None:
        output_plot = args.output_nc.with_name(args.output_nc.stem + "_rotation_diagnostic.png")

    plot_rotation_diagnostics(
        ds_seq=ds_seq,
        output_path=output_plot,
        rolling_window=args.rolling_window,
    )
    print(f"Saved plot:    {output_plot}")

    if args.output_csv is not None:
        write_metadata_csv(ds_seq, args.output_csv)
        print(f"Saved CSV:    {args.output_csv}")


if __name__ == "__main__":
    main()