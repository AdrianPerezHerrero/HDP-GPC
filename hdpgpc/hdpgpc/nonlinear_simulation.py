# -*- coding: utf-8 -*-
"""
High-dimensional attenuation simulation with static trajectory clustering and actual HDP-GPC.

Motivation
----------
This is a fairer example than the previous 2D curve example because the sample
unit for the main comparison is the same:

    one complete high-dimensional trajectory -> one cluster label

The two true systems differ by an attenuation / damping factor. Each observation
is a D-dimensional vector, and each trajectory evolves as a decaying high-
dimensional signal plus nuisance amplitude and offset.

Comparison
----------
1. Cloud-level static DP-GMM on pooled D-dimensional observations.
   This is diagnostic only, not a fair trajectory benchmark.

2. Trajectory-level static DP-GMM on raw flattened trajectories.
   This is fair in sample unit, but can be dominated by amplitude and offsets.

3. Trajectory-level static DP-GMM on shape-normalized flattened trajectories.
   This is a stronger static baseline because it removes much of the nuisance
   amplitude/offset.

4. Actual HDP-GPC using hdpgpc.GPI_HDP on the same [N, T, D] trajectory tensor.

Run
---
From the parent folder containing your hdpgpc/ package:

    python simulate_attenuation_actual_hdpgpc.py

or:

    python simulate_attenuation_actual_hdpgpc.py --repo-root C:/Users/Adrian/Projects/YourProject

Useful options:

    python simulate_attenuation_actual_hdpgpc.py --n-outputs 16 --T 60
    python simulate_attenuation_actual_hdpgpc.py --slow-attenuation 0.985 --fast-attenuation 0.92
    python simulate_attenuation_actual_hdpgpc.py --offset-std 1.0 --amplitude-log-std 0.8
    python simulate_attenuation_actual_hdpgpc.py --skip-hdpgpc

Important interpretation
------------------------
If the shape-normalized static baseline performs well, that is not a problem:
it means the attenuation factor is already recoverable from carefully engineered
static trajectory features. HDP-GPC is then not being compared to an artificially
weak baseline.

The argument for HDP-GPC is stronger when:
    - there are temporal shifts/warping,
    - missing/irregular samples,
    - multi-output smooth functional structure,
    - hierarchical sharing across patients/records,
    - uncertainty in cluster representatives,
    - variable-length trajectories,
    - and dynamic regimes are not easily captured by Euclidean distances between
      flattened trajectories.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes", "y"}:
        return True
    if v.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def ensure_repo_on_path(repo_root):
    repo_root = Path(repo_root).expanduser().resolve()

    candidates = [
        repo_root,
        repo_root.parent,
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
    ]

    for c in candidates:
        if (c / "hdpgpc").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c

    raise FileNotFoundError(
        "Could not find an `hdpgpc` folder. Run from the parent folder "
        "containing `hdpgpc`, or pass --repo-root."
    )


def compact_labels(labels):
    labels = np.asarray(labels)
    unique = []
    for value in labels:
        if value not in unique:
            unique.append(value)
    mapping = {value: i for i, value in enumerate(unique)}
    return np.array([mapping[value] for value in labels], dtype=int)


def count_active_mixture_components(weights, threshold=0.01):
    return int(np.sum(np.asarray(weights) > threshold))


def majority_vote(values):
    values = np.asarray(values)
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def to_numpy_safe(value):
    try:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:
        pass

    try:
        return np.asarray(value)
    except Exception:
        return None


def try_extract_hdpgpc_labels(sw_gp, num_samples):
    """
    Preferred local source:

        self.resp_assigned.append(torch.argmax(resp, axis=1))

    so final labels should be:

        sw_gp.resp_assigned[-1]
    """
    if hasattr(sw_gp, "resp_assigned"):
        try:
            resp_assigned = getattr(sw_gp, "resp_assigned")

            if isinstance(resp_assigned, (list, tuple)) and len(resp_assigned) > 0:
                final_assigned = resp_assigned[-1]
            else:
                final_assigned = resp_assigned

            arr = to_numpy_safe(final_assigned)

            if arr is not None:
                arr = np.asarray(arr)

                if arr.ndim == 1 and arr.shape[0] == num_samples:
                    return compact_labels(arr.astype(int)), "sw_gp.resp_assigned[-1]"

                if arr.ndim == 2:
                    if arr.shape[-1] == num_samples:
                        return compact_labels(arr[-1, :].astype(int)), "sw_gp.resp_assigned[-1][-1, :]"
                    if arr.shape[0] == num_samples:
                        return compact_labels(arr[:, -1].astype(int)), "sw_gp.resp_assigned[-1][:, -1]"

        except Exception as exc:
            warnings.warn(f"Could not extract labels from sw_gp.resp_assigned: {repr(exc)}")

    candidate_attr_names = [
        "labels",
        "label",
        "z",
        "Z",
        "assignments",
        "assignment",
        "cluster_assignments",
        "model_assignments",
        "selected_models",
        "selected_model",
        "model_index",
        "model_indices",
        "main_model",
        "model_id",
        "model_ids",
        "d",
        "s",
    ]

    for name in candidate_attr_names:
        if hasattr(sw_gp, name):
            arr = to_numpy_safe(getattr(sw_gp, name))
            if arr is None:
                continue

            try:
                arr = np.asarray(arr)
                if arr.ndim == 1 and arr.shape[0] == num_samples:
                    return compact_labels(arr.astype(int)), f"sw_gp.{name}"
                if arr.ndim == 2 and arr.shape[0] == num_samples:
                    return compact_labels(arr[:, -1].astype(int)), f"sw_gp.{name}[:, -1]"
                if arr.ndim == 2 and arr.shape[-1] == num_samples:
                    return compact_labels(arr[-1, :].astype(int)), f"sw_gp.{name}[-1, :]"
            except Exception:
                pass

    return None, None


def dp_gmm_fit_predict(
    X,
    max_components,
    alpha,
    max_iter,
    reg_covar,
    covariance_type,
    seed,
):
    model = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type=covariance_type,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=alpha,
        max_iter=max_iter,
        random_state=seed,
        reg_covar=reg_covar,
    )
    labels = compact_labels(model.fit_predict(X))
    active = count_active_mixture_components(model.weights_)
    return labels, active, model


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------
def simulate_attenuation_systems(
    seed=42,
    n_traj_per_system=25,
    T=55,
    n_outputs=12,
    slow_attenuation=0.985,
    fast_attenuation=0.925,
    obs_noise=0.06,
    offset_std=0.75,
    amplitude_log_std=0.65,
    loading_jitter=0.12,
    time_shift_max=0,
):
    """
    Generate high-dimensional trajectories from two attenuation regimes.

    Each trajectory is:

        y_n(t, d) =
            offset_{n,d}
            + amplitude_n * loading_{n,d} * attenuation_k^(tau_n(t))
            + small shared transient
            + noise

    where k in {slow, fast} is the true dynamic system.

    Nuisance terms:
        - random per-trajectory amplitude,
        - random per-output offsets,
        - small random loading perturbations.

    These nuisance terms can make raw flattened static clustering focus on
    amplitude/offset rather than attenuation. Shape-normalized static clustering
    is included as a stronger and fairer baseline.
    """
    rng = np.random.default_rng(seed)

    # A deterministic high-dimensional loading pattern shared by both systems.
    d = np.arange(n_outputs)
    base_loading = (
        1.0
        + 0.35 * np.sin(2 * np.pi * d / max(n_outputs, 2))
        + 0.20 * np.cos(4 * np.pi * d / max(n_outputs, 2))
    )
    base_loading = base_loading / np.linalg.norm(base_loading)

    attenuations = [slow_attenuation, fast_attenuation]

    data = []
    labels = []
    metadata = []

    for system, att in enumerate(attenuations):
        for _ in range(n_traj_per_system):
            amplitude = np.exp(rng.normal(0.0, amplitude_log_std))
            offset = rng.normal(0.0, offset_std, size=n_outputs)
            loading = base_loading + rng.normal(0.0, loading_jitter, size=n_outputs)
            loading = loading / (np.linalg.norm(loading) + 1e-12)

            if time_shift_max > 0:
                shift = rng.integers(-time_shift_max, time_shift_max + 1)
            else:
                shift = 0

            t = np.arange(T)
            tau = np.clip(t + shift, 0, T - 1)

            decay = att ** tau

            # A small common transient prevents the data from being too purely
            # rank-one while preserving the attenuation mechanism.
            transient = 0.12 * np.exp(-0.5 * ((t - 0.28 * T) / (0.10 * T)) ** 2)
            transient_loading = np.sin(np.linspace(0, np.pi, n_outputs))

            Y = (
                offset[None, :]
                + amplitude * decay[:, None] * loading[None, :]
                + amplitude * transient[:, None] * transient_loading[None, :]
            )

            Y = Y + rng.normal(0.0, obs_noise, size=Y.shape)

            data.append(Y)
            labels.append(system)
            metadata.append(
                {
                    "system": system,
                    "attenuation": att,
                    "amplitude": amplitude,
                    "shift": int(shift),
                }
            )

    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    metadata = pd.DataFrame(metadata)

    return data, labels, metadata


def tail_baseline(data, tail_fraction=0.15):
    T = data.shape[1]
    n_tail = max(2, int(np.ceil(T * tail_fraction)))
    return data[:, -n_tail:, :].mean(axis=1, keepdims=True)


def shape_normalize_trajectories(data, tail_fraction=0.15, eps=1e-8):
    """
    Remove per-trajectory offsets and scale.

    This is a strong static preprocessing choice. It gives static trajectory
    clustering a fair chance to recover attenuation shape rather than nuisance
    amplitude/offset.
    """
    centered = data - tail_baseline(data, tail_fraction=tail_fraction)
    scale = np.linalg.norm(centered[:, : max(2, data.shape[1] // 5), :], axis=(1, 2), keepdims=True)
    return centered / (scale + eps)


def energy_curves(data, tail_fraction=0.15, normalize=True):
    centered = data - tail_baseline(data, tail_fraction=tail_fraction)
    energy = np.linalg.norm(centered, axis=2)
    if normalize:
        denom = np.maximum(np.max(energy, axis=1, keepdims=True), 1e-8)
        energy = energy / denom
    return energy


def estimate_attenuation_slope_features(data, tail_fraction=0.15):
    """
    Estimate a simple static feature: the slope of log energy over time.

    This is not HDP-GPC. It is an engineered static baseline showing that if the
    analyst knows the right summary statistic, the attenuation factor can be
    recovered without a full dynamic Bayesian model.
    """
    E = energy_curves(data, tail_fraction=tail_fraction, normalize=False)
    E = np.maximum(E, 1e-8)

    T = E.shape[1]
    x = np.arange(T, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)

    logE = np.log(E)
    slope = np.sum((x[None, :] * (logE - logE.mean(axis=1, keepdims=True))), axis=1)
    slope = slope / np.sum(x**2)

    early = E[:, : max(2, T // 5)].mean(axis=1)
    late = E[:, -max(2, T // 5) :].mean(axis=1)
    ratio = late / np.maximum(early, 1e-8)

    return np.column_stack([slope, ratio])


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def save_energy_curves_plot(data, color_labels, out_path, title):
    E = energy_curves(data)
    plt.figure(figsize=(8.0, 5.0))
    for k in sorted(np.unique(color_labels)):
        idx = np.where(color_labels == k)[0]
        for i in idx:
            plt.plot(E[i], alpha=0.35, linewidth=1.2)
        plt.plot(E[idx].mean(axis=0), linewidth=3.0, label=f"cluster {k}")
    plt.title(title)
    plt.xlabel("time index")
    plt.ylabel("normalized tail-corrected energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_pca_plot(X, color_labels, out_path, title):
    if X.shape[0] < 3:
        return

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(7.2, 5.4))
    for k in sorted(np.unique(color_labels)):
        idx = np.where(color_labels == k)[0]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=35, alpha=0.80, label=f"cluster {k}")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_output_examples(data, labels, out_dir, max_outputs=4):
    n_outputs = data.shape[2]
    for d in range(min(max_outputs, n_outputs)):
        plt.figure(figsize=(8.0, 5.0))
        for k in sorted(np.unique(labels)):
            idx = np.where(labels == k)[0]
            for i in idx:
                plt.plot(data[i, :, d], alpha=0.25, linewidth=1.0)
            plt.plot(data[idx, :, d].mean(axis=0), linewidth=3.0, label=f"system {k}")
        plt.title(f"Output {d}: true attenuation systems")
        plt.xlabel("time index")
        plt.ylabel(f"y(t, output {d})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"true_systems_output{d}.png", dpi=180)
        plt.close()


def save_cloud_pooled_plot(data, point_labels, out_path, title):
    """
    PCA plot for pooled D-dimensional points.
    """
    points = data.reshape(-1, data.shape[-1])
    if points.shape[0] < 3:
        return

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(points)

    plt.figure(figsize=(7.2, 5.4))
    for k in sorted(np.unique(point_labels)):
        idx = np.where(point_labels == k)[0]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=10, alpha=0.45, label=f"cluster {k}")
    plt.title(title)
    plt.xlabel("pooled-point PC1")
    plt.ylabel("pooled-point PC2")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ---------------------------------------------------------------------
# Static baselines
# ---------------------------------------------------------------------
def run_cloud_level_static_dpgmm(data, labels, out_dir, args):
    """
    Diagnostic only: cluster the pooled D-dimensional observations y_t.
    This does not use the same sample unit as HDP-GPC.
    """
    n_traj, T, D = data.shape

    points = data.reshape(n_traj * T, D)
    repeated_labels = np.repeat(labels, T)

    point_clusters, active, _ = dp_gmm_fit_predict(
        points,
        max_components=args.cloud_dpgmm_max_components,
        alpha=args.cloud_dpgmm_alpha,
        max_iter=args.dpgmm_max_iter,
        reg_covar=args.dpgmm_reg_covar,
        covariance_type=args.cloud_dpgmm_covariance_type,
        seed=args.seed,
    )

    point_ari = adjusted_rand_score(repeated_labels, point_clusters)

    point_clusters_by_traj = point_clusters.reshape(n_traj, T)
    traj_majority = np.array([majority_vote(point_clusters_by_traj[i]) for i in range(n_traj)])
    traj_majority_ari = adjusted_rand_score(labels, traj_majority)

    components_per_traj = [
        len(np.unique(point_clusters_by_traj[i]))
        for i in range(n_traj)
    ]

    save_cloud_pooled_plot(
        data,
        point_clusters,
        out_dir / "cloud_level_static_dpgmm_pooled_point_pca.png",
        f"Cloud-level static DP-GMM on pooled observations ({active} active components)",
    )

    return {
        "method": "cloud_level_static_dp_gmm_pooled_observations",
        "comparison_family": "cloud_level_diagnostic",
        "input_used": "pooled D-dimensional observations y_t",
        "active_or_inferred_clusters": active,
        "ARI_vs_true_system": float(point_ari),
        "ARI_trajectory_label_vs_true_system": float(traj_majority_ari),
        "mean_components_per_trajectory": float(np.mean(components_per_traj)),
        "median_components_per_trajectory": float(np.median(components_per_traj)),
    }


def run_trajectory_static_dpgmm_raw(data, labels, out_dir, args):
    """
    Fair sample unit: one complete trajectory is one vector sample.
    """
    n_traj, T, D = data.shape
    X = data.reshape(n_traj, T * D)

    clusters, active, _ = dp_gmm_fit_predict(
        X,
        max_components=args.trajectory_dpgmm_max_components,
        alpha=args.trajectory_dpgmm_alpha,
        max_iter=args.dpgmm_max_iter,
        reg_covar=args.trajectory_dpgmm_reg_covar,
        covariance_type=args.trajectory_dpgmm_covariance_type,
        seed=args.seed,
    )

    ari = adjusted_rand_score(labels, clusters)

    np.save(out_dir / "trajectory_static_dpgmm_raw_labels.npy", clusters)
    pd.DataFrame(
        {
            "trajectory_index": np.arange(n_traj),
            "true_system": labels,
            "cluster": clusters,
        }
    ).to_csv(out_dir / "trajectory_static_dpgmm_raw_labels.csv", index=False)

    save_energy_curves_plot(
        data,
        clusters,
        out_dir / "trajectory_static_dpgmm_raw_energy_curves.png",
        f"Energy curves colored by raw flattened static DP-GMM (ARI={ari:.3f})",
    )

    save_pca_plot(
        X,
        clusters,
        out_dir / "trajectory_static_dpgmm_raw_pca.png",
        f"PCA of raw flattened trajectories colored by static DP-GMM (ARI={ari:.3f})",
    )

    return {
        "method": "trajectory_level_static_dp_gmm_raw_flattened",
        "comparison_family": "trajectory_level_fair_comparison",
        "input_used": "raw complete trajectories flattened to vectors",
        "active_or_inferred_clusters": active,
        "ARI_vs_true_system": float(ari),
        "ARI_trajectory_label_vs_true_system": float(ari),
        "mean_components_per_trajectory": np.nan,
        "median_components_per_trajectory": np.nan,
    }


def run_trajectory_static_dpgmm_shape_normalized(data, labels, out_dir, args):
    """
    Stronger static baseline: trajectory-level DP-GMM after offset/scale
    normalization.
    """
    n_traj, T, D = data.shape
    data_norm = shape_normalize_trajectories(data, tail_fraction=args.tail_fraction)
    X = data_norm.reshape(n_traj, T * D)

    clusters, active, _ = dp_gmm_fit_predict(
        X,
        max_components=args.trajectory_dpgmm_max_components,
        alpha=args.trajectory_dpgmm_alpha,
        max_iter=args.dpgmm_max_iter,
        reg_covar=args.trajectory_dpgmm_reg_covar,
        covariance_type=args.trajectory_dpgmm_covariance_type,
        seed=args.seed,
    )

    ari = adjusted_rand_score(labels, clusters)

    np.save(out_dir / "trajectory_static_dpgmm_shape_normalized_labels.npy", clusters)
    pd.DataFrame(
        {
            "trajectory_index": np.arange(n_traj),
            "true_system": labels,
            "cluster": clusters,
        }
    ).to_csv(out_dir / "trajectory_static_dpgmm_shape_normalized_labels.csv", index=False)

    save_energy_curves_plot(
        data,
        clusters,
        out_dir / "trajectory_static_dpgmm_shape_normalized_energy_curves.png",
        f"Energy curves colored by shape-normalized static DP-GMM (ARI={ari:.3f})",
    )

    save_pca_plot(
        X,
        clusters,
        out_dir / "trajectory_static_dpgmm_shape_normalized_pca.png",
        f"PCA of shape-normalized trajectories colored by static DP-GMM (ARI={ari:.3f})",
    )

    return {
        "method": "trajectory_level_static_dp_gmm_shape_normalized",
        "comparison_family": "trajectory_level_fair_comparison",
        "input_used": "shape-normalized complete trajectories flattened to vectors",
        "active_or_inferred_clusters": active,
        "ARI_vs_true_system": float(ari),
        "ARI_trajectory_label_vs_true_system": float(ari),
        "mean_components_per_trajectory": np.nan,
        "median_components_per_trajectory": np.nan,
    }


def run_engineered_attenuation_feature_dpgmm(data, labels, out_dir, args):
    """
    Static engineered baseline using estimated log-energy slope and late/early
    energy ratio.
    """
    X = estimate_attenuation_slope_features(data, tail_fraction=args.tail_fraction)

    clusters, active, _ = dp_gmm_fit_predict(
        X,
        max_components=args.feature_dpgmm_max_components,
        alpha=args.feature_dpgmm_alpha,
        max_iter=args.dpgmm_max_iter,
        reg_covar=args.dpgmm_reg_covar,
        covariance_type="full",
        seed=args.seed,
    )

    ari = adjusted_rand_score(labels, clusters)

    np.save(out_dir / "engineered_attenuation_feature_dpgmm_labels.npy", clusters)
    pd.DataFrame(
        {
            "trajectory_index": np.arange(len(labels)),
            "true_system": labels,
            "cluster": clusters,
            "log_energy_slope": X[:, 0],
            "late_early_energy_ratio": X[:, 1],
        }
    ).to_csv(out_dir / "engineered_attenuation_feature_dpgmm_labels.csv", index=False)

    plt.figure(figsize=(7.2, 5.4))
    for k in sorted(np.unique(clusters)):
        idx = np.where(clusters == k)[0]
        plt.scatter(X[idx, 0], X[idx, 1], s=40, alpha=0.80, label=f"cluster {k}")
    plt.title(f"Engineered attenuation features clustered by DP-GMM (ARI={ari:.3f})")
    plt.xlabel("log-energy slope")
    plt.ylabel("late / early energy ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "engineered_attenuation_feature_dpgmm.png", dpi=180)
    plt.close()

    return {
        "method": "trajectory_level_static_dp_gmm_engineered_attenuation_features",
        "comparison_family": "trajectory_level_engineered_feature_baseline",
        "input_used": "estimated attenuation features from each trajectory",
        "active_or_inferred_clusters": active,
        "ARI_vs_true_system": float(ari),
        "ARI_trajectory_label_vs_true_system": float(ari),
        "mean_components_per_trajectory": np.nan,
        "median_components_per_trajectory": np.nan,
    }


# ---------------------------------------------------------------------
# Actual HDP-GPC
# ---------------------------------------------------------------------
def run_actual_hdpgpc(data, labels, out_dir, args):
    import hdpgpc.GPI_HDP as hdpgp
    from hdpgpc.get_data import compute_estimators_LDS
    from hdpgpc.util_plots import plot_models_plotly, print_results

    torch.set_default_dtype(torch.float64)

    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)

    num_samples, num_obs_per_sample, num_outputs = data.shape

    std, std_dif, bound_sigma_from_data, bound_gamma = compute_estimators_LDS(data)

    sigma = std * args.sigma_multiplier
    gamma = std_dif * args.gamma_multiplier

    sigma_floor = args.sigma_floor
    bound_sigma = (
        np.maximum(sigma * args.bound_sigma_low_multiplier, sigma_floor),
        np.maximum(sigma * args.bound_sigma_high_multiplier, sigma_floor * 10.0),
    )

    if args.outputscale is None:
        outputscale_ = float(np.max(np.abs(data)) * args.outputscale_multiplier + 1e-8)
    else:
        outputscale_ = float(args.outputscale)

    ini_lengthscale = args.ini_lengthscale
    bound_lengthscale = (args.bound_lengthscale_low, args.bound_lengthscale_high)

    noise_warp = std * args.noise_warp_multiplier
    bound_noise_warp = (
        np.maximum(noise_warp * args.bound_noise_warp_low_multiplier, sigma_floor),
        np.maximum(noise_warp * args.bound_noise_warp_high_multiplier, sigma_floor * 10.0),
    )

    l, L = 0, num_obs_per_sample
    x_basis = np.atleast_2d(np.arange(l, L, 1, dtype=np.float64)).T
    x_basis_warp = np.atleast_2d(np.arange(l, L, args.warp_basis_step, dtype=np.float64)).T
    x_train = np.atleast_2d(np.arange(l, L, dtype=np.float64)).T
    x_trains = np.array([x_train] * num_samples)

    sw_gp = hdpgp.GPI_HDP(
        x_basis,
        x_basis_warp=x_basis_warp,
        n_outputs=num_outputs,
        kernels=None,
        model_type=args.hdpgpc_model_type,
        ini_lengthscale=ini_lengthscale,
        bound_lengthscale=bound_lengthscale,
        ini_gamma=gamma,
        ini_sigma=sigma,
        ini_outputscale=outputscale_,
        noise_warp=noise_warp,
        bound_sigma=bound_sigma,
        bound_gamma=bound_gamma,
        bound_noise_warp=bound_noise_warp,
        warp_updating=args.warp_updating,
        method_compute_warp=args.method_compute_warp,
        verbose=args.verbose_hdpgpc,
        hmm_switch=args.hmm_switch,
        max_models=args.max_models,
        mode_warp=args.mode_warp,
        bayesian_params=args.bayesian_params,
        inducing_points=args.inducing_points,
        reestimate_initial_params=args.reestimate_initial_params,
        n_explore_steps=args.n_explore_steps,
        free_deg_MNIV=args.free_deg_MNIV,
        share_gp=args.share_gp,
        hdp_hyp='less',
    )

    start = time.time()
    sw_gp.include_batch(x_trains, data, warp=args.warp)
    elapsed_min = (time.time() - start) / 60.0

    print("\nHDP-GPC package results:")
    main_model = print_results(sw_gp, labels, 0, error=False)
    selected_gpmodels = sw_gp.selected_gpmodels()

    for lead in range(min(num_outputs, args.max_plot_outputs)):
        try:
            plot_models_plotly(
                sw_gp,
                selected_gpmodels,
                main_model,
                labels,
                0,
                lead=lead,
                save=str(out_dir / f"hdpgpc_offline_clusters_output{lead}.png"),
                step=0.5,
                plot_latent=True,
            )
        except Exception as exc:
            warnings.warn(f"plot_models_plotly failed for output {lead}: {repr(exc)}")

    labels_pred, source = try_extract_hdpgpc_labels(sw_gp, num_samples)

    if labels_pred is not None:
        ari = adjusted_rand_score(labels, labels_pred)
        active = len(np.unique(labels_pred))

        np.save(out_dir / "hdpgpc_extracted_trajectory_labels.npy", labels_pred)
        pd.DataFrame(
            {
                "trajectory_index": np.arange(num_samples),
                "true_system": labels,
                "hdpgpc_cluster": labels_pred,
            }
        ).to_csv(out_dir / "hdpgpc_extracted_trajectory_labels.csv", index=False)

        save_energy_curves_plot(
            data,
            labels_pred,
            out_dir / "hdpgpc_energy_curves_by_cluster.png",
            f"Energy curves colored by HDP-GPC assignment (ARI={ari:.3f})",
        )

        X = data.reshape(num_samples, num_obs_per_sample * num_outputs)
        save_pca_plot(
            X,
            labels_pred,
            out_dir / "hdpgpc_labels_on_raw_trajectory_pca.png",
            f"Raw trajectory PCA colored by HDP-GPC assignment (ARI={ari:.3f})",
        )
    else:
        ari = np.nan
        active = np.nan
        source = "not_found"

    diagnostics = {
        "elapsed_min": elapsed_min,
        "num_samples": int(num_samples),
        "num_obs_per_sample": int(num_obs_per_sample),
        "num_outputs": int(num_outputs),
        "warp": bool(args.warp),
        "outputscale": outputscale_,
        "ini_lengthscale": ini_lengthscale,
        "bound_lengthscale": list(bound_lengthscale),
        "sigma": np.asarray(sigma).tolist(),
        "gamma": np.asarray(gamma).tolist(),
        "bound_sigma_low": np.asarray(bound_sigma[0]).tolist(),
        "bound_sigma_high": np.asarray(bound_sigma[1]).tolist(),
        "noise_warp": np.asarray(noise_warp).tolist(),
        "bound_noise_warp_low": np.asarray(bound_noise_warp[0]).tolist(),
        "bound_noise_warp_high": np.asarray(bound_noise_warp[1]).tolist(),
        "assignment_source": source,
    }

    with open(out_dir / "hdpgpc_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    return {
        "method": "actual_hdpgpc_gpi_hdp",
        "comparison_family": "trajectory_level_fair_comparison",
        "input_used": "full high-dimensional trajectories",
        "active_or_inferred_clusters": active,
        "ARI_vs_true_system": float(ari) if not np.isnan(ari) else np.nan,
        "ARI_trajectory_label_vs_true_system": float(ari) if not np.isnan(ari) else np.nan,
        "mean_components_per_trajectory": np.nan,
        "median_components_per_trajectory": np.nan,
        "elapsed_min": float(elapsed_min),
        "assignment_source": source,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="High-dimensional attenuation simulation with static clustering and actual HDP-GPC."
    )

    parser.add_argument("--repo-root", type=str, default=".", help="Path to parent project containing hdpgpc/.")
    parser.add_argument("--out-dir", type=str, default="results_attenuation_hdpgpc")

    # Simulation
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-traj-per-system", type=int, default=200)
    parser.add_argument("--T", type=int, default=55)
    parser.add_argument("--n-outputs", type=int, default=3)
    parser.add_argument("--slow-attenuation", type=float, default=1.005)
    parser.add_argument("--fast-attenuation", type=float, default=0.985)
    parser.add_argument("--obs-noise", type=float, default=0.05)
    parser.add_argument("--offset-std", type=float, default=0.05)
    parser.add_argument("--amplitude-log-std", type=float, default=0.05)
    parser.add_argument("--loading-jitter", type=float, default=0.12)
    parser.add_argument("--time-shift-max", type=int, default=0)
    parser.add_argument("--tail-fraction", type=float, default=0.15)

    # Shared static clustering
    parser.add_argument("--dpgmm-max-iter", type=int, default=800)
    parser.add_argument("--dpgmm-reg-covar", type=float, default=1e-5)

    # Cloud diagnostic
    parser.add_argument("--skip-cloud-baseline", action="store_true")
    parser.add_argument("--cloud-dpgmm-max-components", type=int, default=15)
    parser.add_argument("--cloud-dpgmm-alpha", type=float, default=0.03)
    parser.add_argument(
        "--cloud-dpgmm-covariance-type",
        type=str,
        default="diag",
        choices=["full", "tied", "diag", "spherical"],
    )

    # Trajectory-level static baselines
    parser.add_argument("--skip-static-baselines", action="store_true")
    parser.add_argument("--trajectory-dpgmm-max-components", type=int, default=8)
    parser.add_argument("--trajectory-dpgmm-alpha", type=float, default=0.1)
    parser.add_argument("--trajectory-dpgmm-reg-covar", type=float, default=1e-4)
    parser.add_argument(
        "--trajectory-dpgmm-covariance-type",
        type=str,
        default="diag",
        choices=["full", "tied", "diag", "spherical"],
    )

    # Engineered attenuation feature baseline
    parser.add_argument("--skip-engineered-feature-baseline", action="store_true")
    parser.add_argument("--feature-dpgmm-max-components", type=int, default=6)
    parser.add_argument("--feature-dpgmm-alpha", type=float, default=0.1)

    # HDP-GPC
    parser.add_argument("--skip-hdpgpc", action="store_true")
    parser.add_argument("--warp", action="store_true")
    parser.add_argument("--warp-updating", action="store_true")
    parser.add_argument("--method-compute-warp", type=str, default="greedy")
    parser.add_argument("--mode-warp", type=str, default="rough")
    parser.add_argument("--warp-basis-step", type=int, default=2)

    parser.add_argument("--hdpgpc-model-type", type=str, default="dynamic")
    parser.add_argument("--hmm-switch", type=str2bool, default=True)
    parser.add_argument("--max-models", type=int, default=100)
    parser.add_argument("--bayesian-params", type=str2bool, default=True)
    parser.add_argument("--inducing-points", type=str2bool, default=False)
    parser.add_argument("--reestimate-initial-params", type=str2bool, default=False)
    parser.add_argument("--n-explore-steps", type=int, default=2)
    parser.add_argument("--free-deg-MNIV", type=int, default=3)
    parser.add_argument("--share-gp", type=str2bool, default=True)
    parser.add_argument("--verbose-hdpgpc", action="store_true")
    parser.add_argument("--max-plot-outputs", type=int, default=4)

    # HDP-GPC priors
    parser.add_argument("--sigma-multiplier", type=float, default=1.0)
    parser.add_argument("--gamma-multiplier", type=float, default=10.0)
    parser.add_argument("--sigma-floor", type=float, default=1e-8)
    parser.add_argument("--bound-sigma-low-multiplier", type=float, default=1e-6)
    parser.add_argument("--bound-sigma-high-multiplier", type=float, default=1e-4)

    parser.add_argument("--outputscale", type=float, default=None)
    parser.add_argument("--outputscale-multiplier", type=float, default=1.2)
    parser.add_argument("--ini-lengthscale", type=float, default=8.0)
    parser.add_argument("--bound-lengthscale-low", type=float, default=1.0)
    parser.add_argument("--bound-lengthscale-high", type=float, default=40.0)

    parser.add_argument("--noise-warp-multiplier", type=float, default=60.0)
    parser.add_argument("--bound-noise-warp-low-multiplier", type=float, default=0.01)
    parser.add_argument("--bound-noise-warp-high-multiplier", type=float, default=60.0)

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    data, labels, metadata = simulate_attenuation_systems(
        seed=args.seed,
        n_traj_per_system=args.n_traj_per_system,
        T=args.T,
        n_outputs=args.n_outputs,
        slow_attenuation=args.slow_attenuation,
        fast_attenuation=args.fast_attenuation,
        obs_noise=args.obs_noise,
        offset_std=args.offset_std,
        amplitude_log_std=args.amplitude_log_std,
        loading_jitter=args.loading_jitter,
        time_shift_max=args.time_shift_max,
    )

    np.save(out_dir / "synthetic_attenuation_data.npy", data)
    np.save(out_dir / "synthetic_attenuation_labels.npy", labels)
    metadata.to_csv(out_dir / "synthetic_attenuation_metadata.csv", index=False)

    print("\nData:")
    print(f"  data.shape   = {data.shape}")
    print(f"  labels.shape = {labels.shape}")
    print(f"  label counts = {np.bincount(labels)}")
    print(f"  output dir   = {out_dir}")

    save_energy_curves_plot(
        data,
        labels,
        out_dir / "true_systems_energy_curves.png",
        "True attenuation systems: normalized tail-corrected energy",
    )

    X_raw = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    save_pca_plot(
        X_raw,
        labels,
        out_dir / "true_systems_raw_trajectory_pca.png",
        "Raw flattened trajectory PCA colored by true system",
    )

    data_shape = shape_normalize_trajectories(data, tail_fraction=args.tail_fraction)
    X_shape = data_shape.reshape(data.shape[0], data.shape[1] * data.shape[2])
    save_pca_plot(
        X_shape,
        labels,
        out_dir / "true_systems_shape_normalized_trajectory_pca.png",
        "Shape-normalized trajectory PCA colored by true system",
    )

    save_output_examples(data, labels, out_dir, max_outputs=args.max_plot_outputs)

    results = []

    if not args.skip_cloud_baseline:
        print("\nRunning cloud-level static DP-GMM diagnostic...")
        res = run_cloud_level_static_dpgmm(data, labels, out_dir, args)
        print(pd.DataFrame([res]).to_string(index=False))
        results.append(res)

    if not args.skip_static_baselines:
        print("\nRunning trajectory-level static DP-GMM on raw flattened trajectories...")
        res = run_trajectory_static_dpgmm_raw(data, labels, out_dir, args)
        print(pd.DataFrame([res]).to_string(index=False))
        results.append(res)

        print("\nRunning trajectory-level static DP-GMM on shape-normalized flattened trajectories...")
        res = run_trajectory_static_dpgmm_shape_normalized(data, labels, out_dir, args)
        print(pd.DataFrame([res]).to_string(index=False))
        results.append(res)

    if not args.skip_engineered_feature_baseline:
        print("\nRunning engineered attenuation-feature DP-GMM baseline...")
        res = run_engineered_attenuation_feature_dpgmm(data, labels, out_dir, args)
        print(pd.DataFrame([res]).to_string(index=False))
        results.append(res)

    if not args.skip_hdpgpc:
        print("\nRunning actual HDP-GPC...")
        try:
            ensure_repo_on_path(args.repo_root)
            res = run_actual_hdpgpc(data, labels, out_dir, args)
            print(pd.DataFrame([res]).to_string(index=False))
            results.append(res)
        except Exception as exc:
            warnings.warn(
                "Actual HDP-GPC run failed. Static baselines and data were saved. "
                f"Error was:\n{repr(exc)}"
            )
            results.append(
                {
                    "method": "actual_hdpgpc_gpi_hdp",
                    "comparison_family": "trajectory_level_fair_comparison",
                    "input_used": "full high-dimensional trajectories",
                    "active_or_inferred_clusters": np.nan,
                    "ARI_vs_true_system": np.nan,
                    "ARI_trajectory_label_vs_true_system": np.nan,
                    "mean_components_per_trajectory": np.nan,
                    "median_components_per_trajectory": np.nan,
                    "elapsed_min": np.nan,
                    "assignment_source": f"failed: {repr(exc)}",
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "comparison_results.csv", index=False)

    with open(out_dir / "simulation_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\nComparison:")
    print(results_df.to_string(index=False))

    print("\nSaved files:")
    for path in sorted(out_dir.iterdir()):
        print(" ", path)

    print(
        "\nRecommended interpretation:\n"
        "  - The cloud-level pooled-observation clustering is diagnostic only.\n"
        "  - The fair static comparison is trajectory-level: raw flattened trajectories\n"
        "    and shape-normalized flattened trajectories.\n"
        "  - The engineered attenuation-feature baseline is an upper-reference static\n"
        "    baseline: it shows what happens if the correct summary statistic is known.\n"
        "  - HDP-GPC receives the same [N, T, D] trajectory tensor as the trajectory-level\n"
        "    static baselines, but models smooth multi-output functional/dynamic structure.\n"
    )


if __name__ == "__main__":
    main()
