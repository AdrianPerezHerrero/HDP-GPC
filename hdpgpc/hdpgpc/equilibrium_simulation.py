# -*- coding: utf-8 -*-
"""
High-dimensional equilibrium-flow simulation with static DP-GMM and actual HDP-GPC.

No warping is used. An optional nonlinear observation map can bend the latent flows into a curved manifold.

Purpose
-------
Simulate a situation where high-dimensional observations evolve smoothly toward
one of two equilibrium points. In PCA, the pooled observations look like smooth
movements from a common initial region toward one equilibrium or another.

A static clustering method applied to the pooled observation cloud tends to
split the continuous motion into several spatial/phase clusters:
    early phase, intermediate phase, final equilibrium region, etc.

HDP-GPC, applied to whole trajectories with model_type='dynamic' and warp=False,
should cluster the underlying dynamical movement and therefore reduce the number
of detected regimes toward the two true systems.

Data-generating process
-----------------------
For each trajectory n and system k:

    y_n(t) = e_k + rho_k^t (y_n(0) - e_k) + noise

where:
    e_k      is the high-dimensional equilibrium of system k,
    rho_k    is a contraction factor,
    y_n(0)   is a noisy initial condition near a common starting region.

There is no temporal warping. All trajectories have the same time grid.

Comparison
----------
1. Cloud-level static DP-GMM on pooled D-dimensional observations y_n(t).
   This is the diagnostic expected to over-segment the movement into multiple
   spatial/phase clusters.

2. Trajectory-level static DP-GMM on flattened trajectories.
   This is a fair sample-unit baseline. It may recover the two systems if the
   final equilibrium dominates the flattened vector.

3. Actual HDP-GPC on the same [N, T, D] trajectory tensor, with warp=False.
   This is the dynamic/functional model expected to recover the two dynamical
   movements.

Run
---
From the parent folder containing hdpgpc/:

    python simulate_equilibrium_flows_actual_hdpgpc.py

or:

    python simulate_equilibrium_flows_actual_hdpgpc.py --repo-root C:/Users/Adrian/Projects/YourProject

Useful options:

    python simulate_equilibrium_flows_actual_hdpgpc.py --skip-hdpgpc
    python simulate_equilibrium_flows_actual_hdpgpc.py --n-outputs 16 --T 70
    python simulate_equilibrium_flows_actual_hdpgpc.py --equilibrium-distance 5.0
    python simulate_equilibrium_flows_actual_hdpgpc.py --rho0 0.965 --rho1 0.94
    python simulate_equilibrium_flows_actual_hdpgpc.py --cloud-dpgmm-alpha 0.02 --cloud-dpgmm-max-components 18

Expected qualitative result
---------------------------
- PCA of pooled points: two smooth branches moving toward two equilibria.
- Cloud DP-GMM: several active clusters along the branches/equilibria.
- HDP-GPC: ideally two trajectory clusters corresponding to the two flows.
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

from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in {"true", "1", "yes", "y"}:
        return True
    if str(v).lower() in {"false", "0", "no", "n"}:
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
def orthonormal_directions(n_outputs, seed=0):
    rng = np.random.default_rng(seed)

    # Build two smooth high-dimensional directions.
    grid = np.linspace(0, 2 * np.pi, n_outputs)
    v1 = np.sin(grid) + 0.35 * np.cos(2 * grid)
    v2 = np.cos(grid) - 0.35 * np.sin(2 * grid)

    v1 = v1 + rng.normal(0.0, 0.02, size=n_outputs)
    v2 = v2 + rng.normal(0.0, 0.02, size=n_outputs)

    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = v2 - np.dot(v2, v1) * v1
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)

    return v1, v2



def apply_nonlinear_observation_map(Y, strength=0.0):
    """
    Optional nonlinear observation map.

    The latent dynamics remain simple contractive flows toward equilibria, but
    the observed high-dimensional signal is passed through a smooth nonlinear
    feature map. This makes the movement lie on a curved observation manifold.

    strength = 0.0 gives the original linear observation model.
    """
    if strength <= 0:
        return Y

    Y = np.asarray(Y, dtype=np.float64)
    D = Y.shape[1]

    # Smooth bounded nonlinearities per coordinate.
    nonlinear = np.tanh(Y)

    # Local coordinate interactions, rolled so dimensionality stays D.
    interactions = Y * np.roll(Y, shift=1, axis=1)

    # A global smooth latent factor that bends all outputs coherently.
    global_factor = np.sin(Y.mean(axis=1, keepdims=True))

    mapped = (
        Y
        + strength * nonlinear
        + 0.35 * strength * interactions
        + 0.50 * strength * global_factor * np.roll(Y, shift=2, axis=1)
    )

    return mapped


def simulate_equilibrium_flows(
    seed=42,
    n_traj_per_system=25,
    T=60,
    n_outputs=12,
    equilibrium_distance=4.0,
    rho0=0.960,
    rho1=0.945,
    init_std=0.35,
    obs_noise=0.06,
    drift_noise=0.015,
    amplitude_log_std=0.10,
    output_offset_std=0.08,
    nonlinear_observation_strength=0.0,
):
    """
    Simulate two high-dimensional contractive flows.

    Both systems start from a shared initial region. System 0 moves to
    equilibrium e0, and system 1 moves to equilibrium e1.

    No warping is applied.
    """
    rng = np.random.default_rng(seed)
    v1, v2 = orthonormal_directions(n_outputs, seed=seed + 100)

    # Two equilibria are separated mostly on v1, with a small v2 component so
    # PCA displays curved/non-collinear movement rather than a trivial line.
    e0 = equilibrium_distance * (0.90 * v1 + 0.25 * v2)
    e1 = equilibrium_distance * (-0.90 * v1 + 0.25 * v2)

    # Common starting region is between the equilibria but displaced on v2.
    start_center = equilibrium_distance * (-0.40 * v2)

    rhos = [rho0, rho1]
    equilibria = [e0, e1]

    data = []
    labels = []
    metadata = []

    for system in [0, 1]:
        rho = rhos[system]
        eq = equilibria[system]

        for _ in range(n_traj_per_system):
            # Shared initial region with nuisance variation.
            y0 = start_center + rng.normal(0.0, init_std, size=n_outputs)

            # Small trajectory-level amplitude and offset nuisance.
            amp = np.exp(rng.normal(0.0, amplitude_log_std))
            offset = rng.normal(0.0, output_offset_std, size=n_outputs)

            Y = []
            y_prev = y0.copy()

            for t in range(T):
                # Deterministic contraction toward equilibrium.
                deterministic = eq + (rho ** t) * (y0 - eq)

                # Mild autocorrelated perturbation around the path.
                if t == 0:
                    drift = rng.normal(0.0, drift_noise, size=n_outputs)
                else:
                    drift = 0.85 * drift + rng.normal(0.0, drift_noise, size=n_outputs)

                y_t = offset + amp * deterministic + drift
                y_t = y_t + rng.normal(0.0, obs_noise, size=n_outputs)
                Y.append(y_t)

                y_prev = y_t

            Y = np.asarray(Y, dtype=np.float64)
            Y = apply_nonlinear_observation_map(
                Y,
                strength=nonlinear_observation_strength,
            )

            data.append(Y)
            labels.append(system)
            metadata.append(
                {
                    "system": system,
                    "rho": rho,
                    "amplitude": amp,
                    "equilibrium_norm": float(np.linalg.norm(eq)),
                }
            )

    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    metadata = pd.DataFrame(metadata)

    return data, labels, metadata, e0, e1, start_center


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def fit_pooled_pca(data):
    points = data.reshape(-1, data.shape[-1])
    pca = PCA(n_components=2, random_state=0)
    pca.fit(points)
    return pca



def fit_pooled_kernel_pca(data, gamma=None):
    """
    Fit an RBF Kernel PCA on the pooled high-dimensional snapshots.

    This is only for visualization. Clustering methods still receive the same
    original data objects as before.
    """
    points = data.reshape(-1, data.shape[-1])

    if gamma is None:
        # Stable heuristic for standardized-ish high-dimensional observations.
        gamma = 1.0 / max(1, points.shape[1])

    kpca = KernelPCA(
        n_components=2,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=False,
        random_state=0,
    )
    kpca.fit(points)
    return kpca


def save_pooled_kernel_pca_true_dynamics(data, labels, out_path, title, gamma=None):
    n_traj, T, D = data.shape
    kpca = fit_pooled_kernel_pca(data, gamma=gamma)
    Z = kpca.transform(data.reshape(-1, D)).reshape(n_traj, T, 2)

    plt.figure(figsize=(7.8, 5.8))
    for k in sorted(np.unique(labels)):
        idx = np.where(labels == k)[0]
        for i in idx:
            plt.plot(Z[i, :, 0], Z[i, :, 1], alpha=0.35, linewidth=1.2)
            plt.scatter(Z[i, -1, 0], Z[i, -1, 1], s=12, alpha=0.55)
        mean_path = Z[idx].mean(axis=0)
        plt.plot(mean_path[:, 0], mean_path[:, 1], linewidth=3.0, label=f"true system {k}")

    plt.title(title)
    plt.xlabel("RBF Kernel PCA 1")
    plt.ylabel("RBF Kernel PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_pooled_kernel_pca_point_clusters(data, point_clusters, out_path, title, gamma=None):
    n_traj, T, D = data.shape
    points = data.reshape(-1, D)
    kpca = fit_pooled_kernel_pca(data, gamma=gamma)
    Z = kpca.transform(points)

    plt.figure(figsize=(7.8, 5.8))
    for k in sorted(np.unique(point_clusters)):
        idx = np.where(point_clusters == k)[0]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=9, alpha=0.50, label=f"cluster {k}")

    plt.title(title)
    plt.xlabel("RBF Kernel PCA 1")
    plt.ylabel("RBF Kernel PCA 2")
    if len(np.unique(point_clusters)) <= 12:
        plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_pooled_kernel_pca_trajectory_clusters(data, traj_clusters, out_path, title, gamma=None):
    n_traj, T, D = data.shape
    kpca = fit_pooled_kernel_pca(data, gamma=gamma)
    Z = kpca.transform(data.reshape(-1, D)).reshape(n_traj, T, 2)

    plt.figure(figsize=(7.8, 5.8))
    for k in sorted(np.unique(traj_clusters)):
        idx = np.where(traj_clusters == k)[0]
        for i in idx:
            plt.plot(Z[i, :, 0], Z[i, :, 1], alpha=0.40, linewidth=1.2)
            plt.scatter(Z[i, -1, 0], Z[i, -1, 1], s=13, alpha=0.60)
        mean_path = Z[idx].mean(axis=0)
        plt.plot(mean_path[:, 0], mean_path[:, 1], linewidth=3.0, label=f"cluster {k}")

    plt.title(title)
    plt.xlabel("RBF Kernel PCA 1")
    plt.ylabel("RBF Kernel PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_pooled_pca_true_dynamics(data, labels, out_path, title):
    n_traj, T, D = data.shape
    pca = fit_pooled_pca(data)
    Z = pca.transform(data.reshape(-1, D)).reshape(n_traj, T, 2)

    plt.figure(figsize=(7.8, 5.8))
    for k in sorted(np.unique(labels)):
        idx = np.where(labels == k)[0]
        for i in idx:
            plt.plot(Z[i, :, 0], Z[i, :, 1], alpha=0.35, linewidth=1.2)
            plt.scatter(Z[i, -1, 0], Z[i, -1, 1], s=12, alpha=0.55)
        mean_path = Z[idx].mean(axis=0)
        plt.plot(mean_path[:, 0], mean_path[:, 1], linewidth=3.0, label=f"true system {k}")

    plt.title(title)
    plt.xlabel("pooled PCA 1")
    plt.ylabel("pooled PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_pooled_pca_point_clusters(data, point_clusters, out_path, title):
    n_traj, T, D = data.shape
    points = data.reshape(-1, D)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(points)

    plt.figure(figsize=(7.8, 5.8))
    for k in sorted(np.unique(point_clusters)):
        idx = np.where(point_clusters == k)[0]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=9, alpha=0.50, label=f"cluster {k}")

    plt.title(title)
    plt.xlabel("pooled PCA 1")
    plt.ylabel("pooled PCA 2")
    if len(np.unique(point_clusters)) <= 12:
        plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_pooled_pca_trajectory_clusters(data, traj_clusters, out_path, title):
    n_traj, T, D = data.shape
    pca = fit_pooled_pca(data)
    Z = pca.transform(data.reshape(-1, D)).reshape(n_traj, T, 2)

    plt.figure(figsize=(7.8, 5.8))
    for k in sorted(np.unique(traj_clusters)):
        idx = np.where(traj_clusters == k)[0]
        for i in idx:
            plt.plot(Z[i, :, 0], Z[i, :, 1], alpha=0.40, linewidth=1.2)
            plt.scatter(Z[i, -1, 0], Z[i, -1, 1], s=13, alpha=0.60)
        mean_path = Z[idx].mean(axis=0)
        plt.plot(mean_path[:, 0], mean_path[:, 1], linewidth=3.0, label=f"cluster {k}")

    plt.title(title)
    plt.xlabel("pooled PCA 1")
    plt.ylabel("pooled PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_trajectory_pca(data, traj_labels, out_path, title):
    X = data.reshape(data.shape[0], -1)
    if X.shape[0] < 3:
        return

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(7.2, 5.4))
    for k in sorted(np.unique(traj_labels)):
        idx = np.where(traj_labels == k)[0]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=38, alpha=0.82, label=f"group {k}")

    plt.title(title)
    plt.xlabel("trajectory PC1")
    plt.ylabel("trajectory PC2")
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
            for i in idx[:12]:
                plt.plot(data[i, :, d], alpha=0.22, linewidth=1.0)
            plt.plot(data[idx, :, d].mean(axis=0), linewidth=3.0, label=f"system {k}")
        plt.title(f"Output {d}: true equilibrium-flow systems")
        plt.xlabel("time index")
        plt.ylabel(f"y(t, output {d})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"true_systems_output{d}.png", dpi=180)
        plt.close()


# ---------------------------------------------------------------------
# Static baselines
# ---------------------------------------------------------------------
def run_cloud_level_dpgmm(data, labels, out_dir, args):
    """
    Diagnostic: cluster all D-dimensional snapshots y_n(t), ignoring trajectory
    identity and temporal evolution.
    """
    n_traj, T, D = data.shape
    points = data.reshape(n_traj * T, D)
    point_true_system = np.repeat(labels, T)

    pred, active, model = dp_gmm_fit_predict(
        points,
        max_components=args.cloud_dpgmm_max_components,
        alpha=args.cloud_dpgmm_alpha,
        max_iter=args.dpgmm_max_iter,
        reg_covar=args.dpgmm_reg_covar,
        covariance_type=args.cloud_dpgmm_covariance_type,
        seed=args.seed,
    )

    point_ari = adjusted_rand_score(point_true_system, pred)

    pred_by_traj = pred.reshape(n_traj, T)
    traj_majority = np.array([majority_vote(pred_by_traj[i]) for i in range(n_traj)])
    traj_majority_ari = adjusted_rand_score(labels, traj_majority)

    components_per_traj = np.array([len(np.unique(pred_by_traj[i])) for i in range(n_traj)])

    np.save(out_dir / "cloud_static_dpgmm_point_labels.npy", pred)
    pd.DataFrame(
        {
            "point_index": np.arange(n_traj * T),
            "trajectory_index": np.repeat(np.arange(n_traj), T),
            "time_index": np.tile(np.arange(T), n_traj),
            "true_system": point_true_system,
            "cloud_static_dpgmm_cluster": pred,
        }
    ).to_csv(out_dir / "cloud_static_dpgmm_point_labels.csv", index=False)

    save_pooled_pca_point_clusters(
        data,
        pred,
        out_dir / "cloud_static_dpgmm_pooled_pca_clusters.png",
        f"Cloud-level DP-GMM: {active} active snapshot clusters",
    )

    save_pooled_kernel_pca_point_clusters(
        data,
        pred,
        out_dir / "cloud_static_dpgmm_pooled_kernel_pca_clusters.png",
        f"Cloud-level DP-GMM in RBF Kernel PCA: {active} active snapshot clusters",
        gamma=args.kernel_pca_gamma,
    )

    return {
        "method": "cloud_level_static_dp_gmm_snapshots",
        "comparison_family": "cloud_level_diagnostic",
        "input_used": "pooled D-dimensional snapshots y(t)",
        "active_or_inferred_clusters": active,
        "ARI_point_clusters_vs_true_system": float(point_ari),
        "ARI_trajectory_majority_vs_true_system": float(traj_majority_ari),
        "mean_snapshot_clusters_per_trajectory": float(components_per_traj.mean()),
        "median_snapshot_clusters_per_trajectory": float(np.median(components_per_traj)),
        "assignment_source": "sklearn.BayesianGaussianMixture",
    }


def run_trajectory_level_dpgmm(data, labels, out_dir, args):
    """
    Fair sample-unit static baseline: one complete trajectory is one flattened
    vector sample.
    """
    n_traj, T, D = data.shape
    X = data.reshape(n_traj, T * D)

    pred, active, model = dp_gmm_fit_predict(
        X,
        max_components=args.trajectory_dpgmm_max_components,
        alpha=args.trajectory_dpgmm_alpha,
        max_iter=args.dpgmm_max_iter,
        reg_covar=args.trajectory_dpgmm_reg_covar,
        covariance_type=args.trajectory_dpgmm_covariance_type,
        seed=args.seed,
    )

    ari = adjusted_rand_score(labels, pred)

    np.save(out_dir / "trajectory_static_dpgmm_labels.npy", pred)
    pd.DataFrame(
        {
            "trajectory_index": np.arange(n_traj),
            "true_system": labels,
            "trajectory_static_dpgmm_cluster": pred,
        }
    ).to_csv(out_dir / "trajectory_static_dpgmm_labels.csv", index=False)

    save_pooled_pca_trajectory_clusters(
        data,
        pred,
        out_dir / "trajectory_static_dpgmm_pooled_pca_paths.png",
        f"Trajectory-level static DP-GMM paths (ARI={ari:.3f})",
    )

    save_pooled_kernel_pca_trajectory_clusters(
        data,
        pred,
        out_dir / "trajectory_static_dpgmm_pooled_kernel_pca_paths.png",
        f"Trajectory-level static DP-GMM paths in RBF Kernel PCA (ARI={ari:.3f})",
        gamma=args.kernel_pca_gamma,
    )

    save_trajectory_pca(
        data,
        pred,
        out_dir / "trajectory_static_dpgmm_trajectory_pca.png",
        f"Flattened trajectory PCA colored by static DP-GMM (ARI={ari:.3f})",
    )

    return {
        "method": "trajectory_level_static_dp_gmm_flattened",
        "comparison_family": "trajectory_level_fair_comparison",
        "input_used": "complete trajectories flattened to vectors",
        "active_or_inferred_clusters": active,
        "ARI_trajectory_vs_true_system": float(ari),
        "assignment_source": "sklearn.BayesianGaussianMixture",
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
        hdp_hyp='less'
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

    pred, source = try_extract_hdpgpc_labels(sw_gp, num_samples)

    if pred is not None:
        ari = adjusted_rand_score(labels, pred)
        active = len(np.unique(pred))

        np.save(out_dir / "hdpgpc_extracted_trajectory_labels.npy", pred)
        pd.DataFrame(
            {
                "trajectory_index": np.arange(num_samples),
                "true_system": labels,
                "hdpgpc_cluster": pred,
            }
        ).to_csv(out_dir / "hdpgpc_extracted_trajectory_labels.csv", index=False)

        save_pooled_pca_trajectory_clusters(
            data,
            pred,
            out_dir / "hdpgpc_pooled_pca_paths_by_cluster.png",
            f"HDP-GPC trajectory clusters in pooled PCA space (ARI={ari:.3f})",
        )

        save_pooled_kernel_pca_trajectory_clusters(
            data,
            pred,
            out_dir / "hdpgpc_pooled_kernel_pca_paths_by_cluster.png",
            f"HDP-GPC trajectory clusters in RBF Kernel PCA space (ARI={ari:.3f})",
            gamma=args.kernel_pca_gamma,
        )

        save_trajectory_pca(
            data,
            pred,
            out_dir / "hdpgpc_trajectory_pca_by_cluster.png",
            f"Flattened trajectory PCA colored by HDP-GPC (ARI={ari:.3f})",
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
        "method": "actual_hdpgpc_gpi_hdp_no_warp",
        "comparison_family": "trajectory_level_fair_comparison",
        "input_used": "full high-dimensional equilibrium-flow trajectories",
        "active_or_inferred_clusters": active,
        "ARI_trajectory_vs_true_system": float(ari) if not np.isnan(ari) else np.nan,
        "elapsed_min": float(elapsed_min),
        "assignment_source": source,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="High-dimensional equilibrium-flow simulation with static DP-GMM and actual HDP-GPC."
    )

    parser.add_argument("--repo-root", type=str, default=".", help="Path to parent project containing hdpgpc/.")
    parser.add_argument("--out-dir", type=str, default="results_equilibrium_flows_hdpgpc")

    # Simulation
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-traj-per-system", type=int, default=25)
    parser.add_argument("--T", type=int, default=60)
    parser.add_argument("--n-outputs", type=int, default=2)
    parser.add_argument("--equilibrium-distance", type=float, default=4.0)
    parser.add_argument("--rho0", type=float, default=0.960)
    parser.add_argument("--rho1", type=float, default=0.945)
    parser.add_argument("--init-std", type=float, default=0.35)
    parser.add_argument("--obs-noise", type=float, default=0.06)
    parser.add_argument("--drift-noise", type=float, default=0.015)
    parser.add_argument("--amplitude-log-std", type=float, default=0.10)
    parser.add_argument("--output-offset-std", type=float, default=0.08)
    parser.add_argument(
        "--nonlinear-observation-strength",
        type=float,
        default=0.65,
        help=(
            "Strength of a smooth nonlinear observation map applied after the "
            "latent contractive flow is generated. Use 0.0 to recover the "
            "purely linear observation model."
        ),
    )
    parser.add_argument(
        "--kernel-pca-gamma",
        type=float,
        default=None,
        help=(
            "Gamma for RBF Kernel PCA visualizations. If omitted, uses 1 / n_outputs."
        ),
    )

    # Static DP-GMM
    parser.add_argument("--skip-cloud-baseline", action="store_true")
    parser.add_argument("--skip-trajectory-static-baseline", action="store_true")

    parser.add_argument("--dpgmm-max-iter", type=int, default=800)
    parser.add_argument("--dpgmm-reg-covar", type=float, default=1e-5)

    parser.add_argument("--cloud-dpgmm-max-components", type=int, default=18)
    parser.add_argument("--cloud-dpgmm-alpha", type=float, default=0.025)
    parser.add_argument(
        "--cloud-dpgmm-covariance-type",
        type=str,
        default="full",
        choices=["full", "tied", "diag", "spherical"],
    )

    parser.add_argument("--trajectory-dpgmm-max-components", type=int, default=8)
    parser.add_argument("--trajectory-dpgmm-alpha", type=float, default=0.1)
    parser.add_argument("--trajectory-dpgmm-reg-covar", type=float, default=1e-4)
    parser.add_argument(
        "--trajectory-dpgmm-covariance-type",
        type=str,
        default="diag",
        choices=["full", "tied", "diag", "spherical"],
    )

    # HDP-GPC
    parser.add_argument("--skip-hdpgpc", action="store_true")
    parser.add_argument("--warp", type=str2bool, default=False)
    parser.add_argument("--warp-updating", type=str2bool, default=False)
    parser.add_argument("--method-compute-warp", type=str, default="greedy")
    parser.add_argument("--mode-warp", type=str, default="rough")
    parser.add_argument("--warp-basis-step", type=int, default=2)

    parser.add_argument("--hdpgpc-model-type", type=str, default="dynamic")
    parser.add_argument("--hmm-switch", type=str2bool, default=True)
    parser.add_argument("--max-models", type=int, default=100)
    parser.add_argument("--bayesian-params", type=str2bool, default=True)
    parser.add_argument("--inducing-points", type=str2bool, default=False)
    parser.add_argument("--reestimate-initial-params", type=str2bool, default=False)
    parser.add_argument("--n-explore-steps", type=int, default=10)
    parser.add_argument("--free-deg-MNIV", type=int, default=3)
    parser.add_argument("--share-gp", type=str2bool, default=True)
    parser.add_argument("--verbose-hdpgpc", action="store_true")
    parser.add_argument("--max-plot-outputs", type=int, default=4)

    # HDP-GPC priors
    parser.add_argument("--sigma-multiplier", type=float, default=0.1)
    parser.add_argument("--gamma-multiplier", type=float, default=0.2)
    parser.add_argument("--sigma-floor", type=float, default=1e-8)
    parser.add_argument("--bound-sigma-low-multiplier", type=float, default=1e-6)
    parser.add_argument("--bound-sigma-high-multiplier", type=float, default=1e-1)

    parser.add_argument("--outputscale", type=float, default=None)
    parser.add_argument("--outputscale-multiplier", type=float, default=1.2)
    parser.add_argument("--ini-lengthscale", type=float, default=8.0)
    parser.add_argument("--bound-lengthscale-low", type=float, default=1.0)
    parser.add_argument("--bound-lengthscale-high", type=float, default=45.0)

    # Kept only because GPI_HDP constructor expects warp-related priors even
    # when include_batch(..., warp=False).
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

    data, labels, metadata, e0, e1, start_center = simulate_equilibrium_flows(
        seed=args.seed,
        n_traj_per_system=args.n_traj_per_system,
        T=args.T,
        n_outputs=args.n_outputs,
        equilibrium_distance=args.equilibrium_distance,
        rho0=args.rho0,
        rho1=args.rho1,
        init_std=args.init_std,
        obs_noise=args.obs_noise,
        drift_noise=args.drift_noise,
        amplitude_log_std=args.amplitude_log_std,
        output_offset_std=args.output_offset_std,
        nonlinear_observation_strength=args.nonlinear_observation_strength,
    )

    np.save(out_dir / "synthetic_equilibrium_flow_data.npy", data)
    np.save(out_dir / "synthetic_equilibrium_flow_labels.npy", labels)
    np.save(out_dir / "equilibrium_0.npy", e0)
    np.save(out_dir / "equilibrium_1.npy", e1)
    np.save(out_dir / "start_center.npy", start_center)
    metadata.to_csv(out_dir / "synthetic_equilibrium_flow_metadata.csv", index=False)

    with open(out_dir / "simulation_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\nData:")
    print(f"  data.shape   = {data.shape}")
    print(f"  labels.shape = {labels.shape}")
    print(f"  label counts = {np.bincount(labels)}")
    print(f"  output dir   = {out_dir}")
    print(f"  HDP-GPC warp = {args.warp}")
    print(f"  nonlinear observation strength = {args.nonlinear_observation_strength}")

    save_pooled_pca_true_dynamics(
        data,
        labels,
        out_dir / "true_equilibrium_flows_pooled_pca_paths.png",
        "True high-dimensional flows projected to pooled PCA space",
    )

    save_pooled_kernel_pca_true_dynamics(
        data,
        labels,
        out_dir / "true_equilibrium_flows_pooled_kernel_pca_paths.png",
        "True high-dimensional flows projected to RBF Kernel PCA space",
        gamma=args.kernel_pca_gamma,
    )

    save_trajectory_pca(
        data,
        labels,
        out_dir / "true_equilibrium_flows_trajectory_pca.png",
        "Flattened trajectory PCA colored by true system",
    )

    save_output_examples(data, labels, out_dir, max_outputs=args.max_plot_outputs)

    results = []

    if not args.skip_cloud_baseline:
        print("\nRunning cloud-level static DP-GMM on pooled snapshots...")
        res = run_cloud_level_dpgmm(data, labels, out_dir, args)
        print(pd.DataFrame([res]).to_string(index=False))
        results.append(res)

    if not args.skip_trajectory_static_baseline:
        print("\nRunning trajectory-level static DP-GMM on flattened trajectories...")
        res = run_trajectory_level_dpgmm(data, labels, out_dir, args)
        print(pd.DataFrame([res]).to_string(index=False))
        results.append(res)

    if not args.skip_hdpgpc:
        print("\nRunning actual HDP-GPC with warp=False by default...")
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
                    "method": "actual_hdpgpc_gpi_hdp_no_warp",
                    "comparison_family": "trajectory_level_fair_comparison",
                    "input_used": "full high-dimensional equilibrium-flow trajectories",
                    "active_or_inferred_clusters": np.nan,
                    "ARI_trajectory_vs_true_system": np.nan,
                    "elapsed_min": np.nan,
                    "assignment_source": f"failed: {repr(exc)}",
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "comparison_results.csv", index=False)

    print("\nComparison:")
    print(results_df.to_string(index=False))

    print("\nSaved files:")
    for path in sorted(out_dir.iterdir()):
        print(" ", path)

    print(
        "\nInterpretation:\n"
        "  - The cloud-level static DP-GMM is expected to infer multiple snapshot\n"
        "    clusters along the smooth movement toward each equilibrium.\n"
        "  - This over-segmentation is not necessarily an error for a cloud model:\n"
        "    it is clustering locations/phases, not dynamical systems.\n"
        "  - HDP-GPC receives complete trajectories and should represent each whole\n"
        "    movement as a dynamic cluster, ideally reducing the result to the two\n"
        "    underlying equilibrium-flow regimes.\n"
        "  - No temporal warping is used. The difference comes from modeling ordered\n"
        "    functional/dynamical evolution instead of an unordered observation cloud.\n"
    )


if __name__ == "__main__":
    main()
