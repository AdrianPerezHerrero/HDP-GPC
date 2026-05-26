# -*- coding: utf-8 -*-
"""
Attractor + transient-excursion simulation with static DP-GMM and actual HDP-GPC.

No warping is used.

Purpose
-------
This simulation is designed for the following argument:

    Static clustering of pooled observations can mistake a transient modulation
    along a dynamical trajectory for a separate cluster.

    A trajectory/dynamical method such as HDP-GPC can keep the transient as part
    of the same ordered behaviour, reducing the inferred structure to the true
    dynamical regimes.

Data-generating story
---------------------
There are two latent attractor systems. Their observations can be quite similar
because the equilibria are close and the same linear observation matrix is used.

System 0:
    smooth contraction toward attractor e0.

System 1:
    smooth contraction toward attractor e1, but around the middle of the
    trajectory it receives a smooth transient perturbation that pushes it away
    from the attractor and then it returns.

In latent space:

    z_n(t) = e_k + rho_k^t (z_n(0) - e_k)
             + 1{k=1} * pulse(t) * q
             + noise

Observed data are a linear projection/mixing of the latent state:

    y_n(t) = W z_n(t) + output_offset_n + noise

The transient pulse is not a third dynamical regime; it is a phase of system 1.
A static cloud method may nonetheless identify it as a third snapshot cluster.

Comparison
----------
1. Cloud-level static DP-GMM on pooled snapshots y_n(t).
   Expected: over-segments into attractor/transition/transient snapshot clusters.

2. Trajectory-level static DP-GMM on flattened full trajectories.
   Fair sample-unit baseline; may or may not recover the two trajectory regimes.

3. Actual HDP-GPC on [N, T, D], with warp=False.
   Expected: ideally two trajectory clusters, preserving the transient as part
   of system 1.

Run
---
From the parent folder containing hdpgpc/:

    python simulate_attractor_transient_actual_hdpgpc.py

or:

    python simulate_attractor_transient_actual_hdpgpc.py --repo-root C:/Users/Adrian/Projects/YourProject

Useful variants:

    python simulate_attractor_transient_actual_hdpgpc.py --skip-hdpgpc
    python simulate_attractor_transient_actual_hdpgpc.py --transient-amplitude 2.2
    python simulate_attractor_transient_actual_hdpgpc.py --equilibrium-distance 1.25
    python simulate_attractor_transient_actual_hdpgpc.py --cloud-dpgmm-alpha 0.02 --cloud-dpgmm-max-components 16

Expected qualitative output
---------------------------
- true_attractor_transient_pooled_pca_paths.png:
    two very similar attractor paths, with one system making a transient lobe.

- cloud_static_dpgmm_pooled_pca_clusters.png:
    static snapshot clusters often include a separate cluster on the transient lobe.

- hdpgpc_pooled_pca_paths_by_cluster.png:
    HDP-GPC trajectory labels should group whole paths by dynamic regime.
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


def count_assigned_clusters(labels):
    return int(len(np.unique(np.asarray(labels))))


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

    for name in [
        "labels", "label", "z", "Z", "assignments", "assignment",
        "cluster_assignments", "model_assignments", "selected_models",
        "selected_model", "model_index", "model_indices", "main_model",
        "model_id", "model_ids", "d", "s",
    ]:
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
# Linear observation model and simulation
# ---------------------------------------------------------------------
def orthonormal_basis(latent_dim, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0, size=(latent_dim, 4))
    Q, _ = np.linalg.qr(A)
    return Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]


def make_linear_observation_matrix(latent_dim, n_outputs, seed=0, projection_condition=3.0):
    """
    Fixed linear observation matrix W for y(t)=Wz(t)+noise.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0, size=(n_outputs, latent_dim))
    U, _, Vt = np.linalg.svd(A, full_matrices=False)

    rank = min(n_outputs, latent_dim)
    if rank == 1:
        singular_values = np.array([1.0])
    else:
        singular_values = np.geomspace(projection_condition, 1.0, rank)

    W = U[:, :rank] @ np.diag(singular_values) @ Vt[:rank, :]
    W = W / np.sqrt(np.mean(W**2) * latent_dim + 1e-12)
    return W


def smooth_pulse(t, center=0.55, width=0.085):
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def simulate_attractor_transient(
    seed=42,
    n_traj_per_system=30,
    T=70,
    latent_dim=6,
    n_outputs=14,
    equilibrium_distance=1.35,
    rho0=0.955,
    rho1=0.955,
    init_std=0.22,
    obs_noise=0.045,
    drift_noise=0.012,
    output_offset_std=0.05,
    amplitude_log_std=0.06,
    transient_amplitude=2.15,
    transient_center=0.56,
    transient_width=0.075,
    projection_condition=3.0,
):
    """
    Generate two attractor systems.

    System 0: contraction toward e0.
    System 1: contraction toward e1 plus a smooth transient excursion.

    The transient is part of system 1, not a separate true cluster.
    """
    rng = np.random.default_rng(seed)

    u_main, u_shared, u_transient, u_curve = orthonormal_basis(latent_dim, seed=seed + 100)

    # Two attractors are close enough that observations are similar, but separated.
    e0 = equilibrium_distance * (+0.5 * u_main + 0.35 * u_shared)
    e1 = equilibrium_distance * (-0.5 * u_main + 0.35 * u_shared)

    # Common start, so early snapshots from both systems are very similar.
    start_center = equilibrium_distance * (-0.75 * u_shared + 0.15 * u_curve)

    # Transient excursion direction is almost orthogonal to the equilibrium separation.
    q = u_transient + 0.25 * u_curve
    q = q / (np.linalg.norm(q) + 1e-12)

    W_obs = make_linear_observation_matrix(
        latent_dim=latent_dim,
        n_outputs=n_outputs,
        seed=seed + 200,
        projection_condition=projection_condition,
    )

    rhos = [rho0, rho1]
    equilibria = [e0, e1]
    normalized_time = np.linspace(0.0, 1.0, T)
    pulse = smooth_pulse(normalized_time, center=transient_center, width=transient_width)

    data = []
    labels = []
    transient_indicator = []
    metadata = []

    for system in [0, 1]:
        rho = rhos[system]
        eq = equilibria[system]

        for n in range(n_traj_per_system):
            z0 = start_center + rng.normal(0.0, init_std, size=latent_dim)

            amp = np.exp(rng.normal(0.0, amplitude_log_std))
            output_offset = rng.normal(0.0, output_offset_std, size=n_outputs)

            Z = []
            drift = np.zeros(latent_dim)

            for ti, tau in enumerate(normalized_time):
                base = eq + (rho ** ti) * (z0 - eq)

                if system == 1:
                    # Smooth excursion and return.
                    transient = transient_amplitude * pulse[ti] * q
                else:
                    transient = 0.0

                drift = 0.85 * drift + rng.normal(0.0, drift_noise, size=latent_dim)

                z_t = amp * (base + transient) + drift
                Z.append(z_t)

            Z = np.asarray(Z, dtype=np.float64)
            Y = Z @ W_obs.T
            Y = output_offset[None, :] + Y
            Y = Y + rng.normal(0.0, obs_noise, size=Y.shape)

            data.append(Y)
            labels.append(system)

            # 1 indicates the artificially perturbed transient segment for system 1.
            transient_indicator.append((system == 1) * (pulse > 0.35).astype(int))

            metadata.append(
                {
                    "system": system,
                    "rho": rho,
                    "amplitude": amp,
                    "transient_amplitude": transient_amplitude if system == 1 else 0.0,
                }
            )

    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    transient_indicator = np.asarray(transient_indicator, dtype=int)
    metadata = pd.DataFrame(metadata)

    params = {
        "e0": e0,
        "e1": e1,
        "start_center": start_center,
        "transient_direction": q,
        "W_obs": W_obs,
        "pulse": pulse,
    }

    return data, labels, transient_indicator, metadata, params


# ---------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------
def fit_pooled_pca(data):
    points = data.reshape(-1, data.shape[-1])
    pca = PCA(n_components=2, random_state=0)
    pca.fit(points)
    return pca


def fit_pooled_kernel_pca(data, gamma=None):
    points = data.reshape(-1, data.shape[-1])
    if gamma is None:
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


def save_pooled_projection_paths(data, traj_labels, out_path, title, method="pca", gamma=None):
    n_traj, T, D = data.shape

    if method == "pca":
        reducer = fit_pooled_pca(data)
        xlabel, ylabel = "pooled PCA 1", "pooled PCA 2"
    elif method == "kpca":
        reducer = fit_pooled_kernel_pca(data, gamma=gamma)
        xlabel, ylabel = "RBF Kernel PCA 1", "RBF Kernel PCA 2"
    else:
        raise ValueError("method must be 'pca' or 'kpca'")

    Z = reducer.transform(data.reshape(-1, D)).reshape(n_traj, T, 2)

    plt.figure(figsize=(8.0, 5.9))
    for k in sorted(np.unique(traj_labels)):
        idx = np.where(traj_labels == k)[0]
        for i in idx:
            plt.plot(Z[i, :, 0], Z[i, :, 1], alpha=0.30, linewidth=1.1)
            plt.scatter(Z[i, -1, 0], Z[i, -1, 1], s=13, alpha=0.60)
        mean_path = Z[idx].mean(axis=0)
        plt.plot(mean_path[:, 0], mean_path[:, 1], linewidth=3.0, label=f"group {k}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_pooled_projection_point_clusters(data, point_clusters, out_path, title, method="pca", gamma=None):
    n_traj, T, D = data.shape
    points = data.reshape(-1, D)

    if method == "pca":
        reducer = PCA(n_components=2, random_state=0)
        Z = reducer.fit_transform(points)
        xlabel, ylabel = "pooled PCA 1", "pooled PCA 2"
    elif method == "kpca":
        reducer = fit_pooled_kernel_pca(data, gamma=gamma)
        Z = reducer.transform(points)
        xlabel, ylabel = "RBF Kernel PCA 1", "RBF Kernel PCA 2"
    else:
        raise ValueError("method must be 'pca' or 'kpca'")

    plt.figure(figsize=(8.0, 5.9))
    for k in sorted(np.unique(point_clusters)):
        idx = np.where(point_clusters == k)[0]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=9, alpha=0.50, label=f"cluster {k}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(np.unique(point_clusters)) <= 14:
        plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_transient_indicator_plot(data, labels, transient_indicator, out_path, method="pca", gamma=None):
    """
    Diagnostic plot: show the true transient segment, not used for fitting.
    """
    n_traj, T, D = data.shape
    points = data.reshape(-1, D)
    transient_flat = transient_indicator.reshape(-1)
    system_flat = np.repeat(labels, T)

    if method == "pca":
        reducer = PCA(n_components=2, random_state=0)
        Z = reducer.fit_transform(points)
        xlabel, ylabel = "pooled PCA 1", "pooled PCA 2"
    else:
        reducer = fit_pooled_kernel_pca(data, gamma=gamma)
        Z = reducer.transform(points)
        xlabel, ylabel = "RBF Kernel PCA 1", "RBF Kernel PCA 2"

    plt.figure(figsize=(8.0, 5.9))
    non_transient = transient_flat == 0
    transient = transient_flat == 1

    plt.scatter(
        Z[non_transient, 0],
        Z[non_transient, 1],
        c=system_flat[non_transient],
        s=8,
        alpha=0.25,
        label="non-transient snapshots",
    )
    plt.scatter(
        Z[transient, 0],
        Z[transient, 1],
        s=20,
        alpha=0.85,
        marker="x",
        label="true transient segment",
    )

    plt.title("True transient segment over pooled observation geometry")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
        plt.title(f"Output {d}: attractor systems with transient excursion")
        plt.xlabel("time index")
        plt.ylabel(f"y(t, output {d})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"true_systems_output{d}.png", dpi=180)
        plt.close()


def save_pulse_plot(pulse, out_path):
    plt.figure(figsize=(7.0, 3.5))
    plt.plot(pulse, linewidth=2.5)
    plt.title("Transient perturbation pulse used only in system 1")
    plt.xlabel("time index")
    plt.ylabel("pulse amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ---------------------------------------------------------------------
# Static baselines
# ---------------------------------------------------------------------
def run_cloud_level_dpgmm(data, labels, transient_indicator, out_dir, args):
    """
    Diagnostic: cluster pooled snapshots, ignoring trajectory identity.
    This is where the transient lobe should tend to become its own static cluster.
    """
    n_traj, T, D = data.shape
    points = data.reshape(n_traj * T, D)
    point_true_system = np.repeat(labels, T)
    transient_flat = transient_indicator.reshape(-1)

    pred, active_by_weight, model = dp_gmm_fit_predict(
        points,
        max_components=args.cloud_dpgmm_max_components,
        alpha=args.cloud_dpgmm_alpha,
        max_iter=args.dpgmm_max_iter,
        reg_covar=args.dpgmm_reg_covar,
        covariance_type=args.cloud_dpgmm_covariance_type,
        seed=args.seed,
    )

    assigned = count_assigned_clusters(pred)
    point_ari = adjusted_rand_score(point_true_system, pred)

    pred_by_traj = pred.reshape(n_traj, T)
    components_per_traj = np.array([len(np.unique(pred_by_traj[i])) for i in range(n_traj)])

    # How concentrated is the true transient in particular static clusters?
    transient_cluster_counts = pd.crosstab(
        pd.Series(pred[transient_flat == 1], name="cloud_cluster"),
        pd.Series(transient_flat[transient_flat == 1], name="is_transient"),
    )

    np.save(out_dir / "cloud_static_dpgmm_point_labels.npy", pred)
    pd.DataFrame(
        {
            "point_index": np.arange(n_traj * T),
            "trajectory_index": np.repeat(np.arange(n_traj), T),
            "time_index": np.tile(np.arange(T), n_traj),
            "true_system": point_true_system,
            "true_transient_segment": transient_flat,
            "cloud_static_dpgmm_cluster": pred,
        }
    ).to_csv(out_dir / "cloud_static_dpgmm_point_labels.csv", index=False)

    transient_cluster_counts.to_csv(out_dir / "cloud_static_dpgmm_transient_cluster_counts.csv")

    save_pooled_projection_point_clusters(
        data,
        pred,
        out_dir / "cloud_static_dpgmm_pooled_pca_clusters.png",
        f"Cloud DP-GMM: {assigned} assigned snapshot clusters, {active_by_weight} active-by-weight",
        method="pca",
    )

    save_pooled_projection_point_clusters(
        data,
        pred,
        out_dir / "cloud_static_dpgmm_pooled_kernel_pca_clusters.png",
        f"Cloud DP-GMM in Kernel PCA: {assigned} assigned snapshot clusters",
        method="kpca",
        gamma=args.kernel_pca_gamma,
    )

    return {
        "method": "cloud_level_static_dp_gmm_snapshots",
        "comparison_family": "cloud_level_diagnostic",
        "input_used": "pooled D-dimensional snapshots y(t)",
        "assigned_clusters": assigned,
        "active_components_by_weight": active_by_weight,
        "active_or_inferred_clusters": assigned,
        "ARI_point_clusters_vs_true_system": float(point_ari),
        "mean_snapshot_clusters_per_trajectory": float(components_per_traj.mean()),
        "median_snapshot_clusters_per_trajectory": float(np.median(components_per_traj)),
        "assignment_source": "sklearn.BayesianGaussianMixture",
    }


def run_trajectory_level_dpgmm(data, labels, out_dir, args):
    """
    Fair sample-unit static baseline: one complete trajectory is one flattened vector.
    """
    n_traj, T, D = data.shape
    X = data.reshape(n_traj, T * D)

    pred, active_by_weight, model = dp_gmm_fit_predict(
        X,
        max_components=args.trajectory_dpgmm_max_components,
        alpha=args.trajectory_dpgmm_alpha,
        max_iter=args.dpgmm_max_iter,
        reg_covar=args.trajectory_dpgmm_reg_covar,
        covariance_type=args.trajectory_dpgmm_covariance_type,
        seed=args.seed,
    )

    assigned = count_assigned_clusters(pred)
    ari = adjusted_rand_score(labels, pred)

    np.save(out_dir / "trajectory_static_dpgmm_labels.npy", pred)
    pd.DataFrame(
        {
            "trajectory_index": np.arange(n_traj),
            "true_system": labels,
            "trajectory_static_dpgmm_cluster": pred,
        }
    ).to_csv(out_dir / "trajectory_static_dpgmm_labels.csv", index=False)

    save_pooled_projection_paths(
        data,
        pred,
        out_dir / "trajectory_static_dpgmm_pooled_pca_paths.png",
        f"Trajectory static DP-GMM paths: {assigned} clusters (ARI={ari:.3f})",
        method="pca",
    )

    save_pooled_projection_paths(
        data,
        pred,
        out_dir / "trajectory_static_dpgmm_pooled_kernel_pca_paths.png",
        f"Trajectory static DP-GMM paths in Kernel PCA: {assigned} clusters (ARI={ari:.3f})",
        method="kpca",
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
        "assigned_clusters": assigned,
        "active_components_by_weight": active_by_weight,
        "active_or_inferred_clusters": assigned,
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
        assigned = count_assigned_clusters(pred)

        np.save(out_dir / "hdpgpc_extracted_trajectory_labels.npy", pred)
        pd.DataFrame(
            {
                "trajectory_index": np.arange(num_samples),
                "true_system": labels,
                "hdpgpc_cluster": pred,
            }
        ).to_csv(out_dir / "hdpgpc_extracted_trajectory_labels.csv", index=False)

        save_pooled_projection_paths(
            data,
            pred,
            out_dir / "hdpgpc_pooled_pca_paths_by_cluster.png",
            f"HDP-GPC trajectory clusters in PCA space ({assigned} clusters, ARI={ari:.3f})",
            method="pca",
        )

        save_pooled_projection_paths(
            data,
            pred,
            out_dir / "hdpgpc_pooled_kernel_pca_paths_by_cluster.png",
            f"HDP-GPC trajectory clusters in Kernel PCA space ({assigned} clusters, ARI={ari:.3f})",
            method="kpca",
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
        assigned = np.nan
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
        "input_used": "full attractor-transient trajectories",
        "assigned_clusters": assigned,
        "active_components_by_weight": np.nan,
        "active_or_inferred_clusters": assigned,
        "ARI_trajectory_vs_true_system": float(ari) if not np.isnan(ari) else np.nan,
        "elapsed_min": float(elapsed_min),
        "assignment_source": source,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Attractor + transient-excursion simulation with static DP-GMM and actual HDP-GPC."
    )

    parser.add_argument("--repo-root", type=str, default=".", help="Path to parent project containing hdpgpc/.")
    parser.add_argument("--out-dir", type=str, default="results_attractor_transient_hdpgpc")

    # Simulation
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-traj-per-system", type=int, default=30)
    parser.add_argument("--T", type=int, default=70)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--n-outputs", type=int, default=14)
    parser.add_argument("--equilibrium-distance", type=float, default=1.35)
    parser.add_argument("--rho0", type=float, default=0.955)
    parser.add_argument("--rho1", type=float, default=0.955)
    parser.add_argument("--init-std", type=float, default=0.22)
    parser.add_argument("--obs-noise", type=float, default=0.045)
    parser.add_argument("--drift-noise", type=float, default=0.012)
    parser.add_argument("--output-offset-std", type=float, default=0.05)
    parser.add_argument("--amplitude-log-std", type=float, default=0.06)
    parser.add_argument("--transient-amplitude", type=float, default=2.15)
    parser.add_argument("--transient-center", type=float, default=0.56)
    parser.add_argument("--transient-width", type=float, default=0.075)
    parser.add_argument("--projection-condition", type=float, default=3.0)

    # Static DP-GMM
    parser.add_argument("--skip-cloud-baseline", action="store_true")
    parser.add_argument("--skip-trajectory-static-baseline", action="store_true")
    parser.add_argument("--dpgmm-max-iter", type=int, default=800)
    parser.add_argument("--dpgmm-reg-covar", type=float, default=1e-5)

    parser.add_argument("--cloud-dpgmm-max-components", type=int, default=16)
    parser.add_argument("--cloud-dpgmm-alpha", type=float, default=0.02)
    parser.add_argument(
        "--cloud-dpgmm-covariance-type",
        type=str,
        default="full",
        choices=["full", "tied", "diag", "spherical"],
    )

    parser.add_argument("--trajectory-dpgmm-max-components", type=int, default=8)
    parser.add_argument("--trajectory-dpgmm-alpha", type=float, default=0.08)
    parser.add_argument("--trajectory-dpgmm-reg-covar", type=float, default=1e-4)
    parser.add_argument(
        "--trajectory-dpgmm-covariance-type",
        type=str,
        default="diag",
        choices=["full", "tied", "diag", "spherical"],
    )

    # Visualization
    parser.add_argument(
        "--kernel-pca-gamma",
        type=float,
        default=None,
        help="Gamma for RBF Kernel PCA plots. If omitted, uses 1 / n_outputs.",
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
    parser.add_argument("--sigma-multiplier", type=float, default=2.0)
    parser.add_argument("--gamma-multiplier", type=float, default=1.0)
    parser.add_argument("--sigma-floor", type=float, default=1e-8)
    parser.add_argument("--bound-sigma-low-multiplier", type=float, default=1e-6)
    parser.add_argument("--bound-sigma-high-multiplier", type=float, default=1e-4)

    parser.add_argument("--outputscale", type=float, default=None)
    parser.add_argument("--outputscale-multiplier", type=float, default=1.2)
    parser.add_argument("--ini-lengthscale", type=float, default=8.0)
    parser.add_argument("--bound-lengthscale-low", type=float, default=1.0)
    parser.add_argument("--bound-lengthscale-high", type=float, default=45.0)

    # GPI_HDP constructor expects warp-related priors even with warp=False.
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

    data, labels, transient_indicator, metadata, params = simulate_attractor_transient(
        seed=args.seed,
        n_traj_per_system=args.n_traj_per_system,
        T=args.T,
        latent_dim=args.latent_dim,
        n_outputs=args.n_outputs,
        equilibrium_distance=args.equilibrium_distance,
        rho0=args.rho0,
        rho1=args.rho1,
        init_std=args.init_std,
        obs_noise=args.obs_noise,
        drift_noise=args.drift_noise,
        output_offset_std=args.output_offset_std,
        amplitude_log_std=args.amplitude_log_std,
        transient_amplitude=args.transient_amplitude,
        transient_center=args.transient_center,
        transient_width=args.transient_width,
        projection_condition=args.projection_condition,
    )

    np.save(out_dir / "synthetic_attractor_transient_data.npy", data)
    np.save(out_dir / "synthetic_attractor_transient_labels.npy", labels)
    np.save(out_dir / "synthetic_attractor_transient_indicator.npy", transient_indicator)
    np.save(out_dir / "linear_observation_matrix.npy", params["W_obs"])
    np.save(out_dir / "equilibrium_0.npy", params["e0"])
    np.save(out_dir / "equilibrium_1.npy", params["e1"])
    np.save(out_dir / "start_center.npy", params["start_center"])
    np.save(out_dir / "transient_direction.npy", params["transient_direction"])
    np.save(out_dir / "transient_pulse.npy", params["pulse"])
    metadata.to_csv(out_dir / "synthetic_attractor_transient_metadata.csv", index=False)

    with open(out_dir / "simulation_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\nData:")
    print(f"  data.shape   = {data.shape}")
    print(f"  labels.shape = {labels.shape}")
    print(f"  label counts = {np.bincount(labels)}")
    print(f"  output dir   = {out_dir}")
    print(f"  HDP-GPC warp = {args.warp}")
    print(f"  transient amplitude = {args.transient_amplitude}")

    # True dynamic geometry.
    save_pooled_projection_paths(
        data,
        labels,
        out_dir / "true_attractor_transient_pooled_pca_paths.png",
        "True attractor systems projected to pooled PCA space",
        method="pca",
    )
    save_pooled_projection_paths(
        data,
        labels,
        out_dir / "true_attractor_transient_pooled_kernel_pca_paths.png",
        "True attractor systems projected to RBF Kernel PCA space",
        method="kpca",
        gamma=args.kernel_pca_gamma,
    )
    save_transient_indicator_plot(
        data,
        labels,
        transient_indicator,
        out_dir / "true_transient_segment_pooled_pca.png",
        method="pca",
    )
    save_transient_indicator_plot(
        data,
        labels,
        transient_indicator,
        out_dir / "true_transient_segment_pooled_kernel_pca.png",
        method="kpca",
        gamma=args.kernel_pca_gamma,
    )
    save_trajectory_pca(
        data,
        labels,
        out_dir / "true_attractor_transient_trajectory_pca.png",
        "Flattened trajectory PCA colored by true dynamic system",
    )
    save_output_examples(data, labels, out_dir, max_outputs=args.max_plot_outputs)
    save_pulse_plot(params["pulse"], out_dir / "transient_pulse.png")

    results = []

    if not args.skip_cloud_baseline:
        print("\nRunning cloud-level static DP-GMM on pooled snapshots...")
        res = run_cloud_level_dpgmm(data, labels, transient_indicator, out_dir, args)
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
                    "input_used": "full attractor-transient trajectories",
                    "assigned_clusters": np.nan,
                    "active_components_by_weight": np.nan,
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
        "  - The cloud-level static DP-GMM clusters individual snapshots. It may create\n"
        "    a distinct snapshot cluster for the transient excursion, even though that\n"
        "    excursion is only a phase of system 1.\n"
        "  - HDP-GPC receives complete ordered trajectories. The desired behaviour is\n"
        "    to keep the excursion inside the same trajectory-level dynamic cluster\n"
        "    rather than declaring it a separate global cluster.\n"
        "  - No temporal warping is used. The observation model is linear: y(t)=Wz(t)+noise.\n"
        "  - Use cloud_static_dpgmm_transient_cluster_counts.csv to check whether the\n"
        "    static method isolated the transient segment as one or more clusters.\n"
    )


if __name__ == "__main__":
    main()
