# -*- coding: utf-8 -*-
"""
Warped high-dimensional event-order simulation: a case where shape-normalized
static DP-GMM should struggle, while HDP-GPC with warping can recover the two
dynamic systems.

Core idea
---------
Each trajectory is high-dimensional:

    y_n(t) in R^D

There are two true systems. They contain the same two high-dimensional events,
but in opposite temporal order:

    System 0: event A -> event B
    System 1: event B -> event A

Every trajectory has:
    - random onset,
    - random tempo/duration,
    - nonlinear time warp,
    - random amplitude,
    - random per-output offset,
    - random per-output scaling,
    - noise.

Shape normalization removes amplitude and offset, but it does NOT remove random
temporal warping. Therefore a DP-GMM on flattened, shape-normalized trajectories
usually clusters by onset/tempo/warp artifacts rather than by the A->B vs B->A
dynamical template.

HDP-GPC, run with warp=True, receives the same trajectory tensor [N, T, D] but
can align warped trajectories and cluster the underlying smooth functional
templates.

This is a fairer argument than clustering a 2D cloud, because the main static
baseline and HDP-GPC both receive the same sample unit:

    one complete trajectory -> one cluster label.

Run
---
From the parent folder containing hdpgpc/:

    python simulate_warped_event_order_actual_hdpgpc.py

or:

    python simulate_warped_event_order_actual_hdpgpc.py --repo-root C:/Users/Adrian/Projects/YourProject

Useful variants:

    python simulate_warped_event_order_actual_hdpgpc.py --skip-hdpgpc
    python simulate_warped_event_order_actual_hdpgpc.py --n-traj-per-system 30 --T 90 --n-outputs 16
    python simulate_warped_event_order_actual_hdpgpc.py --time-warp-strength 0.65
    python simulate_warped_event_order_actual_hdpgpc.py --warp false

Expected qualitative result
---------------------------
- Static shape-normalized DP-GMM:
    low ARI or many components driven by timing nuisance.

- HDP-GPC with warp=True:
    should be much closer to the two true event-order systems, if the local
    HDP-GPC warping machinery succeeds on this synthetic data.

If HDP-GPC does not recover two clusters immediately, try:
    --n-traj-per-system 20 --T 70 --n-outputs 8
    --obs-noise 0.035
    --time-warp-strength 0.45
    --n-explore-steps 20
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
def make_event_loadings(n_outputs, seed=0):
    """
    Create two approximately orthogonal high-dimensional event signatures A and B.
    """
    rng = np.random.default_rng(seed)

    a = np.zeros(n_outputs)
    b = np.zeros(n_outputs)

    split = n_outputs // 2
    a[:split] = 1.0
    b[split:] = 1.0

    # Add deterministic smooth structure so the outputs are not just two blocks.
    grid = np.linspace(0, 2 * np.pi, n_outputs)
    a = a + 0.25 * np.sin(grid) + 0.15 * np.cos(2 * grid)
    b = b + 0.25 * np.cos(grid) - 0.15 * np.sin(2 * grid)

    # Small random perturbation to avoid a perfectly artificial structure.
    a = a + rng.normal(0.0, 0.03, size=n_outputs)
    b = b + rng.normal(0.0, 0.03, size=n_outputs)

    a = a / (np.linalg.norm(a) + 1e-12)
    b = b - np.dot(b, a) * a
    b = b / (np.linalg.norm(b) + 1e-12)

    return a, b


def nonlinear_time_warp(t, rng, onset_range, duration_range, strength):
    """
    Random monotone map from observed time t in [0,1] to latent time tau in [0,1].

    onset and duration create shifts and speed variation.
    gamma creates nonlinear acceleration/deceleration.
    """
    onset = rng.uniform(*onset_range)
    duration = rng.uniform(*duration_range)

    tau = np.clip((t - onset) / duration, 0.0, 1.0)

    # gamma < 1 accelerates early part; gamma > 1 delays it.
    gamma = np.exp(rng.normal(0.0, strength))
    tau = tau ** gamma

    return tau, onset, duration, gamma


def simulate_warped_event_order_systems(
    seed=42,
    n_traj_per_system=25,
    T=80,
    n_outputs=12,
    obs_noise=0.045,
    amplitude_log_std=0.70,
    offset_std=0.70,
    per_output_scale_std=0.20,
    loading_jitter=0.06,
    onset_low=-0.25,
    onset_high=0.25,
    duration_low=0.52,
    duration_high=1.50,
    time_warp_strength=0.55,
    pulse_width=0.065,
    attenuation=0.65,
):
    """
    High-dimensional event-order trajectories.

    System 0: A then B
    System 1: B then A

    Both systems have the same event signatures and the same nuisance
    distributions. Only the latent temporal order differs.
    """
    rng = np.random.default_rng(seed)

    event_a, event_b = make_event_loadings(n_outputs, seed=seed + 123)

    t = np.linspace(0.0, 1.0, T)

    data = []
    labels = []
    metadata = []

    for system in [0, 1]:
        for _ in range(n_traj_per_system):
            tau, onset, duration, gamma = nonlinear_time_warp(
                t,
                rng,
                onset_range=(onset_low, onset_high),
                duration_range=(duration_low, duration_high),
                strength=time_warp_strength,
            )

            width = pulse_width * np.exp(rng.normal(0.0, 0.12))
            early = np.exp(-0.5 * ((tau - 0.32) / width) ** 2)
            late = np.exp(-0.5 * ((tau - 0.68) / width) ** 2)

            # A mild attenuation envelope: the late event is slightly attenuated.
            env = np.exp(-attenuation * tau)

            if system == 0:
                template = early[:, None] * event_a[None, :] + late[:, None] * event_b[None, :]
            else:
                template = early[:, None] * event_b[None, :] + late[:, None] * event_a[None, :]

            template = template * env[:, None]

            # Nuisance: random perturbation of the output loadings.
            loading_noise = rng.normal(0.0, loading_jitter, size=n_outputs)
            per_output_scale = np.exp(rng.normal(0.0, per_output_scale_std, size=n_outputs))
            template = template * (per_output_scale + loading_noise)[None, :]

            amplitude = np.exp(rng.normal(0.0, amplitude_log_std))
            offset = rng.normal(0.0, offset_std, size=n_outputs)

            Y = offset[None, :] + amplitude * template
            Y = Y + rng.normal(0.0, obs_noise, size=Y.shape)

            data.append(Y)
            labels.append(system)
            metadata.append(
                {
                    "system": system,
                    "onset": onset,
                    "duration": duration,
                    "gamma": gamma,
                    "pulse_width": width,
                    "amplitude": amplitude,
                }
            )

    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    metadata = pd.DataFrame(metadata)

    return data, labels, metadata, event_a, event_b


def shape_normalize_trajectories(data, eps=1e-8):
    """
    Remove per-trajectory/per-output offset and global scale.

    This normalization deliberately does not align or warp time.
    """
    # Use temporal median to remove offsets robustly.
    centered = data - np.median(data, axis=1, keepdims=True)

    # Use one global trajectory norm so cross-output event ordering remains.
    scale = np.linalg.norm(centered, axis=(1, 2), keepdims=True)

    return centered / (scale + eps)


def project_events(data, event_a, event_b):
    """
    Project high-dimensional observations onto the two known event directions.
    Used only for diagnostic plots, not for fitting static DPGMM or HDP-GPC.
    """
    centered = data - np.median(data, axis=1, keepdims=True)
    pa = centered @ event_a
    pb = centered @ event_b
    return pa, pb


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def save_event_projection_plot(data, labels_for_color, event_a, event_b, out_path, title):
    pa, pb = project_events(data, event_a, event_b)

    plt.figure(figsize=(8.2, 5.2))
    for k in sorted(np.unique(labels_for_color)):
        idx = np.where(labels_for_color == k)[0]

        # Plot mean event projections per cluster/system.
        plt.plot(pa[idx].mean(axis=0), linewidth=3, label=f"A projection, group {k}")
        plt.plot(pb[idx].mean(axis=0), linewidth=3, linestyle="--", label=f"B projection, group {k}")

    plt.title(title)
    plt.xlabel("observed time index")
    plt.ylabel("median-centered projection")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_individual_projection_examples(data, labels, event_a, event_b, out_path, title, max_per_system=8):
    pa, pb = project_events(data, event_a, event_b)

    plt.figure(figsize=(8.2, 5.2))
    for k in sorted(np.unique(labels)):
        idx = np.where(labels == k)[0][:max_per_system]
        for i in idx:
            plt.plot(pa[i], alpha=0.35, linewidth=1.0)
            plt.plot(pb[i], alpha=0.35, linewidth=1.0, linestyle="--")

    plt.title(title)
    plt.xlabel("observed time index")
    plt.ylabel("median-centered projection")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_pca_plot(X, color_labels, out_path, title):
    if X.shape[0] < 3:
        return

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(7.4, 5.5))
    for k in sorted(np.unique(color_labels)):
        idx = np.where(color_labels == k)[0]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=38, alpha=0.82, label=f"group {k}")
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
            for i in idx[:12]:
                plt.plot(data[i, :, d], alpha=0.22, linewidth=1.0)
            plt.plot(data[idx, :, d].mean(axis=0), linewidth=3.0, label=f"system {k}")
        plt.title(f"Output {d}: true event-order systems")
        plt.xlabel("observed time index")
        plt.ylabel(f"y(t, output {d})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"true_systems_output{d}.png", dpi=180)
        plt.close()


# ---------------------------------------------------------------------
# Static baselines
# ---------------------------------------------------------------------
def run_static_dpgmm_shape_normalized(data, labels, event_a, event_b, out_dir, args):
    """
    Fair sample unit baseline:
        one normalized complete trajectory -> one label.

    This is the baseline that should struggle under strong time warping.
    """
    n_traj, T, D = data.shape

    data_norm = shape_normalize_trajectories(data)
    X = data_norm.reshape(n_traj, T * D)

    pred, active, _ = dp_gmm_fit_predict(
        X,
        max_components=args.trajectory_dpgmm_max_components,
        alpha=args.trajectory_dpgmm_alpha,
        max_iter=args.dpgmm_max_iter,
        reg_covar=args.trajectory_dpgmm_reg_covar,
        covariance_type=args.trajectory_dpgmm_covariance_type,
        seed=args.seed,
    )

    ari = adjusted_rand_score(labels, pred)

    np.save(out_dir / "static_shape_normalized_dpgmm_labels.npy", pred)
    pd.DataFrame(
        {
            "trajectory_index": np.arange(n_traj),
            "true_system": labels,
            "static_shape_normalized_dpgmm_cluster": pred,
        }
    ).to_csv(out_dir / "static_shape_normalized_dpgmm_labels.csv", index=False)

    save_pca_plot(
        X,
        pred,
        out_dir / "static_shape_normalized_dpgmm_pca_by_predicted_cluster.png",
        f"Shape-normalized flattened trajectories: DP-GMM clusters (ARI={ari:.3f})",
    )

    save_pca_plot(
        X,
        labels,
        out_dir / "static_shape_normalized_dpgmm_pca_by_true_system.png",
        "Shape-normalized flattened trajectories: true systems",
    )

    save_event_projection_plot(
        data,
        pred,
        event_a,
        event_b,
        out_dir / "static_shape_normalized_dpgmm_event_projections_by_cluster.png",
        f"Event projections colored by static shape-normalized DP-GMM (ARI={ari:.3f})",
    )

    return {
        "method": "trajectory_level_static_dp_gmm_shape_normalized",
        "comparison_family": "trajectory_level_fair_comparison",
        "input_used": "shape-normalized complete trajectories flattened to vectors",
        "active_or_inferred_clusters": active,
        "ARI_vs_true_system": float(ari),
        "assignment_source": "sklearn.BayesianGaussianMixture",
    }


def run_static_dpgmm_raw(data, labels, event_a, event_b, out_dir, args):
    """
    Additional fair baseline without shape normalization.
    """
    n_traj, T, D = data.shape
    X = data.reshape(n_traj, T * D)

    pred, active, _ = dp_gmm_fit_predict(
        X,
        max_components=args.trajectory_dpgmm_max_components,
        alpha=args.trajectory_dpgmm_alpha,
        max_iter=args.dpgmm_max_iter,
        reg_covar=args.trajectory_dpgmm_reg_covar,
        covariance_type=args.trajectory_dpgmm_covariance_type,
        seed=args.seed,
    )

    ari = adjusted_rand_score(labels, pred)

    np.save(out_dir / "static_raw_dpgmm_labels.npy", pred)
    pd.DataFrame(
        {
            "trajectory_index": np.arange(n_traj),
            "true_system": labels,
            "static_raw_dpgmm_cluster": pred,
        }
    ).to_csv(out_dir / "static_raw_dpgmm_labels.csv", index=False)

    save_pca_plot(
        X,
        pred,
        out_dir / "static_raw_dpgmm_pca_by_predicted_cluster.png",
        f"Raw flattened trajectories: DP-GMM clusters (ARI={ari:.3f})",
    )

    return {
        "method": "trajectory_level_static_dp_gmm_raw",
        "comparison_family": "trajectory_level_fair_comparison",
        "input_used": "raw complete trajectories flattened to vectors",
        "active_or_inferred_clusters": active,
        "ARI_vs_true_system": float(ari),
        "assignment_source": "sklearn.BayesianGaussianMixture",
    }


def run_cloud_level_dpgmm(data, labels, out_dir, args):
    """
    Diagnostic only: cluster individual D-dimensional observations y(t).
    """
    n_traj, T, D = data.shape
    points = data.reshape(n_traj * T, D)
    true_point_labels = np.repeat(labels, T)

    pred, active, _ = dp_gmm_fit_predict(
        points,
        max_components=args.cloud_dpgmm_max_components,
        alpha=args.cloud_dpgmm_alpha,
        max_iter=args.dpgmm_max_iter,
        reg_covar=args.dpgmm_reg_covar,
        covariance_type=args.cloud_dpgmm_covariance_type,
        seed=args.seed,
    )

    ari = adjusted_rand_score(true_point_labels, pred)

    return {
        "method": "cloud_level_static_dp_gmm_pooled_observations",
        "comparison_family": "cloud_level_diagnostic",
        "input_used": "pooled D-dimensional observations y(t)",
        "active_or_inferred_clusters": active,
        "ARI_vs_true_system": float(ari),
        "assignment_source": "sklearn.BayesianGaussianMixture",
    }


# ---------------------------------------------------------------------
# Actual HDP-GPC
# ---------------------------------------------------------------------
def run_actual_hdpgpc(data, labels, event_a, event_b, out_dir, args):
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
        hdp_hyp='min',
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

        save_event_projection_plot(
            data,
            pred,
            event_a,
            event_b,
            out_dir / "hdpgpc_event_projections_by_cluster.png",
            f"Event projections colored by HDP-GPC assignment (ARI={ari:.3f})",
        )

        X = shape_normalize_trajectories(data).reshape(num_samples, -1)
        save_pca_plot(
            X,
            pred,
            out_dir / "hdpgpc_labels_on_shape_normalized_pca.png",
            f"Shape-normalized trajectory PCA colored by HDP-GPC assignment (ARI={ari:.3f})",
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
        "input_used": "full high-dimensional warped trajectories",
        "active_or_inferred_clusters": active,
        "ARI_vs_true_system": float(ari) if not np.isnan(ari) else np.nan,
        "elapsed_min": float(elapsed_min),
        "assignment_source": source,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Warped high-dimensional event-order simulation with static DP-GMM and actual HDP-GPC."
    )

    parser.add_argument("--repo-root", type=str, default=".", help="Path to parent project containing hdpgpc/.")
    parser.add_argument("--out-dir", type=str, default="results_warped_event_order_hdpgpc")

    # Simulation
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-traj-per-system", type=int, default=25)
    parser.add_argument("--T", type=int, default=80)
    parser.add_argument("--n-outputs", type=int, default=12)
    parser.add_argument("--obs-noise", type=float, default=0.045)
    parser.add_argument("--amplitude-log-std", type=float, default=0.70)
    parser.add_argument("--offset-std", type=float, default=0.70)
    parser.add_argument("--per-output-scale-std", type=float, default=0.20)
    parser.add_argument("--loading-jitter", type=float, default=0.06)
    parser.add_argument("--onset-low", type=float, default=-0.25)
    parser.add_argument("--onset-high", type=float, default=0.25)
    parser.add_argument("--duration-low", type=float, default=0.52)
    parser.add_argument("--duration-high", type=float, default=1.50)
    parser.add_argument("--time-warp-strength", type=float, default=0.55)
    parser.add_argument("--pulse-width", type=float, default=0.065)
    parser.add_argument("--attenuation", type=float, default=0.65)

    # Static DP-GMM
    parser.add_argument("--skip-cloud-baseline", action="store_true")
    parser.add_argument("--skip-static-raw-baseline", action="store_true")
    parser.add_argument("--skip-static-shape-baseline", action="store_true")

    parser.add_argument("--dpgmm-max-iter", type=int, default=800)
    parser.add_argument("--dpgmm-reg-covar", type=float, default=1e-5)

    parser.add_argument("--cloud-dpgmm-max-components", type=int, default=15)
    parser.add_argument("--cloud-dpgmm-alpha", type=float, default=0.03)
    parser.add_argument(
        "--cloud-dpgmm-covariance-type",
        type=str,
        default="diag",
        choices=["full", "tied", "diag", "spherical"],
    )

    parser.add_argument("--trajectory-dpgmm-max-components", type=int, default=15)
    parser.add_argument("--trajectory-dpgmm-alpha", type=float, default=0.05)
    parser.add_argument("--trajectory-dpgmm-reg-covar", type=float, default=1e-5)
    parser.add_argument(
        "--trajectory-dpgmm-covariance-type",
        type=str,
        default="diag",
        choices=["full", "tied", "diag", "spherical"],
    )

    # HDP-GPC
    parser.add_argument("--skip-hdpgpc", action="store_true")
    parser.add_argument("--warp", type=str2bool, default=True)
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
    parser.add_argument("--n-explore-steps", type=int, default=3)
    parser.add_argument("--free-deg-MNIV", type=int, default=3)
    parser.add_argument("--share-gp", type=str2bool, default=True)
    parser.add_argument("--verbose-hdpgpc", action="store_true")
    parser.add_argument("--max-plot-outputs", type=int, default=4)

    # HDP-GPC priors
    parser.add_argument("--sigma-multiplier", type=float, default=0.0005)
    parser.add_argument("--gamma-multiplier", type=float, default=0.0001)
    parser.add_argument("--sigma-floor", type=float, default=1e-8)
    parser.add_argument("--bound-sigma-low-multiplier", type=float, default=1e-6)
    parser.add_argument("--bound-sigma-high-multiplier", type=float, default=1e-1)

    parser.add_argument("--outputscale", type=float, default=None)
    parser.add_argument("--outputscale-multiplier", type=float, default=1.2)
    parser.add_argument("--ini-lengthscale", type=float, default=8.0)
    parser.add_argument("--bound-lengthscale-low", type=float, default=1.0)
    parser.add_argument("--bound-lengthscale-high", type=float, default=45.0)

    parser.add_argument("--noise-warp-multiplier", type=float, default=1e8)
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

    data, labels, metadata, event_a, event_b = simulate_warped_event_order_systems(
        seed=args.seed,
        n_traj_per_system=args.n_traj_per_system,
        T=args.T,
        n_outputs=args.n_outputs,
        obs_noise=args.obs_noise,
        amplitude_log_std=args.amplitude_log_std,
        offset_std=args.offset_std,
        per_output_scale_std=args.per_output_scale_std,
        loading_jitter=args.loading_jitter,
        onset_low=args.onset_low,
        onset_high=args.onset_high,
        duration_low=args.duration_low,
        duration_high=args.duration_high,
        time_warp_strength=args.time_warp_strength,
        pulse_width=args.pulse_width,
        attenuation=args.attenuation,
    )

    np.save(out_dir / "synthetic_warped_event_order_data.npy", data)
    np.save(out_dir / "synthetic_warped_event_order_labels.npy", labels)
    np.save(out_dir / "event_loading_A.npy", event_a)
    np.save(out_dir / "event_loading_B.npy", event_b)
    metadata.to_csv(out_dir / "synthetic_warped_event_order_metadata.csv", index=False)

    with open(out_dir / "simulation_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\nData:")
    print(f"  data.shape   = {data.shape}")
    print(f"  labels.shape = {labels.shape}")
    print(f"  label counts = {np.bincount(labels)}")
    print(f"  output dir   = {out_dir}")

    save_individual_projection_examples(
        data,
        labels,
        event_a,
        event_b,
        out_dir / "individual_event_projection_examples_true_systems.png",
        "Individual warped trajectories projected onto event A and B directions",
    )

    save_event_projection_plot(
        data,
        labels,
        event_a,
        event_b,
        out_dir / "mean_event_projections_true_systems.png",
        "Mean event projections by true system",
    )

    save_output_examples(data, labels, out_dir, max_outputs=args.max_plot_outputs)

    X_shape = shape_normalize_trajectories(data).reshape(data.shape[0], -1)
    save_pca_plot(
        X_shape,
        labels,
        out_dir / "shape_normalized_trajectory_pca_by_true_system.png",
        "Shape-normalized flattened trajectories colored by true system",
    )

    results = []

    if not args.skip_cloud_baseline:
        print("\nRunning cloud-level pooled-observation DP-GMM diagnostic...")
        res = run_cloud_level_dpgmm(data, labels, out_dir, args)
        print(pd.DataFrame([res]).to_string(index=False))
        results.append(res)

    if not args.skip_static_raw_baseline:
        print("\nRunning trajectory-level static DP-GMM on raw flattened trajectories...")
        res = run_static_dpgmm_raw(data, labels, event_a, event_b, out_dir, args)
        print(pd.DataFrame([res]).to_string(index=False))
        results.append(res)

    if not args.skip_static_shape_baseline:
        print("\nRunning trajectory-level static DP-GMM on shape-normalized flattened trajectories...")
        res = run_static_dpgmm_shape_normalized(data, labels, event_a, event_b, out_dir, args)
        print(pd.DataFrame([res]).to_string(index=False))
        results.append(res)

    if not args.skip_hdpgpc:
        print("\nRunning actual HDP-GPC...")
        try:
            ensure_repo_on_path(args.repo_root)
            res = run_actual_hdpgpc(data, labels, event_a, event_b, out_dir, args)
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
                    "input_used": "full high-dimensional warped trajectories",
                    "active_or_inferred_clusters": np.nan,
                    "ARI_vs_true_system": np.nan,
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
        "  - This is a trajectory-level fair comparison: static DP-GMM and HDP-GPC\n"
        "    both receive one complete high-dimensional trajectory per sample.\n"
        "  - Shape normalization removes amplitude and offset, but not temporal warping.\n"
        "  - Static flattened DP-GMM is expected to split trajectories by nuisance\n"
        "    onset/duration/gamma effects, especially when time_warp_strength is large.\n"
        "  - HDP-GPC with warp=True should be better matched to the data-generating\n"
        "    process because it can align smooth warped trajectories while clustering\n"
        "    the underlying event-order templates A->B and B->A.\n"
    )


if __name__ == "__main__":
    main()
